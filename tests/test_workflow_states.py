"""Workflow state transitions: step outcomes and claim status, including every allowed and refused move."""

from claimflow.agents.base import Agent
from claimflow.errors import AgentError, WorkflowError
from claimflow.models import (
    ALLOWED_TRANSITIONS, AgentResult, ClaimStatus, ReviewDecision, StepStatus, can_transition,
)
from claimflow.orchestrator import WorkflowOrchestrator
from claimflow.pipeline import PIPELINE, step_label
from tests.helpers import make_claim, quiet_logs, temp_container


class Stub(Agent):
    def __init__(self, name, behavior="ok", probe=None):
        self.name, self.label, self.behavior, self.probe = name, name, behavior, probe

    def run(self, ctx):
        if self.probe:
            self.probe()
        if self.behavior == "boom":
            raise RuntimeError("internal detail")
        if self.behavior == "fail":
            raise AgentError("Could not finish.", hint="Try again.")
        if self.behavior == "warn":
            return self.warning("careful", ["a note"])
        if self.behavior == "skip":
            return self.skipped("nothing to do")
        return self.done("fine")


def build(c, probe=None, **behaviors):
    agents = [Stub(s.name, behaviors.get(s.name, "ok"), probe if s.name == "intake" else None) for s in PIPELINE]
    return WorkflowOrchestrator(c.db, c.llm, c.settings, agents)


def raises(exc_type, fn, *args):
    try:
        fn(*args)
    except exc_type as exc:
        return exc
    raise AssertionError(f"{exc_type.__name__} was not raised")


# ---------------------------------------------------------------- step outcomes

def test_only_done_and_warning_count_as_finished():
    finished = {s for s in StepStatus if AgentResult("x", s).succeeded}
    assert finished == {StepStatus.DONE, StepStatus.WARNING}


def test_success_warning_failure_and_skipped_are_all_produced_stored_and_counted():
    with temp_container() as (c, _):
        orchestrator = build(c, policy="warn", damage="skip", evidence="boom")
        claim = make_claim(c.db)
        with quiet_logs():
            summary = orchestrator.run(claim.id)
        statuses = {name: r.status for name, r in summary.results.items()}
        assert statuses["intake"] == StepStatus.DONE and statuses["policy"] == StepStatus.WARNING
        assert statuses["damage"] == StepStatus.SKIPPED and statuses["evidence"] == StepStatus.FAILED
        assert statuses["compliance"] == StepStatus.SKIPPED                     # needs evidence, which failed
        stored = {name: r.status for name, r in c.db.latest_agent_results(claim.id).items()}
        assert stored == statuses                                              # what ran is what was saved
        stats = c.db.dashboard_stats()
        assert stats["failed_steps"] == 1 and stats["claims_with_warnings"] == 1


def test_a_failed_step_carries_a_readable_error_and_no_internals():
    with temp_container() as (c, _):
        with quiet_logs():
            summary = build(c, intake="boom", policy="fail").run(make_claim(c.db).id)
        assert summary.results["intake"].error == "Unexpected error (RuntimeError). The log has the details."
        assert summary.results["policy"].error == "Could not finish. Try again."


# ---------------------------------------------------------------- claim status during and after a run

def test_the_claim_is_processing_during_the_run_and_never_left_there():
    with temp_container() as (c, _):
        claim = make_claim(c.db)
        seen = []
        probe = lambda: seen.append(c.db.get_claim(claim.id).status)
        for behaviors in ({}, {"intake": "fail"}, {"policy": "boom"}, {"compliance": "skip"}):
            with quiet_logs():
                result = build(c, probe=probe, **behaviors).run(claim.id)
            assert c.db.get_claim(claim.id).status != ClaimStatus.PROCESSING
            assert result.final_status == c.db.get_claim(claim.id).status
            if c.db.get_claim(claim.id).status == ClaimStatus.AWAITING_REVIEW:
                c.db.update_claim_status(claim.id, ClaimStatus.PROCESSING)       # allowed: a manual rerun
                c.db.update_claim_status(claim.id, ClaimStatus.SUBMITTED)
        assert seen and set(seen) == {ClaimStatus.PROCESSING}


def test_final_status_for_each_kind_of_run():
    expectations = [
        ({}, ClaimStatus.AWAITING_REVIEW),                       # everything finished
        ({"risk": "warn"}, ClaimStatus.AWAITING_REVIEW),         # a warning still finishes
        ({"policy": "boom"}, ClaimStatus.AWAITING_REVIEW),       # policy is not a hard requirement: a person still gets the claim
        ({"damage": "skip"}, ClaimStatus.AWAITING_REVIEW),
        ({"intake": "fail"}, ClaimStatus.FAILED),                # nothing downstream could run
        ({"compliance": "fail"}, ClaimStatus.FAILED),
        ({"compliance": "skip"}, ClaimStatus.SUBMITTED),         # no failure, but no recommendation either
    ]
    for behaviors, expected in expectations:
        with temp_container() as (c, _):
            claim = make_claim(c.db)
            with quiet_logs():
                build(c, **behaviors).run(claim.id)
            assert c.db.get_claim(claim.id).status == expected, behaviors


def test_trace_events_start_with_running_and_blocked_steps_only_report_an_end():
    with temp_container() as (c, _):
        events = []
        with quiet_logs():
            build(c, intake="fail").run(make_claim(c.db).id, on_event=events.append)
        by_step = {}
        for event in events:
            by_step.setdefault(event.step, []).append(event.status)
        assert by_step["intake"] == [StepStatus.RUNNING, StepStatus.FAILED]
        assert by_step["policy"] == [StepStatus.RUNNING, StepStatus.DONE]
        assert by_step["evidence"] == [StepStatus.SKIPPED]                      # never started: it was blocked
        assert [e.step for e in events if e.status == StepStatus.RUNNING] == ["intake", "policy", "damage"]


# ---------------------------------------------------------------- every transition

def claim_in(c, status):
    """Walk a new claim to `status` using only moves the system itself allows."""
    claim = make_claim(c.db)
    if status == ClaimStatus.SUBMITTED:
        return claim
    c.db.update_claim_status(claim.id, ClaimStatus.PROCESSING)
    if status == ClaimStatus.FAILED:
        c.db.update_claim_status(claim.id, ClaimStatus.FAILED)
    elif status != ClaimStatus.PROCESSING:
        c.db.update_claim_status(claim.id, ClaimStatus.AWAITING_REVIEW)
        decision = {ClaimStatus.APPROVED: ReviewDecision.APPROVE, ClaimStatus.INFO_REQUESTED: ReviewDecision.REQUEST_INFO,
                    ClaimStatus.ESCALATED: ReviewDecision.ESCALATE}.get(status)
        if decision:
            c.db.add_review(claim.id, decision, "Sara Ahmed")
    assert c.db.get_claim(claim.id).status == status
    return claim


def test_the_transition_table_covers_every_status_and_final_states_have_no_exits():
    assert set(ALLOWED_TRANSITIONS) == set(ClaimStatus)
    assert ALLOWED_TRANSITIONS[ClaimStatus.APPROVED] == set() and ALLOWED_TRANSITIONS[ClaimStatus.ESCALATED] == set()
    assert not can_transition(ClaimStatus.SUBMITTED, ClaimStatus.AWAITING_REVIEW)    # a claim cannot skip processing


def test_the_database_allows_exactly_the_transitions_in_the_table():
    checked = 0
    for start in ClaimStatus:
        for target in ClaimStatus:
            with temp_container() as (c, _):
                claim = claim_in(c, start)
                if target == start or can_transition(start, target):
                    assert c.db.update_claim_status(claim.id, target).status == target
                else:
                    raises(WorkflowError, c.db.update_claim_status, claim.id, target)
                    assert c.db.get_claim(claim.id).status == start                 # a refused move changes nothing
                checked += 1
    assert checked == len(ClaimStatus) ** 2


def test_only_a_claim_awaiting_review_can_be_decided():
    for start in ClaimStatus:
        if start == ClaimStatus.AWAITING_REVIEW:
            continue
        with temp_container() as (c, _):
            claim = claim_in(c, start)
            reviews_before = len(c.db.list_reviews(claim.id))       # a decided claim already has its one review
            raises(WorkflowError, c.db.add_review, claim.id, ReviewDecision.APPROVE, "Sara Ahmed")
            assert c.db.get_claim(claim.id).status == start and len(c.db.list_reviews(claim.id)) == reviews_before


def test_the_agents_can_run_only_where_processing_is_a_legal_next_state():
    for start in ClaimStatus:
        with temp_container() as (c, _):
            claim = claim_in(c, start)
            orchestrator = build(c)
            if start == ClaimStatus.PROCESSING:
                continue                                          # a run in progress is not started twice here
            if can_transition(start, ClaimStatus.PROCESSING):
                assert orchestrator.run(claim.id).final_status == ClaimStatus.AWAITING_REVIEW
            else:
                assert "no longer be processed" in raises(WorkflowError, orchestrator.run, claim.id).message
                assert c.db.agent_history(claim.id) == []         # a refused run records nothing


def test_the_full_lifecycle_with_a_request_for_more_information():
    with temp_container() as (c, _):
        orchestrator = build(c)
        claim = make_claim(c.db)
        path = [c.db.get_claim(claim.id).status]
        orchestrator.run(claim.id); path.append(c.db.get_claim(claim.id).status)
        c.db.add_review(claim.id, ReviewDecision.REQUEST_INFO, "Sara Ahmed", "need the quotation")
        path.append(c.db.get_claim(claim.id).status)
        orchestrator.rerun(claim.id, {"intake"}); path.append(c.db.get_claim(claim.id).status)
        c.db.add_review(claim.id, ReviewDecision.APPROVE, "Sara Ahmed"); path.append(c.db.get_claim(claim.id).status)
        assert path == [ClaimStatus.SUBMITTED, ClaimStatus.AWAITING_REVIEW, ClaimStatus.INFO_REQUESTED,
                        ClaimStatus.AWAITING_REVIEW, ClaimStatus.APPROVED]
        assert [r.decision for r in c.db.list_reviews(claim.id)] == [ReviewDecision.APPROVE, ReviewDecision.REQUEST_INFO]


def test_history_from_an_older_pipeline_still_loads_and_displays():
    with temp_container() as (c, _):
        claim = make_claim(c.db)
        c.db.save_agent_result(claim.id, AgentResult("verification", StepStatus.DONE, "from an older version"))
        assert "verification" in c.db.latest_agent_results(claim.id)
        assert step_label("verification") == "verification"                      # unknown steps show their raw name
        assert step_label("risk") == "Risk analysis"
