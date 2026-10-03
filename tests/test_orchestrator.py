from claimflow.agents import default_agents
from claimflow.agents.base import Agent, AgentContext
from claimflow.errors import AgentError, WorkflowError
from claimflow.llm import FakeProvider, LLMRequest
from claimflow.models import AgentResult, ClaimState, ClaimStatus, ReviewDecision, StepStatus
from claimflow.orchestrator import WorkflowOrchestrator, final_claim_status
from claimflow.pipeline import PIPELINE, StepSpec, affected_steps, validate_pipeline
from tests.helpers import make_claim, quiet_logs, temp_container


class StubAgent(Agent):
    """An agent whose behaviour the test chooses."""
    def __init__(self, name, behavior="ok"):
        self.name, self.label, self.behavior, self.ran = name, name, behavior, 0

    def run(self, ctx: AgentContext) -> AgentResult:
        self.ran += 1
        if self.behavior == "boom":
            raise RuntimeError("secret internal detail")
        if self.behavior == "fail":
            raise AgentError("The policy file is unreadable.", hint="Upload a clearer PDF.")
        if self.behavior == "warn":
            return self.warning("careful", ["a risk indicator"])
        if self.behavior == "skip":
            return self.skipped("no photos")
        if self.behavior == "wrongstep":
            return AgentResult("somebody_else", StepStatus.DONE)
        if self.behavior == "llm":
            return self.done("asked the model", ctx.ask_json(LLMRequest(prompt="give me data")))
        return self.done(f"{self.name} ok", {"run": self.ran})


def build(container, **behaviors):
    agents = {s.name: StubAgent(s.name, behaviors.get(s.name, "ok")) for s in PIPELINE}
    orchestrator = WorkflowOrchestrator(container.db, container.llm, container.settings, list(agents.values()))
    return orchestrator, agents


def raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        return exc
    raise AssertionError(f"{exc_type.__name__} was not raised")


# ---------------------------------------------------------------- pipeline definition

def test_the_real_pipeline_is_valid_and_ordered():
    validate_pipeline(PIPELINE)
    assert [s.name for s in PIPELINE] == ["intake", "policy", "damage", "evidence", "risk", "assessment", "compliance"]


def test_a_broken_pipeline_is_rejected():
    forward = (StepSpec("a", "A", "", requires=("b",)), StepSpec("b", "B", ""))
    assert "not defined before" in raises(WorkflowError, validate_pipeline, forward).message
    assert "twice" in raises(WorkflowError, validate_pipeline, (StepSpec("a", "A", ""), StepSpec("a", "A", ""))).message


def test_affected_steps_follow_the_dependencies():
    assert affected_steps({"intake"}) == ["intake", "evidence", "risk", "assessment", "compliance"]   # policy and damage do not need intake
    assert affected_steps({"policy"}) == ["policy", "evidence", "risk", "assessment", "compliance"]
    assert affected_steps({"damage"}) == ["damage", "evidence", "risk", "compliance"]                   # via "uses"
    assert affected_steps({"risk"}) == ["risk", "compliance"]
    assert affected_steps({"assessment"}) == ["assessment", "compliance"]
    assert affected_steps({"compliance"}) == ["compliance"]
    raises(WorkflowError, affected_steps, {"nonsense"})


# ---------------------------------------------------------------- running

def test_a_clean_run_goes_through_every_step_and_waits_for_review():
    with temp_container() as (c, _):
        orchestrator, agents = build(c)
        claim = make_claim(c.db)
        summary = orchestrator.run(claim.id)
        assert summary.ran == [s.name for s in PIPELINE]
        assert all(a.ran == 1 for a in agents.values())
        assert summary.final_status == ClaimStatus.AWAITING_REVIEW
        assert c.db.get_claim(claim.id).status == ClaimStatus.AWAITING_REVIEW
        assert set(c.db.latest_agent_results(claim.id)) == {s.name for s in PIPELINE}


def test_a_crashing_agent_is_contained_and_its_details_are_not_leaked():
    with temp_container() as (c, _):
        orchestrator, agents = build(c, policy="boom")
        claim = make_claim(c.db)
        with quiet_logs():
            summary = orchestrator.run(claim.id)
        policy = summary.results["policy"]
        assert policy.status == StepStatus.FAILED
        assert "secret internal detail" not in policy.error and "RuntimeError" in policy.error
        assert summary.results["intake"].status == StepStatus.DONE
        # Policy is not a hard requirement for later steps, so the claim still reaches a person.
        assert summary.results["compliance"].status == StepStatus.DONE
        assert c.db.get_claim(claim.id).status == ClaimStatus.AWAITING_REVIEW


def test_a_failed_intake_blocks_what_needs_it_and_says_why():
    with temp_container() as (c, _):
        orchestrator, agents = build(c, intake="fail")
        claim = make_claim(c.db)
        summary = orchestrator.run(claim.id)
        assert summary.results["intake"].error == "The policy file is unreadable. Upload a clearer PDF."
        for name in ("evidence", "risk", "assessment", "compliance"):
            assert summary.results[name].status == StepStatus.SKIPPED and "needs" in summary.results[name].summary
            assert agents[name].ran == 0
        assert summary.results["policy"].status == StepStatus.DONE    # independent of intake
        assert summary.results["damage"].status == StepStatus.DONE
        assert c.db.get_claim(claim.id).status == ClaimStatus.FAILED


def test_an_intentionally_skipped_optional_step_does_not_block_the_rest():
    with temp_container() as (c, _):
        orchestrator, agents = build(c, damage="skip")
        summary = orchestrator.run(make_claim(c.db).id)
        assert summary.results["damage"].status == StepStatus.SKIPPED
        assert summary.results["evidence"].status == StepStatus.DONE
        assert summary.final_status == ClaimStatus.AWAITING_REVIEW


def test_a_warning_counts_as_finished():
    with temp_container() as (c, _):
        orchestrator, _ = build(c, risk="warn")
        summary = orchestrator.run(make_claim(c.db).id)
        assert summary.results["compliance"].status == StepStatus.DONE
        assert summary.final_status == ClaimStatus.AWAITING_REVIEW


def test_an_agent_returning_the_wrong_result_is_a_failure():
    with temp_container() as (c, _):
        orchestrator, _ = build(c, intake="wrongstep")
        summary = orchestrator.run(make_claim(c.db).id)
        assert summary.results["intake"].status == StepStatus.FAILED
        assert "invalid result" in summary.results["intake"].error


def test_model_and_cache_details_are_recorded_on_the_result():
    with temp_container() as (c, _):
        orchestrator, _ = build(c, intake="llm")
        orchestrator.llm = FakeProvider(default='{"claimant": "Ayesha"}', model="test-model")
        summary = orchestrator.run(make_claim(c.db).id)
        intake = summary.results["intake"]
        assert intake.output == {"claimant": "Ayesha"} and intake.model == "test-model" and intake.cached is False
        assert orchestrator.llm.calls[0].json_mode is True


# ---------------------------------------------------------------- selective rerun

def test_rerun_runs_only_the_changed_step_and_its_dependents():
    with temp_container() as (c, _):
        orchestrator, agents = build(c)
        claim = make_claim(c.db)
        orchestrator.run(claim.id)
        summary = orchestrator.rerun(claim.id, {"risk"})
        assert summary.ran == ["risk", "compliance"]
        assert {n: a.ran for n, a in agents.items()} == {
            "intake": 1, "policy": 1, "damage": 1, "evidence": 1, "risk": 2, "assessment": 1, "compliance": 2}
        assert c.db.get_claim(claim.id).status == ClaimStatus.AWAITING_REVIEW
        latest = c.db.latest_agent_results(claim.id)
        assert latest["risk"].output == {"run": 2} and latest["intake"].output == {"run": 1}
        assert len(c.db.agent_history(claim.id)) == 9


def test_rerun_after_a_failure_recovers_the_claim():
    with temp_container() as (c, _):
        orchestrator, agents = build(c, intake="fail")
        claim = make_claim(c.db)
        orchestrator.run(claim.id)
        assert c.db.get_claim(claim.id).status == ClaimStatus.FAILED
        agents["intake"].behavior = "ok"
        summary = orchestrator.rerun(claim.id, {"intake"})
        assert summary.ran == ["intake", "evidence", "risk", "assessment", "compliance"]
        assert summary.final_status == ClaimStatus.AWAITING_REVIEW


def test_rerun_without_earlier_results_skips_what_it_cannot_support():
    with temp_container() as (c, _):
        orchestrator, agents = build(c)
        summary = orchestrator.rerun(make_claim(c.db).id, {"risk"})
        assert summary.ran == ["risk", "compliance"]
        assert summary.results["risk"].status == StepStatus.SKIPPED and agents["risk"].ran == 0


# ---------------------------------------------------------------- guard rails

def test_a_claim_with_a_human_decision_cannot_be_reprocessed():
    with temp_container() as (c, _):
        orchestrator, _ = build(c)
        claim = make_claim(c.db)
        orchestrator.run(claim.id)
        c.db.add_review(claim.id, ReviewDecision.ESCALATE, "Sara Ahmed")
        assert "no longer be processed" in raises(WorkflowError, orchestrator.run, claim.id).message


def test_a_request_for_more_information_can_be_rerun():
    with temp_container() as (c, _):
        orchestrator, _ = build(c)
        claim = make_claim(c.db)
        orchestrator.run(claim.id)
        c.db.add_review(claim.id, ReviewDecision.REQUEST_INFO, "Sara Ahmed", "need the quotation")
        assert c.db.get_claim(claim.id).status == ClaimStatus.INFO_REQUESTED
        assert orchestrator.rerun(claim.id, {"intake"}).final_status == ClaimStatus.AWAITING_REVIEW


def test_unknown_steps_and_mismatched_agents_are_rejected():
    with temp_container() as (c, _):
        orchestrator, _ = build(c)
        raises(WorkflowError, orchestrator.run, make_claim(c.db).id, only={"nonsense"})
        raises(WorkflowError, WorkflowOrchestrator, c.db, c.llm, c.settings, [StubAgent("intake")])


def test_trace_events_reach_the_ui_callback_and_a_broken_callback_is_harmless():
    with temp_container() as (c, _):
        orchestrator, _ = build(c)
        seen = []
        summary = orchestrator.run(make_claim(c.db).id, on_event=lambda e: seen.append((e.step, e.status)))
        assert seen[0] == ("intake", StepStatus.RUNNING) and seen[1] == ("intake", StepStatus.DONE)
        assert len(seen) == 14 and len(summary.trace) == 14      # a start and an end event for each of 7 steps

        def broken(event):
            raise RuntimeError("UI blew up")
        with quiet_logs():
            result = build(c)[0].run(make_claim(c.db).id, on_event=broken)
        assert result.final_status == ClaimStatus.AWAITING_REVIEW


def test_final_status_rules():
    done, failed, skipped = (AgentResult("x", s) for s in (StepStatus.DONE, StepStatus.FAILED, StepStatus.SKIPPED))
    review_done = AgentResult("compliance", StepStatus.DONE)
    assert final_claim_status({"compliance": review_done, "intake": done}) == ClaimStatus.AWAITING_REVIEW
    assert final_claim_status({"compliance": review_done, "policy": failed}) == ClaimStatus.AWAITING_REVIEW   # a person still gets it
    assert final_claim_status({"intake": failed}) == ClaimStatus.FAILED
    assert final_claim_status({"intake": skipped}) == ClaimStatus.SUBMITTED
    assert final_claim_status({}) == ClaimStatus.SUBMITTED


# ---------------------------------------------------------------- what ships today

def test_default_agents_run_end_to_end_on_the_offline_provider():
    with temp_container() as (c, _):
        claim = make_claim(c.db)
        summary = c.orchestrator.run(claim.id)
        assert summary.results["intake"].status == StepStatus.DONE
        assert summary.results["policy"].status == StepStatus.SKIPPED and "No policy document" in summary.results["policy"].summary
        assert summary.results["damage"].status == StepStatus.SKIPPED and "No damage photos" in summary.results["damage"].summary
        assert summary.results["assessment"].status == StepStatus.WARNING            # nothing to calculate yet
        # A claim with nothing attached still reaches a person, with a clear request for more information.
        assert summary.final_status == ClaimStatus.AWAITING_REVIEW
        assert summary.results["compliance"].output["recommendation"] == "request_more_information"


def test_the_default_agents_cover_the_pipeline_exactly():
    assert [a.name for a in default_agents()] == [s.name for s in PIPELINE]


def test_agent_context_records_model_calls_and_forces_json_mode():
    with temp_container() as (c, _):
        fake = FakeProvider(default='{"ok": true}')
        ctx = AgentContext(state=ClaimState(make_claim(c.db)), llm=fake, settings=c.settings)
        assert ctx.ask_json(LLMRequest(prompt="x")) == {"ok": True}
        assert fake.calls[0].json_mode is True and len(ctx.llm_calls) == 1
