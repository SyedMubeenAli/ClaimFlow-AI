"""The Policy Analyst Agent on the sample policy: cited findings, explicit not-found, nothing invented."""

import json

from claimflow.agents import default_agents
from claimflow.llm import FakeProvider
from claimflow.models import DocumentKind, StepStatus
from claimflow.orchestrator import WorkflowOrchestrator
from claimflow.rag import HashingEmbedder
from claimflow.schemas import PolicyOutput
from tests.helpers import attach, attach_sample_policy, make_claim, temp_container
from tests.test_agent_policy import chunk_id
from tests.test_rag_index_answer import CountingEmbedder

COVER = "We pay for accidental damage to the insured vehicle caused by collision with another vehicle, a person, an animal or an object."
EXCESS = "The excess for a collision and accidental damage claim is PKR 25,000 for each claim."
LIMIT = "The sum insured of PKR 1,500,000 is the most we will pay for any one claim"
RACING = "We do not cover damage that happens while the insured vehicle is used for racing, rallying, pace-making, speed testing or track days."
QUERIES = '{"queries": ["collision cover conditions", "racing speed testing exclusion", "excess for collision claim"]}'


def cited(prompt, snippet, quote):
    return {"chunk": chunk_id(prompt, snippet), "quote": quote}


def full_reply(request):
    p = request.prompt
    return json.dumps({
        "coverage": {"applies": "yes", **cited(p, "caused by collision with another vehicle", COVER)},
        "deductible": {"kind": "fixed", "amount": "25,000", **cited(p, "excess for a collision and accidental damage claim", EXCESS)},
        "coverage_limit": {"amount": "1,500,000", **cited(p, "most we will pay for any one claim", LIMIT)},
        "exclusions": [{"title": "Racing and speed testing", "applies_to_claim": False, "reason": "No racing was reported.",
                        **cited(p, "used for racing, rallying", RACING)}],
    })


def run_policy(c, claim, extraction, queries=QUERIES, rag=None):
    fake = FakeProvider({"TASK: extract claim facts": '{"summary": "Rear-ended at a traffic light."}',
                         "TASK: write search queries": queries, "TASK: extract policy terms": extraction})
    orchestrator = WorkflowOrchestrator(c.db, fake, c.settings, default_agents(), rag=rag or c.rag)
    return orchestrator.run(claim.id, only={"intake", "policy"}).results["policy"], fake


def sample_claim(c):
    claim = make_claim(c.db)
    return claim, attach_sample_policy(c, claim)


# ---------------------------------------------------------------- cited findings

def test_every_finding_names_its_page_section_and_source_file():
    with temp_container() as (c, _):
        claim, doc = sample_claim(c)
        result, _ = run_policy(c, claim, full_reply)
        out = PolicyOutput.from_dict(result.output)
        assert result.status == StepStatus.DONE and result.warnings == [] and out.not_found == []
        by_topic = {f["topic"]: f for f in out.findings}
        assert set(by_topic) == {"coverage", "deductible", "coverage_limit", "exclusion"}
        assert (by_topic["deductible"]["page"], by_topic["deductible"]["section"]) == (3, "Section 4. Excess")
        assert by_topic["deductible"]["value"] == "25000.00" and by_topic["coverage_limit"]["value"] == "1500000.00"
        assert by_topic["coverage_limit"]["page"] == 1 and by_topic["coverage"]["page"] == 2
        assert by_topic["exclusion"]["page"] == 4 and "5.1 Racing" in by_topic["exclusion"]["section"]
        assert all(f["source"] == "sample_motor_policy.pdf" and f["doc_ref"] == f"d{doc.id}" and f["quote"] for f in out.findings)
        assert [e.page for e in result.evidence if e.detail.startswith("the deductible")] == [3]
        assert next(e for e in result.evidence if e.detail.startswith("the deductible")).section == "Section 4. Excess"


def test_the_agents_own_queries_pull_in_the_clause_it_asked_about():
    with temp_container() as (c, _):
        claim, _ = sample_claim(c)
        result, fake = run_policy(c, claim, full_reply)
        queries = PolicyOutput.from_dict(result.output).queries
        assert "racing speed testing exclusion" in queries and "deductible excess amount you pay" in queries
        assert any("used for racing" in call.prompt for call in fake.calls)


def test_index_details_are_part_of_the_result_and_the_index_is_saved_and_reused():
    with temp_container() as (c, tmp):
        claim, doc = sample_claim(c)
        counting = CountingEmbedder()
        c.rag.embedder = counting
        result, _ = run_policy(c, claim, full_reply)
        info = PolicyOutput.from_dict(result.output).index_info
        assert info["documents"] == ["sample_motor_policy.pdf"] and info["chunks"] > 20 and info["sections"] > 10
        assert info["embedder"].startswith("hashing:") and info["vector_store"] in ("numpy", "faiss")
        assert counting.batches == 1 and any((c.settings.index_dir).glob("doc*/manifest.json"))
        run_policy(c, claim, full_reply)                                              # a second run
        assert counting.batches == 1                                                  # reused, not re-embedded


# ---------------------------------------------------------------- not found, stated explicitly

def test_topics_the_policy_search_could_not_answer_are_reported_as_not_found():
    with temp_container() as (c, _):
        claim, _ = sample_claim(c)
        result, _ = run_policy(c, claim, '{"coverage": null, "deductible": null, "coverage_limit": null, "exclusions": []}')
        out = PolicyOutput.from_dict(result.output)
        assert result.status == StepStatus.WARNING and out.findings == []
        assert {n["topic"] for n in out.not_found} == {"coverage", "deductible", "coverage_limit", "exclusions"}
        assert "The deductible was not found in the policy." in result.warnings
        assert "The coverage limit was not found in the policy." in result.warnings
        assert "Deductible: not found" in result.summary and "Limit: not found" in result.summary and "Coverage: not found" in result.summary
        assert (out.deductible, out.coverage_limit, out.coverage_applies) == (None, None, "unclear")


def test_not_found_covers_only_what_is_actually_missing():
    with temp_container() as (c, _):
        claim, _ = sample_claim(c)
        def only_deductible(request):
            data = json.loads(full_reply(request))
            return json.dumps({"coverage": None, "deductible": data["deductible"], "coverage_limit": None, "exclusions": []})
        result, _ = run_policy(c, claim, only_deductible)
        out = PolicyOutput.from_dict(result.output)
        assert {n["topic"] for n in out.not_found} == {"coverage", "coverage_limit", "exclusions"}
        assert [f["topic"] for f in out.findings] == ["deductible"] and out.deductible == "25000.00"
        exclusions = next(n for n in out.not_found if n["topic"] == "exclusions")
        assert "does not prove the policy has none" in exclusions["message"]


def test_an_invented_clause_never_appears_as_a_finding():
    with temp_container() as (c, _):
        claim, _ = sample_claim(c)
        def invents(request):
            data = json.loads(full_reply(request))
            data["exclusions"].append({"title": "Rental car cover", "applies_to_claim": True, "reason": "A rental car is provided.",
                                       "chunk": data["exclusions"][0]["chunk"], "quote": "We provide a rental car for up to 30 days after any claim."})
            data["coverage_limit"] = {"amount": "9,000,000", "chunk": data["coverage_limit"]["chunk"],
                                      "quote": "The most we will pay for any one claim is PKR 9,000,000 including a rental car."}
            return json.dumps(data)
        result, _ = run_policy(c, claim, invents)
        out = PolicyOutput.from_dict(result.output)
        assert {r["item"] for r in out.rejected} == {"the exclusion 'Rental car cover'", "the coverage limit"}
        everything = json.dumps({"findings": out.findings, "exclusions": out.exclusions, "limit": out.coverage_limit})
        assert "rental" not in everything.lower() and "9,000,000" not in everything and out.coverage_limit is None
        assert not any("Possible exclusion" in w for w in result.warnings)                  # the invented one never reached the reviewer


def test_nothing_relevant_in_the_policy_is_stated_plainly_and_the_model_is_not_asked_to_extract():
    with temp_container() as (c, _):
        claim = make_claim(c.db)
        attach(c, claim, DocumentKind.POLICY, "other.pdf", [["Lorem ipsum dolor sit amet", "consectetur adipiscing elit"]])
        result, fake = run_policy(c, claim, full_reply)
        out = PolicyOutput.from_dict(result.output)
        assert result.status == StepStatus.WARNING and "Nothing relevant was found" in result.summary
        assert {n["topic"] for n in out.not_found} == {"coverage", "deductible", "coverage_limit", "exclusions"}
        assert out.findings == [] and out.chunks_considered == []
        assert not any("extract policy terms" in call.prompt for call in fake.calls)       # nothing to ground an answer on


def test_a_scanned_file_among_the_policies_is_reported_and_the_readable_one_is_used():
    with temp_container() as (c, _):
        claim, _ = sample_claim(c)
        attach(c, claim, DocumentKind.POLICY, "scan.pdf", [[]])
        result, _ = run_policy(c, claim, full_reply)
        assert any("scan.pdf" in w and "could not be parsed as readable text" in w for w in result.warnings)
        assert PolicyOutput.from_dict(result.output).policy_documents == ["sample_motor_policy.pdf"]


def test_the_agent_works_without_a_shared_index_service():
    with temp_container() as (c, tmp):
        claim, _ = sample_claim(c)
        fake = FakeProvider({"TASK: write search queries": QUERIES, "TASK: extract policy terms": full_reply})
        orchestrator = WorkflowOrchestrator(c.db, fake, c.settings, default_agents())          # no rag argument
        result = orchestrator.run(claim.id, only={"intake", "policy"}).results["policy"]
        assert result.status == StepStatus.DONE and list((tmp / "index").glob("doc*")) == []      # in memory: nothing saved


# ---------------------------------------------------------------- the deductible must belong to this kind of claim

from decimal import Decimal as D

from claimflow.agents.policy import excess_problem
from claimflow.models import IncidentType

THEFT_EXCESS = "The excess for a theft claim is PKR 50,000 for each claim."
YOUNG_DRIVER = "If the driver was under 25 years old at the time of the incident, an additional excess of PKR 10,000 applies on top of the excess shown above."
BOTH = (EXCESS + " " + THEFT_EXCESS)


def test_an_excess_for_another_claim_type_or_an_additional_excess_is_not_this_claims_deductible():
    assert excess_problem(EXCESS, D("25000"), IncidentType.COLLISION) is None
    assert excess_problem(THEFT_EXCESS, D("50000"), IncidentType.THEFT) is None
    assert "different type of claim" in excess_problem(THEFT_EXCESS, D("50000"), IncidentType.COLLISION)
    assert "different type of claim" in excess_problem(EXCESS, D("25000"), IncidentType.THEFT)
    assert "additional excess" in excess_problem(YOUNG_DRIVER, D("10000"), IncidentType.COLLISION)
    assert excess_problem("You must pay the first 25,000 of every claim.", D("25000"), IncidentType.COLLISION) is None   # a general excess
    assert excess_problem(THEFT_EXCESS, D("50000"), IncidentType.OTHER) is None                                        # unknown type: no basis to refuse


def test_a_quote_listing_both_excesses_is_judged_by_the_words_next_to_the_figure():
    assert excess_problem(BOTH, D("25000"), IncidentType.COLLISION) is None
    assert "different type of claim" in excess_problem(BOTH, D("50000"), IncidentType.COLLISION)
    assert excess_problem(BOTH, D("50000"), IncidentType.THEFT) is None


def deductible_reply(quote, amount, snippet):
    def reply(request):
        data = json.loads(full_reply(request))
        data["deductible"] = {"kind": "fixed", "amount": amount, "quote": quote, "chunk": chunk_id(request.prompt, snippet)}
        return json.dumps(data)
    return reply


def test_the_agent_refuses_the_wrong_excess_and_the_calculation_then_says_what_is_missing():
    with temp_container() as (c, _):
        claim, _ = sample_claim(c)                                              # a collision claim
        for quote, amount, snippet, why in ((THEFT_EXCESS, "50,000", "excess for a theft claim", "different type of claim"),
                                            (YOUNG_DRIVER, "10,000", "additional excess of PKR 10,000", "additional excess")):
            result, _ = run_policy(c, claim, deductible_reply(quote, amount, snippet))
            out = PolicyOutput.from_dict(result.output)
            assert out.deductible is None and result.status == StepStatus.WARNING
            assert {"item": "the deductible", "why": next(r["why"] for r in out.rejected)} in out.rejected and why in out.rejected[0]["why"]
            assert any(why in w for w in result.warnings)
            assert not any(f["topic"] == "deductible" for f in out.findings)


def test_the_right_excess_is_accepted_for_a_theft_claim():
    with temp_container() as (c, _):
        claim = make_claim(c.db, incident_type="theft", description="My car was stolen from outside my house overnight.")
        attach_sample_policy(c, claim)
        def theft_reply(request):
            p = request.prompt
            return json.dumps({
                "coverage": {"applies": "yes", **cited(p, "if it is stolen", "We pay for the loss of the insured vehicle if it is stolen")},
                "deductible": {"kind": "fixed", "amount": "50,000", **cited(p, "excess for a theft claim", THEFT_EXCESS)},
                "coverage_limit": {"amount": "1,500,000", **cited(p, "most we will pay for any one claim", LIMIT)}, "exclusions": []})
        result, _ = run_policy(c, claim, theft_reply, queries='{"queries": ["theft cover", "theft excess"]}')
        out = PolicyOutput.from_dict(result.output)
        assert out.deductible == "50000.00" and next(f for f in out.findings if f["topic"] == "deductible")["page"] == 3
