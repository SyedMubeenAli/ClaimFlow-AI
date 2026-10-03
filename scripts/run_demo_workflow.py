"""
Runs one demo claim through the whole workflow in Demo Mode (offline, no key), records a human
decision, and writes the final claim package PDF. It uses a temporary database, so it never touches
your real data.

    python scripts/run_demo_workflow.py                 # the clean claim
    python scripts/run_demo_workflow.py --claim missing # request more information, attach, re-run, approve
    python scripts/run_demo_workflow.py --claim conflict
    python scripts/run_demo_workflow.py --claim upload  # the four sample files, uploaded as a new claim
"""

import argparse
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from claimflow import demo  # noqa: E402
from claimflow.container import build_container  # noqa: E402
from claimflow.models import DocumentKind, ReviewDecision  # noqa: E402
from claimflow.package import build_claim_package  # noqa: E402
from claimflow.pipeline import steps_changed_by_document  # noqa: E402
from claimflow.schemas import ComplianceOutput, PolicyOutput  # noqa: E402
from claimflow.uploads import store_document  # noqa: E402

KEYS = {"clean": "Ayesha Khan", "missing": "Bilal Raza", "conflict": "Sara Ahmed"}
DECISIONS = {"approve_for_processing": ReviewDecision.APPROVE, "request_more_information": ReviewDecision.REQUEST_INFO,
             "escalate_for_investigation": ReviewDecision.ESCALATE}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--claim", choices=[*KEYS, "upload"], default="clean")
    parser.add_argument("--out", default="demo_output", help="folder for the claim package PDF")
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        c = build_container({"DATABASE_PATH": f"{tmp}/demo.db", "UPLOAD_DIR": f"{tmp}/uploads", "LLM_CACHE_DIR": f"{tmp}/cache",
                             "INDEX_DIR": f"{tmp}/index", "LOG_DIR": f"{tmp}/logs", "LOG_LEVEL": "ERROR", "DEMO_MODE": "true"})
        if args.claim == "upload":
            claim = demo.create_upload_sample_claim(c)
        else:
            claim = {cl.claimant_name: cl for cl, _ in demo.load_demo_claims(c)}[KEYS[args.claim]]
        print(f"Claim {claim.claim_number}: {claim.claimant_name}, mode: Demo (offline)\n")

        def trace(event):
            if event.status.value != "running":
                print(f"  [{event.status.value:7}] {event.label}: {event.message[:100]}")

        summary = c.orchestrator.run(claim.id, on_event=trace)
        recommendation = ComplianceOutput.from_dict(summary.results["compliance"].output).recommendation
        notes = "Checked the findings against the cited policy pages."
        if recommendation == "request_more_information" and demo.held_back_quotation(claim, c.db.list_documents(claim.id)):
            c.db.add_review(claim.id, ReviewDecision.REQUEST_INFO, "Demo Reviewer", "Please send the repair quotation.", recommendation)
            name, data = demo.held_back_quotation(claim, c.db.list_documents(claim.id))
            store_document(c.db, c.settings.upload_dir, claim.id, DocumentKind.QUOTATION, name, data, c.settings.max_upload_bytes)
            print("\n  Human asked for the quotation. It was attached. Re-running only the affected steps:")
            summary = c.orchestrator.rerun(claim.id, steps_changed_by_document(DocumentKind.QUOTATION), on_event=trace)
            recommendation = ComplianceOutput.from_dict(summary.results["compliance"].output).recommendation
        c.db.add_review(claim.id, DECISIONS[recommendation], "Demo Reviewer", notes, recommendation)

        policy = PolicyOutput.from_dict(c.db.latest_agent_results(claim.id)["policy"].output)
        print("\nPolicy findings with citations:")
        for f in policy.findings:
            print(f"  {f['label']}: {f['value']}  (page {f['page']}, {f['section']})")
        amount = c.db.latest_agent_results(claim.id)["assessment"].output
        print(f"\nPotential claim amount: {amount.get('potential_amount')}  (calculated in code)")
        print(f"AI recommendation: {recommendation}\nHuman decision: {c.db.list_reviews(claim.id)[0].decision.outcome}")

        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        pdf = build_claim_package(c.db.get_claim(claim.id), c.db.list_documents(claim.id), c.db.latest_agent_results(claim.id),
                                  c.db.list_reviews(claim.id), "Demo mode (scripted offline model)")
        path = out / f"{claim.claim_number}_{args.claim}_claim_package.pdf"   # the story is in the name, so runs do not overwrite each other
        path.write_bytes(pdf)
        print(f"\nClaim package written to {path.resolve()} ({len(pdf) / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
