# ClaimFlow AI

ClaimFlow AI helps an insurance claims reviewer check a vehicle claim. A set of agents reads the claim, the policy and the evidence, and recommends a next action. A person makes the final decision.

The multi-agent system is built: seven agents, a workflow orchestrator, a human review step, and a UI that shows the workflow as it runs.

## What works today

- **The whole claim workflow, end to end.** Create or load a claim, run seven agents, review the recommendation, record a decision, download the claim package.
- **Seven agents** with typed inputs and outputs: claim intake, policy analyst, damage analysis, evidence verification, risk analysis, claim assessment and compliance review. A workflow orchestrator runs them in dependency order, passes results between them, stores every run, and contains failures.
- **Selective rerun.** Attach a new document and only the steps it affects run again.
- **Policy RAG.** Upload a policy PDF. It is read, split by section, embedded and searched. Every policy finding and every answer cites its page and section, and anything the policy does not say is reported as "not found". See "Policy RAG".
- **Deterministic money.** The potential claim amount is calculated in code with exact decimal arithmetic. The language model never does arithmetic. The deductible must belong to the claim's type, and an "additional excess" is never used as the base deductible.
- **Human in the loop.** Only a person approves, requests information or escalates. The decision is stored with what the AI recommended, and deciding differently from the AI needs a written reason.
- **Final claim package.** One PDF with the decision, the calculation with its working, the policy findings with source pages, the evidence check, the risk indicators, the damage description and the audit trail of every agent step. It is marked draft until a person decides.
- **Demo Mode and Live AI.** Demo Mode runs the entire workflow offline with a scripted model, so a demo cannot be broken by Wi-Fi or a rate limit. A switch in the sidebar flips between the two. See "Demo Mode".
- **Safety.** Upload checks, quote verification, a wording guard so risk analysis never judges anyone's honesty, secret redaction in logs, and an audit trail.

## Known limits

- Scanned PDFs have no text layer and are reported as unreadable. There is no OCR.
- A percentage deductible is flagged for a person, not calculated. Additional excesses (for example for young drivers) are not applied automatically. A reviewer must check them.
- Damage analysis needs a vision model. Without one the step is skipped with a stated reason, and the photos stay visible. In Demo Mode the "description" comes from file names, and it says so.
- Live AI has not been run by the authors of this build. Run Settings, "Send a test request" once with your key before a demo. Model names change often, so check them.
- The pages were exercised against a stand-in for Streamlit, not the real library. The first run on your machine is the real test of the UI.
- Streamlit Community Cloud resets its file system on restart, so claims and uploads do not persist there. That is fine for a demo.

## Documentation

The final submission documents are in `docs/`:

- `ClaimFlow_AI_PRD.docx` and `.md`: the product requirements document, written from the application as built.
- `ClaimFlow_AI_Presentation.pptx` and `.pdf`: 12 slides.
- `ClaimFlow_AI_Demo_Script.docx` and `.md`: a five-minute demo script with timings, expected results and fallbacks.

The team names, hackathon name and date are placeholders in the PRD and on the title slide. Fill them in before submitting.

## Quick start

You need Python 3.10 or newer.

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501. With no configuration it starts in Demo Mode, so nothing needs a key or an internet connection.

If `pip install` fails on `faiss-cpu` or `fastembed` (some platforms and Python versions have no wheel yet), delete that line from `requirements.txt` and install again. The app detects what is installed and falls back on its own: NumPy replaces FAISS, offline hashing replaces the embedding model, and pypdf replaces PyMuPDF. The Settings page shows what is in use.

Try this first: on the Dashboard press "Load the demo claims", open a claim in the claim workspace, and press Run agents. Then open Review and package, record a decision, and prepare the claim package.

## Configuration

Copy `.env.example` to `.env` and edit it. Real environment variables win over `.env`. The variables:

| Variable | Default | Meaning |
|---|---|---|
| LLM_PROVIDER | fake | fake, gemini, groq or openrouter |
| GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY | empty | Key for the chosen provider |
| LLM_MODEL | per provider | Model name. Check the provider's current list |
| VISION_MODEL | see below | Model for damage photos |
| DEMO_MODE | true if no provider is set, otherwise false | Use the scripted offline model instead of a live provider |
| LLM_CACHE | true | Save and reuse model answers |
| DATABASE_PATH | data/claimflow.db | SQLite file |
| UPLOAD_DIR | data/uploads | Where uploaded files go |
| LLM_CACHE_DIR | data/llm_cache | Where live model answers are saved for reuse |
| LOG_DIR, LOG_LEVEL | logs, INFO | Log folder and level |
| MAX_UPLOAD_MB | 10 | Upload limit, 1 to 50 |
| LLM_TIMEOUT_S, LLM_RETRY_DELAY_S | 60, 1.5 | Provider timeout and retry pause |

VISION_MODEL defaults to the main model for Gemini. Groq and OpenRouter have no vision model unless you name one.

If you raise MAX_UPLOAD_MB, also raise `maxUploadSize` in `.streamlit/config.toml`. Streamlit enforces its own limit first.

## Switching AI providers

Set LLM_PROVIDER and the matching key in `.env`, then restart. No code changes are needed. Model names change often, and the defaults in this repository were not checked against the providers' current lists. If a name is rejected, the Settings page shows "does not know the model" and tells you to set LLM_MODEL.

The Settings page has a button that sends one tiny live request, so you can check the key and model before a demo.

## Demo Mode

Demo Mode runs the whole workflow with no internet and no API key. A scripted model stands in for the language model. It reads the same prompts a live model would, and answers the same questions by finding dates, amounts and clauses in the text it is given and quoting them back. Embeddings use the offline fallback too.

What stays real in Demo Mode: PDF reading, section-aware chunking, vector search, quote verification, the evidence rules, the risk rules, the calculator, the orchestrator, the human review and the package PDF. What is scripted: the model's reading of the documents. It does not understand language and it cannot see photos, so damage descriptions come from file names and basic image checks, and the answer says so. The sidebar shows "Demo mode" whenever it is active, and the package PDF records the mode.

Demo Mode is on by default when no provider is configured. To use live AI, set `LLM_PROVIDER` and its key in `.env`, then turn the Demo mode switch in the sidebar off. If no key is configured the switch tells you so and stays on.

### The demo claims

"Load the demo claims" on the Dashboard creates three made-up claims with real generated documents (accident report, quotation, damage photo) and the sample policy. Dates are relative to today.

| Claim | What happens | Result |
|---|---|---|
| Ayesha Khan, clean collision | Everything is present and consistent | Approve for processing. Potential amount 160,000.00 (185,000 estimate minus the 25,000 collision excess) |
| Bilal Raza, missing quotation | No repair quotation yet | Request more information. Press "Attach the repair quotation (demo)" in the Documents tab, re-run only the affected steps, then approve. Potential amount 67,500.00 |
| Sara Ahmed, date conflict | The report's date differs from the form, and the report contains an instruction aimed at an AI | Escalate for investigation. The instruction is reported and not followed. Approving anyway needs a written reason |

### Sample files for a live upload demo

The New claim page has an expander, "Sample documents for a live upload demo". It gives you the values to type into the form and four readable files that agree with each other: the policy, an accident report, a repair quotation and a damage photo. Upload them to one new claim and run the agents. The expected result is Approve for processing with a potential claim amount of 118,500.00 (the 143,500 estimate minus the 25,000 collision excess). The files use made-up data and are dated relative to today. Upload each file to one claim only. The same file on two claims is reported as a risk indicator.

### Run one story from the command line

```
python scripts/run_demo_workflow.py                  # the clean claim
python scripts/run_demo_workflow.py --claim missing  # request more information, attach, re-run, approve
python scripts/run_demo_workflow.py --claim conflict
python scripts/run_demo_workflow.py --claim upload   # the sample files, uploaded as a new claim
```

It runs offline with a temporary database, prints the live trace, the policy findings with their pages and the calculated amount, records a human decision, and writes the claim package PDF to `demo_output/`.

If a document cannot be read as text, the system says so and refuses to guess. It does not invent policy clauses or amounts. That is the intended behaviour for scanned files. The demo files are text PDFs, and the demo loader checks they can be read before it creates a claim.

### A three-minute walkthrough

1. Dashboard: load the demo claims. Open "How the AI is used in this app".
2. Claim workspace, Ayesha Khan: Run agents. Watch the seven steps. Open the Agent workflow tab to see what each step needed and passed on.
3. Policy search: ask "What is the excess for a collision claim?" (answered with page 3) and "Is a rental car provided?" (not found).
4. Review and package: record "Approve for processing", then prepare and download the package.
5. Bilal Raza: run, see "request more information", attach the quotation, re-run the affected steps, approve.
6. Sara Ahmed: run, see the escalation. Try to approve without a note, then with one, and open the package to see "Overrode the AI".

## Project layout

```
app.py                 entry point and navigation
views/                 the six pages (dashboard, new_claim, workspace, policy, review, settings)
samples/               sample_motor_policy.pdf, a fictional policy for testing
scripts/               make_sample_policy.py regenerates the sample (needs reportlab)
claimflow/
  config.py            settings from environment variables
  logging_setup.py     console and file logging with secret redaction
  errors.py            error types with messages written for people
  models.py            claim, document, review and agent result types
  pipeline.py          the seven workflow steps and their dependencies
  db.py                SQLite: claims, documents, agent_runs, reviews
  llm.py               provider abstraction, response cache, JSON parsing
  uploads.py           upload validation and storage
  extract.py           PDF reading: PyMuPDF first, pypdf as fallback, page images
  retrieval.py         the Chunk type and BM25 keyword search
  rag/                 chunker, embeddings, vector store, policy index, grounded answers
  money.py             Decimal amounts and the "is this figure in the text" check
  orchestrator.py      runs the pipeline, handles failures, reruns selectively
  container.py         wires the pieces together and switches Demo Mode on and off
  package.py           the final claim package PDF (ReportLab)
  demo.py              the three demo claims and their generated documents
  demo_model.py        the scripted offline model behind Demo Mode
  ui.py                shared Streamlit helpers
  schemas.py           typed output of each agent: the contract between agents
  calculator.py        the claim arithmetic, in Decimal, no model
  language.py          the wording guard that blocks accusatory text
  images.py            prepares photos for a vision model
  agents/              base.py plus intake, policy, damage, evidence, risk, assessment, compliance
tests/                 tests that need no network and no API key
```

## How the pieces fit

Agents share one claim state. They read earlier results from it and return a single result of their own. They never call each other, and they never write to the database. The orchestrator does the saving, skips a step when something it needs failed, and turns a crashing agent into a failed step instead of a crashed app.

## How the agents communicate

Agents never call each other. They communicate through one shared object, the claim state, and the orchestrator is the only thing that moves data.

1. The orchestrator walks the steps in `pipeline.py` in order. For each step it checks `requires`. If a required step did not finish (done or warning), the step is skipped with a stated reason.
2. Otherwise it builds the claim state: the claim, its documents, the results of earlier steps, and any files shared with other claims. Each agent gets its own copy, so an agent cannot change what the next one sees.
3. The agent runs and returns one result with a status, a summary, a structured output, evidence, and warnings. It never writes to the database.
4. The orchestrator saves the result and adds it to the state. The next agent reads it with a typed call such as `ctx.state.typed("intake", IntakeOutput)`. If that step was skipped or failed, the call returns safe defaults instead of crashing.

`requires` means the step cannot run without the other. `uses` means the step reads the other's output when it exists, and is also rerun when it changes. Only intake is required by most steps, so a claim with no policy still reaches a person with a clear request, instead of stalling.

Example: intake reads a different accident date in the report. The risk agent reads that from the intake output and raises a date indicator. The compliance agent reads the indicator and recommends requesting more information, and cites it in its reasons.

## What the agents will not do

- No agent says or implies that a person committed fraud. Risk indicators use fixed wording about documents and data. A model may point out an inconsistency, but it needs two quotes that really appear in the claim text, its wording must pass the wording guard, and the most severe label it can earn is low. Only fixed rules can raise a severity. The compliance agent audits every agent's text before a reviewer sees it.
- No model does arithmetic that code trusts. The claim assessment agent calls `calculator.py` and makes no model call at all.
- The model never decides the recommendation. Fixed rules in `compliance.py` choose it. The model only writes the explanation, and that text is replaced by a template if it contains a figure that is not in the facts it was given.
- A repair estimate is kept only if that figure appears in a document. A policy statement is kept only if its quote appears word for word in the cited chunk, and a deductible or limit only if the amount is in the quote.
- Text inside documents is treated as evidence, never as instructions. Phrases that read like orders to an AI are reported as an indicator.

The recommendation is advice. Nothing happens until a person records a decision.

## Policy RAG

How a policy becomes an answer:

1. **Read.** PyMuPDF extracts the text of each page, with the file's metadata. If PyMuPDF is missing or fails on a file, pypdf is used instead. The Policy search page shows which one read the file.
2. **Split.** The text is cut into chunks that follow the policy's structure. A chunk never crosses a page or a heading, and it carries its page and its section label, for example "Section 5. What we do not cover / 5.1 Racing and speed testing". The chunk text is the page's own words, so quotes can be checked against it exactly.
3. **Embed.** Each chunk becomes a vector. By default a small local model (BAAI/bge-small-en-v1.5, run with fastembed) does this. It is downloaded once, on first use. If it cannot start, offline hashing embeddings take over and the Settings page says so.
4. **Store.** Vectors go into a FAISS index. Without faiss-cpu, a NumPy search gives the same answer. The index is saved in `data/policy_index` and reused until the PDF or the embedding model changes.
5. **Retrieve.** Search is hybrid: vector similarity finds clauses that mean the same thing, keyword ranking finds exact terms such as "deductible", and the two lists are merged. A chunk counts only if it matches on keywords or clears the similarity floor, so an unrelated question returns nothing.
6. **Analyse.** The Policy Analyst Agent writes its own search queries for the claim, always adds three fixed ones, and passes the retrieved chunks to the model. The model must quote the policy word for word.
7. **Verify.** Code checks every quote against the chunk it cites. A statement is kept only if its quote is really there, and an amount only if that figure is in the quote. Anything else is dropped and listed as unverified.

What is guaranteed, and what is not:

- A finding or an answer is shown only with a quote that code found in the policy. A made-up clause cannot appear, even if the model offers one.
- A topic the search could not answer is reported as "not found". It is never left blank and never filled in. For exclusions the message adds that finding none in the searched sections does not prove the policy has none.
- If the model is unavailable, the page shows the closest passages and says the AI answer is unavailable. It does not present them as an answer.
- Not guaranteed: that the model reads a clause correctly. A real quote can still be a misreading, which is why the source page is one click away and a person decides.

Using it: open Policy search, pick a claim, upload a PDF or press "Use the sample policy", look at the indexed information, then ask a question or read the findings. Each finding and each answer has a "View source page" control. With PyMuPDF it shows the real page image next to the page text with the quoted words in bold. Without it, the text is shown.

The sample policy is `samples/sample_motor_policy.pdf`, a fictional six-page motor policy. Questions to try, and what the policy says:

| Question | Expected |
|---|---|
| What is the excess for a collision claim? | PKR 25,000, page 3, Section 4 |
| What is the excess for a theft claim? | PKR 50,000, page 3, Section 4 |
| Is racing covered? | No, excluded, page 4, section 5.1 |
| How long do I have to report a theft to the police? | 24 hours, page 2 (3.3) and page 5 (6.1) |
| What is the sum insured? | PKR 1,500,000, page 1 |
| Is a rental car provided while my car is repaired? | Not found. The policy does not mention it |
| What is the roadside assistance phone number? | Not found. The policy does not mention it |

These are what the policy contains. The pipeline was tested with scripted model answers, so how a real model phrases them is not yet checked.

Limits to know about:

- Only the first 60 pages are read. Scanned PDFs have no text layer and are reported as unreadable. There is no OCR.
- Tables are read as text in reading order, so a table row can come out split. The sample states its key figures in sentences as well for that reason.
- Headings are recognised by shape ("Section 4. Excess", "4.2 Exclusions", "EXCLUSIONS"). A policy in another style still works, but its chunks carry no section label.
- The offline hashing fallback matches wording, not meaning. "Maximum payout" will not find "the most we will pay". The fastembed model closes that gap. The similarity floors are my estimates and have not been tuned on real policies.
- Questions on the Policy search page use one model call each and are outside the six-calls-per-claim budget of the workflow.

Settings: `EMBEDDING_BACKEND` (auto, fastembed or hashing), `EMBEDDING_MODEL` and `INDEX_DIR`. To regenerate the sample after editing it: `pip install reportlab`, then `python scripts/make_sample_policy.py`.

## Model calls per claim

At most six: intake 1, policy 2, damage 1, risk 1, compliance 1. The evidence check and the claim assessment use none. Risk is skipped when there are no documents to compare, and damage is skipped when there are no photos.

To add an agent, write a class with a `name` that matches a step in `pipeline.py` and a `run(ctx)` method, and add it to `default_agents()` in `claimflow/agents/__init__.py`.

## Tests

```
python -m tests.runner
```

This needs no extra packages. If you have pytest, `python -m pytest tests` also works.

## Deploying to Streamlit Community Cloud

Put the keys in the app's secrets as top-level entries, for example `GEMINI_API_KEY = "..."`. Streamlit exposes them as environment variables, so the same names work. The hosted file system resets on restart, so the database and uploads do not persist. That is fine for a demo.

## Security notes

Uploads are limited to PDF, PNG and JPG, checked by content as well as name, and stored under random names. Text inside a document is treated as evidence and never as instructions to the agents. API keys are read from the environment, hidden from the Settings page and replaced with [REDACTED] in logs.
