# ClaimFlow AI: Demo and presentation script

*5 minutes, using the clean demo claim (Ayesha Khan), in Demo Mode*

The spoken lines below come to about 506 words, which is about 3.6 minutes at 140 words a minute, plus the time you spend clicking. Times are targets. The demo covers: load the claim, view the documents, run the seven agents, Policy RAG citations, evidence verification, risk analysis, the calculation, human review and the claim package PDF.

## Before you start

- Start the app with: streamlit run app.py. Open http://localhost:8501.
- Check the sidebar. It should read "Demo mode: scripted offline model". Demo Mode is what makes this safe without internet.
- Start with no claims if you can. Load the demo claims is safe to press twice, it only adds what is missing.
- Decide the reviewer name you will type at the decision step.
- Keep a PDF viewer ready. Keep a backup package: python scripts/run_demo_workflow.py writes demo_output/CF-..._clean_claim_package.pdf.
- Set the browser zoom to about 110 percent so the back row can read.

## Timeline

| Time | On screen | Say |
| --- | --- | --- |
| 0:00 | Dashboard with no claims. The sidebar shows Demo mode. | Every claim review comes down to four questions. Is it covered? Is the evidence complete? Do the documents agree? How much could be paid? ClaimFlow AI prepares that file with seven agents, and a person decides. I'm running in Demo Mode, so this works with no internet. |
| 0:20 | Press Load the demo claims. Open "How the AI is used in this app" for a few seconds, then close it. Open Claim workspace and choose the claim for Ayesha Khan. | I'll load three made-up claims. This one is a rear-end collision with a clean file. This panel shows where each idea appears: generative AI, RAG, a multi-agent workflow, and a human in the loop. |
| 0:45 | Documents and evidence tab. Point at the four files and the photo. Show the Add evidence form but do not submit it. | Four documents are attached: the policy, an accident report, a repair quotation and a damage photo. Uploads are checked for type, size and real content before they are stored. I can add evidence here, or create a claim on the New claim page with sample files. |
| 1:10 | Agent workflow tab. Press Run agents and let the trace run. Then click Policy analysis in the row of steps. | One click runs seven agents. Each ends as done, warning, failed or skipped, and each result is saved. Intake reads the report and the quotation. Policy analysis searches the policy. Damage analysis looks at the photo. Then evidence, risk, assessment and compliance. This table shows what each step needs and what it passes on. All seven are done. |
| 1:50 | Policy search page. Show the indexed information. Ask: What is the excess for a collision claim? Press View source page. Then ask: Is a rental car provided while my car is repaired? | This is the policy RAG. The policy was read, split by section and indexed: six pages, twenty-nine chunks. I ask what the excess is for a collision claim. The answer is twenty-five thousand, with the quote, page three, Section 4. The page itself is one click away. Now something the policy does not say: is a rental car provided? The answer is: not found. It does not guess. |
| 2:40 | Back in the Claim workspace. Risk and verification tab: evidence table, then the risk result. Then the Assessment tab: the damage section with its LIMITED DEMO ANALYSIS label. | Evidence verification checked what a collision claim needs. Four of four items are present, and the policy number and the claimant's name match the documents. Risk analysis lists indicators for a person to look at. It never judges anyone's honesty, and it says so when a check could not run. Here every check ran and found no indicators. One honest note: in Demo Mode the damage step does not look at the photo. It is labelled limited demo analysis and works from the file name. With a live vision model it describes the photo. |
| 3:10 | Assessment tab, Claim assessment: the working. | The amount comes from code, not from a model. The repair estimate is one hundred eighty-five thousand. The policy's excess is twenty-five thousand. The potential amount is one hundred sixty thousand. If an input were missing, nothing would be calculated and it would say which one. |
| 3:35 | Review and package page. Point at HUMAN DECISION (Pending) and AI RECOMMENDATION (Approve for Processing). Type your name and a note. Press Approve for processing. | Compliance review chose a recommendation with fixed rules: approve for processing. That is advice. The human decision is still pending. I type my name, add a note, and approve. If I disagreed with the AI, the app would require a written reason, and the package would record that I overrode it. |
| 4:15 | Press Prepare the claim package, then Download. Open the PDF on page 1. Scroll to the policy findings on page 2. | The claim package is one PDF. It shows the human decision beside the AI recommendation, the calculation with its working, each policy finding with its page, the evidence check, the risk result, and an audit trail of every agent step. |
| 4:50 | Stay on the PDF or the dashboard. | The agents prepare the file. Code does the arithmetic. The policy says what is covered. A person decides the claim. Thank you. |

## Say this about Demo Mode

> Demo Mode uses a scripted model. It does not understand language and it cannot see the photo. The damage step is labelled LIMITED DEMO ANALYSIS and works from the file name and basic image checks. Everything after the model is real application code: PDF reading, section-aware chunking, vector search, quote checks, the evidence rules, the risk rules, the calculator, the orchestrator, the human review and the PDF.

## What you should see

| Moment | Expected on screen |
| --- | --- |
| After Run agents | Seven steps, all done. Intake: repair estimate 185000.00. Policy: coverage yes, deductible 25000.00, limit 1500000.00. Evidence: 4 of 4 required items are present. Risk: No risk indicators were found. Every planned check was performed. Assessment: potential claim amount 160000.00. Compliance: Approve for processing. |
| Policy search | 6 pages, 29 chunks. Answer: the excess for a collision claim is PKR 25,000, Page 3, Section 4. Excess. Rental car: This was not found in the policy. |
| Review page | HUMAN DECISION: Pending. AI RECOMMENDATION: Approve for Processing. After approval: HUMAN DECISION: Approved for Processing, and the line "The reviewer agreed with the AI recommendation." |
| Package PDF | Banner: FINAL: approved for processing by a person. HUMAN DECISION and AI RECOMMENDATION as separate rows. Calculation: 185000.00, 25000.00, 160000.00. Policy findings with pages. Damage analysis (limited demo analysis). Audit trail of the seven steps. |

## Key numbers

| Item | Value |
| --- | --- |
| Repair estimate, excess, potential amount | 185,000.00, 25,000.00, 160,000.00 |
| Where the excess comes from | Page 3, Section 4. Excess |
| Coverage limit | 1,500,000.00 (page 1) |
| Sample policy | 6 pages, 29 chunks, 28 section labels |
| Agents | 7 |
| Model calls per claim | At most 6 |
| Automated tests | 338 pass, 3 run only where PyMuPDF, FAISS and fastembed are installed |

## If you have 30 extra seconds

Use the Bilal Raza claim. Run the agents. The evidence check shows the repair quotation is missing, the risk result says the assessment is incomplete, and the recommendation is to request more information. On the Documents tab press Attach the repair quotation (demo). On the Agent workflow tab press Re-run 5 affected steps. Say: "Only five of seven steps run again. The policy and the photo are not repeated. The amount is sixty-seven thousand five hundred."

## If you are running long

- Skip the rental car question.
- Skip opening the How the AI is used panel.
- Do not open the PDF. Show the Download button and say what the PDF contains.

## If something goes wrong

| Problem | Do this |
| --- | --- |
| The app will not start or a page errors | Run python scripts/run_demo_workflow.py in a terminal. Show the trace it prints and open the PDF it writes. |
| Load the demo claims shows an error | Read the message aloud. It names the problem. Use the sample files on the New claim page as the fallback, or the terminal script. |
| The network drops during a Live AI demo | Turn the Demo mode switch in the sidebar on and continue. |
| The answer to a policy question is not what you expected | Ask one of the two prepared questions instead. |
| You are asked to run Live AI and have not set a key | Say the switch needs a key in .env. Show the Settings page and the provider test. |

## Do not say

- That Demo Mode looks at or understands the photo. It does not.
- That Live AI has been tested end to end, unless you have done that before the demo.
- That the system detects fraud. It lists risk indicators and inconsistencies for a person.
- That the system approves claims. A person decides.
- Any figure for time saved or accuracy. Neither was measured.
- That it reads scanned documents. There is no OCR.

## Likely questions

| Question | Answer |
| --- | --- |
| How do you stop the model inventing policy clauses? | A finding is kept only if its quote is found word for word in the cited chunk, and an amount only if it is inside the quote. Otherwise the result is not found. |
| Who calculates the money? | Code. A fixed formula in Decimal: the larger of (estimate minus deductible) and zero, capped at the limit. The assessment step makes no model call. |
| Is the demo real AI? | Demo Mode uses a scripted model so it runs offline. It is labelled as such. Live AI uses Gemini, Groq or OpenRouter through the same workflow, and the provider layer is tested with mocked responses. |
| What if the policy is a scan? | The system says the policy could not be parsed as readable text and does not guess. There is no OCR yet. |
| What stops a document from giving instructions to the AI? | Document text is framed as untrusted and instruction-like phrases are reported as a risk indicator, not followed. The recommendation is chosen by fixed rules, not by the model. |
| Does it decide claims? | No. It recommends. Only a person can approve, request information or escalate, and overriding the AI needs a written reason. |
| What next? | OCR, percentage and additional excesses, a policy period check, accounts and roles, and evaluation on real labelled claims. |
