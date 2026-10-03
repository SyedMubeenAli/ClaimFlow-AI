"""
Regenerates samples/sample_motor_policy.pdf, the demo policy used for testing.

Only needed if you want to change the sample. The generated PDF is already in the
repository. This script needs reportlab:   pip install reportlab
Run from the project folder:               python scripts/make_sample_policy.py

The policy is fictional. It is written to contain the things the policy analyst must
find (cover, excess per claim type, sum insured, exclusions) and some things it must
report as not found (rental car cover, roadside assistance), so retrieval and the
"not found" behaviour can both be tested.
"""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

OUT = Path(__file__).resolve().parent.parent / "samples" / "sample_motor_policy.pdf"

styles = getSampleStyleSheet()
TITLE = ParagraphStyle("T", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=20, spaceAfter=4)
SUB = ParagraphStyle("S", parent=styles["Normal"], fontSize=10, textColor=colors.HexColor("#444444"), spaceAfter=10)
H1 = ParagraphStyle("H1", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=14, spaceBefore=12, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=11, spaceBefore=8, spaceAfter=3)
BODY = ParagraphStyle("B", parent=styles["BodyText"], fontName="Helvetica", fontSize=10, leading=14, spaceAfter=5)

story = []


def h1(text): story.append(Paragraph(text, H1))
def h2(text): story.append(Paragraph(text, H2))
def p(text): story.append(Paragraph(text, BODY))


# ------------------------------------------------------------------ page 1: schedule
story.append(Paragraph("Northgate Mutual Motor Insurance Policy", TITLE))
story.append(Paragraph("SAMPLE DOCUMENT FOR TESTING. This is a fictional policy and not a real contract.", SUB))
h1("Section 1. Policy schedule")
table = Table([
    ["Policy number", "POL-1001"],
    ["Policyholder", "Sample Policyholder"],
    ["Insured vehicle", "Honda City 2019, private use"],
    ["Period of insurance", "1 January 2026 to 31 December 2026"],
    ["Sum insured", "PKR 1,500,000"],
], colWidths=[55 * mm, 100 * mm])
table.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 10),
                           ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#EEEEEE"))]))
story += [table, Spacer(1, 8)]
p("The sum insured of PKR 1,500,000 is the most we will pay for any one claim. "
  "The excess for each type of claim is shown in Section 4.")
h1("Section 2. Definitions")
p("<b>Accident</b> means a sudden and unexpected event that causes damage to the insured vehicle. "
  "<b>Excess</b> means the amount you must pay towards each claim. <b>Insured vehicle</b> means the vehicle "
  "shown in the schedule. <b>We</b> and <b>us</b> mean Northgate Mutual. <b>You</b> means the policyholder.")
story.append(PageBreak())

# ------------------------------------------------------------------ page 2-3: cover
h1("Section 3. What we cover")
p("We cover loss of or damage to the insured vehicle during the period of insurance, as described below, "
  "up to the sum insured.")
h2("3.1 Collision and accidental damage")
p("We pay for accidental damage to the insured vehicle caused by collision with another vehicle, a person, an animal "
  "or an object. This includes damage caused when the insured vehicle overturns.")
h2("3.2 Fire and explosion")
p("We pay for loss of or damage to the insured vehicle caused by fire, lightning or explosion.")
h2("3.3 Theft")
p("We pay for the loss of the insured vehicle if it is stolen, and for damage caused during a theft or an attempted theft. "
  "You must report a theft to the police within 24 hours of discovering it.")
h2("3.4 Weather and natural disaster")
p("We pay for damage to the insured vehicle caused by flood, storm, hail or earthquake.")
h2("3.5 Malicious damage")
p("We pay for damage deliberately caused to the insured vehicle by someone who is not you and who is not acting "
  "with your permission. You must report malicious damage to the police within 24 hours.")
h2("3.6 Windscreen and glass")
p("We pay for the repair or replacement of broken glass in the insured vehicle. The most we will pay for glass "
  "damage is PKR 40,000 for each claim, and no excess applies to a glass-only claim.")
story.append(PageBreak())

# ------------------------------------------------------------------ page 4: excess
h1("Section 4. Excess")
p("You must pay the excess shown below towards each claim. We pay the cost of the claim after the excess has been "
  "taken off, up to the sum insured.")
excess = Table([
    ["Type of claim", "Excess for each claim"],
    ["Collision and accidental damage", "PKR 25,000"],
    ["Fire and explosion", "PKR 25,000"],
    ["Theft", "PKR 50,000"],
    ["Weather and natural disaster", "PKR 25,000"],
    ["Malicious damage", "PKR 25,000"],
], colWidths=[85 * mm, 60 * mm])
excess.setStyle(TableStyle([("FONTNAME", (0, 0), (-1, -1), "Helvetica"), ("FONTSIZE", (0, 0), (-1, -1), 10),
                            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EEEEEE"))]))
story += [excess, Spacer(1, 8)]
p("The excess for a collision and accidental damage claim is PKR 25,000 for each claim. The excess for a theft claim "
  "is PKR 50,000 for each claim.")
h2("4.1 Additional excess for young drivers")
p("If the driver was under 25 years old at the time of the incident, an additional excess of PKR 10,000 applies "
  "on top of the excess shown above.")
h2("4.2 Additional excess for unapproved repairers")
p("If you have the insured vehicle repaired at a garage that we have not approved, an additional excess of PKR 5,000 applies.")
story.append(PageBreak())

# ------------------------------------------------------------------ page 5: exclusions
h1("Section 5. What we do not cover")
p("We do not pay for any claim that arises from, or is connected with, the following.")
h2("5.1 Racing and speed testing")
p("We do not cover damage that happens while the insured vehicle is used for racing, rallying, pace-making, "
  "speed testing or track days.")
h2("5.2 Wear and tear")
p("We do not cover wear and tear, rust, corrosion, or mechanical or electrical breakdown or failure.")
h2("5.3 Alcohol and drugs")
p("We do not cover damage that happens while the driver is under the influence of alcohol or drugs, "
  "above the legal limit.")
h2("5.4 Unlicensed drivers")
p("We do not cover damage that happens while the insured vehicle is driven by a person who does not hold a valid "
  "driving licence.")
h2("5.5 Hire or reward")
p("We do not cover damage that happens while the insured vehicle is used as a taxi or for hire or reward.")
h2("5.6 Deliberate damage")
p("We do not cover damage that is caused deliberately by you or by anyone acting with your permission.")
h2("5.7 War and nuclear risks")
p("We do not cover loss or damage caused by war, invasion, civil war, or nuclear reaction or radiation.")
story.append(PageBreak())

# ------------------------------------------------------------------ page 6: claims
h1("Section 6. Making a claim")
h2("6.1 Telling us")
p("You must tell us about any incident that may lead to a claim within 7 days of the incident. "
  "For theft and malicious damage, you must also report the incident to the police within 24 hours.")
h2("6.2 What we need from you")
p("To assess a claim we need the completed claim form, a written account or accident report, photographs of the "
  "damage, and a repair quotation showing the total cost of repair. For theft and malicious damage we also need "
  "the police report number.")
h2("6.3 Our assessment")
p("We may appoint an assessor to inspect the insured vehicle. You must not begin repairs before we have agreed "
  "the repair estimate, unless the repair is needed to make the insured vehicle safe.")
h2("6.4 Settlement")
p("We aim to settle a valid claim within 30 days of receiving all the documents we have asked for.")
story.append(PageBreak())

# ------------------------------------------------------------------ page 7: general conditions
h1("Section 7. General conditions")
h2("7.1 Fraud")
p("If a claim is fraudulent in any way, we will not pay the claim and we may cancel this policy from the date of the "
  "fraudulent act.")
h2("7.2 Reasonable care")
p("You must take reasonable care to protect the insured vehicle from loss or damage, and to keep it in a "
  "roadworthy condition.")
h2("7.3 Cancellation")
p("You may cancel this policy at any time by writing to us. We may cancel it by giving you 14 days' notice in writing.")
h2("7.4 Governing law")
p("This policy is governed by the laws of Pakistan.")

OUT.parent.mkdir(parents=True, exist_ok=True)
SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=22 * mm, rightMargin=22 * mm, topMargin=20 * mm, bottomMargin=20 * mm,
                  title="Northgate Mutual Motor Insurance Policy (sample)", author="ClaimFlow AI sample data",
                  subject="Fictional policy for testing").build(story)
print("wrote", OUT)
