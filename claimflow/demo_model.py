"""
Demo Mode's model: a scripted, offline stand-in for the language model.

It exists so the whole workflow can run with no internet and no API key. It reads the
same prompts a live model would read and answers the same JSON questions, using plain
rules: it finds dates, amounts and clauses in the text it is given and quotes them back.

What this is and is not
  * It does not understand language and it cannot see photos. Damage descriptions come from
    photo file names and basic image statistics, and the answer says so.
  * Everything it returns still goes through the same code checks as a live model's output:
    quote verification, amount-in-text checks, the wording guard, and the deterministic
    calculator. So Demo Mode exercises the real pipeline, retrieval and verification included.
  * The UI says "Demo mode" wherever it is active. Nothing here pretends to be a live model.
"""

from __future__ import annotations

import io
import json
import re
from datetime import date

from .errors import LLMResponseError
from .llm import LLMProvider, LLMRequest, LLMResponse
from .retrieval import tokenize

_MONTHS = {name: i for i, name in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october",
     "november", "december"], 1)}
_AMOUNT = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SENTENCE_START = re.compile(r"\b(We|The|You|If|This|Any)\b")

_INCIDENT_WORDS = {
    "collision": ["collision", "accidental damage"], "theft": ["theft", "stolen"],
    "fire": ["fire", "explosion"], "weather": ["flood", "storm", "hail", "weather"],
    "vandalism": ["malicious", "vandal", "deliberately caused"],
}
# What in the claim description makes a policy exclusion worth flagging for a person.
_EXCLUSION_TRIGGERS = {
    "racing": ["racing", " race", "rally", "track day", "speed test"],
    "wear": ["wear and tear", "rust", "corrosion", "breakdown"],
    "alcohol": ["alcohol", "drunk", "drinking", "drugs"],
    "unlicensed": ["unlicensed", "no licence", "no license", "without a licence"],
    "hire": ["taxi", "careem", "uber", "for hire", "for reward"],
    "deliberate": ["deliberate", "on purpose", "intentional"],
    "war": [" war ", "nuclear"],
}
# Photo file names are the demo model's only clue. The first specific word found wins; generic ones are a fallback.
_PHOTO_SPECIFIC = [
    ("left", ["left front door", "left front wing"]), ("right", ["right front door", "right front wing"]),
    ("rear", ["rear bumper", "boot lid"]), ("back", ["rear bumper", "boot lid"]), ("front", ["front bumper"]),
    ("windscreen", ["windscreen"]), ("glass", ["windscreen"]), ("roof", ["roof"]), ("bonnet", ["bonnet"]),
]
_PHOTO_GENERIC = [("bumper", ["bumper"]), ("door", ["door panel"])]
_GENERIC_QUESTION_WORDS = {"what", "does", "do", "how", "long", "many", "much", "is", "are", "can", "my", "me", "mine",
                           "while", "when", "which", "who", "policy", "provide", "provided", "covered", "cover",
                           "have", "need", "get", "there", "any", "about", "tell", "give"}


def _dates_in(text: str) -> list[date]:
    found: list[date] = []
    for y, m, d in re.findall(r"\b(\d{4})-(\d{2})-(\d{2})\b", text):
        try:
            found.append(date(int(y), int(m), int(d)))
        except ValueError:
            pass
    for d, month, y in re.findall(r"\b(\d{1,2})(?:st|nd|rd|th)?\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\b", text):
        number = _MONTHS.get(month.lower())
        if number:
            try:
                found.append(date(int(y), number, int(d)))
            except ValueError:
                pass
    return found


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.;:!?])\s+", text) if len(s.strip()) > 8]


def _chunks(prompt: str) -> list[tuple[str, str, str]]:
    """(chunk id, section label, text) for every [CHUNK id] block in a prompt."""
    blocks = re.findall(r"\[CHUNK ([^\]]+)\] \(([^)]*)\)\n(.*?)(?=\n\n\[CHUNK |\n\nQuote the policy|\Z)", prompt, re.S)
    return [(cid, header.split(", page ", 1)[-1].split(", ", 1)[-1] if ", page " in header and ", " in header.split(", page ", 1)[-1] else "",
             " ".join(text.split())) for cid, header, text in blocks]


def _tighten(sentence: str, wanted: set[str]) -> str:
    """A table row can run into the sentence after it. Keep the last clause that still matches the question."""
    if len(sentence) <= 140:
        return sentence
    need = min(2, len(wanted)) or 1
    for match in reversed(list(_SENTENCE_START.finditer(sentence))):
        tail = sentence[match.start():]
        if len(wanted & set(tokenize(tail))) >= need:
            return tail
    return sentence


def _quote(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(0).strip() if match else None


def _incident_words(prompt: str) -> list[str]:
    label = (re.search(r"Incident type: ([^\n]+)", prompt) or [None, ""])[1].lower()
    for key, words in _INCIDENT_WORDS.items():
        if label.startswith(key):
            return words
    return []


class DemoProvider(LLMProvider):
    name = "demo"
    model = "demo-scripted-model"
    vision_model = "demo-scripted-model"

    def complete(self, request: LLMRequest) -> LLMResponse:
        task = (re.match(r"TASK: ([^\n]+)", request.prompt) or [None, ""])[1].strip()
        handler = {
            "extract claim facts": self._intake, "write search queries": self._queries,
            "extract policy terms": self._policy_terms, "describe damage photos": self._damage,
            "find inconsistencies": lambda r: {"inconsistencies": []},
            "explain recommendation": self._explain, "answer policy question": self._answer,
        }.get(task)
        if handler is None:
            raise LLMResponseError("The demo model has no script for this request.",
                                   hint="Turn off demo mode and use a live provider.")
        return LLMResponse(json.dumps(handler(request)), self.name, self.model)

    # ------------------------------------------------------------ intake

    @staticmethod
    def _intake(request: LLMRequest) -> dict:
        prompt = request.prompt
        description = (re.search(r"<description>\n(.*?)\n</description>", prompt, re.S) or [None, ""])[1]
        documents = re.findall(r'<document kind="([^"]+)" name="[^"]*">\n(.*?)\n</document>', prompt, re.S)
        reports = " \n".join(text for kind, text in documents if kind == "Accident report")
        quotes = " \n".join(text for kind, text in documents if kind == "Repair quotation")
        everything = f"{description}\n{reports}\n{quotes}"

        dated = re.search(r"(?im)^date of (?:accident|incident)[^\n]*", reports)
        dates = _dates_in(dated.group(0)) if dated else _dates_in(reports)
        total = re.findall(r"(?im)^[^\n]*\btotal\b[^\n]*?(\d[\d,]*(?:\.\d+)?)\s*\.?\s*$", quotes)
        police = re.search(r"(?i)police report (?:number|no\.?)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]{3,})", everything)
        location = re.search(r"(?im)^location\s*:\s*(.+?)\.?\s*$", reports)
        vehicle = re.search(r"(?im)^vehicle\s*:\s*(.+?)\.?\s*$", reports)
        injuries = False if re.search(r"(?i)\bno (?:one was )?injur|nobody (?:was )?injured|no injuries", everything) \
            else True if re.search(r"(?i)injur", everything) else None
        return {
            "incident_date": dates[0].isoformat() if dates else None,
            "location": location.group(1) if location else None,
            "vehicle": vehicle.group(1) if vehicle else None,
            "police_report_number": police.group(1) if police else None,
            "injuries_mentioned": injuries,
            "repair_estimate": total[-1] if total else None,
            "summary": " ".join(_sentences(description)[:2]) or None,
        }

    # ------------------------------------------------------------ policy

    @staticmethod
    def _queries(request: LLMRequest) -> dict:
        label = (re.search(r"Incident type: ([^\n]+)", request.prompt) or [None, "claim"])[1].lower()
        return {"queries": [f"{label} cover accidental damage conditions", f"excess for {label} claim",
                            "exclusions racing wear and tear alcohol unlicensed"]}

    @staticmethod
    def _policy_terms(request: LLMRequest) -> dict:
        prompt = request.prompt
        words = _incident_words(prompt)
        kw = "(?:" + "|".join(re.escape(w) for w in words) + ")" if words else None
        happened = (re.search(r"What happened: ([^\n]*)", prompt) or [None, ""])[1].lower()
        result: dict = {"coverage": None, "deductible": None, "coverage_limit": None, "exclusions": []}
        for chunk_id, section, text in _chunks(prompt):
            if kw and result["coverage"] is None:
                quote = _quote(rf"We pay [^.]*\b{kw}\b[^.]*\.", text)
                if quote:
                    result["coverage"] = {"applies": "yes", "quote": quote, "chunk": chunk_id}
            if kw and result["deductible"] is None:
                quote = _quote(rf"The excess for [^.]*?\b{kw}\b[^.]*?\d[\d,]*[^.]*\.", text)
                if quote and "additional" not in quote.lower():
                    result["deductible"] = {"kind": "fixed", "amount": _AMOUNT.search(quote).group(0),
                                            "quote": quote, "chunk": chunk_id}
            if result["coverage_limit"] is None:
                quote = _quote(r"The [^.]*most we will pay[^.]*\.", text)
                if quote and _AMOUNT.search(quote):
                    result["coverage_limit"] = {"amount": _AMOUNT.search(quote).group(0), "quote": quote, "chunk": chunk_id}
            quote = _quote(r"We do not cover [^.]*\.", text)
            if quote and len(result["exclusions"]) < 8:
                title = re.sub(r"^\d+(?:\.\d+)*\s+", "", section.split(" / ")[-1]).strip() or "Exclusion"
                key = next((k for k in _EXCLUSION_TRIGGERS if k in title.lower()), None)
                hit = next((t for t in _EXCLUSION_TRIGGERS.get(key, []) if t in f" {happened} "), None)
                result["exclusions"].append({
                    "title": title, "applies_to_claim": hit is not None,
                    "reason": (f"The description mentions '{hit.strip()}'." if hit
                               else "Nothing in the description suggests this applies."),
                    "quote": quote, "chunk": chunk_id})
        return result

    @staticmethod
    def _answer(request: LLMRequest) -> dict:
        prompt = request.prompt
        question = (re.search(r"Question: ([^\n]+)", prompt) or [None, ""])[1]
        wanted = {t for t in tokenize(question) if t not in _GENERIC_QUESTION_WORDS}
        best: tuple[tuple[int, int, int], str, str] | None = None
        for chunk_id, _section, text in _chunks(prompt):
            for sentence in _sentences(text):
                score = len(wanted & set(tokenize(sentence)))
                if not score:
                    continue
                start = _SENTENCE_START.search(sentence)
                quote = _tighten(sentence[start.start():] if start and re.match(r"\d", sentence) else sentence, wanted)
                # More matching words first, then a sentence with a figure, then the shorter and more focused one.
                rank = (score, int(bool(re.search(r"\d", quote))), -len(quote))
                if best is None or rank > best[0]:
                    best = (rank, chunk_id, quote)
        enough = best is not None and (best[0][0] >= 2 or len(wanted) == 1)
        if not enough:
            return {"found": False, "answer": "", "citations": []}
        return {"found": True, "answer": best[2], "citations": [{"chunk": best[1], "quote": best[2]}]}

    # ------------------------------------------------------------ damage (no vision: file names and image statistics)

    @staticmethod
    def _damage(request: LLMRequest) -> dict:
        names = re.findall(r"(?m)^\d+\. (.+)$", request.prompt.split("Photos, in the order attached:")[-1].split("Return JSON")[0])
        description = (re.search(r"description \(untrusted\): ([^\n]*)", request.prompt) or [None, ""])[1].lower()
        photos, overlaps = [], 0
        for index, name in enumerate(names, 1):
            stem = name.lower().rsplit(".", 1)[0].replace("_", " ").replace("-", " ")
            areas = next((parts for word, parts in _PHOTO_SPECIFIC + _PHOTO_GENERIC if re.search(rf"\b{word}\b", stem)), [])
            image = request.images[index - 1].data if index <= len(request.images) else b""
            quality = _quality(image)
            overlaps += any(w in description for w in ("front", "rear", "left", "right", "door", "bumper") if w in stem)
            photos.append({"photo": index, "visible_damage": [f"damage to the {a}" for a in areas[:3]],
                           "affected_areas": areas[:4], "severity": "moderate" if areas else "unclear",
                           "image_quality": quality})
        return {"photos": photos,
                "overall_summary": "Demo mode cannot see images. This description comes from the photo file names "
                                   "and basic image checks only. Use a live vision model for a real reading.",
                "consistent_with_description": "yes" if overlaps else "unclear"}

    # ------------------------------------------------------------ compliance

    @staticmethod
    def _explain(request: LLMRequest) -> dict:
        prompt = request.prompt
        recommendation = (re.search(r"Recommendation: ([^\n]+)", prompt) or [None, "a decision"])[1]
        reasons = re.findall(r"(?m)^- (.+)$", prompt)
        text = f"The recommendation is to {recommendation.lower()}. {' '.join(reasons[:3])} A person makes the final decision."
        return {"explanation": text}


def _quality(data: bytes) -> str:
    """good or poor, from size and contrast. A real model judges blur and framing. This only catches the obvious."""
    try:
        from PIL import Image, ImageStat
        with Image.open(io.BytesIO(data)) as image:
            if min(image.size) < 200:
                return "poor"
            spread = ImageStat.Stat(image.convert("L")).stddev[0]
        return "good" if spread >= 12 else "poor"
    except Exception:
        return "unclear"
