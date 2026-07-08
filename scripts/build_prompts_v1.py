#!/usr/bin/env python3
"""Build prompts/v1 of the decibel-tts English text corpus.

Subcommands:
  build  — parse Arctic + Harvard, extract/filter/sample Gutenberg prose,
           template-generate the normalization set. Writes 4 JSONL files.
  merge  — merge agent-authored parts (prosody_a/b, tags_a/b, edge), validate
           every category file, write manifest.json with size estimates.

All randomness is seeded: same sources in -> same corpus out.
See TTS_RESEARCH_AND_ROADMAP.txt (repo root) for the design rationale.
"""

import argparse
import html as htmllib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

SEED = 42
WORDS_PER_HOUR = 10_000  # speech rate rule of thumb used across the roadmap
CHARS_PER_HOUR = 55_000

GUTENBERG_BOOKS = {
    "1342": "Pride and Prejudice",
    "1661": "The Adventures of Sherlock Holmes",
    "345": "Dracula",
    "174": "The Picture of Dorian Gray",
    "205": "Walden",
    "84": "Frankenstein",
    "11": "Alice's Adventures in Wonderland",
}
GUTENBERG_TARGET = 7000
GUTENBERG_PER_BOOK_CAP = 1300

CANONICAL_TAGS = ["<laugh>", "<chuckle>", "<sigh>", "<gasp>", "<whisper>", "<pause>"]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"  wrote {len(rows):>6} -> {path.name}")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                sys.exit(f"invalid JSON at {path}:{i}: {e}")
    return rows


# ---------------------------------------------------------------- arctic ----

def build_arctic(sources: Path) -> list[dict]:
    pat = re.compile(r'^\(\s*(arctic_\w+)\s+"(.+)"\s*\)\s*$')
    rows = []
    for line in (sources / "cmuarctic.data").read_text(encoding="utf-8").splitlines():
        m = pat.match(line.strip())
        if not m:
            continue
        text = m.group(2).strip()
        if not text.endswith((".", "!", "?")):
            text += "."
        rows.append({"id": m.group(1), "text": text, "category": "phonetic",
                     "source": "arctic", "prompt_version": "v1"})
    assert len(rows) == 1132, f"expected 1132 arctic prompts, got {len(rows)}"
    return rows


# --------------------------------------------------------------- harvard ----

def build_harvard(sources: Path) -> list[dict]:
    raw = (sources / "harvard.html").read_text(encoding="utf-8", errors="replace")
    items = re.findall(r"<li>(.*?)</li>", raw, flags=re.S | re.I)
    rows = []
    for i, item in enumerate(items, 1):
        text = htmllib.unescape(re.sub(r"<[^>]+>", "", item)).strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        list_no, sent_no = (i - 1) // 10 + 1, (i - 1) % 10 + 1
        rows.append({"id": f"harvard_l{list_no:02d}_s{sent_no:02d}", "text": text,
                     "category": "phonetic", "source": "harvard", "prompt_version": "v1"})
    assert len(rows) == 720, f"expected 720 harvard sentences, got {len(rows)}"
    return rows


# ------------------------------------------------------------- gutenberg ----

ALLOWED_CHARS = re.compile(r"^[A-Za-z0-9 .,;:!?'\"()—-]+$")
SENT_SPLIT = re.compile(r"(?<=[.!?])[\"']?\s+(?=[\"'A-Z])")


def clean_gutenberg_text(raw: str) -> str:
    start = re.search(r"\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", raw)
    end = re.search(r"\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", raw)
    body = raw[start.end() if start else 0: end.start() if end else len(raw)]
    body = body.replace("“", '"').replace("”", '"')
    body = body.replace("‘", "'").replace("’", "'")
    body = body.replace("_", "")  # Gutenberg italics markers
    return body


def extract_sentences(body: str) -> list[str]:
    out = []
    for para in re.split(r"\n\s*\n", body):
        para = re.sub(r"\s+", " ", para).strip()
        if not para or para.isupper():
            continue
        if re.match(r"^(CHAPTER|BOOK|VOLUME|LETTER|PART)\b", para, re.I):
            continue
        for sent in SENT_SPLIT.split(para):
            sent = sent.strip().strip('"').strip()
            words = sent.split()
            if not (3 <= len(words) <= 45):
                continue
            if not sent[0].isupper() or not sent.endswith((".", "!", "?")):
                continue
            if not ALLOWED_CHARS.match(sent):
                continue
            if sent.count('"') % 2 != 0:
                continue
            caps = sum(1 for w in words if len(w) > 1 and w.isupper())
            if caps > 1 or "Gutenberg" in sent or "ebook" in sent.lower():
                continue
            out.append(sent)
    return out


def build_gutenberg(sources: Path, rng: random.Random) -> list[dict]:
    per_book: dict[str, list[str]] = {}
    seen = set()
    for book_id in GUTENBERG_BOOKS:
        raw = (sources / f"pg{book_id}.txt").read_text(encoding="utf-8", errors="replace")
        sents = []
        for s in extract_sentences(clean_gutenberg_text(raw)):
            key = s.lower()
            if key not in seen:
                seen.add(key)
                sents.append(s)
        rng.shuffle(sents)
        per_book[book_id] = sents[:GUTENBERG_PER_BOOK_CAP]
        print(f"  pg{book_id} ({GUTENBERG_BOOKS[book_id]}): "
              f"{len(sents)} usable, kept {len(per_book[book_id])}")

    pool = [(bid, s) for bid, sents in per_book.items() for s in sents]
    rng.shuffle(pool)
    pool = pool[:GUTENBERG_TARGET]
    return [{"id": f"gut_{i:05d}", "text": s, "category": "general",
             "source": f"gutenberg:{bid}", "prompt_version": "v1"}
            for i, (bid, s) in enumerate(pool, 1)]


# ---------------------------------------------------------- normalization ----

FIRST = ["Priya", "Marcus", "Wei", "Fatima", "Diego", "Anya", "Kenji", "Zara",
         "Tunde", "Lars", "Meera", "Ravi", "Sofia", "Omar", "Nina", "Arjun"]
CITY = ["Mumbai", "Austin", "Berlin", "Nairobi", "Osaka", "Toronto", "Pune",
        "Lisbon", "Denver", "Jakarta", "Manchester", "Seoul"]
MONTH = ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"]
DAY = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
STORE = ["the hardware store", "that new cafe", "the pharmacy", "the airport lounge",
         "the corner bakery", "the electronics shop"]
ITEM = ["headphones", "a mechanical keyboard", "the standing desk", "winter boots",
        "a coffee grinder", "the microphone", "new tires", "a graphics card"]
ACRO = ["NASA", "HTML", "GPU", "API", "USB", "HDMI", "PDF", "GDP", "FAQ", "CEO",
        "ETA", "DIY", "RSVP", "ASAP", "NGO", "ATM", "GPS", "URL", "SIM", "OTP"]
UNITS = [("kilometers", "drive"), ("miles", "run"), ("kilograms", "lift"),
         ("pounds", "carry"), ("liters", "pour"), ("gallons", "haul"),
         ("meters", "measure"), ("degrees Celsius", "reach"),
         ("gigabytes", "download"), ("megabits per second", "get")]
SITE = ["reso.audio", "decibel.dev", "example.com", "openweather.org",
        "translate.google.com", "news.ycombinator.com"]


def _price(rng):
    cur = rng.choice(["$", "₹", "€", "£"])
    amt = rng.choice([f"{rng.randint(1, 99)}.{rng.randint(0, 99):02d}",
                      f"{rng.randint(100, 9999):,}", f"{rng.randint(1, 20)}"])
    t = [f"It's on sale for {cur}{amt} until {rng.choice(DAY)}.",
         f"They quoted me {cur}{amt} for {rng.choice(ITEM)}, which feels steep.",
         f"Wait, {cur}{amt}? Last week it was half that!",
         f"{rng.choice(FIRST)} paid {cur}{amt} at {rng.choice(STORE)} without blinking.",
         f"The subscription jumped from {cur}{amt} to {cur}{rng.randint(10, 99)}.99 a month.",
         f"Shipping alone costs {cur}{amt}, so I'll pick it up myself."]
    return rng.choice(t)


def _date(rng):
    d, m, y = rng.randint(1, 28), rng.choice(MONTH), rng.randint(1987, 2027)
    t = [f"The wedding is on {m} {d}, {y}, somewhere near {rng.choice(CITY)}.",
         f"Her visa expires {d}/{rng.randint(1, 12)}/{y}, so we should hurry.",
         f"We moved here back in {y}, the same week as the {m} storms.",
         f"Registration closes on the {d}th of {m}.",
         f"My flight got moved from {m} {d} to {m} {min(d + 3, 28)}.",
         f"He was born on {rng.choice(DAY)}, {m} {d}, {y}, at dawn."]
    return rng.choice(t)


def _time(rng):
    h12, mi = rng.randint(1, 12), rng.choice([0, 5, 10, 15, 20, 30, 40, 45, 50])
    ampm = rng.choice(["a.m.", "p.m."])
    t = [f"The standup moved to {h12}:{mi:02d} {ampm}, again.",
         f"Last train leaves at {rng.randint(0, 23)}:{mi:02d}, don't be late.",
         f"I set three alarms for {h12}:{mi:02d} {ampm} and slept through all of them.",
         f"Doors open at {h12} {ampm} sharp, {rng.choice(DAY)}.",
         f"It took 2 hours and {rng.randint(5, 55)} minutes end to end.",
         f"We landed at {h12}:{mi:02d} {ampm} local time in {rng.choice(CITY)}."]
    return rng.choice(t)


def _phone(rng):
    us = f"{rng.randint(200, 989)}-555-{rng.randint(1000, 9999):04d}"
    inn = f"+91 {rng.randint(70000, 99999)} {rng.randint(10000, 99999)}"
    t = [f"Call the front desk at {us} and ask for {rng.choice(FIRST)}.",
         f"My new number is {inn}, save it this time.",
         f"The helpline, {us}, keeps you on hold forever.",
         f"Text {rng.randint(40000, 59999)} to confirm your booking.",
         f"Dial {us}, extension {rng.randint(100, 999)}, for billing."]
    return rng.choice(t)


def _url(rng):
    s = rng.choice(SITE)
    user = rng.choice(FIRST).lower() + rng.choice(["", str(rng.randint(1, 99))])
    path = rng.choice(["docs", "blog", "pricing", "status", "careers", "support"])
    t = [f"Just go to {s} and click the second tab.",
         f"The docs live at {s}/api/v{rng.randint(1, 3)} now.",
         f"Email it to {user}@{s} before the demo.",
         f"I found the fix on {s}, of all places.",
         f"Sign-ups open at https://{s} at midnight.",
         f"Check {s}/{path} for the announcement.",
         f"His handle is @{user} on basically every platform.",
         f"The form bounced, so try {user}@{s} instead.",
         f"Bookmark {s}/{path}/{rng.randint(2019, 2026)} before they move it again.",
         f"Everything's mirrored at www.{s}, same login."]
    return rng.choice(t)


def _percent(rng):
    p = rng.choice([rng.randint(1, 99), round(rng.uniform(0.1, 9.9), 1)])
    t = [f"Revenue grew {p}% quarter over quarter, somehow.",
         f"Battery's at {p}%, we're not going to make it.",
         f"About {p}% of users never open the settings page.",
         f"They offered a {p}% discount if we pay annually.",
         f"Only three quarters of the class passed; that's {p}% worse than last year.",
         f"Humidity hit {p}% in {rng.choice(CITY)} today."]
    return rng.choice(t)


def _fraction(rng):
    fr = rng.choice(["1/2", "1/3", "2/3", "3/4", "1/4", "3/8", "5/8", "7/8",
                     "1 1/2", "2 1/2", "3 3/4", "one and a half", "two thirds",
                     "a quarter", "three fifths"])
    who, thing = rng.choice(FIRST), rng.choice(ITEM)
    t = [f"Add {fr} cups of flour, then whisk.",
         f"We're roughly {fr} of the way through the migration.",
         f"He ate {fr} of the pizza before anyone sat down.",
         f"The recipe calls for {fr} teaspoons of salt, not tablespoons.",
         f"{who} only finished {fr} of the assignment by {rng.choice(DAY)}.",
         f"Cut the board to {fr} inches, no more.",
         f"About {fr} of the audience left before the encore.",
         f"The tank is {fr} full, which should get us to {rng.choice(CITY)}.",
         f"Mix {fr} parts vinegar to one part water.",
         f"They refunded {fr} of the ticket price for {thing}.",
         f"It shrank to {fr} of its original size in the wash.",
         f"Turn the dial {fr} of a rotation to the left."]
    return rng.choice(t)


def _unit(rng):
    n = rng.choice([rng.randint(2, 500), round(rng.uniform(0.5, 99.9), 1)])
    unit, verb = rng.choice(UNITS)
    t = [f"It's a {n} {unit} {verb} from {rng.choice(CITY)}.",
         f"The package weighs {n} {unit}, believe it or not.",
         f"We managed to {verb} {n} {unit} before lunch.",
         f"Sensors read {n} {unit} at the peak.",
         f"That's {n} {unit} more than the spec allows."]
    return rng.choice(t)


def _acronym(rng):
    a, b = rng.sample(ACRO, 2)
    t = [f"The {a} report contradicts what the {b} team claimed.",
         f"Send me the {a} file as a {b} link, whichever's faster.",
         f"She joined {a} right after finishing her PhD.",
         f"Our {a} integration broke when the {b} changed.",
         f"Ask the {a} desk; they handle every {b} request."]
    return rng.choice(t)


def _ordinal(rng):
    o = rng.choice(["1st", "2nd", "3rd", "4th", "5th", "12th", "21st", "42nd", "100th"])
    y = rng.randint(1850, 2026)
    t = [f"She finished {o} in her age group, out of hundreds.",
         f"This is the {o} time I've explained the invoice.",
         f"The {o} floor button never works in that elevator.",
         f"Founded in {y}, it's celebrating its {o} anniversary.",
         f"Turn left on {o} Avenue, past the fountain."]
    return rng.choice(t)


def _address(rng):
    n = rng.randint(2, 9999)
    t = [f"Deliveries go to {n} Juniper Lane, apartment {rng.randint(1, 40)}B.",
         f"The office is at {n} MG Road, {rng.choice(CITY)}, {rng.randint(110001, 999999)}.",
         f"We're at {n} West {rng.randint(2, 99)}th Street, buzz twice.",
         f"Mail it to P.O. Box {n}, {rng.choice(CITY)}."]
    return rng.choice(t)


def _version(rng):
    v = f"{rng.randint(0, 12)}.{rng.randint(0, 20)}.{rng.randint(0, 9)}"
    t = [f"The bug only shows up in version {v} on ARM.",
         f"Please upgrade to v{v} before filing tickets.",
         f"Model {rng.choice(['X', 'Z', 'RS'])}-{rng.randint(100, 900)} shipped with firmware {v}.",
         f"Rollback to {v} took exactly {rng.randint(4, 59)} minutes.",
         f"Order number {rng.randint(10 ** 7, 10 ** 8 - 1)} still says processing."]
    return rng.choice(t)


def _math(rng):
    a, b = rng.randint(2, 99), rng.randint(2, 99)
    t = [f"Quick, what's {a} times {b}? No calculator.",
         f"Room {a} is next to room {a + 1}, obviously.",
         f"Temperatures swing from minus {rng.randint(1, 20)} to plus {rng.randint(21, 45)} here.",
         f"Chapter {a} spoils everything from chapter {b}, skip ahead carefully.",
         f"The score was {a} to {b} at halftime."]
    return rng.choice(t)


NORM_SUBTYPES = [("price", _price), ("date", _date), ("time", _time),
                 ("phone", _phone), ("url_email", _url), ("percent", _percent),
                 ("fraction", _fraction), ("unit", _unit), ("acronym", _acronym),
                 ("ordinal", _ordinal), ("address", _address),
                 ("version_id", _version), ("math_misc", _math)]
NORM_PER_SUBTYPE = 140  # 13 subtypes x 140 = 1820


def build_normalization(rng: random.Random) -> list[dict]:
    rows, n = [], 0
    for subtype, gen in NORM_SUBTYPES:
        seen, tries = set(), 0
        while len(seen) < NORM_PER_SUBTYPE and tries < NORM_PER_SUBTYPE * 200:
            tries += 1
            s = gen(rng)
            if s not in seen:
                seen.add(s)
                n += 1
                rows.append({"id": f"norm_{subtype}_{len(seen):04d}", "text": s,
                             "category": "normalization", "subtype": subtype,
                             "source": "generated", "prompt_version": "v1"})
        if len(seen) < NORM_PER_SUBTYPE:
            print(f"  warning: {subtype} exhausted at {len(seen)} unique sentences")
    return rows


# ----------------------------------------------------------------- merge ----

def validate(rows: list[dict], fname: str) -> None:
    ids = set()
    for r in rows:
        for field in ("id", "text", "category", "source"):
            assert field in r and r[field], f"{fname}: missing {field}: {r}"
        assert r["id"] not in ids, f"{fname}: duplicate id {r['id']}"
        ids.add(r["id"])
        if r["category"] == "tags":
            assert any(t in r["text"] for t in CANONICAL_TAGS), \
                f"{fname}: tag row without tag token: {r['id']}"
        else:
            assert not re.search(r"<[a-z]+>", r["text"]), \
                f"{fname}: unexpected tag token in {r['id']}"
        if r["category"] == "prosody":
            assert r.get("conversation_id") and r.get("turn_index"), \
                f"{fname}: prosody row missing conversation fields: {r['id']}"


def word_count(text: str) -> int:
    return len([w for w in re.sub(r"<[a-z]+>", "", text).split() if w])


def cmd_merge(out: Path) -> None:
    parts = out / "parts"
    merged = {
        "prosody.jsonl": read_jsonl(parts / "prosody_a.jsonl") + read_jsonl(parts / "prosody_b.jsonl"),
        "tags.jsonl": read_jsonl(parts / "tags_a.jsonl") + read_jsonl(parts / "tags_b.jsonl"),
        "edge.jsonl": read_jsonl(parts / "edge.jsonl"),
    }
    for fname, rows in merged.items():
        for r in rows:
            r.setdefault("prompt_version", "v1")
        write_jsonl(out / fname, rows)

    manifest = {"prompt_version": "v1", "seed": SEED, "files": {}, "totals": {}}
    tot_rows = tot_words = tot_chars = 0
    for f in sorted(out.glob("*.jsonl")):
        rows = read_jsonl(f)
        validate(rows, f.name)
        words = sum(word_count(r["text"]) for r in rows)
        chars = sum(len(r["text"]) for r in rows)
        manifest["files"][f.name] = {
            "rows": len(rows), "words": words, "chars": chars,
            "est_hours": round(words / WORDS_PER_HOUR, 2),
            "categories": dict(Counter(r["category"] for r in rows)),
        }
        tot_rows += len(rows); tot_words += words; tot_chars += chars

    tag_counts = Counter()
    for r in read_jsonl(out / "tags.jsonl"):
        for t in CANONICAL_TAGS:
            tag_counts[t] += r["text"].count(t)
    manifest["totals"] = {
        "rows": tot_rows, "words": tot_words, "chars": tot_chars,
        "est_hours_by_words": round(tot_words / WORDS_PER_HOUR, 2),
        "est_hours_by_chars": round(tot_chars / CHARS_PER_HOUR, 2),
        "tag_token_counts": dict(tag_counts),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["totals"], indent=2))


def cmd_build(sources: Path, out: Path) -> None:
    rng = random.Random(SEED)
    write_jsonl(out / "arctic.jsonl", build_arctic(sources))
    write_jsonl(out / "harvard.jsonl", build_harvard(sources))
    write_jsonl(out / "general.jsonl", build_gutenberg(sources, rng))
    write_jsonl(out / "normalization.jsonl", build_normalization(rng))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=["build", "merge"])
    ap.add_argument("--sources", type=Path, help="dir with downloaded source files")
    ap.add_argument("--out", type=Path, required=True, help="prompts/v1 output dir")
    args = ap.parse_args()
    if args.command == "build":
        if not args.sources:
            ap.error("build requires --sources")
        cmd_build(args.sources, args.out)
    else:
        cmd_merge(args.out)


if __name__ == "__main__":
    main()
