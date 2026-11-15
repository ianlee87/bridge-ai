"""Evaluate Bridge on the cases in test_phrases.txt, from text only (no audio).

Two steps:
  1. Generate: each case goes through what main.py does after speech-to-text
     (urgent keyword check, then Bonsai + the first-person filter). The 14 normal
     cases run twice: WITH the pointed-at object and WITHOUT it.
  2. Judge: YOU read each case's options and type y (one of them says what the
     patient meant) or n. Answers are saved to eval_results.csv as you go.

A keyword-overlap score is kept as a second, rough column. It counts shared
words, not meaning, so trust your y/n over it.

Run:
    .venv/bin/python eval.py           generate, then judge (if run in a terminal)
    .venv/bin/python eval.py --judge   judge rows not marked yet (no Bonsai calls)
"""
import argparse
import csv
import re
import sys
import time

from rich.console import Console
from rich.table import Table

from brain import Brain
from fragments import check_urgent

console = Console()
RESULTS_CSV = "eval_results.csv"
COLUMNS = ["case", "run", "caregiver", "fragments", "pointing_at", "expected",
           "option_1", "option_2", "option_3", "filtered_out",
           "keyword_score", "judge", "bonsai_ms"]

KEYWORD_HIT_AT = 0.5   # keyword score at or above this counts as a keyword "hit"

# Small words that don't carry meaning, ignored by the keyword score.
STOP_WORDS = set("""
a an the i i'd i'm im me my to of in on for and or is it it's its be am are
was can could would will you your please some that this just with by at so
do does like want yes no not
""".split())


# ---------------------------------------------------------------------------
# Loading cases and the keyword score
# ---------------------------------------------------------------------------

def load_cases(path="test_phrases.txt"):
    cases = []
    for line in open(path):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        caregiver, frags, pointing, expected = [p.strip() for p in line.split("|")]
        cases.append({
            "caregiver": None if caregiver == "none" else caregiver.strip('"'),
            "fragments": [f.strip() for f in frags.split("...") if f.strip()],
            "pointing": None if pointing == "none" else pointing,
            "expected": expected,
        })
    return cases


def content_words(sentence):
    words = re.findall(r"[a-z']+", sentence.lower().replace("’", "'"))
    # Very simple stemming: drop trailing "s" so "blankets" matches "blanket".
    return {w.rstrip("s") for w in words if w not in STOP_WORDS}


def keyword_score(option, expected):
    """Fraction of the expected sentence's content words found in the option."""
    want = content_words(expected)
    if not want:
        return 0.0
    return len(want & content_words(option)) / len(want)


# ---------------------------------------------------------------------------
# Step 1: generate options with Bonsai and save them
# ---------------------------------------------------------------------------

def generate():
    brain = Brain()
    t0 = time.perf_counter()
    brain.warm_up()
    console.print(f"[green]Bonsai warm[/] [dim]({(time.perf_counter() - t0) * 1000:.0f} ms)[/]")

    rows = []
    for number, c in enumerate(load_cases(), start=1):
        base = {"case": number, "caregiver": c["caregiver"] or "",
                "fragments": " ... ".join(c["fragments"]), "expected": c["expected"]}

        # The urgent case: the keyword alert must fire and Bonsai must not run.
        # This is judged automatically.
        if c["expected"] == "URGENT":
            reason = check_urgent(c["fragments"])
            rows.append({**base, "run": "urgent", "pointing_at": c["pointing"] or "",
                         "option_1": f"URGENT alert ({reason})" if reason else "no alert",
                         "judge": "y" if reason else "n"})
            continue

        for run in ("with pointing", "without pointing"):
            pointing = c["pointing"] if run == "with pointing" else None
            row = {**base, "run": run, "pointing_at": pointing or ""}
            reason = check_urgent(c["fragments"])
            if reason:
                # A wrong alarm on a normal case: Bonsai never runs, so it's a miss.
                row.update(option_1=f"FALSE URGENT ({reason})", judge="n")
            else:
                t0 = time.perf_counter()
                result = brain.suggest(c["fragments"], c["caregiver"], pointing)
                row["bonsai_ms"] = round((time.perf_counter() - t0) * 1000)
                for i, option in enumerate(result["options"][:3], start=1):
                    row[f"option_{i}"] = option
                row["filtered_out"] = " | ".join(result["dropped"])
                row["keyword_score"] = round(max(
                    (keyword_score(o, c["expected"]) for o in result["options"]),
                    default=0.0), 2)
            rows.append(row)
            console.print(f"[dim]case {number:>2} {run:<16} done[/]")
    keep_old_judgments(rows)
    save(rows)
    console.print(f"Saved {len(rows)} rows to {RESULTS_CSV}\n")
    return rows


def options_of(row):
    """The row's options, sorted, so the same options in a new order count as unchanged."""
    return sorted(row.get(f"option_{i}") or "" for i in (1, 2, 3))


def keep_old_judgments(rows):
    """Compare with the previous eval_results.csv (if any).
    - Same options as last time: copy your old y/n, so you don't judge it twice.
    - Different options: print what changed, and leave it for you to judge."""
    try:
        old = {(r["case"], r["run"]): r for r in load_results()}
    except FileNotFoundError:
        return
    changed = 0
    for row in rows:
        before = old.get((str(row["case"]), row["run"]))
        if not before or row["run"] == "urgent":
            continue
        if options_of(before) == options_of(row):
            row["judge"] = before.get("judge", "")
            continue
        changed += 1
        console.print(f"[bold]Changed: case {row['case']} ({row['run']})[/]  {row['fragments']}")
        for o in options_of(before):
            if o and o not in options_of(row):
                console.print(f"  [red]- {o}[/]")
        for o in options_of(row):
            if o and o not in options_of(before):
                console.print(f"  [green]+ {o}[/]")
    console.print(f"\n{changed} rows changed since last run (those need judging again).")


def save(rows):
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in COLUMNS})


def load_results():
    with open(RESULTS_CSV, newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Step 2: you judge each row with y / n
# ---------------------------------------------------------------------------

def show_score(row):
    """Keyword score as text. (0.0 is a real score, so don't treat it as empty.)"""
    score = row.get("keyword_score")
    return "-" if score in (None, "") else str(score)


def show_row(row):
    console.rule(f"[bold]Case {row['case']}  ({row['run']})")
    console.print(f"Caregiver asked: {row['caregiver'] or '-'}")
    console.print(f"Fragments:       [cyan]{row['fragments']}[/]")
    console.print(f"Pointing at:     {row['pointing_at'] or '-'}")
    console.print(f"Expected:        [bold]{row['expected']}[/]")
    for i in (1, 2, 3):
        if row.get(f"option_{i}"):
            console.print(f"  {i}. {row[f'option_{i}']}")
    if row.get("filtered_out"):
        console.print(f"[dim]  filtered out: {row['filtered_out']}[/]")
    console.print(f"[dim]  keyword score: {show_score(row)}[/]")


def judge(rows):
    """Ask y/n for every row that isn't judged yet. Saves after each answer."""
    todo = [r for r in rows if not r.get("judge")]
    if not todo:
        console.print("All rows are already judged.")
        return
    console.print(f"[bold]{len(todo)} rows to judge.[/] For each: does ANY option say what "
                  "the patient meant?\n  y = yes   n = no   s = skip   q = save and quit\n")
    for row in todo:
        show_row(row)
        while True:
            answer = input("Match? [y/n/s/q] ").strip().lower()
            if answer in ("y", "n", "s", "q"):
                break
        if answer == "q":
            break
        if answer in ("y", "n"):
            row["judge"] = answer
            save(rows)      # save every answer so nothing is lost
    console.print()


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def summary(rows):
    table = Table(title="Results (your y/n first, keyword score second)")
    table.add_column("#", justify="right")
    table.add_column("Fragments")
    table.add_column("With pointing", justify="center")
    table.add_column("Without", justify="center")

    def cell(row):
        mark = {"y": "[green]y[/]", "n": "[red]n[/]"}.get(row.get("judge"), "[yellow]?[/]")
        return f"{mark}  [dim]kw {show_score(row)}[/]"

    by_run = {"with pointing": [], "without pointing": []}
    for row in rows:
        if row["run"] in by_run:
            by_run[row["run"]].append(row)
    for w, wo in zip(by_run["with pointing"], by_run["without pointing"]):
        table.add_row(str(w["case"]), w["fragments"], cell(w), cell(wo))

    def judged_rate(rs):
        judged = [r for r in rs if r.get("judge") in ("y", "n")]
        yes = sum(r["judge"] == "y" for r in judged)
        if not judged:
            return "not judged yet"
        text = f"{yes}/{len(judged)} ({yes / len(judged):.0%})"
        if len(judged) < len(rs):
            text += f", {len(rs) - len(judged)} unjudged"
        return text

    def keyword_rate(rs):
        hits = sum(float(r["keyword_score"] or 0) >= KEYWORD_HIT_AT for r in rs)
        return f"{hits}/{len(rs)}"

    def avg_ms(rs):
        times = [int(r["bonsai_ms"]) for r in rs if r.get("bonsai_ms")]
        return f"{sum(times) / len(times):.0f} ms" if times else "-"

    w, wo = by_run["with pointing"], by_run["without pointing"]
    table.add_section()
    table.add_row("", "[bold]Your hit rate[/]", f"[bold]{judged_rate(w)}[/]",
                  f"[bold]{judged_rate(wo)}[/]")
    table.add_row("", f"keyword hits (>= {KEYWORD_HIT_AT})", keyword_rate(w), keyword_rate(wo))
    table.add_row("", "avg Bonsai time", avg_ms(w), avg_ms(wo))
    console.print(table)

    for row in rows:
        if row["run"] == "urgent":
            ok = "[green]FIRED[/]" if row["judge"] == "y" else "[red]DID NOT FIRE[/]"
            console.print(f"Urgent case {row['case']} ({row['fragments']}): {ok}  {row['option_1']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Bridge on test_phrases.txt")
    parser.add_argument("--judge", action="store_true",
                        help="judge the saved eval_results.csv without calling Bonsai again")
    args = parser.parse_args()

    rows = load_results() if args.judge else generate()
    if sys.stdin.isatty():
        judge(rows)
    else:
        console.print("[yellow]Not running in a terminal, so judging is skipped. "
                      "Run: .venv/bin/python eval.py --judge[/]\n")
    summary(rows)


if __name__ == "__main__":
    main()
