"""Per-anchor grounding evidence as one offline page.

  python3 tools/arkitscenes_grounding_evidence.py \
      --grounding runs/arkit_relation_challenge/grounding.json \
      --report runs/arkit_relation_challenge/report.json \
      --out runs/arkit_relation_challenge/grounding_evidence.html

Shows, for every anchor, what the bridge ranked and why it admitted or
abstained -- the top candidates with their aggregate and per-slot scores, which
entity won each view slot, and whether the admitted uid matched the owner's
mapping. Renders committed values only; computes no metric and runs no script.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def build(grounding: dict, report: dict) -> str:
    resolution = report.get("anchor_resolution", {})
    by_anchor = {(r["scene_id"], r["anchor"]): r
                 for r in resolution.get("rows", [])}
    summary = resolution.get("summary", {})

    sections = []
    for scene in grounding["scenes"]:
        scene_id = scene["scene_id"]
        cards = []
        for entry in scene["anchors"]:
            outcome = by_anchor.get((scene_id, entry["anchor"]), {})
            state = outcome.get("outcome", "unscored")
            human = outcome.get("human_uid")
            rows = "".join(
                f'<tr class="{"top" if i == 0 else ""}">'
                f'<td><code>{esc(c["uid"])}</code></td>'
                f'<td class="n">{c["aggregate"]:.4f}</td>'
                f'<td class="n">{", ".join(f"{v:.4f}" for v in c["slot_scores"])}</td>'
                f'</tr>' for i, c in enumerate(entry["ranking"]))
            winners = ", ".join(f'slot {k} → <code>{esc(v)}</code>'
                                for k, v in entry["slot_winners"].items())
            verdict = {
                "correct": f'admitted <code>{esc(entry["uid"])}</code> — matches the owner',
                "wrong": (f'admitted <code>{esc(entry["uid"])}</code> — owner says '
                          f'<code>{esc(human)}</code>' if human else
                          f'admitted <code>{esc(entry["uid"])}</code> — the owner '
                          f'could not map this object at all'),
                "abstained": esc(entry.get("reason") or "abstained"),
            }.get(state, esc(state))
            cards.append(f"""
    <section class="card {esc(state)}">
      <h3>{esc(entry['anchor'])}</h3>
      <div class="verdict">{verdict}</div>
      <div class="meta">phrases: {", ".join(f"<code>{esc(p)}</code>" for p in entry["phrases"])}</div>
      <div class="meta">view-slot winners: {winners or "none"} ·
        agreeing slots: <b>{entry.get('agreeing_slots', 0)}</b> of the 2 required ·
        ranked over {entry.get('n_entities_ranked', 0)} entities</div>
      <table><thead><tr><th>uid</th><th>aggregate</th><th>per-slot scores</th></tr></thead>
        <tbody>{rows}</tbody></table>
    </section>""")
        sections.append(f'<h2>{esc(scene_id)}</h2>{"".join(cards)}')

    return f"""<!DOCTYPE html>
<meta charset="utf-8">
<title>Grounding evidence — {esc(grounding.get('prediction_sha256','')[:12])}</title>
<style>
:root {{ color-scheme: light dark; --bg:#fbfbfa; --fg:#1a1a19; --mut:#6b6b66;
  --line:#dcdcd6; --card:#fff; --ok:#1a7f4b; --okbg:#e8f5ee; --bad:#a3372c;
  --badbg:#fbecea; --idk:#6b6b66; --idkbg:#f0f0ec; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg:#16161a; --fg:#e8e8e4; --mut:#9a9a94; --line:#33333a; --card:#1e1e24;
  --ok:#6ede9f; --okbg:#163023; --bad:#f0938a; --badbg:#331d1a;
  --idk:#9a9a94; --idkbg:#26262c; }} }}
* {{ box-sizing:border-box; }}
body {{ margin:0 auto; padding:2rem 1.25rem 4rem; max-width:60rem; background:var(--bg);
  color:var(--fg); font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",sans-serif; }}
h1 {{ font-size:1.5rem; margin:0 0 .3rem; }}
h2 {{ font-size:1.05rem; margin:2rem 0 .75rem; padding-bottom:.3rem;
  border-bottom:1px solid var(--line); }}
h3 {{ font-size:.98rem; margin:0 0 .4rem; }}
p.sub {{ color:var(--mut); margin:0 0 .3rem; }}
code {{ font:12.5px ui-monospace,Menlo,monospace; background:var(--idkbg);
  padding:.08em .35em; border-radius:3px; }}
.card {{ border:1px solid var(--line); border-radius:8px; padding:.9rem 1rem;
  margin:0 0 .8rem; background:var(--card); }}
.card.correct {{ background:var(--okbg); border-color:var(--ok); }}
.card.wrong {{ background:var(--badbg); border-color:var(--bad); }}
.card.abstained {{ background:var(--idkbg); }}
.verdict {{ font-size:.9rem; margin-bottom:.5rem; }}
.card.correct .verdict {{ color:var(--ok); }}
.card.wrong .verdict {{ color:var(--bad); }}
.meta {{ color:var(--mut); font-size:.8rem; margin-bottom:.3rem; }}
table {{ border-collapse:collapse; width:100%; font-size:13px; margin-top:.5rem; }}
th,td {{ text-align:left; padding:.3rem .5rem; border-bottom:1px solid var(--line); }}
th {{ color:var(--mut); font-size:.72rem; text-transform:uppercase; }}
td.n {{ font-variant-numeric:tabular-nums; }}
tr.top td {{ font-weight:600; }}
footer {{ margin-top:2.5rem; padding-top:1rem; border-top:1px solid var(--line);
  color:var(--mut); font-size:.8rem; }}
</style>

<h1>Grounding bridge — per-anchor evidence</h1>
<p class="sub">prediction <code>{esc(grounding.get('prediction_sha256','')[:24])}…</code> ·
admission rule: the top aggregate entity must also win at least two independent
view slots · no confidence threshold</p>
<p class="sub">precision <b>{summary.get('precision')}</b> ·
coverage <b>{summary.get('coverage')}</b> ·
{summary.get('n_admitted')} admitted of {summary.get('n_anchors')} anchors,
{summary.get('n_correct')} correct</p>

{"".join(sections)}

<footer>Every value here is read from the committed grounding sidecar and
report. The page computes nothing and runs no script. A ceiling or oracle
number never appears on it: these are the oracle-free bridge's own
predictions, scored after the sidecar was sealed.</footer>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--grounding", type=Path, required=True)
    ap.add_argument("--report", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    page = build(json.loads(args.grounding.read_text()),
                 json.loads(args.report.read_text()))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(page)
    print(f"per-anchor evidence -> {args.out} "
          f"({args.out.stat().st_size / 1e3:.0f} kB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
