"""Support-relation review sheet: collect human resting truth, change nothing.

  python3 tools/arkitscenes_support_review_sheet.py --scene 41069025

Builds a fillable sheet asking the owner which target-owner pairs are real
gravity-resting contacts. It reads the frozen resting-evidence artifact and
writes a form. **It changes no support logic and no threshold**, and a test
asserts it imports nothing from the relations package.

THE DEFINITION UNDER REVIEW, IN THE OWNER'S WORDS
-------------------------------------------------
    gravity-resting contact on the supporter

Not proximity, not overlap, not "above". A cushion on a sofa qualifies. A
picture hanging over a table does not, however close it is. That distinction is
the whole point of collecting truth rather than lowering a threshold.

WHY THE SHEET IS NOT JUST THE 77 QUALIFYING PATCHES
----------------------------------------------------
Sampling only what the current algorithm likes would produce a key that can
confirm the algorithm and never contradict it -- every false negative would be
invisible by construction. Three sources are pooled instead:

  owner_confirmed     pairs the owner has already asserted, INCLUDED WHETHER OR
                      NOT the algorithm proposes them. obj_9 -> obj_13 is a
                      candidate; obj_23 -> obj_13 was rejected at footprint
                      overlap 0.4545 against a 0.50 gate. A key built from
                      candidates alone would have recorded only the first and
                      learned nothing.
  qualifying_patch    every one of the 77 geometry-qualifying patches, each via
                      its best target pairing.
  stratified_reject   rejected pairs sampled across fixed overlap bands, so the
                      near-misses AND the clear negatives are both represented.
                      Deterministic: highest overlap first inside each band,
                      ties by uid. No randomness, no seed to remember.

SEPARATION, ENFORCED BY THE OUTPUT SHAPE
----------------------------------------
The emitted JSON has two blocks that share only a pair id:

  human_relation_truth        what the owner says: supports / does_not_support
                              / unsure. No metric appears in it.
  system_candidate_evidence   candidate status, rejection reasons, footprint
                              overlap, vertical gap, patch identity and areas.
                              No human judgement appears in it.

Mixing them is how a threshold ends up fitted to the evidence that produced it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_REST = (REPO_ROOT / "runs" / "arkit_vertical_slice" / "sealed_pair_rest"
                / "41069025" / "entity_patch_rest.json")
DEFAULT_ENTITIES = (REPO_ROOT / "runs" / "arkit_label_image_ab_41069025"
                    / "rgb_tight" / "entities" / "manifest.json")
DEFAULT_IDS = (REPO_ROOT / "runs" / "arkitscenes_mask3d_transfer"
               / "bundle_arkitscenes_41069025" / "vertex_instance_ids.npy")
DEFAULT_DATA_ROOT = Path.home() / "Desktop/datasets/arkitscenes/Validation"
DEFAULT_OUT = REPO_ROOT / "eval" / "human_feedback"

# Owner-asserted resting contacts, included regardless of algorithm status.
OWNER_CONFIRMED = [("obj_9", "obj_13"), ("obj_23", "obj_13")]

# Fixed overlap bands and a fixed row budget each. Bands straddle the
# provisional 0.50 gate deliberately: the band just below it is where a false
# negative would hide, and the band above it is where a false positive would.
BANDS = (
    ("overlap >= 0.50", 0.50, 1.01, 10),
    ("0.40 <= overlap < 0.50", 0.40, 0.50, 8),
    ("0.25 <= overlap < 0.40", 0.25, 0.40, 6),
    ("0.10 <= overlap < 0.25", 0.10, 0.25, 6),
    ("0 < overlap < 0.10", 1e-9, 0.10, 6),
    ("overlap == 0", -1.0, 1e-9, 4),
)

DEFINITION = "gravity-resting contact on the supporter"


def _round(value, digits: int = 4):
    """Null-safe. A patch with no footprint overlap has no gap to report, and
    the artifact stores None rather than 0.0 there."""
    return None if value is None else round(value, digits)


def best_patch(pair: dict) -> dict | None:
    if not pair["evaluated_patches"]:
        return None
    return max(pair["evaluated_patches"],
               key=lambda p: p["overlap_ratio_target"])


def row_for(pair: dict, sources: list[str]) -> dict:
    patch = best_patch(pair)
    return {
        "pair_id": f'{pair["target_entity_uid"]}->{pair["owner_entity_uid"]}',
        "target_uid": pair["target_entity_uid"],
        "owner_uid": pair["owner_entity_uid"],
        "sources": sources,
        "system": {
            "relation_candidate": bool(pair["relation_candidate"]),
            "rejection_reasons": list(pair["rejection_reasons"]),
            "best_patch_uid": patch["patch_uid"] if patch else None,
            "overlap_ratio_target": _round(patch["overlap_ratio_target"])
            if patch else None,
            "vertical_gap_m": _round(
                patch["vertical_gap_at_overlap_centroid_m"]) if patch else None,
            "patch_footprint_m2": _round(patch["patch_footprint_xy_area_m2"])
            if patch else None,
            "target_footprint_m2": _round(patch["target_footprint_area_m2"])
            if patch else None,
            "patch_rejection_reasons": list(patch["rejection_reasons"])
            if patch else [],
        },
    }


def select_rows(pairs: list[dict]) -> tuple[list[dict], dict]:
    by_pair = {(p["target_entity_uid"], p["owner_entity_uid"]): p
               for p in pairs}
    chosen: dict[tuple[str, str], list[str]] = {}

    for key in OWNER_CONFIRMED:
        if key not in by_pair:
            raise ValueError(
                f"owner-confirmed pair {key} is absent from the resting "
                "artifact; the sheet would silently drop it")
        chosen.setdefault(key, []).append("owner_confirmed")

    # Every qualifying patch, via its best target pairing.
    per_patch: dict[str, tuple[float, tuple[str, str]]] = {}
    for pair in pairs:
        key = (pair["target_entity_uid"], pair["owner_entity_uid"])
        for patch in pair["evaluated_patches"]:
            score = patch["overlap_ratio_target"]
            if (patch["patch_uid"] not in per_patch
                    or score > per_patch[patch["patch_uid"]][0]):
                per_patch[patch["patch_uid"]] = (score, key)
    for _, key in per_patch.values():
        chosen.setdefault(key, []).append("qualifying_patch")

    # Stratified rejects, deterministic inside each band.
    evaluated = [p for p in pairs if p["evaluated_patches"]
                 and not p["relation_candidate"]]
    ranked = sorted(
        evaluated,
        key=lambda p: (-best_patch(p)["overlap_ratio_target"],
                       p["target_entity_uid"], p["owner_entity_uid"]))
    band_counts = {}
    for name, low, high, budget in BANDS:
        taken = 0
        for pair in ranked:
            if taken >= budget:
                break
            overlap = best_patch(pair)["overlap_ratio_target"]
            if not (low <= overlap < high):
                continue
            key = (pair["target_entity_uid"], pair["owner_entity_uid"])
            chosen.setdefault(key, []).append("stratified_reject")
            taken += 1
        band_counts[name] = taken

    rows = [row_for(by_pair[key], sorted(set(sources)))
            for key, sources in chosen.items()]
    rows.sort(key=lambda r: (-(r["system"]["overlap_ratio_target"] or -1.0),
                             r["pair_id"]))
    return rows, band_counts


def build_html(scene_id: str, rows: list[dict], band_counts: dict,
               provenance: dict) -> str:
    confirmed = {f"{t}->{o}" for t, o in OWNER_CONFIRMED}
    body = []
    for r in rows:
        s = r["system"]
        mark = ' <span class="tag">owner-confirmed</span>' \
            if r["pair_id"] in confirmed else ""
        reasons = ", ".join(s["rejection_reasons"] + s["patch_rejection_reasons"]) or "—"
        pics = r.get("renders")
        imgs = "" if not pics else f"""
        <div class="imgs">
          <div><img src="{pics['side']}" alt="side-on"><span>side-on</span></div>
          <div><img src="{pics['ctx'][0]}" alt="room A"><span>room A</span></div>
          <div><img src="{pics['ctx'][1]}" alt="room B"><span>room B</span></div>
        </div>"""
        body.append(f"""
      <article class="pair {'conf' if mark else ''}" data-pair="{r['pair_id']}">
        <div class="hd">
          <span class="tgt">{r['target_uid']}</span>
          <span class="arrow">rests on?</span>
          <span class="own">{r['owner_uid']}</span>{mark}
        </div>
        {imgs}
        <div class="ev">{'candidate' if s['relation_candidate'] else 'rejected'}
          · overlap {'—' if s['overlap_ratio_target'] is None else s['overlap_ratio_target']}
          · gap {'—' if s['vertical_gap_m'] is None else str(s['vertical_gap_m']) + ' m'}
          <br><span class="rsn">{reasons}</span></div>
        <div class="ans">
          <label><input type="radio" name="{r['pair_id']}" value="supports"> rests on</label>
          <label><input type="radio" name="{r['pair_id']}" value="does_not_support"> no</label>
          <label><input type="radio" name="{r['pair_id']}" value="unsure"> unsure</label>
        </div>
      </article>""")
    bands = "".join(f"<tr><td>{n}</td><td>{c}</td></tr>"
                    for n, c in band_counts.items())
    return f"""<title>Support-relation review — {scene_id}</title>
<style>
 :root {{ --bg:#fff; --fg:#15171a; --mut:#666; --line:#e3e5e8; --ac:#0a7d55; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --bg:#14161a; --fg:#e9ebee;
   --mut:#9aa1aa; --line:#2b2f36; --ac:#3fd39b; }} }}
 :root[data-theme="dark"] {{ --bg:#14161a; --fg:#e9ebee; --mut:#9aa1aa;
   --line:#2b2f36; --ac:#3fd39b; }}
 :root[data-theme="light"] {{ --bg:#fff; --fg:#15171a; --mut:#666;
   --line:#e3e5e8; --ac:#0a7d55; }}
 body {{ background:var(--bg); color:var(--fg); margin:0 auto; padding:24px;
   max-width:1080px; font:15px/1.5 ui-sans-serif,system-ui,sans-serif; }}
 h1 {{ font-size:20px; margin:0 0 4px; }}
 h2 {{ font-size:16px; margin:26px 0 8px; padding-top:12px;
   border-top:1px solid var(--line); }}
 .sub {{ color:var(--mut); font-size:13px; margin:0 0 14px; }}
 .def {{ border-left:3px solid var(--ac); padding:8px 14px; font-size:14.5px; }}
 table {{ border-collapse:collapse; width:100%; font-size:13.5px; }}
 th,td {{ text-align:left; padding:5px 7px; border-bottom:1px solid var(--line); }}
 th {{ color:var(--mut); font-weight:600; position:sticky; top:0;
   background:var(--bg); }}
 .num {{ text-align:right; font-variant-numeric:tabular-nums; }}
 .pairs {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(320px,1fr));
   gap:12px; }}
 .pair {{ border:1px solid var(--line); border-radius:8px; padding:9px; }}
 .pair.conf {{ background:rgba(10,125,85,.10); border-color:var(--ac); }}
 .hd {{ font-size:14px; margin-bottom:6px; }}
 .tgt {{ font-weight:700; color:#c0158f; }}
 .own {{ font-weight:700; color:#00857a; }}
 @media (prefers-color-scheme: dark) {{
   .tgt {{ color:#ff5cc8; }} .own {{ color:#3fd39b; }} }}
 .arrow {{ color:var(--mut); font-size:12px; margin:0 4px; }}
 .tag {{ font-size:10.5px; background:var(--ac); color:#fff; padding:1px 6px;
   border-radius:9px; margin-left:4px; }}
 .imgs {{ display:flex; gap:4px; margin-bottom:6px; }}
 .imgs div {{ flex:1; text-align:center; }}
 .imgs img {{ width:100%; border:1px solid var(--line); border-radius:4px;
   display:block; }}
 .imgs span {{ font-size:10px; color:var(--mut); }}
 .ev {{ font-size:12px; color:var(--mut); margin-bottom:6px; }}
 .ans label {{ display:inline-block; }}
 .sw {{ display:inline-block; width:10px; height:10px; border-radius:2px;
   vertical-align:middle; }}
 .tgt-sw {{ background:#ff28be; }} .own-sw {{ background:#00b3a4; }}
 .rsn {{ color:var(--mut); font-size:12px; }}
 .sysc {{ font-size:12.5px; }}
 tr.conf {{ background:rgba(10,125,85,.10); }}
 label {{ margin-right:9px; white-space:nowrap; font-size:12.5px; }}
 button {{ font:inherit; padding:7px 14px; border-radius:6px; cursor:pointer;
   border:1px solid var(--line); background:var(--ac); color:#fff; }}
 pre {{ background:rgba(128,128,128,.10); padding:12px; border-radius:6px;
   overflow-x:auto; font-size:12.5px; }}
 .wrap {{ overflow-x:auto; }}
</style>

<h1>Support-relation review — {scene_id}</h1>
<p class="sub">{len(rows)} pairs · from {provenance['n_pairs']} target-owner
pairs, {provenance['n_evaluated']} evaluated, {provenance['n_candidates']}
proposed by the current algorithm</p>

<h2>1 · The question</h2>
<p class="def"><b>Does the target rest on the owner?</b><br>
Definition in use: <b>{DEFINITION}</b>. Not proximity, not overlap, not
"above". A cushion on a sofa qualifies; a picture hanging over a table does
not, however close it is.</p>
<p class="sub">The two pairs highlighted green are ones you have already
confirmed. They are shown so the sheet is self-consistent — leave or change
them as you see fit.</p>

<h2>2 · Pairs to judge</h2>
<p class="sub"><b><span class="sw tgt-sw"></span> pink = the target</b> (the
thing that might be resting) ·
<b><span class="sw own-sw"></span> teal = the owner</b> (what it might rest on)
· grey = the rest of the room, ceiling clipped. The <b>side-on</b> view is the
one to judge from: resting contact is a vertical question and the room views
flatten it.</p>
<p class="sub">The evidence line is evidence, not a suggestion. One pair the
algorithm proposes and one it rejects are both known-good, so it cannot be used
as a shortcut.</p>
<div class="pairs">{''.join(body)}</div>
<p style="margin-top:14px"><button onclick="emit()">Build JSON</button></p>
<pre id="out">(judge the rows above, then press Build JSON)</pre>

<h2>3 · How these rows were chosen</h2>
<p class="sub">Sampling only the algorithm's own candidates would build a key
that can confirm it and never contradict it. Fixed budget per overlap band,
highest overlap first, ties by uid — deterministic, no seed.</p>
<table><tr><th>band</th><th>rejected pairs sampled</th></tr>{bands}</table>

<script>
function emit() {{
  const truth = [];
  document.querySelectorAll('[data-pair]').forEach(tr => {{
    const pick = tr.querySelector('input[type=radio]:checked');
    if (pick) truth.push({{pair_id: tr.dataset.pair, judgement: pick.value}});
  }});
  document.getElementById('out').textContent = JSON.stringify({{
    schema: 'arkitscenes_support_relation_truth_v1',
    scene_id: '{scene_id}',
    definition: '{DEFINITION}',
    status: 'OWNER_REVIEWED',
    human_relation_truth: truth
  }}, null, 1);
}}
</script>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="41069025")
    ap.add_argument("--rest", type=Path, default=DEFAULT_REST)
    ap.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    ap.add_argument("--ids", type=Path, default=DEFAULT_IDS)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--no-renders", action="store_true",
                    help="text-only sheet; renders are the point, so this is "
                         "for debugging the sampling")
    args = ap.parse_args(argv)

    scene_id = f"arkitscenes_{args.scene}"
    artifact = json.loads(args.rest.read_text())
    pairs = artifact["pairs"]
    rows, band_counts = select_rows(pairs)
    n_patches = len({patch["patch_uid"] for pair in pairs
                     for patch in pair["evaluated_patches"]})

    # Pictures, so a pair can be judged without cross-referencing the UID sheet
    # by hand. Same renderer, so the colours and framing match.
    if not args.no_renders:
        import numpy as np
        from adapters.arkitscenes import MESH_SUFFIX, read_mesh
        from tools.arkitscenes_uid_visual_sheet import SceneRenderer

        mesh = read_mesh(args.data_root / args.scene
                         / f"{args.scene}{MESH_SUFFIX}")
        instance_ids = np.load(args.ids)
        if len(instance_ids) != len(mesh.xyz):
            raise ValueError(
                f"{len(instance_ids)} instance ids against {len(mesh.xyz)} "
                "mesh vertices; the sidecar does not belong to this mesh")
        index_of = {e["identity"]["object_uid"]:
                    int(e["geometry_handle"].rsplit("#", 1)[1])
                    for e in json.loads(args.entities.read_text())["entities"]}
        renderer = SceneRenderer(mesh.xyz, mesh.rgb)
        cache: dict[str, np.ndarray] = {}

        def vertices_of(uid: str) -> np.ndarray:
            if uid not in cache:
                cache[uid] = np.flatnonzero(instance_ids == index_of[uid])
            return cache[uid]

        for row in rows:
            target = vertices_of(row["target_uid"])
            owner = vertices_of(row["owner_uid"])
            row["renders"] = {
                "side": renderer.isolated_pair(target, owner),
                "ctx": renderer.context_pair(target, owner),
            }

    provenance = {
        "resting_artifact": str(args.rest.relative_to(REPO_ROOT)),
        "resting_artifact_sha256": hashlib.sha256(
            args.rest.read_bytes()).hexdigest(),
        "n_pairs": len(pairs),
        "n_evaluated": sum(1 for p in pairs if p["evaluated_patches"]),
        "n_candidates": sum(1 for p in pairs if p["relation_candidate"]),
        "n_qualifying_patches": n_patches,
        "patch_to_pair_note": (
            "the qualifying patches collapse to fewer target-owner pairs "
            "because several patches share an owner and the same best target; "
            "every patch is represented by the pair it scores highest against, "
            "none is dropped"),
        "algorithm_config": artifact["config"],
        "floor_evidence_status": artifact["floor_evidence"]["status"],
        "logic_changed": False,
        "thresholds_changed": False,
    }

    args.out_root.mkdir(parents=True, exist_ok=True)
    html_path = args.out_root / f"{scene_id}_support_review.html"
    html_path.write_text(build_html(scene_id, rows, band_counts, provenance))

    form = {
        "schema": "arkitscenes_support_relation_review_v1",
        "scene_id": scene_id,
        "status": "AWAITING_OWNER",
        "definition": DEFINITION,
        # Two blocks, joined only by pair_id. Human judgement never sits in the
        # same object as a metric.
        "human_relation_truth": [
            {"pair_id": r["pair_id"],
             "judgement": ("supports" if r["pair_id"] in
                           {f"{t}->{o}" for t, o in OWNER_CONFIRMED} else None),
             "source": ("owner_confirmed_2026-08-10"
                        if r["pair_id"] in
                        {f"{t}->{o}" for t, o in OWNER_CONFIRMED} else None)}
            for r in rows],
        "system_candidate_evidence": [
            {"pair_id": r["pair_id"], "target_uid": r["target_uid"],
             "owner_uid": r["owner_uid"], "row_sources": r["sources"],
             **r["system"]}
            for r in rows],
        "renders_embedded": not args.no_renders,
        "sampling": {"bands": [{"band": n, "low": lo, "high": hi,
                                "budget": b, "taken": band_counts[n]}
                               for n, lo, hi, b in BANDS],
                     "rule": "highest overlap first, ties by uid; deterministic",
                     "owner_confirmed_always_included": [
                         f"{t}->{o}" for t, o in OWNER_CONFIRMED]},
        "provenance": provenance,
    }
    json_path = args.out_root / f"{scene_id}_support_review.json"
    json_path.write_text(json.dumps(form, indent=1) + "\n")

    by_source: dict[str, int] = {}
    for r in rows:
        for s in r["sources"]:
            by_source[s] = by_source.get(s, 0) + 1
    print(f"=== {scene_id} support-relation review sheet")
    print(f"    pairs in artifact : {provenance['n_pairs']} "
          f"({provenance['n_evaluated']} evaluated, "
          f"{provenance['n_candidates']} proposed)")
    print(f"    rows to review    : {len(rows)}")
    print("    by source         : " + ", ".join(
        f"{k}={v}" for k, v in sorted(by_source.items())))
    print(f"    qualifying patches: {n_patches} -> "
          f"{by_source.get('qualifying_patch', 0)} distinct pairs "
          "(all represented; several patches share an owner and best target)")
    print("    band sample       :")
    for name, count in band_counts.items():
        print(f"        {name:<24s} {count}")
    print(f"    definition        : {DEFINITION}")
    print(f"    logic/thresholds  : unchanged")
    print(f"    -> {html_path.relative_to(REPO_ROOT)}")
    print(f"    -> {json_path.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
