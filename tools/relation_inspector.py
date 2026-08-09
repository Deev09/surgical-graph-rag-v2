"""Relation Inspector — see WHY an edge exists, or why an answer was withheld.

  python3 tools/relation_inspector.py                     # replica_room_2
  python3 tools/relation_inspector.py --scene replica_room_1

Writes ONE self-contained HTML file (default
`runs/inspector/<scene>_inspector.html`): no CDN, no external requests, no
build step. Deterministic — byte-identical output for identical inputs.

WHY THIS EXISTS, given `tools/mvp_viewer.py` already renders the scene:
that viewer surfaces ANSWERS — text, citations, status — and contains no
notion of an edge's evidence or of a rejected edge. The measurements that
justify a relation are already computed and stored (`Edge.evidence`), and
so are the ones that killed a candidate (`EdgeRejection.rejected_reason` +
evidence), but nothing displays either. This tool is that missing half. It
computes NO new geometry and re-derives NO verdicts.

On verdicts, deliberately: an emitted Edge passed every predicate its
extractor applied — that is what emission means — so this renders its
measurements beside the thresholds they were tested against and does not
recompute pass/fail. A rejection carries the failing predicate by name in
`rejected_reason`. Reverse-engineering comparison operators out of key
names would invent semantics the extractors never promised, and would be
wrong silently.

Everything shown comes from `graph.builder.build_graph` and
`reasoner.router.Router` — the same objects the scorecard scores, built the
same way, so the inspector cannot show a scene the evaluation never saw.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUT = REPO_ROOT / "runs" / "inspector"

SCENE_ROOMS = {
    "replica_room_0": "room_0", "replica_room_1": "room_1",
    "replica_room_2": "room_2", "replica_office_0": "office_0",
    "replica_frl_apartment_0": "frl_apartment_0",
}

# Evidence keys that are thresholds/config rather than measurements. Split
# out so the panel can show "measured against" instead of one flat blob.
THRESHOLD_KEYS = frozenset({
    "contact_threshold_m", "penetration_tolerance_m", "max_tilt_deg",
    "max_wall_tilt_deg", "footprint_tolerance_m", "near_surface_threshold_m",
    "threshold_m", "sparse_max_distance", "support_class_allowlist",
})
# Bulky provenance — kept in the payload but collapsed by default.
BULKY_KEYS = frozenset({
    "entity_surface_polygon", "entity_surface_plane", "up",
    "support_class_allowlist",
})


def _room_dir(scene_id: str) -> Path:
    lock = json.loads((REPO_ROOT / "tools" / "replica_scenes.lock.json")
                      .read_text())
    root = Path(lock["data_root_relative_to_repo"])
    if not root.is_absolute():
        root = (REPO_ROOT / root).resolve()
    return root / SCENE_ROOMS[scene_id]


def run_questions(scene_id: str, bundle) -> list[dict]:
    """The frozen battery through the real Router. `outcome` is the router's
    own five-way verdict; `abstain` and `empty` are the interesting ones here
    because they are what 'the system declined' actually looks like."""
    from reasoner.base import CompletenessProfile, ExecutionContext
    from reasoner.compiler_rules import RulesCompiler
    from reasoner.executor import RulesExecutor
    from reasoner.router import Router
    from reasoner.verbalizer import StandardVerbalizer

    mvp = REPO_ROOT / "runs" / "mvp_v0" / f"{scene_id}_mvp.json"
    if not mvp.is_file():
        return []
    battery = json.loads(mvp.read_text())["key_questions"]
    router = Router(compiler=RulesCompiler(), executor=RulesExecutor(),
                    verbalizer=StandardVerbalizer())
    ctx = ExecutionContext(completeness=CompletenessProfile(
        source="oracle", entity_recall_by_class={}, edge_recall_by_type={}))

    out = []
    for qid in sorted(battery):
        q = battery[qid]
        a = router.answer(q["question"], bundle, ctx)
        out.append({
            "qid": qid,
            "question": q["question"],
            "expected_outcome": q.get("expected_outcome"),
            "outcome": a.outcome,
            "answered_by": a.answered_by,
            "text": a.text,
            "confidence": a.confidence,
            "cited_uids": sorted(a.cited_uids),
            "cited_edges": sorted(a.cited_edges),
        })
    return out


def payload(scene_id: str, bundle, diag, questions: list[dict]) -> dict:
    labels = {n.id: n.label for n in bundle.nodes}
    nodes = []
    for n in sorted(bundle.nodes, key=lambda x: x.id):
        lo, hi = n.bbox_aabb
        nodes.append({
            "id": n.id, "label": n.label,
            "c": [round(float(v), 4) for v in n.centroid],
            "lo": [round(float(v), 4) for v in lo],
            "hi": [round(float(v), 4) for v in hi],
        })
    edges = []
    for e in sorted(bundle.edges, key=lambda x: x.edge_id):
        edges.append({
            "id": e.edge_id, "type": e.type,
            "src": e.source.uid, "srcKind": e.source.kind,
            "dst": e.target.uid, "dstKind": e.target.kind,
            "conf": round(float(e.confidence), 4),
            "extractor": f"{e.extractor} v{e.extractor_version}",
            "ev": _clean(e.evidence),
        })
    rejections = []
    for r in diag.rejection_samples:
        rejections.append({
            "type": r.type, "src": r.source.uid, "dst": r.target.uid,
            "reason": r.rejected_reason, "extractor": r.extractor,
            "ev": _clean(r.evidence),
        })
    return {
        "scene_id": scene_id,
        "labels": labels,
        "nodes": nodes,
        "edges": edges,
        "rejections": rejections,
        "rejections_total": {k: int(v) for k, v in
                             sorted(diag.rejections_per_type.items())},
        "rejections_sampled": len(rejections),
        "questions": questions,
        "threshold_keys": sorted(THRESHOLD_KEYS),
        "bulky_keys": sorted(BULKY_KEYS),
        "edges_by_type": {k: int(v) for k, v in
                          sorted(diag.edges_emitted_per_type.items())},
    }


def _clean(ev: dict) -> dict:
    """JSON-safe, rounded, key-sorted. Rounding is display-only and stated as
    such in the panel; the stored evidence is untouched."""
    out = {}
    for k in sorted(ev):
        v = ev[k]
        if isinstance(v, float):
            out[k] = round(v, 6)
        elif isinstance(v, (list, tuple)):
            out[k] = [round(x, 6) if isinstance(x, float) else x for x in v]
        else:
            out[k] = v
    return out


def render(data: dict) -> str:
    blob = json.dumps(data, sort_keys=True, separators=(",", ":"))
    title = f"Relation Inspector — {data['scene_id']}"
    return _TEMPLATE.replace("__TITLE__", html.escape(title)).replace(
        "__DATA__", blob.replace("</", "<\\/"))


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{
  --bg:#0e1116; --panel:#161b22; --line:#2a3139; --ink:#e6edf3;
  --dim:#8b949e; --accent:#4a9eff; --ok:#3fb950; --bad:#f85149;
  --warn:#d29922; --sel:#ffd33d;
}
@media (prefers-color-scheme:light){:root{
  --bg:#ffffff; --panel:#f6f8fa; --line:#d0d7de; --ink:#1f2328;
  --dim:#656d76; --accent:#0969da; --ok:#1a7f37; --bad:#cf222e;
  --warn:#9a6700; --sel:#bf8700;}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:13px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace}
header{padding:10px 14px;border-bottom:1px solid var(--line);
  display:flex;gap:16px;align-items:baseline;flex-wrap:wrap}
h1{font-size:14px;margin:0;font-weight:600}
.meta{color:var(--dim);font-size:12px}
.wrap{display:grid;grid-template-columns:minmax(320px,1.1fr) minmax(280px,.9fr) minmax(300px,1fr);
  gap:1px;background:var(--line);min-height:calc(100vh - 46px)}
@media(max-width:1100px){.wrap{grid-template-columns:1fr}}
.col{background:var(--bg);padding:12px;overflow:auto;max-height:calc(100vh - 46px)}
@media(max-width:1100px){.col{max-height:none}}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.06em;
  color:var(--dim);margin:0 0 8px;font-weight:600}
svg{width:100%;height:auto;background:var(--panel);border:1px solid var(--line);
  border-radius:6px;display:block;margin-bottom:10px}
.ent{fill:rgba(139,148,158,.10);stroke:var(--dim);stroke-width:.5;
  stroke-opacity:.65;cursor:pointer}
.ent:hover{fill:rgba(74,158,255,.28);stroke:var(--accent)}
.ent.sel{fill:rgba(255,211,61,.34);stroke:var(--sel);stroke-width:1.6}
.ent.cited{fill:rgba(63,185,80,.30);stroke:var(--ok);stroke-width:1.3}
.ent.rel{fill:rgba(74,158,255,.26);stroke:var(--accent);stroke-width:1.1}
.lnk{stroke:var(--accent);stroke-width:1.2;opacity:.85}
.lnk.cited{stroke:var(--ok);stroke-width:1.6}
/* NB: no font-size here. The viewBox is in METRES, so a px size in this
   rule is read as user units and a 5px label renders five metres tall.
   Size is set inline, derived from the scene extent. */
text.lbl{font-family:ui-monospace,monospace;fill:var(--ink);
  paint-order:stroke;stroke:var(--bg);stroke-width:.012;pointer-events:none}
.row{padding:5px 7px;border:1px solid transparent;border-radius:5px;
  cursor:pointer;display:flex;gap:8px;align-items:baseline}
.row:hover{background:var(--panel)}
.row.on{background:var(--panel);border-color:var(--accent)}
.row .t{color:var(--accent);flex:0 0 auto}
.row .n{flex:1 1 auto;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.row .c{color:var(--dim);flex:0 0 auto;font-size:11px}
.pill{display:inline-block;padding:1px 6px;border-radius:9px;font-size:11px;
  border:1px solid var(--line);color:var(--dim)}
.pill.bindings{color:var(--ok);border-color:var(--ok)}
.pill.empty,.pill.unknown{color:var(--warn);border-color:var(--warn)}
.pill.abstain,.pill.parser_failure{color:var(--bad);border-color:var(--bad)}
table{width:100%;border-collapse:collapse;margin:6px 0 12px}
td,th{padding:3px 6px;border-bottom:1px solid var(--line);
  text-align:left;vertical-align:top;font-size:12px}
th{color:var(--dim);font-weight:600}
td.v{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.true{color:var(--ok)} .false{color:var(--bad)}
.k{color:var(--dim)}
.note{color:var(--dim);font-size:11px;margin:6px 0 10px;line-height:1.45}
.empty{color:var(--dim);padding:10px 0;font-style:italic}
button{font:inherit;background:var(--panel);color:var(--ink);
  border:1px solid var(--line);border-radius:5px;padding:3px 9px;cursor:pointer}
button:hover{border-color:var(--accent)}
button.on{border-color:var(--accent);color:var(--accent)}
.bar{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:9px}
.hdr{font-weight:600;margin:2px 0 6px}
.sub{color:var(--dim);font-size:11px;margin-bottom:8px}
</style></head><body>
<header>
  <h1>Relation Inspector</h1>
  <span class="meta" id="hdr"></span>
</header>
<div class="wrap">
  <div class="col">
    <h2>Scene — click an object</h2>
    <div class="bar">
      <button id="btop" class="on">plan (x·y)</button>
      <button id="bside">elevation (x·z)</button>
      <button id="bclear">clear</button>
    </div>
    <svg id="plan" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet"></svg>
    <div class="note" id="legend"></div>
    <h2>Objects</h2>
    <div id="nodes"></div>
  </div>

  <div class="col">
    <h2 id="edgesHdr">Relations</h2>
    <div class="sub" id="edgesSub">Select an object to see its relations.</div>
    <div id="edges"></div>
    <h2 style="margin-top:14px">Rejected candidates</h2>
    <div class="sub" id="rejSub"></div>
    <div id="rejs"></div>
  </div>

  <div class="col">
    <h2>Evidence</h2>
    <div id="evi"><div class="empty">Select a relation or a rejected candidate.</div></div>
    <h2 style="margin-top:14px">Questions</h2>
    <div class="sub">Click a question to highlight the entities and edges it cited.</div>
    <div id="qs"></div>
  </div>
</div>
<script>
const D = __DATA__;
const NODE = Object.fromEntries(D.nodes.map(n=>[n.id,n]));
const THR = new Set(D.threshold_keys), BULK = new Set(D.bulky_keys);
let axis='top', sel=null, selEdge=null, selRej=null, selQ=null;

const name = u => D.labels[u] || u;
const esc = s => String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));

// ---- projection -------------------------------------------------------
function proj(){ return axis==='top' ? [0,1] : [0,2]; }
function bounds(){
  const [a,b]=proj(); let x0=1e9,y0=1e9,x1=-1e9,y1=-1e9;
  for(const n of D.nodes){
    x0=Math.min(x0,n.lo[a]); x1=Math.max(x1,n.hi[a]);
    y0=Math.min(y0,n.lo[b]); y1=Math.max(y1,n.hi[b]);
  }
  const px=(x1-x0)*0.04+0.2, py=(y1-y0)*0.04+0.2;
  return [x0-px,y0-py,x1+px,y1+py];
}
function drawScene(){
  const [a,b]=proj(), [x0,y0,x1,y1]=bounds();
  const W=x1-x0, H=y1-y0;
  const fs=Math.max(W,H)/55;   // label height in metres, not pixels
  const cited = new Set(selQ ? selQ.cited_uids : []);
  const citedE = new Set(selQ ? selQ.cited_edges : []);
  // related = the other end of every edge touching the selected entity
  const rel = new Set();
  if(sel) for(const e of D.edges){
    if(e.src===sel) rel.add(e.dst); else if(e.dst===sel) rel.add(e.src);
  }
  let s = '';
  // edges first, so boxes sit on top
  const lines = [];
  const push = (e,cl) => {
    const p=NODE[e.src], q=NODE[e.dst]; if(!p||!q) return;
    lines.push(`<line class="lnk ${cl}" x1="${p.c[a]}" y1="${-p.c[b]}" x2="${q.c[a]}" y2="${-q.c[b]}"/>`);
  };
  for(const e of D.edges){
    if(citedE.has(e.id)) push(e,'cited');
    else if(selEdge && e.id===selEdge.id) push(e,'');
    else if(sel && !selQ && (e.src===sel||e.dst===sel)) push(e,'');
  }
  s += lines.join('');
  // Largest footprint first so small objects land on top. Without this a rug
  // or a table buries every object resting on it -- which are exactly the
  // ones a support relation is about.
  const order = D.nodes.slice().sort((p,q)=>
      ((q.hi[a]-q.lo[a])*(q.hi[b]-q.lo[b])) - ((p.hi[a]-p.lo[a])*(p.hi[b]-p.lo[b])));
  for(const n of order){
    const w=Math.max(n.hi[a]-n.lo[a],0.04), h=Math.max(n.hi[b]-n.lo[b],0.04);
    let cl='ent';
    if(n.id===sel) cl+=' sel';
    else if(cited.has(n.id)) cl+=' cited';
    else if(rel.has(n.id)) cl+=' rel';
    s += `<rect class="${cl}" data-id="${n.id}" x="${n.lo[a]}" y="${-n.hi[b]}" `
      +  `width="${w}" height="${h}" rx="0.04"><title>${esc(n.label)} (${n.id})</title></rect>`;
    if(n.id===sel||cited.has(n.id))
      s += `<text class="lbl" font-size="${fs}" x="${n.c[a]}" `
        +  `y="${-n.c[b]}" text-anchor="middle">${esc(n.label)}</text>`;
  }
  const svg=document.getElementById('plan');
  svg.setAttribute('viewBox',`${x0} ${-y1} ${W} ${H}`);
  svg.innerHTML=s;
  svg.querySelectorAll('rect').forEach(r=>r.onclick=()=>{
    sel = r.dataset.id===sel ? null : r.dataset.id;
    selEdge=null; selRej=null; render();
  });
  document.getElementById('legend').innerHTML =
    'Boxes are axis-aligned entity bounds projected onto the plane — '
    + (axis==='top'?'floor plan, up is +z out of the page.'
                   :'elevation, height is +z upward.')
    + ' <span style="color:var(--sel)">selected</span> · '
    + '<span style="color:var(--accent)">directly related</span> · '
    + '<span style="color:var(--ok)">cited by the question</span>.';
}

// ---- panels -----------------------------------------------------------
function renderNodes(){
  const q=D.nodes.slice().sort((x,y)=>x.label.localeCompare(y.label)||x.id.localeCompare(y.id));
  document.getElementById('nodes').innerHTML = q.map(n=>{
    const deg=D.edges.filter(e=>e.src===n.id||e.dst===n.id).length;
    return `<div class="row ${n.id===sel?'on':''}" data-id="${n.id}">`
      +`<span class="n">${esc(n.label)}</span>`
      +`<span class="c">${n.id} · ${deg} rel</span></div>`;
  }).join('');
  document.querySelectorAll('#nodes .row').forEach(r=>r.onclick=()=>{
    sel = r.dataset.id===sel?null:r.dataset.id; selEdge=null; selRej=null; render();
  });
}

function renderEdges(){
  const box=document.getElementById('edges'), sub=document.getElementById('edgesSub');
  if(!sel){ box.innerHTML=''; sub.textContent='Select an object to see its relations.'; }
  else{
    const mine=D.edges.filter(e=>e.src===sel||e.dst===sel)
      .sort((a,b)=>a.type.localeCompare(b.type)||a.id.localeCompare(b.id));
    sub.innerHTML = `<b>${esc(name(sel))}</b> — ${mine.length} relation(s)`;
    box.innerHTML = mine.length ? mine.map(e=>{
      const out=e.src===sel;
      const other=out?e.dst:e.src;
      return `<div class="row ${selEdge&&selEdge.id===e.id?'on':''}" data-eid="${e.id}">`
        +`<span class="t">${e.type}</span>`
        +`<span class="n">${out?'→':'←'} ${esc(name(other))}</span>`
        +`<span class="c">${e.conf.toFixed(2)}</span></div>`;
    }).join('') : '<div class="empty">no relations</div>';
    box.querySelectorAll('.row').forEach(r=>r.onclick=()=>{
      selEdge=D.edges.find(x=>x.id===r.dataset.eid); selRej=null; render();
    });
  }
  // rejections
  const rbox=document.getElementById('rejs'), rsub=document.getElementById('rejSub');
  const tot=Object.values(D.rejections_total).reduce((a,b)=>a+b,0);
  if(!sel){
    rbox.innerHTML='';
    rsub.innerHTML=`${tot} candidates rejected scene-wide; ${D.rejections_sampled} sampled by the builder.`;
  }else{
    const mine=D.rejections.filter(r=>r.src===sel||r.dst===sel)
      .sort((a,b)=>a.type.localeCompare(b.type)||a.reason.localeCompare(b.reason));
    rsub.innerHTML=`Why <b>${esc(name(sel))}</b> did <i>not</i> get an edge `
      +`— ${mine.length} of ${D.rejections_sampled} sampled rejections.`;
    rbox.innerHTML = mine.length ? mine.map((r,i)=>
      `<div class="row ${selRej&&selRej._i===i&&selRej.src===r.src&&selRej.type===r.type?'on':''}" data-ri="${i}">`
      +`<span class="t">${r.type}</span>`
      +`<span class="n">${r.src===sel?'→':'←'} ${esc(name(r.src===sel?r.dst:r.src))}</span>`
      +`<span class="c">${esc(r.reason)}</span></div>`).join('')
      : '<div class="empty">no sampled rejections for this object</div>';
    rbox.querySelectorAll('.row').forEach(el=>el.onclick=()=>{
      const m=D.rejections.filter(r=>r.src===sel||r.dst===sel)
        .sort((a,b)=>a.type.localeCompare(b.type)||a.reason.localeCompare(b.reason));
      selRej=Object.assign({_i:+el.dataset.ri}, m[+el.dataset.ri]); selEdge=null; render();
    });
  }
}

function evTable(ev){
  const meas=[], thr=[], bulk=[];
  for(const k of Object.keys(ev).sort()){
    (BULK.has(k)?bulk:THR.has(k)?thr:meas).push(k);
  }
  const cell=v=>{
    if(typeof v==='boolean') return `<span class="${v?'true':'false'}">${v}</span>`;
    if(typeof v==='number') return Number.isInteger(v)?v:v.toFixed(6).replace(/0+$/,'').replace(/\.$/,'');
    if(Array.isArray(v)) return esc('['+v.join(', ')+']');
    return esc(String(v));
  };
  const rows=ks=>ks.map(k=>`<tr><td class="k">${esc(k)}</td><td class="v">${cell(ev[k])}</td></tr>`).join('');
  let s='';
  if(meas.length) s+=`<table><tr><th>measurement</th><th style="text-align:right">value</th></tr>${rows(meas)}</table>`;
  if(thr.length) s+=`<table><tr><th>tested against</th><th style="text-align:right">threshold</th></tr>${rows(thr)}</table>`;
  if(bulk.length) s+=`<details><summary class="k" style="cursor:pointer;margin:4px 0">provenance (${bulk.length})</summary>`
    +`<table>${rows(bulk)}</table></details>`;
  return s || '<div class="empty">no evidence recorded</div>';
}

function renderEvidence(){
  const box=document.getElementById('evi');
  if(selEdge){
    const e=selEdge;
    box.innerHTML=`<div class="hdr">${esc(name(e.src))} <span class="k">—${e.type}→</span> ${esc(name(e.dst))}</div>`
      +`<div class="sub">${esc(e.extractor)} · confidence ${e.conf} · edge <span class="k">${esc(e.id)}</span></div>`
      +`<div class="note">This edge was <b>emitted</b>, which means every predicate its extractor applied passed. `
      +`Values below are the stored measurements and the thresholds they were tested against — `
      +`shown, not recomputed. Displayed to 6 dp; stored values are full precision.</div>`
      +evTable(e.ev);
  } else if(selRej){
    const r=selRej;
    box.innerHTML=`<div class="hdr">${esc(name(r.src))} <span class="k">—${r.type}✗→</span> ${esc(name(r.dst))}</div>`
      +`<div class="sub">${esc(r.extractor)} · <span class="pill abstain">rejected</span> `
      +`<b>${esc(r.reason)}</b></div>`
      +`<div class="note">The extractor considered this pair and declined it for the reason named above. `
      +`This is what "the system does not claim a relation" looks like from the inside.</div>`
      +evTable(r.ev);
  } else {
    box.innerHTML='<div class="empty">Select a relation or a rejected candidate.</div>';
  }
}

function renderQs(){
  document.getElementById('qs').innerHTML = D.questions.length
    ? D.questions.map(q=>{
        const on=selQ&&selQ.qid===q.qid;
        return `<div class="row ${on?'on':''}" data-q="${q.qid}">`
          +`<span class="n">${esc(q.question)}</span>`
          +`<span class="pill ${q.outcome}">${q.outcome}</span></div>`
          +(on?`<div class="note" style="padding-left:7px">${esc(q.text||'(no text)')}<br>`
              +`cited ${q.cited_uids.length} entit${q.cited_uids.length===1?'y':'ies'}, `
              +`${q.cited_edges.length} edge(s)`
              +(q.confidence!=null?` · confidence ${q.confidence}`:'')
              +(q.expected_outcome?` · expected <b>${esc(q.expected_outcome)}</b>`:'')
              +`</div>`:'');
      }).join('')
    : '<div class="empty">no question battery for this scene</div>';
  document.querySelectorAll('#qs .row').forEach(r=>r.onclick=()=>{
    const q=D.questions.find(x=>x.qid===r.dataset.q);
    selQ = (selQ&&selQ.qid===q.qid)?null:q; render();
  });
}

function render(){ drawScene(); renderNodes(); renderEdges(); renderEvidence(); renderQs(); }

document.getElementById('btop').onclick=()=>{axis='top';
  document.getElementById('btop').classList.add('on');
  document.getElementById('bside').classList.remove('on'); drawScene();};
document.getElementById('bside').onclick=()=>{axis='side';
  document.getElementById('bside').classList.add('on');
  document.getElementById('btop').classList.remove('on'); drawScene();};
document.getElementById('bclear').onclick=()=>{sel=selEdge=selRej=selQ=null; render();};

const et=Object.entries(D.edges_by_type).map(([k,v])=>`${k} ${v}`).join(' · ');
document.getElementById('hdr').innerHTML =
  `${esc(D.scene_id)} — ${D.nodes.length} entities · ${D.edges.length} edges · `
  + `${D.questions.length} questions<br><span style="font-size:11px">${esc(et)}</span>`;
render();
</script></body></html>
"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default="replica_room_2",
                    choices=sorted(SCENE_ROOMS))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    from demo.question_battery import _runs
    from demo.replica_habitat_import import import_habitat_room
    from graph.builder import build_graph

    room_dir = _room_dir(args.scene)
    if not room_dir.is_dir():
        print(f"scene data not found: {room_dir}")
        return 1
    arts = import_habitat_room(room_dir, args.scene)
    bundle, diag = build_graph(arts, _runs(),
                               density_policy="phase2_telemetry_only")
    questions = run_questions(args.scene, bundle)
    data = payload(args.scene, bundle, diag, questions)

    out = args.out or (DEFAULT_OUT / f"{args.scene}_inspector.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data), encoding="utf-8")

    outcomes: dict[str, int] = {}
    for q in questions:
        outcomes[q["outcome"]] = outcomes.get(q["outcome"], 0) + 1
    print(f"{args.scene}: {len(data['nodes'])} entities, "
          f"{len(data['edges'])} edges, "
          f"{data['rejections_sampled']} sampled rejections")
    print(f"  edges by type : "
          + "  ".join(f"{k}={v}" for k, v in data["edges_by_type"].items()))
    print(f"  question outcomes: "
          + "  ".join(f"{k}={v}" for k, v in sorted(outcomes.items())))
    print(f"  -> {out}  ({out.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
