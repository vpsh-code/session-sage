"""Generate a self-contained interactive D3.js HTML knowledge graph.

D3.js is embedded inline (no CDN dependency) so the output works fully offline
and private session data never touches an external server.
"""

from __future__ import annotations

import json
from pathlib import Path

# D3 v7 min — vendored inline to avoid CDN dependency on private-data output
_D3_JS_PATH = Path(__file__).parent / "_d3.v7.min.js"


def _d3_js() -> str:
    """Return D3 source from the vendored file, falling back to CDN with SRI if missing."""
    if _D3_JS_PATH.exists():
        return f"<script>{_D3_JS_PATH.read_text()}</script>"
    # Fallback: CDN with SRI hash — last-resort only
    return (
        '<script src="https://d3js.org/d3.v7.min.js" '
        'integrity="sha384-NjAc5fxVLXQIhEpvf3z1FUbFK0A3cNd9sHGbJFP3g2QTMm0z4mM5gGdsMlbblxWQ" '
        'crossorigin="anonymous"></script>'
    )


# ---------------------------------------------------------------------------
# Group colour palette
# ---------------------------------------------------------------------------

GROUP_COLOURS = {
    "user":         "#f0c040",   # gold
    "topic":        "#4a9eff",   # blue
    "tool":         "#50c878",   # emerald
    "skill":        "#f97316",   # orange — Copilot skills
    "preference":   "#a78bfa",   # violet
    "correction":   "#ff6b6b",   # red-orange
    "persuasion":   "#fb923c",   # amber — LLM convinced user
    "methodology":  "#22d3ee",   # cyan — analytical challenges
    "knowledge":    "#86efac",   # light green — learning moments
    "stakeholder":  "#fbbf24",   # yellow — audience context
    "decision":     "#818cf8",   # indigo — decision patterns
    "frustration":  "#f87171",   # coral — friction triggers
    "trust":        "#34d399",   # teal — trust calibration
    "architecture": "#c084fc",   # purple — mental models
    "urgency":      "#fb7185",   # rose — time pressure
    "quality":      "#a3e635",   # lime — quality standards
    "agency":       "#38bdf8",   # sky — autonomy style
    "compliance":   "#f472b6",   # pink — MIP/label compliance
    "scope":        "#67e8f9",   # light cyan — scope disambiguation
    "tool_directive": "#fde68a", # light yellow — tool routing rules
    "terminology":  "#d8b4fe",   # lavender — term corrections
    "format":       "#6ee7b7",   # mint — output format preferences
    "model_cost":   "#fca5a5",   # light red — model cost awareness
    "concept":      "#94a3b8",   # slate
}


def render_html(graph_data: dict, title: str = "Session Sage — Knowledge Graph") -> str:
    graph_json = json.dumps(graph_data, indent=2)
    colours_json = json.dumps(GROUP_COLOURS)
    d3_tag = _d3_js()

    return f"""<!DOCTYPE html>
<!-- ⚠️  SENSITIVE: This file embeds raw Copilot CLI session data. Treat as internal/confidential. -->
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>{title}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0f1117; color: #e2e8f0; font-family: 'Segoe UI', system-ui, sans-serif; height: 100vh; display: flex; flex-direction: column; overflow: hidden; }}

  #header {{ background: #1a1d27; border-bottom: 1px solid #2d3148; padding: 10px 18px; display: flex; align-items: center; gap: 16px; flex-shrink: 0; }}
  #header h1 {{ font-size: 15px; font-weight: 600; color: #f0c040; letter-spacing: 0.5px; }}
  #header .subtitle {{ font-size: 11px; color: #64748b; }}

  #controls {{ background: #1a1d27; border-bottom: 1px solid #2d3148; padding: 8px 18px; display: flex; align-items: center; gap: 10px; flex-shrink: 0; flex-wrap: wrap; }}
  #search {{ background: #0f1117; border: 1px solid #334155; color: #e2e8f0; padding: 5px 10px; border-radius: 6px; font-size: 12px; width: 200px; outline: none; }}
  #search:focus {{ border-color: #4a9eff; }}

  .filter-btn {{ background: #1e2235; border: 1px solid #334155; color: #94a3b8; padding: 4px 10px; border-radius: 12px; font-size: 11px; cursor: pointer; transition: all 0.15s; }}
  .filter-btn:hover {{ border-color: #4a9eff; color: #e2e8f0; }}
  .filter-btn.active {{ border-color: var(--c); color: var(--c); background: color-mix(in srgb, var(--c) 15%, transparent); }}

  #main {{ display: flex; flex: 1; overflow: hidden; }}
  #graph-container {{ flex: 1; position: relative; }}
  svg {{ width: 100%; height: 100%; }}

  #sidebar {{ width: 300px; background: #1a1d27; border-left: 1px solid #2d3148; padding: 14px; overflow-y: auto; font-size: 12px; flex-shrink: 0; }}
  #sidebar h2 {{ font-size: 13px; color: #94a3b8; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.8px; }}
  #node-detail {{ display: none; }}
  #node-detail .node-label {{ font-size: 16px; font-weight: 700; margin-bottom: 4px; }}
  #node-detail .node-group {{ font-size: 10px; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; opacity: 0.7; }}
  #node-detail .node-desc {{ color: #94a3b8; margin-bottom: 10px; line-height: 1.5; }}
  #node-detail .node-temporal {{ font-size: 10px; color: #50c878; margin-bottom: 6px; font-family: monospace; }}
  #node-detail .node-count {{ font-size: 11px; color: #a78bfa; margin-bottom: 6px; }}
  #node-detail .examples-title {{ font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 6px; }}
  #node-detail .example {{ background: #0f1117; border-left: 2px solid #334155; padding: 6px 8px; margin-bottom: 5px; border-radius: 3px; color: #cbd5e1; line-height: 1.4; font-style: italic; }}

  #legend {{ margin-top: 20px; padding-top: 14px; border-top: 1px solid #2d3148; }}
  #legend h2 {{ font-size: 11px; color: #64748b; text-transform: uppercase; letter-spacing: 0.8px; margin-bottom: 8px; }}
  .legend-item {{ display: flex; align-items: center; gap: 7px; margin-bottom: 6px; font-size: 11px; color: #94a3b8; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}

  #stats {{ margin-top: 14px; padding-top: 14px; border-top: 1px solid #2d3148; }}
  .stat-row {{ display: flex; justify-content: space-between; font-size: 11px; padding: 3px 0; color: #64748b; }}
  .stat-val {{ color: #e2e8f0; font-weight: 600; }}

  .node circle {{ stroke-width: 1.5px; cursor: pointer; transition: opacity 0.15s; }}
  .node circle:hover {{ stroke-width: 3px; filter: brightness(1.3); }}
  .node text {{ pointer-events: none; user-select: none; }}
  .link {{ stroke-opacity: 0.4; }}
  .link:hover {{ stroke-opacity: 0.9; }}

  #hint {{ position: absolute; bottom: 12px; left: 50%; transform: translateX(-50%); font-size: 10px; color: #334155; pointer-events: none; }}
</style>
</head>
<body>

<div id="header">
  <h1>🧠 Session Sage</h1>
  <span class="subtitle">Knowledge graph mined from your Copilot CLI session history</span>
</div>

<div id="controls">
  <input id="search" type="text" placeholder="Search nodes…"/>
  <button class="filter-btn active" style="--c:#f0c040" data-group="user">You</button>
  <button class="filter-btn active" style="--c:#4a9eff" data-group="topic">Topics</button>
  <button class="filter-btn active" style="--c:#50c878" data-group="tool">Tools</button>
  <button class="filter-btn active" style="--c:#f97316" data-group="skill">Skills</button>
  <button class="filter-btn active" style="--c:#a78bfa" data-group="preference">Preferences</button>
  <button class="filter-btn active" style="--c:#ff6b6b" data-group="correction">Corrections</button>
  <button id="reset-zoom" class="filter-btn" style="--c:#94a3b8">Reset Zoom</button>
</div>

<div id="main">
  <div id="graph-container">
    <svg id="svg"></svg>
    <div id="hint">Scroll to zoom · Drag to pan · Click node for details</div>
  </div>

  <div id="sidebar">
    <div id="node-detail">
      <div class="node-label" id="det-label"></div>
      <div class="node-group" id="det-group"></div>
      <div class="node-count" id="det-count"></div>
      <div class="node-temporal" id="det-temporal"></div>
      <div class="node-desc" id="det-desc"></div>
      <div class="examples-title" id="ex-title"></div>
      <div id="det-examples"></div>
    </div>
    <div id="legend">
      <h2>Node Types</h2>
      <div class="legend-item"><div class="legend-dot" style="background:#f0c040"></div> You (the user)</div>
      <div class="legend-item"><div class="legend-dot" style="background:#4a9eff"></div> Topic domain</div>
      <div class="legend-item"><div class="legend-dot" style="background:#50c878"></div> Tool / library</div>
      <div class="legend-item"><div class="legend-dot" style="background:#f97316"></div> Copilot skill</div>
      <div class="legend-item"><div class="legend-dot" style="background:#a78bfa"></div> Standing preference</div>
      <div class="legend-item"><div class="legend-dot" style="background:#ff6b6b"></div> Recurring correction</div>
    </div>
    <div id="stats">
      <h2 style="font-size:11px;color:#64748b;text-transform:uppercase;letter-spacing:.8px;margin-bottom:8px;">Session Stats</h2>
    </div>
  </div>
</div>

{d3_tag}
<script>
const GRAPH = {graph_json};
const COLOURS = {colours_json};

// ── Stats panel ──────────────────────────────────────────────────────────
const userNode = GRAPH.nodes.find(n => n.id === 'user');
const statsEl = document.getElementById('stats');
const meta = userNode?.metadata || {{}};
const topTopics = (meta.top_topics || []).slice(0,5);

const addStat = (label, val) => {{
  const div = document.createElement('div');
  div.className = 'stat-row';
  div.innerHTML = `<span>${{label}}</span><span class="stat-val">${{val}}</span>`;
  statsEl.appendChild(div);
}};

addStat('Sessions', meta.total_sessions || '—');
addStat('Turns', meta.total_turns || '—');
addStat('Nodes', GRAPH.nodes.length);
addStat('Edges', GRAPH.links.length);
addStat('Corrections', GRAPH.nodes.filter(n=>n.group==='correction').length);
addStat('Skills', GRAPH.nodes.filter(n=>n.group==='skill').length);
addStat('Preferences', GRAPH.nodes.filter(n=>n.group==='preference').length);

if (meta.first_session) addStat('From', meta.first_session.slice(0,10));
if (meta.last_session)  addStat('To', meta.last_session.slice(0,10));

// ── D3 force graph ───────────────────────────────────────────────────────
const svg = d3.select('#svg');
const container = d3.select('#graph-container');
const W = () => container.node().clientWidth;
const H = () => container.node().clientHeight;

const g = svg.append('g');

// Zoom
const zoom = d3.zoom()
  .scaleExtent([0.15, 6])
  .on('zoom', e => g.attr('transform', e.transform));
svg.call(zoom);
document.getElementById('reset-zoom').onclick = () =>
  svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity);

// Groups tracking
const visibleGroups = new Set(['user','topic','tool','skill','preference','correction','concept']);

// Build nodes/links with mutable visibility
const allNodes = GRAPH.nodes.map(d => ({{...d, visible: true}}));
const nodeById = Object.fromEntries(allNodes.map(n => [n.id, n]));
const allLinks = GRAPH.links.map(d => ({{...d}}));

// Force simulation
const sim = d3.forceSimulation(allNodes)
  .force('link', d3.forceLink(allLinks).id(d => d.id).distance(d => 80 + d.weight * 0.5))
  .force('charge', d3.forceManyBody().strength(-220))
  .force('center', d3.forceCenter(W() / 2, H() / 2))
  .force('collision', d3.forceCollide(d => d.size + 6))
  .force('x', d3.forceX(W()/2).strength(0.03))
  .force('y', d3.forceY(H()/2).strength(0.03));

// Draw links
const linkEl = g.append('g').attr('class','links')
  .selectAll('line')
  .data(allLinks)
  .join('line')
  .attr('class','link')
  .attr('stroke', d => {{
    const typeColours = {{ WORKS_ON:'#4a9eff', USES:'#50c878', PREFERS:'#a78bfa', CORRECTED:'#ff6b6b', RELATED_TO:'#475569' }};
    return typeColours[d.type] || '#475569';
  }})
  .attr('stroke-width', d => Math.max(1, Math.log(d.weight + 1)));

// Draw nodes
const nodeEl = g.append('g').attr('class','nodes')
  .selectAll('g')
  .data(allNodes)
  .join('g')
  .attr('class','node')
  .call(d3.drag()
    .on('start', (e,d) => {{ if (!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; }})
    .on('drag',  (e,d) => {{ d.fx=e.x; d.fy=e.y; }})
    .on('end',   (e,d) => {{ if (!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }}))
  .on('click', (e, d) => showDetail(d));

nodeEl.append('circle')
  .attr('r', d => d.size)
  .attr('fill', d => COLOURS[d.group] || '#94a3b8')
  .attr('stroke', d => d.group==='user' ? '#fff' : 'rgba(255,255,255,0.15)');

nodeEl.append('text')
  .attr('dy', d => d.size + 11)
  .attr('text-anchor','middle')
  .attr('fill','#cbd5e1')
  .attr('font-size', d => d.group==='user' ? 13 : Math.max(9, Math.min(12, d.size * 0.75)))
  .attr('font-weight', d => d.group==='user' ? '700' : '400')
  .text(d => d.label.length > 22 ? d.label.slice(0,21)+'…' : d.label);

sim.on('tick', () => {{
  linkEl
    .attr('x1', d => d.source.x).attr('y1', d => d.source.y)
    .attr('x2', d => d.target.x).attr('y2', d => d.target.y);
  nodeEl.attr('transform', d => `translate(${{d.x}},${{d.y}})`);
}});

// ── Apply initial visibility (must run after D3 binds nodes/links) ────────
function applyVisibility() {{
  nodeEl.attr('display', d => visibleGroups.has(d.group) ? null : 'none');
  linkEl.attr('display', l => {{
    const src = typeof l.source === 'object' ? l.source : nodeById[l.source];
    const tgt = typeof l.target === 'object' ? l.target : nodeById[l.target];
    return (visibleGroups.has(src?.group) && visibleGroups.has(tgt?.group)) ? null : 'none';
  }});
}}
applyVisibility();

// ── Detail panel ─────────────────────────────────────────────────────────
function showDetail(d) {{
  document.getElementById('node-detail').style.display = 'block';
  document.getElementById('det-label').textContent = d.label;
  document.getElementById('det-label').style.color = COLOURS[d.group] || '#e2e8f0';
  document.getElementById('det-group').textContent = d.group.toUpperCase();
  document.getElementById('det-count').textContent =
    d.count > 0 ? `Evidence: ${{d.count}} session(s) / turn(s)` : '';

  const fs = d.first_seen ? d.first_seen.slice(0,10) : '';
  const ls = d.last_seen  ? d.last_seen.slice(0,10)  : '';
  document.getElementById('det-temporal').textContent =
    fs ? (fs === ls ? `Seen: ${{fs}}` : `First: ${{fs}}  →  Last: ${{ls}}`) : '';

  document.getElementById('det-desc').textContent = d.description || '';

  const examplesEl = document.getElementById('det-examples');
  examplesEl.innerHTML = '';

  if (d.examples && d.examples.length) {{
    document.getElementById('ex-title').textContent = 'Examples from sessions:';
    d.examples.forEach(ex => {{
      const div = document.createElement('div');
      div.className = 'example';
      div.textContent = ex;
      examplesEl.appendChild(div);
    }});
  }} else {{
    document.getElementById('ex-title').textContent = '';
  }}

  // Highlight connected nodes
  const connectedIds = new Set([d.id]);
  allLinks.forEach(l => {{
    if (l.source.id === d.id) connectedIds.add(l.target.id);
    if (l.target.id === d.id) connectedIds.add(l.source.id);
  }});
  nodeEl.selectAll('circle')
    .attr('opacity', n => connectedIds.has(n.id) ? 1 : 0.2);
  linkEl.attr('opacity', l =>
    (l.source.id === d.id || l.target.id === d.id) ? 1 : 0.05);
}}

svg.on('click', e => {{
  if (e.target.tagName === 'svg' || e.target.closest('g.links')) {{
    nodeEl.selectAll('circle').attr('opacity', 1);
    linkEl.attr('opacity', 1);
  }}
}});

// ── Search ────────────────────────────────────────────────────────────────
document.getElementById('search').addEventListener('input', e => {{
  const q = e.target.value.toLowerCase().trim();
  if (!q) {{
    nodeEl.selectAll('circle').attr('opacity', 1);
    linkEl.attr('opacity', 1);
    return;
  }}
  const matched = new Set(allNodes.filter(n =>
    n.label.toLowerCase().includes(q) || (n.description||'').toLowerCase().includes(q)
  ).map(n => n.id));
  nodeEl.selectAll('circle').attr('opacity', n => matched.has(n.id) ? 1 : 0.1);
  linkEl.attr('opacity', l =>
    matched.has(l.source.id) || matched.has(l.target.id) ? 0.8 : 0.05);
}});

// ── Filter buttons ────────────────────────────────────────────────────────
document.querySelectorAll('.filter-btn[data-group]').forEach(btn => {{
  btn.addEventListener('click', () => {{
    const grp = btn.dataset.group;
    btn.classList.toggle('active');
    if (btn.classList.contains('active')) visibleGroups.add(grp);
    else visibleGroups.delete(grp);
    applyVisibility();
  }});
}});

// Responsive resize
window.addEventListener('resize', () => {{
  sim.force('center', d3.forceCenter(W()/2, H()/2));
  sim.alpha(0.1).restart();
}});
</script>
</body>
</html>
"""


def write_html(graph_data: dict, output_path: Path, title: str = "Session Sage — Knowledge Graph") -> None:
    html = render_html(graph_data, title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
