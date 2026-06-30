"""
Build a typed knowledge graph from classified session signals.

Node types:
  USER        — the user (singleton)
  TOPIC       — domain area (e.g. "Workforce Metrics")
  TOOL        — specific CLI tool or library
  PREFERENCE  — stable preference extracted from correction/preference signals
  CORRECTION  — recurring correction pattern (things agent kept getting wrong)

Edge types:
  WORKS_ON      — user → topic (weighted by session count)
  USES          — user → tool / topic → tool
  PREFERS       — user → preference
  CORRECTED     — user → correction
  RELATED_TO    — topic ↔ topic (co-occurrence)

Every node carries:
  first_seen, last_seen  — temporal provenance
  source_turns           — list of (session_id, turn_index) evidence pairs (up to 5)
  count                  — number of evidence occurrences
  examples               — short excerpts from real turns

Zero-evidence nodes are never added to the graph.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .classify import TurnSignal, _clean_message
from .discover import discover_all
from .extract import SessionMeta



# ---------------------------------------------------------------------------
# Graph node / edge
# ---------------------------------------------------------------------------

@dataclass
class GraphNode:
    id: str
    label: str
    group: str
    size: int = 10
    count: int = 0
    description: str = ""
    examples: list[str] = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    source_turns: list[dict] = field(default_factory=list)   # [{session_id, turn_index, timestamp}]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source: str
    target: str
    type: str
    weight: float = 1.0
    label: str = ""


def _short(msg: str, max_len: int = 120) -> str:
    """Clean boilerplate and truncate for display."""
    msg = _clean_message(msg)
    msg = msg.strip().replace("\n", " ")
    return msg[:max_len] + "…" if len(msg) > max_len else msg


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_graph(
    sessions: list[SessionMeta],
    signals: list[TurnSignal],
) -> tuple[list[GraphNode], list[GraphEdge]]:

    nodes: dict[str, GraphNode] = {}
    edges_raw: dict[tuple[str, str, str], float] = defaultdict(float)

    def upsert(n: GraphNode) -> GraphNode:
        if n.id not in nodes:
            nodes[n.id] = n
        return nodes[n.id]

    def add_edge(src: str, tgt: str, etype: str, w: float = 1.0):
        if src in nodes and tgt in nodes:
            edges_raw[(src, tgt, etype)] += w

    def _update_temporal(node: GraphNode, timestamp: str):
        if not node.first_seen or timestamp < node.first_seen:
            node.first_seen = timestamp
        if not node.last_seen or timestamp > node.last_seen:
            node.last_seen = timestamp

    # ── Run statistical discovery — no hardcoded archetypes ───────────────
    dk = discover_all(sessions, signals)

    # ── User node — description derived from data, not hardcoded ────────────
    top_tool = dk.tools[0].name if dk.tools else "unknown"
    top_pref = dk.preferences[0].phrase if dk.preferences else ""
    user_node = GraphNode(
        id="user",
        label="You",
        group="user",
        size=40,
        count=len(sessions),
        description=(
            f"{len(sessions)} sessions · {dk.total_turns} turns · "
            f"top tool: {top_tool}"
        ),
        first_seen=sessions[0].created_at if sessions else "",
        last_seen=sessions[-1].created_at if sessions else "",
        metadata={
            "total_sessions": len(sessions),
            "total_turns": dk.total_turns,
            "top_tool": top_tool,
            "top_preference": top_pref,
        },
    )
    upsert(user_node)

    # ── Topic nodes — TF-IDF discovered, labels emerge from data ────────────
    for topic in dk.topics:
        n = upsert(GraphNode(
            id=topic.id,
            label=topic.label,
            group="topic",
            size=max(8, min(35, topic.session_count // 2 + 8)),
            count=topic.session_count,
            description=f"Discovered topic — active in {topic.session_count} sessions",
            first_seen=topic.first_seen,
            last_seen=topic.last_seen,
            metadata={"top_terms": topic.top_terms},
        ))
        edges_raw[("user", n.id, "WORKS_ON")] += topic.session_count

    # Topic co-occurrence edges via shared TF-IDF terms
    for i, ti in enumerate(dk.topics):
        for tj in dk.topics[i + 1:]:
            shared = len(set(ti.top_terms) & set(tj.top_terms))
            if shared >= 2:
                edges_raw[(ti.id, tj.id, "RELATED_TO")] += shared

    # ── Tool nodes — syntactically extracted, session-spread ranked ──────────
    for tool in dk.tools[:25]:
        nid = f"tool_{re.sub(r'[^a-z0-9]', '_', tool.name.lower())}"
        n = upsert(GraphNode(
            id=nid,
            label=tool.name,
            group="tool",
            size=max(6, min(25, tool.session_count // 3 + 6)),
            count=tool.total_count,
            description=f"Used in {tool.session_count} sessions · {tool.total_count} mentions",
            examples=tool.examples,
            first_seen=tool.first_seen,
            last_seen=tool.last_seen,
        ))
        edges_raw[("user", nid, "USES")] += tool.session_count

    # ── Path nodes — canonical directories mined from text ──────────────────
    for path in dk.paths[:12]:
        nid = f"path_{re.sub(r'[^a-z0-9]', '_', path.root.lower())}"
        n = upsert(GraphNode(
            id=nid,
            label=path.root,
            group="path",
            size=max(6, min(20, path.session_count // 2 + 6)),
            count=path.total_count,
            description=f"Referenced in {path.session_count} sessions",
            examples=path.examples,
        ))
        edges_raw[("user", nid, "WORKS_IN")] += path.session_count

    # ── Preference nodes — session-spread n-grams, no hardcoded list ────────
    for pref in dk.preferences[:30]:
        nid = f"pref_{re.sub(r'[^a-z0-9]', '_', pref.phrase.lower())[:40]}"
        n = upsert(GraphNode(
            id=nid,
            label=pref.phrase.title(),
            group="preference",
            size=max(7, min(22, pref.session_count + 5)),
            count=pref.session_count,
            description=(
                f"Standing preference — {pref.session_count} sessions, "
                f"{pref.total_count} occurrences"
            ),
            examples=pref.examples,
            first_seen=pref.first_seen,
            last_seen=pref.last_seen,
            metadata={"score": round(pref.score, 1)},
        ))
        edges_raw[("user", nid, "PREFERS")] += pref.session_count

    # ── Correction nodes — behavioural detection + content extraction ────────
    for corr in dk.corrections[:20]:
        nid = f"corr_{re.sub(r'[^a-z0-9]', '_', corr.subject.lower())[:40]}"
        n = upsert(GraphNode(
            id=nid,
            label=corr.subject.title(),
            group="correction",
            size=max(8, min(22, corr.session_count * 2 + 6)),
            count=corr.count,
            description=(
                f"Recurring correction — {corr.count} occurrence(s) "
                f"across {corr.session_count} session(s)"
            ),
            examples=corr.examples,
            first_seen=corr.first_seen,
            last_seen=corr.last_seen,
        ))
        edges_raw[("user", nid, "CORRECTED")] += corr.count

    # ── Skill nodes — Copilot skills used across sessions ───────────────────
    for skill in dk.skills[:20]:
        nid = f"skill_{re.sub(r'[^a-z0-9]', '_', skill.name.lower())}"
        label = skill.name.replace("-", " ").title()
        n = upsert(GraphNode(
            id=nid,
            label=label,
            group="skill",
            size=max(7, min(28, int(skill.session_count ** 0.8) + 7)),
            count=skill.session_count,
            description=(
                f"Skill '{skill.name}' — used in {skill.session_count} sessions"
                f", {skill.total_count} invocation(s)"
            ),
            examples=skill.examples,
            first_seen=skill.first_seen,
            last_seen=skill.last_seen,
            metadata={"skill_type": skill.skill_type},
        ))
        edges_raw[("user", nid, "USES_SKILL")] += skill.session_count

    # ── User metadata (data-derived) ─────────────────────────────────────────
    nodes["user"].metadata.update({
        "top_topics":    [(t.label, t.session_count) for t in dk.topics[:5]],
        "top_tools":     [(t.name, t.session_count) for t in dk.tools[:5]],
        "top_paths":     [(p.root, p.session_count) for p in dk.paths[:3]],
        "first_session": sessions[0].created_at if sessions else None,
        "last_session":  sessions[-1].created_at if sessions else None,
    })

    # ── Behavioural signal nodes (10 types from classify.py) ─────────────
    # Generic builder: groups signals by sub-label, creates one node per label
    def _build_signal_nodes(
        signals: list,
        bucket_attr: str,
        group: str,
        edge_type: str,
        node_prefix: str,
        description_template: str,
    ) -> dict[str, str]:
        """Build archetype nodes for a behavioural signal bucket.
        Returns {node_id: sub_label} for cross-signal edge wiring."""
        bucket: dict[str, list[tuple[str, str, str, int, str]]] = defaultdict(list)
        for sig in signals:
            # Bug fix (Opus): dedup sub_labels per signal to prevent edge weight inflation
            for sub_label in set(getattr(sig, bucket_attr, [])):
                bucket[sub_label].append((
                    sub_label,
                    _short(sig.cleaned_message),
                    sig.turn.timestamp,
                    sig.turn.turn_index,
                    sig.turn.session_id,
                ))
        created: dict[str, str] = {}
        for sub_label, matches in bucket.items():
            if not matches:
                continue
            nid = f"{node_prefix}_{re.sub(r'[^a-z0-9]', '_', sub_label.lower())}"
            label = sub_label.replace("_", " ").title()
            timestamps = [m[2] for m in matches]
            # Bug fix (Opus): dedup source_turns by (session_id, turn_index)
            seen_turns: set[tuple] = set()
            source_turns = []
            for m in matches:
                key = (m[4], m[3])
                if key not in seen_turns:
                    seen_turns.add(key)
                    source_turns.append({"session_id": m[4], "turn_index": m[3], "timestamp": m[2]})
                if len(source_turns) == 5:
                    break
            n = upsert(GraphNode(
                id=nid,
                label=label,
                group=group,
                size=max(7, min(22, len(matches) + 5)),
                count=len(matches),
                description=description_template.format(n=len(matches)),
                examples=list(dict.fromkeys(m[1] for m in matches))[:4],
                first_seen=min(timestamps),
                last_seen=max(timestamps),
                source_turns=source_turns,
            ))
            edges_raw[("user", nid, edge_type)] += len(matches)
            created[nid] = sub_label
        return created

    persuasion_nodes   = _build_signal_nodes(signals, "persuasion_types",         "persuasion",    "DEFERRED_TO",   "persuasion",   "LLM convinced user — {n} instance(s)")
    methodology_nodes  = _build_signal_nodes(signals, "methodology_types",        "methodology",   "CHALLENGES",    "method",       "Methodology challenge — {n} instance(s)")
    knowledge_nodes    = _build_signal_nodes(signals, "knowledge_boundary_types", "knowledge",     "LEARNING",      "know",         "Knowledge boundary — {n} instance(s)")
    stakeholder_nodes  = _build_signal_nodes(signals, "stakeholder_types",        "stakeholder",   "SERVES",        "stake",        "Stakeholder context — {n} instance(s)")
    decision_nodes     = _build_signal_nodes(signals, "decision_types",           "decision",      "DECIDES",       "decision",     "Decision pattern — {n} instance(s)")
    frustration_nodes  = _build_signal_nodes(signals, "frustration_types",        "frustration",   "FRUSTRATED_BY", "friction",     "Frustration trigger — {n} instance(s)")
    trust_nodes        = _build_signal_nodes(signals, "trust_types",              "trust",         "TRUSTS_WHEN",   "trust",        "Trust calibration — {n} instance(s)")
    arch_nodes         = _build_signal_nodes(signals, "architecture_types",       "architecture",  "THINKS_IN",     "arch",         "Mental model — {n} instance(s)")
    urgency_nodes      = _build_signal_nodes(signals, "urgency_types",            "urgency",       "URGENT_ABOUT",  "urgency",      "Urgency pattern — {n} instance(s)")
    quality_nodes      = _build_signal_nodes(signals, "quality_types",            "quality",       "EXPECTS",       "quality",      "Quality standard — {n} instance(s)")
    agency_nodes       = _build_signal_nodes(signals, "agency_types",             "agency",        "PREFERS_STYLE", "agency",       "Agency preference — {n} instance(s)")
    tool_dir_nodes     = _build_signal_nodes(signals, "tool_directive_types",     "tool_directive","DIRECTS_TOOL",  "tooldir",      "Tool directive — {n} instance(s)")
    term_nodes         = _build_signal_nodes(signals, "terminology_types",        "terminology",   "ENFORCES_TERM", "term",         "Terminology rule — {n} instance(s)")
    format_nodes       = _build_signal_nodes(signals, "format_types",             "format",        "PREFERS_FORMAT","fmt",          "Format preference — {n} instance(s)")
    mip_nodes          = _build_signal_nodes(signals, "mip_types",                "compliance",    "REQUIRES",      "mip",          "MIP/label compliance — {n} instance(s)")
    scope_nodes        = _build_signal_nodes(signals, "scope_types",              "scope",         "SCOPES",        "scope",        "Scope disambiguation — {n} instance(s)")
    redirect_nodes     = _build_signal_nodes(signals, "redirect_types",           "frustration",   "REDIRECTS",     "redirect",     "User redirect — {n} instance(s)")
    model_nodes        = _build_signal_nodes(signals, "model_cost_types",         "model_cost",    "COSTS_AWARE",   "model",        "Model cost rule — {n} instance(s)")

    # ── Cross-signal edges (per GPT-5.5 design) ───────────────────────────
    # StakeholderContext → raises QualityStandard
    for snid in stakeholder_nodes:
        for qnid in quality_nodes:
            if snid in nodes and qnid in nodes:
                edges_raw[(snid, qnid, "RAISES_QUALITY_BAR")] += 1

    # StakeholderContext → creates Urgency
    for snid in stakeholder_nodes:
        for unid in urgency_nodes:
            if snid in nodes and unid in nodes:
                edges_raw[(snid, unid, "CREATES_URGENCY")] += 1

    # FrustrationTrigger → reduces TrustCalibration
    for fnid in frustration_nodes:
        for tnid in trust_nodes:
            if fnid in nodes and tnid in nodes:
                edges_raw[(fnid, tnid, "REDUCES_TRUST_IN")] += 1

    # Methodology → requires evidence (TrustCalibration)
    for mnid in methodology_nodes:
        for tnid in trust_nodes:
            if mnid in nodes and tnid in nodes:
                edges_raw[(mnid, tnid, "REQUIRES_EVIDENCE")] += 1

    # Architecture mental model → informs DecisionPattern
    for anid in arch_nodes:
        for dnid in decision_nodes:
            if anid in nodes and dnid in nodes:
                edges_raw[(anid, dnid, "INFORMS_DECISION")] += 1

    # AgencyPreference + UrgencyPattern co-signal
    for agnid in agency_nodes:
        for unid in urgency_nodes:
            if agnid in nodes and unid in nodes:
                edges_raw[(agnid, unid, "MODIFIES_STYLE")] += 1

    # QualityBar → most-visited topic (data-derived, not hardcoded)
    if dk.topics:
        top_topic_id = dk.topics[0].id
        for qnid in quality_nodes:
            if qnid in nodes and top_topic_id in nodes:
                edges_raw[(qnid, top_topic_id, "RELATED_TO")] += 1

    # ── Finalise edge list ────────────────────────────────────────────────
    edges: list[GraphEdge] = [
        GraphEdge(source=src, target=tgt, type=etype, weight=w, label=etype)
        for (src, tgt, etype), w in edges_raw.items()
        if src in nodes and tgt in nodes
    ]

    return list(nodes.values()), edges


def to_json(nodes: list[GraphNode], edges: list[GraphEdge]) -> dict:
    return {
        "nodes": [
            {
                "id": n.id,
                "label": n.label,
                "group": n.group,
                "size": n.size,
                "count": n.count,
                "description": n.description,
                "examples": n.examples,
                "first_seen": n.first_seen,
                "last_seen": n.last_seen,
                "source_turns": n.source_turns,
                "metadata": n.metadata,
            }
            for n in nodes
        ],
        "links": [
            {
                "source": e.source,
                "target": e.target,
                "type": e.type,
                "weight": e.weight,
                "label": e.label,
            }
            for e in edges
        ],
    }
