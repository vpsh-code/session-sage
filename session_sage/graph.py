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
from .extract import SessionMeta


# ---------------------------------------------------------------------------
# Preference / Correction archetypes
# Each entry: (pattern_on_lowercased_user_message, node_id, label)
# Patterns intentionally specific — broad patterns are false positive magnets.
# ---------------------------------------------------------------------------

PREFERENCE_ARCHETYPES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"org.{0,10}l?2|org.{0,10}level.{0,5}2|business unit", re.I),
     "pref_org_l2", "Always Org Level 2"),
    # xlsx preference: require preference signal context (not bare Excel mention)
    (re.compile(r"(output|format|save|export|generate|write).{0,20}(xlsx|excel|workbook)", re.I),
     "pref_xlsx_output", "Output as .xlsx"),
    (re.compile(r"muted|accessible.{0,20}(colour|color|palette)|palette.{0,20}(muted|accessible)", re.I),
     "pref_muted_colours", "Muted accessible colour palette"),
    (re.compile(r"autofit|auto.?fit.{0,15}(column|width)", re.I),
     "pref_autofit", "Autofit column widths"),
    # wf_duck preference: require explicit comparison or instruction context
    (re.compile(r"(use|prefer|should use).{0,20}wf_duck|wf_duck.{0,20}(not|instead of).{0,20}wf_metrics", re.I),
     "pref_wf_duck_primary", "wf_duck.py over wf_metrics.py"),
    (re.compile(r"(proprietary|mip|sensitivity|label).{0,40}(always|standing|instruction|rule)", re.I),
     "pref_mip_label", "Apply MIP Proprietary label"),
    (re.compile(r"(named sheet|sheet name|clear header|human.?readable.{0,20}(sheet|output))", re.I),
     "pref_named_sheets", "Human-readable sheet names"),
    (re.compile(r"autofilter|auto.?filter.{0,15}(enabled|on all|every)", re.I),
     "pref_autofilter", "AutoFilter on all tables"),
    (re.compile(r"co.?authored.?by|commit.{0,20}trailer", re.I),
     "pref_coauthor", "Co-authored-by in every commit"),
    (re.compile(r"(both|dual).{0,20}(github|forgejo|remote)|two.{0,10}remote", re.I),
     "pref_dual_remote", "Dual git remotes (GitHub + Forgejo)"),
    (re.compile(r"cdsid.{0,40}(not|never|instead).{0,20}worker.{0,10}id", re.I),
     "pref_cdsid_term", "Use 'CDSID' not 'Worker ID'"),
    (re.compile(r"question topic.{0,20}(not|never|instead).{0,20}dimension", re.I),
     "pref_question_topic", "Use 'Question Topic' not 'Dimension'"),
    (re.compile(r"(literal|actual).{0,20}percent|percentages?.{0,20}literal|21\.9.{0,10}not.{0,10}0\.", re.I),
     "pref_literal_pct", "Percentages as literals (21.9 not 0.219)"),
]

CORRECTION_ARCHETYPES: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"org.{0,20}(l1|level.{0,5}1).{0,40}(should|wrong|not|instead)", re.I),
     "corr_org_level", "Org Level: L2 not L1"),
    (re.compile(r"(mip|proprietary|label|sensitivity).{0,60}(not done|forgot|missed|why|standing instruction)", re.I),
     "corr_mip_label", "MIP label: forgot to apply"),
    (re.compile(r"(wf_metrics|hand.?craft|hand.?written sql).{0,40}(wrong|not|never|avoid|don.t)", re.I),
     "corr_wrong_tool", "Used wrong tool (wf_metrics / hand-SQL)"),
    (re.compile(r"(americas|region).{0,40}(geographic|geographical|not what i meant|i did not mean)", re.I),
     "corr_americas_intent", "Ambiguous 'region' → org node not geography"),
    (re.compile(r"not (current|latest|up.?to.?date)\b", re.I),
     "corr_stale_data", "Returned stale data instead of current"),
    (re.compile(r"(model|month model|people.?master).{0,40}(not aware|super concerned|missing|3.{0,5}model)", re.I),
     "corr_data_model_gap", "Agent unaware of data model (MONTH/PEOPLE_MASTER)"),
    (re.compile(r"only.{0,25}(xlsx|excel|one).{0,20}attach", re.I),
     "corr_single_attach", "Only one attachment sent instead of all"),
    (re.compile(r"worker.?id.{0,30}(not|never|should|cdsid)", re.I),
     "corr_worker_id_term", "Used 'Worker ID' — should be CDSID"),
    (re.compile(r"(after all.{0,20}sessions|super concerned.{0,40}sessions|still not.{0,40}after)", re.I),
     "corr_repeated_miss", "Repeated failure across many sessions"),
    (re.compile(r"\bdimension\b.{0,30}(question topic|should be)", re.I),
     "corr_dimension_term", "Used 'Dimension' — should be 'Question Topic'"),
]


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

    # ── User node ────────────────────────────────────────────────────────
    user_node = GraphNode(
        id="user",
        label="You",
        group="user",
        size=40,
        count=len(sessions),
        description="People analytics professional — Volvo Cars",
        first_seen=sessions[0].created_at if sessions else "",
        last_seen=sessions[-1].created_at if sessions else "",
        metadata={
            "total_sessions": len(sessions),
            "total_turns": len(signals),
        },
    )
    upsert(user_node)

    # ── Topic nodes — frequency from session summaries, turns, checkpoints ──
    topic_sessions: dict[str, list[str]] = defaultdict(list)   # topic → [session_id]
    topic_timestamps: dict[str, list[str]] = defaultdict(list)
    topic_examples: dict[str, list[str]] = defaultdict(list)
    topic_source_turns: dict[str, list[dict]] = defaultdict(list)

    from .classify import TOPIC_BUCKETS

    def _check_topic(text: str, session_id: str, timestamp: str, example: str, turn_idx: int = -1):
        clean_text = _clean_message(text)
        text_lower = clean_text.lower()
        clean_example = _short(example)
        for topic, keywords in TOPIC_BUCKETS.items():
            for kw in keywords:
                if kw in text_lower:
                    topic_sessions[topic].append(session_id)
                    topic_timestamps[topic].append(timestamp)
                    if len(topic_examples[topic]) < 6 and clean_example:
                        topic_examples[topic].append(clean_example)
                    if len(topic_source_turns[topic]) < 5 and turn_idx >= 0:
                        topic_source_turns[topic].append(
                            {"session_id": session_id, "turn_index": turn_idx, "timestamp": timestamp}
                        )
                    break

    for s in sessions:
        if s.summary:
            clean_summary = _clean_message(s.summary)
            _check_topic(clean_summary, s.id, s.created_at, clean_summary)
        for cp in s.checkpoints:
            for field_text in [cp.title, cp.overview, cp.work_done, cp.technical_details]:
                if field_text:
                    # Clean the FULL field before truncating so end-markers are visible
                    clean_field = _clean_message(field_text)
                    if clean_field:
                        _check_topic(clean_field, s.id, cp.created_at, clean_field[:200])

    for sig in signals:
        for topic in sig.topics:
            topic_sessions[topic].append(sig.turn.session_id)
            topic_timestamps[topic].append(sig.turn.timestamp)
            clean_summary = _short(sig.turn.session_summary or "")
            if len(topic_examples[topic]) < 6 and clean_summary:
                topic_examples[topic].append(clean_summary)
            if len(topic_source_turns[topic]) < 5:
                topic_source_turns[topic].append({
                    "session_id": sig.turn.session_id,
                    "turn_index": sig.turn.turn_index,
                    "timestamp": sig.turn.timestamp,
                })

    # Deduplicate session counts per topic
    topic_unique_sessions: dict[str, set[str]] = defaultdict(set)
    for topic, sids in topic_sessions.items():
        topic_unique_sessions[topic].update(sids)

    for topic, sids in topic_unique_sessions.items():
        timestamps = topic_timestamps[topic]
        n = upsert(GraphNode(
            id=f"topic_{re.sub(r'[^a-z0-9]', '_', topic.lower())}",
            label=topic,
            group="topic",
            size=max(8, min(35, len(sids) // 2 + 8)),
            count=len(sids),
            description=f"Active in {len(sids)} sessions",
            examples=list(dict.fromkeys(topic_examples[topic]))[:5],
            first_seen=min(timestamps) if timestamps else "",
            last_seen=max(timestamps) if timestamps else "",
            source_turns=topic_source_turns[topic][:5],
        ))
        edges_raw[("user", n.id, "WORKS_ON")] += len(sids)

    # ── Tool nodes ───────────────────────────────────────────────────────
    tool_turns: dict[str, list[TurnSignal]] = defaultdict(list)
    for sig in signals:
        for tool in sig.tools:
            tool_turns[tool].append(sig)

    for tool, sigs in tool_turns.items():
        timestamps = [s.turn.timestamp for s in sigs]
        n = upsert(GraphNode(
            id=f"tool_{re.sub(r'[^a-z0-9]', '_', tool.lower())}",
            label=tool,
            group="tool",
            size=max(6, min(25, len(sigs) // 3 + 6)),
            count=len(sigs),
            description=f"Mentioned in {len(sigs)} turns",
            examples=[_short(s.cleaned_message) for s in sigs[:4]],
            first_seen=min(timestamps),
            last_seen=max(timestamps),
            source_turns=[
                {"session_id": s.turn.session_id, "turn_index": s.turn.turn_index, "timestamp": s.turn.timestamp}
                for s in sigs[:5]
            ],
        ))
        edges_raw[("user", n.id, "USES")] += len(sigs)

    # Connect tools to primary topics
    tool_topic_map = {
        "wf_duck.py": "DuckDB",
        "wf_query.py": "Org Structure",
        "sf.py": "Snowflake / SQL",
        "nsc_harness.py": "NSC Analytics",
        "wf_extract.py": "Workforce Metrics",
        "wf_metrics.py": "Snowflake / SQL",
        "openpyxl": "Excel Output",
        "xlsxwriter": "Excel Output",
        "pandas": "Data Pipeline",
        "streamlit": "Visualisation",
        "duckdb": "DuckDB",
        "snowflake": "Snowflake / SQL",
    }
    for tool, topic in tool_topic_map.items():
        tid = f"tool_{re.sub(r'[^a-z0-9]', '_', tool.lower())}"
        topic_id = f"topic_{re.sub(r'[^a-z0-9]', '_', topic.lower())}"
        if tid in nodes and topic_id in nodes:
            edges_raw[(topic_id, tid, "USES")] += 1

    # ── File extension analysis → topic signals ──────────────────────────
    ext_topic: dict[str, str] = {
        ".xlsx": "Excel Output", ".pptx": "PowerPoint / Docs", ".docx": "PowerPoint / Docs",
        ".py": "Python Scripting", ".sql": "Snowflake / SQL", ".md": "Git / Repos",
        ".csv": "Data Pipeline", ".json": "Data Pipeline",
    }
    for s in sessions:
        for fp in s.files_touched:
            suffix = "." + fp.rsplit(".", 1)[-1].lower() if "." in fp else ""
            topic = ext_topic.get(suffix)
            if topic:
                topic_id = f"topic_{re.sub(r'[^a-z0-9]', '_', topic.lower())}"
                if topic_id in nodes:
                    edges_raw[("user", topic_id, "WORKS_ON")] += 0.5

    # ── Preference archetype nodes ────────────────────────────────────────
    # Only match turns that carry a preference or correction signal to reduce false positives
    pref_matches: dict[str, list[tuple[str, str, str, int, str]]] = defaultdict(list)  # node_id → [(label, excerpt, timestamp, turn_idx, session_id)]

    for sig in signals:
        if not sig.is_preference and not sig.is_correction:
            continue
        msg_lower = sig.cleaned_message.lower()
        for pattern, node_id, label in PREFERENCE_ARCHETYPES:
            if pattern.search(msg_lower):
                pref_matches[node_id].append((
                    label,
                    _short(sig.cleaned_message),
                    sig.turn.timestamp,
                    sig.turn.turn_index,
                    sig.turn.session_id,
                ))

    # Only create nodes with at least 1 match
    for node_id, matches in pref_matches.items():
        if not matches:
            continue
        label = matches[0][0]
        timestamps = [m[2] for m in matches]
        n = upsert(GraphNode(
            id=node_id,
            label=label,
            group="preference",
            size=max(8, min(22, len(matches) + 6)),
            count=len(matches),
            description=f"Standing preference — {len(matches)} evidence point(s)",
            examples=list(dict.fromkeys(m[1] for m in matches))[:4],
            first_seen=min(timestamps),
            last_seen=max(timestamps),
            source_turns=[
                {"session_id": m[4], "turn_index": m[3], "timestamp": m[2]}
                for m in matches[:5]
            ],
        ))
        edges_raw[("user", node_id, "PREFERS")] += len(matches)

    # ── Correction archetype nodes ────────────────────────────────────────
    corr_matches: dict[str, list[tuple[str, str, str, int, str]]] = defaultdict(list)

    for sig in signals:
        if not sig.is_correction:
            continue
        msg_lower = sig.cleaned_message.lower()
        for pattern, node_id, label in CORRECTION_ARCHETYPES:
            if pattern.search(msg_lower):
                corr_matches[node_id].append((
                    label,
                    _short(sig.cleaned_message),
                    sig.turn.timestamp,
                    sig.turn.turn_index,
                    sig.turn.session_id,
                ))

    for node_id, matches in corr_matches.items():
        if not matches:
            continue
        label = matches[0][0]
        timestamps = [m[2] for m in matches]
        n = upsert(GraphNode(
            id=node_id,
            label=label,
            group="correction",
            size=max(8, min(22, len(matches) * 2 + 6)),
            count=len(matches),
            description=f"Recurring correction — {len(matches)} occurrence(s)",
            examples=list(dict.fromkeys(m[1] for m in matches))[:4],
            first_seen=min(timestamps),
            last_seen=max(timestamps),
            source_turns=[
                {"session_id": m[4], "turn_index": m[3], "timestamp": m[2]}
                for m in matches[:5]
            ],
        ))
        edges_raw[("user", node_id, "CORRECTED")] += len(matches)

    # Link pref/corr nodes to relevant topics
    pref_corr_topic_links: dict[str, str] = {
        "corr_org_level":       "Org Structure",
        "corr_mip_label":       "Sensitivity Labels",
        "corr_wrong_tool":      "Workforce Metrics",
        "corr_stale_data":      "Workforce Metrics",
        "corr_data_model_gap":  "Workforce Metrics",
        "corr_worker_id_term":  "Volvo Domain",
        "corr_dimension_term":  "Volvo Domain",
        "corr_single_attach":   "Excel Output",
        "corr_americas_intent": "Org Structure",
        "corr_repeated_miss":   "Volvo Domain",
        "pref_org_l2":          "Org Structure",
        "pref_xlsx_output":     "Excel Output",
        "pref_muted_colours":   "Visualisation",
        "pref_wf_duck_primary": "DuckDB",
        "pref_mip_label":       "Sensitivity Labels",
        "pref_named_sheets":    "Excel Output",
        "pref_dual_remote":     "Git / Repos",
        "pref_coauthor":        "Git / Repos",
        "pref_literal_pct":     "Workforce Metrics",
        "pref_autofit":         "Excel Output",
        "pref_autofilter":      "Excel Output",
        "pref_cdsid_term":      "Volvo Domain",
        "pref_question_topic":  "Volvo Domain",
    }
    for nid, topic in pref_corr_topic_links.items():
        topic_id = f"topic_{re.sub(r'[^a-z0-9]', '_', topic.lower())}"
        if nid in nodes and topic_id in nodes:
            edges_raw[(nid, topic_id, "RELATED_TO")] = 1

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

    # QualityBar + StakeholderContext → topic links
    quality_topic_links = {
        "quality_executive_quality": "Volvo Domain",
        "quality_production_standard": "Data Pipeline",
        "quality_human_readability": "Excel Output",
        "quality_precision_standard": "Workforce Metrics",
    }
    for qnid, topic in quality_topic_links.items():
        tid = f"topic_{re.sub(r'[^a-z0-9]', '_', topic.lower())}"
        if qnid in nodes and tid in nodes:
            edges_raw[(qnid, tid, "RELATED_TO")] += 1


    # ── RELATED_TO topic-topic edges via co-occurrence ────────────────────
    topic_co: dict[tuple[str, str], int] = defaultdict(int)
    for sig in signals:
        tids = [f"topic_{re.sub(r'[^a-z0-9]', '_', t.lower())}" for t in sig.topics if f"topic_{re.sub(r'[^a-z0-9]', '_', t.lower())}" in nodes]
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                pair = tuple(sorted([tids[i], tids[j]]))
                topic_co[pair] += 1  # type: ignore[arg-type]

    for (t1, t2), count in topic_co.items():
        if count >= 5:  # only meaningful co-occurrences
            edges_raw[(t1, t2, "RELATED_TO")] += count

    # ── User metadata ─────────────────────────────────────────────────────
    top_topics = sorted(topic_unique_sessions.items(), key=lambda x: -len(x[1]))[:5]
    nodes["user"].metadata.update({
        "top_topics": [(t, len(s)) for t, s in top_topics],
        "first_session": sessions[0].created_at if sessions else None,
        "last_session": sessions[-1].created_at if sessions else None,
    })

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
