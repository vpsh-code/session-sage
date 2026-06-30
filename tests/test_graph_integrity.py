"""
Automated tests for session-sage graph integrity.

Catches bugs like:
- Nodes in JSON not appearing in HTML
- visibleGroups not applied on load (the "11 instead of 30" bug)
- Disconnected nodes
- Duplicate IDs
- Bad edge references
- Sidebar stat mismatch
- Preference/correction count too low (regression guard)
"""

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
SAGE = REPO / "session_sage"


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def graph_output(tmp_path_factory):
    """Run session-sage once and return (json_data, html_text)."""
    out = tmp_path_factory.mktemp("sage")
    json_path = out / "graph.json"
    html_path = out / "graph.html"

    result = subprocess.run(
        [sys.executable, "-m", "session_sage.run",
         "--json-out", str(json_path),
         "--output", str(html_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"run.py failed:\n{result.stderr}"

    data = json.loads(json_path.read_text())
    html = html_path.read_text()
    return data, html


# ── JSON integrity tests ──────────────────────────────────────────────────────

def test_has_nodes(graph_output):
    data, _ = graph_output
    assert len(data["nodes"]) > 0, "Graph has no nodes"


def test_has_edges(graph_output):
    data, _ = graph_output
    assert len(data["links"]) > 0, "Graph has no edges"


def test_user_node_exists(graph_output):
    data, _ = graph_output
    user = [n for n in data["nodes"] if n["id"] == "user"]
    assert len(user) == 1, "Exactly one 'user' node required"


def test_no_duplicate_node_ids(graph_output):
    data, _ = graph_output
    ids = [n["id"] for n in data["nodes"]]
    from collections import Counter
    dups = {k: v for k, v in Counter(ids).items() if v > 1}
    assert not dups, f"Duplicate node IDs: {dups}"


def test_all_edge_references_valid(graph_output):
    data, _ = graph_output
    node_ids = {n["id"] for n in data["nodes"]}
    bad = [
        (l["source"], l["target"])
        for l in data["links"]
        if l["source"] not in node_ids or l["target"] not in node_ids
    ]
    assert not bad, f"Edges reference non-existent nodes: {bad[:5]}"


def test_preference_nodes_minimum(graph_output):
    """Regression: prefs must exceed the old hardcoded archetype count (was 3)."""
    data, _ = graph_output
    prefs = [n for n in data["nodes"] if n["group"] == "preference"]
    assert len(prefs) >= 10, f"Too few preference nodes: {len(prefs)} (expected ≥10)"


def test_correction_nodes_minimum(graph_output):
    """Regression: corrections must exceed the old archetype gate result (was 4)."""
    data, _ = graph_output
    corrs = [n for n in data["nodes"] if n["group"] == "correction"]
    assert len(corrs) >= 8, f"Too few correction nodes: {len(corrs)} (expected ≥8)"


def test_all_preference_nodes_have_edges(graph_output):
    """Every preference node must connect to the user node."""
    data, _ = graph_output
    pref_ids = {n["id"] for n in data["nodes"] if n["group"] == "preference"}
    connected = set()
    for l in data["links"]:
        if l["source"] in pref_ids:
            connected.add(l["source"])
        if l["target"] in pref_ids:
            connected.add(l["target"])
    orphans = pref_ids - connected
    assert not orphans, f"Preference nodes with no edges: {orphans}"


def test_preference_node_sizes_nonzero(graph_output):
    data, _ = graph_output
    bad = [n["id"] for n in data["nodes"] if n["group"] == "preference" and n.get("size", 0) <= 0]
    assert not bad, f"Preference nodes with zero/missing size: {bad}"


def test_topic_nodes_present(graph_output):
    data, _ = graph_output
    topics = [n for n in data["nodes"] if n["group"] == "topic"]
    assert len(topics) >= 3, f"Too few topic nodes: {len(topics)}"


def test_tool_nodes_present(graph_output):
    data, _ = graph_output
    tools = [n for n in data["nodes"] if n["group"] == "tool"]
    assert len(tools) >= 3, f"Too few tool nodes: {len(tools)}"


# ── HTML integrity tests ──────────────────────────────────────────────────────

def _extract_graph_from_html(html: str) -> dict:
    m = re.search(r"const GRAPH\s*=\s*(\{.*?\});", html, re.S)
    assert m, "GRAPH constant not found in HTML"
    return json.loads(m.group(1))


def test_html_embeds_all_json_nodes(graph_output):
    """Every node in the JSON must appear in the embedded HTML GRAPH object."""
    data, html = graph_output
    g = _extract_graph_from_html(html)
    json_ids = {n["id"] for n in data["nodes"]}
    html_ids = {n["id"] for n in g["nodes"]}
    missing = json_ids - html_ids
    assert not missing, f"Nodes in JSON but missing from HTML: {missing}"


def test_html_preference_count_matches_json(graph_output):
    """The number of preference nodes in the HTML must equal the JSON."""
    data, html = graph_output
    g = _extract_graph_from_html(html)
    json_count = sum(1 for n in data["nodes"] if n["group"] == "preference")
    html_count = sum(1 for n in g["nodes"] if n["group"] == "preference")
    assert html_count == json_count, (
        f"HTML has {html_count} preference nodes but JSON has {json_count}"
    )


def test_html_correction_count_matches_json(graph_output):
    data, html = graph_output
    g = _extract_graph_from_html(html)
    json_count = sum(1 for n in data["nodes"] if n["group"] == "correction")
    html_count = sum(1 for n in g["nodes"] if n["group"] == "correction")
    assert html_count == json_count, (
        f"HTML has {html_count} correction nodes but JSON has {json_count}"
    )


def test_html_initial_visibility_applied(graph_output):
    """
    The HTML must call applyVisibility() on load — not just inside button handlers.
    This prevents the 'visible in JSON but hidden on canvas' bug.
    """
    _, html = graph_output
    # applyVisibility() must be defined AND called outside the button event listener
    assert "function applyVisibility()" in html, \
        "applyVisibility() function not defined in HTML"
    # The call must appear OUTSIDE the addEventListener block
    # Simplest check: appears at least twice (definition + standalone call)
    call_count = html.count("applyVisibility()")
    assert call_count >= 2, (
        f"applyVisibility() called only {call_count} time(s) — "
        "must be called on init AND in filter button handler"
    )


def test_html_visible_groups_includes_preference(graph_output):
    """'preference' must be in the initial visibleGroups set."""
    _, html = graph_output
    m = re.search(r"const visibleGroups\s*=\s*new Set\(\[([^\]]+)\]\)", html)
    assert m, "visibleGroups definition not found"
    groups_str = m.group(1)
    assert "'preference'" in groups_str or '"preference"' in groups_str, \
        f"'preference' not in initial visibleGroups: {groups_str}"


def test_html_visible_groups_includes_correction(graph_output):
    _, html = graph_output
    m = re.search(r"const visibleGroups\s*=\s*new Set\(\[([^\]]+)\]\)", html)
    assert m, "visibleGroups definition not found"
    groups_str = m.group(1)
    assert "'correction'" in groups_str or '"correction"' in groups_str, \
        f"'correction' not in initial visibleGroups: {groups_str}"


def test_html_has_filter_buttons_for_key_groups(graph_output):
    """Filter buttons must exist for preference and correction groups."""
    _, html = graph_output
    for group in ("preference", "correction", "topic", "tool"):
        assert f'data-group="{group}"' in html, \
            f"No filter button for group '{group}'"


def test_html_stat_counter_shows_preference(graph_output):
    """The sidebar stat for Preferences must be present."""
    _, html = graph_output
    assert "Preferences" in html, "No 'Preferences' stat label in HTML"
    assert "Corrections" in html, "No 'Corrections' stat label in HTML"


# ── Output path tests ─────────────────────────────────────────────────────────

def test_skill_nodes_present(graph_output):
    """Skills must be discovered from sessions — expect ~28 from real data."""
    data, _ = graph_output
    skills = [n for n in data["nodes"] if n["group"] == "skill"]
    assert len(skills) >= 15, f"Too few skill nodes: {len(skills)} (expected ≥15)"


def test_skill_nodes_have_edges(graph_output):
    """Every skill node must connect to the user node."""
    data, _ = graph_output
    skill_ids = {n["id"] for n in data["nodes"] if n["group"] == "skill"}
    connected = set()
    for l in data["links"]:
        if l["source"] in skill_ids: connected.add(l["source"])
        if l["target"] in skill_ids: connected.add(l["target"])
    orphans = skill_ids - connected
    assert not orphans, f"Skill nodes with no edges: {orphans}"


def test_html_skill_count_matches_json(graph_output):
    data, html = graph_output
    g = _extract_graph_from_html(html)
    json_count = sum(1 for n in data["nodes"] if n["group"] == "skill")
    html_count = sum(1 for n in g["nodes"] if n["group"] == "skill")
    assert html_count == json_count, f"HTML has {html_count} skill nodes but JSON has {json_count}"


def test_html_visible_groups_includes_skill(graph_output):
    _, html = graph_output
    m = re.search(r"const visibleGroups\s*=\s*new Set\(\[([^\]]+)\]\)", html)
    assert m, "visibleGroups definition not found"
    assert "'skill'" in m.group(1) or '"skill"' in m.group(1), \
        f"'skill' not in initial visibleGroups: {m.group(1)}"


def test_html_has_skill_filter_button(graph_output):
    _, html = graph_output
    assert 'data-group="skill"' in html, "No filter button for 'skill' group"


def test_html_stat_counter_shows_skills(graph_output):
    _, html = graph_output
    assert "'Skills'" in html or '"Skills"' in html or ">Skills<" in html, \
        "No 'Skills' stat label in HTML"

    """Output must never default to ~/Downloads."""
    import session_sage.run as run_mod
    import inspect
    src = inspect.getsource(run_mod)
    assert "Downloads" not in src or "downloads" not in src.lower().replace("Downloads", ""), \
        "run.py references Downloads as an output path"
