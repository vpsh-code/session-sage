"""
Classify user turns into typed signals:
  - CORRECTION  : user pushes back on something the agent did wrong
  - PREFERENCE  : user states or implies a standing preference / rule
  - TOPIC       : what domain area the turn belongs to
  - TOOL        : specific tool / file mentioned
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .extract import Turn


# ---------------------------------------------------------------------------
# Boilerplate stripping
# System-injected blocks contaminate classification if not removed first.
# ---------------------------------------------------------------------------

_STRIP_PATTERNS: list[re.Pattern] = [
    re.compile(r"<skill-context[^>]*>.*?</skill-context>", re.I | re.S),
    re.compile(r"<system_reminder[^>]*>.*?</system_reminder>", re.I | re.S),
    re.compile(r"<file:[^>]+>", re.I),
    # Full COPILOT-GUARD block with end marker
    re.compile(r"\[COPILOT-GUARD:.*?\[END SESSION CONTEXT\]", re.I | re.S),
    # Partial COPILOT-GUARD block (no end marker — truncated checkpoints / raw turns):
    # strip from [COPILOT-GUARD: to "User's first message:" OR end-of-string
    re.compile(r"\[COPILOT-GUARD:.*?(?=User'?s? first message:|\Z)", re.I | re.S),
    re.compile(r"<!-- cold-start:begin.*?cold-start:end -->", re.I | re.S),
    # "User's first message:" prefix that follows the guard block
    re.compile(r"^User'?s? first message:\s*", re.I | re.M),
    # Pasted Copilot conversation transcripts (contaminates "try again" etc.)
    re.compile(r"\n?(You said:|Copilot said:|GitHub Copilot:)\s*", re.I),
    re.compile(r"please try again.*$", re.I | re.M),
]


def _clean_message(msg: str) -> str:
    """Remove system-injected wrappers; return only the human-authored portion."""
    for pat in _STRIP_PATTERNS:
        msg = pat.sub("", msg)
    msg = msg.strip()
    return msg[:2000] if len(msg) > 2000 else msg


# ---------------------------------------------------------------------------
# Signal patterns — kept tight to reduce false positives.
# Each tuple is (compiled pattern, label).
# ---------------------------------------------------------------------------

CORRECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Intent mismatch — explicit disambiguation
    (re.compile(r"\b(not what i (meant|said|wanted|asked|need))\b", re.I), "intent_mismatch"),
    (re.compile(r"\bi did not mean\b", re.I), "intent_mismatch"),
    (re.compile(r"\bi (am|was) not (asking|talking) about\b", re.I), "intent_mismatch"),
    (re.compile(r"\b(that'?s? not (right|correct|what))\b", re.I), "intent_mismatch"),
    # Wrong/incorrect — require context to reduce false positives on code discussions
    (re.compile(r"\b(that'?s? wrong|you('re| are) wrong|was wrong|is wrong)\b", re.I), "factual_error"),
    (re.compile(r"\b(that'?s? incorrect|you('re| are) incorrect)\b", re.I), "factual_error"),
    # Repeated failure — strong signal of persistent issue
    # Tightened (Opus): exclude "still not sure", "still not done" — require agent-failure verbs
    (re.compile(r"\bstill\s+(?:same|wrong|broken|failing)\b|\bstill\s+(?:an?\s+)?(?:issue|problem|error|bug)\b|\bstill\s+not\s+(?:working|fixed|applied|labell?ed|done|correct|right|complete|there|showing|loading|saved|updated|refreshed)\b", re.I), "repeated_failure"),
    (re.compile(r"\bsame issue\b", re.I), "repeated_failure"),
    (re.compile(r"\bafter all the (many )?sessions\b", re.I), "accumulated_frustration"),
    # Partial result — output was incomplete
    (re.compile(r"\b(only|just) .{0,30} (attached|loaded|shown|returned|visible)\b", re.I), "partial_result"),
    # Compliance miss — standing instruction not followed
    (re.compile(r"\b(not applied|not labell?ed|no label)\b", re.I), "compliance_miss"),
    (re.compile(r"\b(standing|existing) (instruction|rule|convention)\b", re.I), "rule_reminder"),
    (re.compile(r"\bwhy (is it not|didn.t you|haven.t you|are you not)\b", re.I), "compliance_miss"),
    (re.compile(r"\b(super |very |extremely )?concerned\b.{0,60}(instruction|rule|label|session)", re.I), "compliance_miss"),
    # Stale/wrong data
    (re.compile(r"\bnot (current|latest|up.?to.?date)\b", re.I), "stale_data"),
    # Knowledge gap
    (re.compile(r"\byou('re| are) (not aware|unaware|missing)\b", re.I), "knowledge_gap"),
    (re.compile(r"\bare.{0,10}(3|three) models not.{0,20}mentioned\b", re.I), "knowledge_gap"),
    # Omission — explicit "you forgot/missed"
    (re.compile(r"\b(you )(missed|forgot|omitted|skipped)\b", re.I), "omission"),
    (re.compile(r"\bwhy (is|was) it not (done|applied|used|there)\b", re.I), "omission"),
    # Tightened (Opus+GPT-5.5): require agent-attributing frame; "not sure but" / "not bad but" are benign
    (re.compile(r"\b(?:(?:that'?s|it'?s|you'?re|you\s+(?:said|wrote|did|made|gave\s+me))\s+not|not\s+(?:what|how|where|when|why|the\s+way))\b.{1,50}\bbut\b", re.I), "natural_negation"),
    # "no,? i (did not|didn't|never) (say|ask|mean|want)"
    (re.compile(r"\bno[,\s].{0,10}i (did not|didn.t|never) (say|ask|mean|want|mention)\b", re.I), "natural_negation"),
    # "try again" — only at message start/standalone; avoids echoed agent text "please try again"
    (re.compile(r"^try again\b", re.I | re.M), "redo_request"),
    # "i don't understand why" — signals agent did something unexpected
    (re.compile(r"\bi (don.t|do not) understand why\b", re.I), "intent_mismatch"),
]

PERSUASION_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Explicit concession — user acknowledges LLM was right
    (re.compile(r"\b(you'?re|you are) right\b", re.I), "llm_was_right"),
    (re.compile(r"\bfair (point|enough|call)\b", re.I), "llm_was_right"),
    # Tightened (GPT-5.5): require attributing subject; bare "good idea" at work meeting != persuasion
    (re.compile(r"\b(?:that'?s|this\s+is|your|you\s+made\s+a)\s+(?:a\s+)?good\s+(?:point|catch|call|idea|suggestion)\b|\bgood\s+(?:point|catch|call)[.!]?$", re.I | re.M), "llm_was_right"),
    (re.compile(r"\b(ok|okay|yes)[,\s].{0,30}(go (with|ahead)|use that|do that|proceed|makes sense)\b", re.I), "deferred_to_llm"),
    # Direction reversal — user switches from their original to LLM's suggestion
    (re.compile(r"\bactually[,\s].{0,50}(makes sense|better|right|good|go with)\b", re.I), "direction_reversal"),
    (re.compile(r"\b(your|that) (approach|suggestion|way|idea) (is |sounds )?(better|cleaner|simpler|makes more sense)\b", re.I), "direction_reversal"),
    (re.compile(r"\bi('ll| will) go with (your|that|the)\b", re.I), "direction_reversal"),
    (re.compile(r"\b(let'?s?|let us) (go with|use|do) (your|that)\b", re.I), "direction_reversal"),
    # Convinced / persuaded — explicit
    (re.compile(r"\b(ok|okay)[,\s].{0,20}convinced\b", re.I), "explicitly_convinced"),
    (re.compile(r"\b(convinced|persuaded)\b.{0,30}(by you|your point|that approach|your reasoning)\b", re.I), "explicitly_convinced"),
    # Acknowledging a missed insight
    (re.compile(r"\bi (didn.t|did not|hadn.t|have not) (think|consider|realise|realize|know) (about )?that\b", re.I), "missed_insight"),
    (re.compile(r"\bdidn.t (occur|think) to me\b", re.I), "missed_insight"),
    (re.compile(r"\bdidn.t (realise|realize)\b", re.I), "missed_insight"),
    # Conceding own approach was wrong
    (re.compile(r"\bi was (wrong|mistaken|off)\b", re.I), "self_correction"),
    (re.compile(r"\bmy (approach|assumption|idea|thinking) was (wrong|off|incorrect)\b", re.I), "self_correction"),
]

PREFERENCE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bshould (always|never)\b", re.I), "standing_rule"),
    (re.compile(r"\balways (use|apply|include|add|show|put)\b", re.I), "standing_rule"),
    (re.compile(r"\bnever (use|show|include)\b", re.I), "standing_rule"),
    (re.compile(r"\binstead (of|use)\b", re.I), "alternative_preference"),
    # Tightened (GPT-5.5): require output-object context; bare "I want/need" is too conversational
    (re.compile(r"\bi\s+prefer\b|\bi\s+expect\s+(?:you|the|this|it)\b|\bi\s+(?:want|need)\s+(?:it|this|the\s+(?:output|answer|result|format|file|response))\s+to\b", re.I), "explicit_preference"),
    # Tightened (GPT-5.5): exclude "must be a bug/mistake/error" — require actionable verb
    (re.compile(r"\b(must|have to)\s+(?:use|include|apply)\b|\b(?:must|has\s+to|have\s+to)\s+be\s+(?:used|included|applied|shown|present|available|formatted|labell?ed|human[- ]?readable|exact|correct|complete)\b", re.I), "hard_requirement"),
    (re.compile(r"\buse .{1,20} not .{1,20}\b", re.I), "tool_preference"),
    (re.compile(r"\bnot .{1,20}\buse\b.{1,20}\binstead\b", re.I), "tool_preference"),
    (re.compile(r"\b(org|level) (l?2|two)\b", re.I), "org_level_preference"),
    (re.compile(r"\bfrom now on\b", re.I), "standing_rule"),
    (re.compile(r"\b(in this (repo|project)|for (this|all) (file|workbook)s?)\b", re.I), "scoped_preference"),
]


# ---------------------------------------------------------------------------
# Expanded behavioural signal patterns (10 types)
# ---------------------------------------------------------------------------

METHODOLOGY_CHALLENGE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bthat('?s| is) not (right|correct|valid|how it works)\b", re.I), "invalid_method"),
    (re.compile(r"\bwhy (are|did|would) you (using|use|assuming|assume)\b", re.I), "assumption_challenge"),
    (re.compile(r"\b(source of truth|canonical source|actual source)\b", re.I), "source_of_truth"),
    (re.compile(r"\bwhat('?s| is) the (logic|methodology|definition|basis)\b", re.I), "definition_probe"),
    (re.compile(r"\bcompare (against|to) (the actual|historical|baseline|source)\b", re.I), "baseline_demand"),
    (re.compile(r"\bwe (cannot|can'?t|shouldn'?t) (derive|calculate|infer|assume)\b", re.I), "anti_inference"),
    (re.compile(r"\b(do(es)? this|that) (make sense|hold up)\b", re.I), "validity_check"),
]

KNOWLEDGE_BOUNDARY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(i don'?t know|not sure (about|how|why)|unclear|confused)\b", re.I), "explicit_uncertainty"),
    (re.compile(r"\bexplain (this|that|why|how)\b", re.I), "explanation_request"),
    (re.compile(r"\bwhat('?s| is) the difference between\b", re.I), "concept_distinction"),
    (re.compile(r"\bhelp me understand\b", re.I), "understanding_request"),
    (re.compile(r"\bwalk me through\b", re.I), "stepwise_learning"),
    (re.compile(r"\bwhy does this happen\b", re.I), "causal_learning"),
    (re.compile(r"\bi didn'?t (know|realise|realize) (that|this|you could)\b", re.I), "learning_moment"),
    (re.compile(r"\bnow i (understand|see|get it|get why)\b", re.I), "learning_moment"),
]

STAKEHOLDER_CONTEXT_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Tightened: "exec" alone matches "execute/executed"; require full form or paired context
    (re.compile(r"\b(?:c[- ]?suite|senior leadership|hrlt|chro|cfo|ceo|coo)\b|\bexec(?:utive)?s?\b.{0,40}\b(?:audience|ready|review|presentation|will ask|care about)\b", re.I), "executive_audience"),
    (re.compile(r"\b(board|steerco|leadership team|management team)\b", re.I), "formal_audience"),
    (re.compile(r"\b(presenting to|showing to|sharing with|this (will|needs to) go to)\b", re.I), "handoff_context"),
    (re.compile(r"\bpoliticall?y? sensitive\b|\bsensitive topic\b", re.I), "political_sensitivity"),
    (re.compile(r"\bmy (manager|boss|director|stakeholder)\b", re.I), "named_stakeholder"),
    (re.compile(r"\bmake it (exec|executive|leadership)[ -]?(ready|friendly)\b", re.I), "exec_ready_output"),
    (re.compile(r"\bwhat (they|leadership|execs?) (care about|will ask)\b", re.I), "stakeholder_need"),
]

DECISION_PATTERN_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\blet'?s (go with|do|use|take|proceed with)\b", re.I), "decision_commit"),
    (re.compile(r"\bnot worth it\b|\boverkill\b|\btoo much (effort|work|complexity)\b", re.I), "scope_rejection"),
    (re.compile(r"\btrade[- ]?off[s]?\b", re.I), "tradeoff_reasoning"),
    (re.compile(r"\bfast(est)? (path|way|approach)\b|\bsurgical\b", re.I), "speed_priority"),
    (re.compile(r"\brobust\b|\bproduction[- ]?ready\b|\bdurable\b", re.I), "durability_priority"),
    # Tightened: require verb+object — not bare "later"
    (re.compile(r"\b(?:defer|postpone|park|table)\s+(?:it|this|that|for now)\b|\b(?:not now|skip\s+(?:it|this|that)\s+for now)\b|\bcome back to (?:it|this|that) later\b", re.I), "deferral"),
    (re.compile(r"\bsticking with\b|\bgoing with\b|\bconfirmed[:\s]\b", re.I), "decision_confirmed"),
]

FRUSTRATION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bthis is (annoying|frustrating|ridiculous|not ok)\b", re.I), "explicit_frustration"),
    (re.compile(r"\bwhy (is|are) (you|this|it) still\b", re.I), "repeat_failure"),
    (re.compile(r"\b(not again|again\?*$|same (thing|issue|error|problem) again)\b", re.I | re.M), "recurrence"),
    (re.compile(r"\bstop (doing|using|asking|adding|explaining)\b", re.I), "behavior_stop"),
    # Tightened: require "already/now" to avoid neutral instructions like "just run it"
    (re.compile(r"\bjust\s+(?:do|fix|run|answer|use)\s+it\s+(?:already|now)\b", re.I), "impatience"),
    (re.compile(r"\b(too verbose|too much explanation|stop explaining)\b", re.I), "verbosity_friction"),
    (re.compile(r"\bhow (many times|often) (do i|have i|must i)\b", re.I), "repeat_instruction"),
]

TRUST_CALIBRATION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bare you sure\b|\bdouble[- ]?check\b|\bsanity[- ]?check\b", re.I), "confidence_probe"),
    # Tightened (GPT-5.5): require output noun for ALL verbs; "validate this" alone is too broad
    (re.compile(r"\bverify\s+(the\s+)?(answer|result|output|number|count|claim|finding|math|total)s?\b|\bvalidate\s+the\s+(answer|result|output|finding)s?\b|\bconfirm\s+(the\s+)?(number|answer|result|count)\b|\bdouble[- ]?check\s+(the|your)\s+(answer|result|output|number|count)\b", re.I), "verification_request"),
    (re.compile(r"\bdon'?t (guess|hallucinate|assume|make up)\b", re.I), "anti_hallucination"),
    (re.compile(r"\bread (the )?(file|docs?|source|schema) first\b", re.I), "source_first"),
    (re.compile(r"\bprove it\b|\bshow (me )?(proof|evidence|source)\b", re.I), "proof_demand"),
    (re.compile(r"\bi trust (you|your judgment)\b|\bgo ahead (autonomously)?\b", re.I), "autonomy_granted"),
    # Tightened: require explicit ordering intent
    (re.compile(r"\b(?:check|verify|validate|confirm)\s+(?:first|before\s+(?:doing|changing|running|writing|proceeding))\b|\bbefore you\s+(?:do|change|run|write|proceed)\b", re.I), "pre_check_required"),
]

ARCHITECTURE_MENTAL_MODEL_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bsource of truth\b|\bcanonical\b", re.I), "canonical_layer"),
    (re.compile(r"\bend[- ]?to[- ]?end\b|\bpipeline\b", re.I), "pipeline_model"),
    (re.compile(r"\btyped\b.*\b(schema|model|entity|signal)\b", re.I), "typed_model"),
    (re.compile(r"\bknowledge graph\b|\bgraph (model|structure|nodes?)\b", re.I), "graph_model"),
    # Tightened (Opus): "interface" alone fires on every UI/API mention; require qualifier
    (re.compile(r"\b(?:leaky|clean|proper|right|wrong)\s+abstraction\b|\babstraction\s+(?:boundary|layer|leak)\b|\binterface\s+(?:boundary|contract|between)\b|\b(?:respect|cross|violate)\s+(?:the\s+)?(?:abstraction|interface|contract|boundary)\b", re.I), "abstraction_boundary"),
    (re.compile(r"\bseparate (concerns?|layers?|logic|from)\b", re.I), "separation_of_concerns"),
    (re.compile(r"\bnot just a keyword matcher\b|\bintelligent system\b", re.I), "semantic_intelligence"),
    (re.compile(r"\breusable\b|\bgenerali[sz]able\b", re.I), "reuse_scale"),
]

URGENCY_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(asap|urgent|immediately|right now)\b", re.I), "immediate_urgency"),
    # Tightened (Opus): require deadline framing — "today" alone fires on "today's data"
    (re.compile(r"\b(?:by|before|due|needs?\s+(?:it|this)\s+by|deadline\s+(?:is\s+)?)\s+(?:today|tonight|this\s+morning|this\s+afternoon|eod)\b|\b(?:today|tonight)\s+(?:by|at)\s+\d", re.I), "same_day_deadline"),
    (re.compile(r"\bbefore (the )?(meeting|call|review|presentation)\b", re.I), "event_deadline"),
    (re.compile(r"\btime[- ]sensitive\b|\bdeadline\b", re.I), "time_sensitivity"),
    (re.compile(r"\b(eod|end of (day|week))\b", re.I), "eod_deadline"),
    (re.compile(r"\bneed (it|this) (fast|quickly|now|today)\b", re.I), "explicit_time_pressure"),
]

QUALITY_BAR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bexecutive[- ]?ready\b|\bleadership[- ]?ready\b", re.I), "executive_quality"),
    (re.compile(r"\bpolish(ed)?\b|\bmake it look (good|professional|clean)\b", re.I), "presentation_polish"),
    (re.compile(r"\bnot (good enough|acceptable|usable|clean enough)\b", re.I), "quality_rejection"),
    (re.compile(r"\bproduction[- ]?ready\b|\bdurable\b", re.I), "production_standard"),
    (re.compile(r"\bno ai slop\b|\bnot ai slop\b", re.I), "anti_ai_slop"),
    # Tightened (Opus): require intent context — bare "exact" fires on "exact path", "exact column"
    (re.compile(r"\bprecise(ly)?\b|\b(be|needs? to be|must be|want it)\s+exact\b|\bexactly\s+(right|correct|accurate|what)\b|\bno hand[- ]?waving\b|\bpinpoint\b", re.I), "precision_standard"),
    (re.compile(r"\bhuman[- ]?readable\b", re.I), "human_readability"),
    (re.compile(r"\bcomplete\b.{0,30}(solution|output|answer)\b", re.I), "completeness_standard"),
]

AGENCY_PREFERENCE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bdon'?t (ask|wait for) (me|permission|confirmation)\b", re.I), "no_permission_loop"),
    (re.compile(r"\bmake (reasonable )?assumptions\b", re.I), "assumption_autonomy"),
    # Tightened (GPT-5.5): exclude "don't proceed", "before you proceed" — require explicit autonomy framing
    (re.compile(r"\bgo\s+ahead\s+(?:and\s+)?\w+|\bjust\s+(?:go\s+ahead|proceed)\b|\bplease\s+proceed\b|\bjust\s+(?:do|fix|run|ship|implement)\s+it\b|\bdo\s+it\s+(?:now|autonomously|without\s+asking)\b", re.I), "execution_autonomy"),
    (re.compile(r"\bchallenge (me|the methodology|the assumption|my)\b", re.I), "challenge_expected"),
    (re.compile(r"\b(be concise|short answer|brief(ly)?|≤\s*\d+ words?)\b", re.I), "conciseness"),
    (re.compile(r"\bdon'?t over[- ]?explain\b|\bskip the explanation\b", re.I), "low_explanation"),
    (re.compile(r"\b(use (agents?|subagents?)|run (in )?parallel)\b", re.I), "delegation_preference"),
    (re.compile(r"\bloop (until|till) (green|done|complete|fixed)\b", re.I), "autonomous_loop"),
]

# ---------------------------------------------------------------------------
# Additional high-value domain-specific signals (Opus Round 1)
# ---------------------------------------------------------------------------

TOOL_DIRECTIVE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Tightened (Opus): require a known tool name, not any word pair
    (re.compile(r"\b(use|run)\s+[\w./_-]*(wf_duck|wf_query|wf_extract|sf\.py|nsc_harness|snowflake|duckdb|markitdown)[\w./_-]*\s+(not|instead\s+of)\s+\w+\b", re.I), "tool_directive"),
    # Tightened (GPT-5.5): require known tool name to prevent "never use global variables"
    (re.compile(r"\bnever\s+(use|run|call)\s+(wf_duck|wf_query|wf_extract|sf\.py|nsc_harness|snowflake|duckdb|markitdown|opus|gpt|claude|sonnet)\b", re.I), "tool_prohibition"),
    (re.compile(r"\balways\s+(use|run|call)\s+(wf_duck|snowflake|sf\.py|nsc_harness)\b", re.I), "tool_mandate"),
]

TERMINOLOGY_CORRECTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bit'?s\s+(called|named)\s+\w+\s+not\s+\w+\b", re.I), "term_correction"),
    (re.compile(r"\bdon'?t\s+(call|say)\s+(it|that)\b", re.I), "term_correction"),
    (re.compile(r"\bnever\s+(call|say|use)\s+['\"]?\w+['\"]?\b.{0,30}\b(instead|use)\b", re.I), "term_correction"),
    (re.compile(r"\bcorrect\s+term\s+is\b|\bshould be called\b", re.I), "term_correction"),
    # Tightened (GPT-5.5): require the negative context ("not", "never") to fire on worker id
    (re.compile(r"\bworker\s+id\b.{0,40}\b(not|never|don.?t|instead)\b|\b(not|never|don.?t)\b.{0,40}\bworker\s+id\b|\bcdsid\b.{0,20}\b(not|never|don.?t)\b|\bglobal\s+id\b.{0,20}\b(not|never|don.?t)\b", re.I), "id_terminology"),
    (re.compile(r"\bquestion topic\b.{0,20}\b(not|never)\b.{0,20}\bdimension\b", re.I), "dimension_terminology"),
]

FORMAT_PREFERENCE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(output|save|export|write|generate)\s+(as\s+)?(xlsx|excel|docx|pptx)\b", re.I), "xlsx_output"),
    (re.compile(r"\bnever\s+json\b|\bnot\s+json\b|\bjson\s+is\s+not\b", re.I), "no_json_output"),
    (re.compile(r"\bhuman[- ]?readable\s+(output|format|file|sheet)\b", re.I), "human_readable_format"),
    (re.compile(r"\bnamed?\s+sheet\b|\bsheet\s+name\b|\bsheet names\b", re.I), "named_sheets"),
]

MIP_COMPLIANCE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(mip|proprietary)\s+label\b", re.I), "mip_label"),
    (re.compile(r"\bapply.{0,20}\b(label|sensitivity)\b", re.I), "apply_label"),
    (re.compile(r"\blabelinfo\.xml\b|\bsensitivity label\b", re.I), "label_artifact"),
    (re.compile(r"\bforgot.{0,20}label\b|\bnot labell?ed\b|\bmissing.{0,10}label\b", re.I), "label_miss"),
]

SCOPE_DISAMBIGUATION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bin\s+this\s+(dataset|file|extract|snapshot|workbook)\b", re.I), "local_scope"),
    (re.compile(r"\bcompany[- ]?wide\b|\ball\s+(employees?|workers?|staff)\b", re.I), "company_scope"),
    (re.compile(r"\bscope\s+(of\s+)?(this|the)\s+(question|query|analysis)\b", re.I), "scope_clarify"),
    (re.compile(r"\b(local|snowflake|duckdb)\b.{0,30}\b(scope|source|data)\b", re.I), "data_source_scope"),
]

USER_REDIRECT_PATTERNS: list[tuple[re.Pattern, str]] = [
    # Fix (Opus+GPT-5.5): \A anchor to start-of-string only; exclude "no problem", "no worries", "not"
    (re.compile(r"\A\s*(no[,.](?!\s*(problem|worries|issue|worries))|wrong[,.]\s|that'?s\s+wrong[,.\s]|stop[,.]?\s)", re.I), "direct_redirect"),
    # Fix (Opus): require prior-state adverb or direct-object frame to avoid FP
    (re.compile(r"\bi\s+(already\s+)?(said|asked|told\s+you|mentioned)\b.{0,60}\b(already|again|before|earlier)\b|\b(as|like)\s+i\s+(said|asked|mentioned)\b|\bi\s+(just\s+)?(told\s+you|asked\s+you\s+to)\b", re.I), "reminder_redirect"),
    (re.compile(r"\bdid\s+i\s+ask\b|\bnot\s+what\s+i\s+(asked|wanted|said|meant)\b", re.I), "explicit_mismatch"),
]

MODEL_COST_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bnever\s+(use\s+)?opus\b|\bno\s+opus\b", re.I), "opus_prohibition"),
    (re.compile(r"\b(cheaper|free|0x|0\.33x|multiplier)\b.{0,40}(model|claude|gpt)\b", re.I), "cost_awareness"),
    (re.compile(r"\b(gpt[- ]?5[\.\-]?\d?[- ]?mini|haiku|sonnet)\b.{0,30}\b(use|prefer|instead)\b", re.I), "model_preference"),
    (re.compile(r"\bmodel\s+(multiplier|cost|tier|selection)\b", re.I), "model_routing"),
]


# ---------------------------------------------------------------------------
# Pasted-prompt filter — skip turns that are pasted templates, not user voice
# ---------------------------------------------------------------------------

def _is_pasted_prompt(text: str) -> bool:
    """Return True if this turn looks like a pasted external prompt, not the user's own words."""
    if len(text) < 400:
        return False
    text_stripped = text.lstrip()
    # "You are a/an ..." or "You're a/an ..." framing
    if re.match(r"(?i)^you(\s+are|'re)\s+an?\b", text_stripped):
        return True
    # Code block + many newlines → pasted technical prompt
    if text.count("\n") >= 15 and "```" in text:
        return True
    # Fix (GPT-5.5): markdown headings alone is too broad; require system-prompt keywords too
    if (re.search(r"(?im)^#{1,3}\s+(?:system|developer|instructions?|role|task|prompt|constraints?|output\s+format)\b", text)
            and re.search(r"(?i)\b(you\s+are|act\s+as|your\s+task\s+is|follow\s+these\s+instructions)\b", text)):
        return True
    return False



TOPIC_BUCKETS: dict[str, list[str]] = {
    "Workforce Metrics":   ["fte", "headcount", " hc ", "ext hc", "workforce", "metric", "head count"],
    "NSC Analytics":       ["nsc", "national sales", "nscs", " usa ", " uk ", "nordics", "latam", "canada"],
    "Org Structure":       ["org level", "org l", "l2", "l3", "l4", "l5", "business unit", "reporting line", "hierarchy"],
    "Career Level":        ["career level", "korn ferry", "grade band", "senior leader", "associate professional"],
    "Excel Output":        ["xlsx", "workbook", "pivot", "autofit", "autofilter", "openpyxl", "xlsxwriter"],
    "Snowflake / SQL":     ["snowflake", " sql ", "select ", "sf.py", "wf_metrics"],
    "DuckDB":              ["duckdb", "wf_duck", "duck db"],
    "Python Scripting":    ["python", "script", "pandas", "plotly", "streamlit", "uv run"],
    "Git / Repos":         ["git commit", "git push", "github", "forgejo", "pull request", " pr ", " branch"],
    "LLM / AI":            ["claude", " gpt", "llm", "copilot", "skill", "agent", "mcp"],
    "Sensitivity Labels":  ["proprietary", "mip", "sensitivity label", "labeling", "privileged"],
    "Volvo Domain":        ["volvo", "cdsid", "people analytics", "smart pipeline", "hpo", "soc update"],
    "PowerPoint / Docs":   ["pptx", "docx", "powerpoint", "presentation", "slide"],
    "Data Pipeline":       ["pipeline", "batch append", "extract", "delta", "snapshot", "etl"],
    "Visualisation":       ["chart", "plotly", "visual", "palette", "colour", "color", "graph"],
}

TOOL_KEYWORDS: dict[str, list[str]] = {
    "wf_duck.py":      ["wf_duck"],
    "wf_query.py":     ["wf_query"],
    "sf.py":           ["sf.py"],
    "nsc_harness.py":  ["nsc_harness"],
    "wf_extract.py":   ["wf_extract"],
    "wf_metrics.py":   ["wf_metrics"],
    "openpyxl":        ["openpyxl"],
    "xlsxwriter":      ["xlsxwriter"],
    "pandas":          ["pandas"],
    "streamlit":       ["streamlit"],
    "duckdb":          ["duckdb"],
    "snowflake":       ["snowflake"],
    "graphify":        ["graphify"],
    "mempalace":       ["mempalace"],
    "dream":           [" dream "],
    "pos-chain":       ["pos-chain", "pos_chain"],
    "AppleScript":     ["applescript"],
}


# ---------------------------------------------------------------------------
# Signal dataclass
# ---------------------------------------------------------------------------

@dataclass
class TurnSignal:
    turn: Turn
    cleaned_message: str = ""
    correction_types: list[str] = field(default_factory=list)
    preference_types: list[str] = field(default_factory=list)
    persuasion_types: list[str] = field(default_factory=list)
    methodology_types: list[str] = field(default_factory=list)
    knowledge_boundary_types: list[str] = field(default_factory=list)
    stakeholder_types: list[str] = field(default_factory=list)
    decision_types: list[str] = field(default_factory=list)
    frustration_types: list[str] = field(default_factory=list)
    trust_types: list[str] = field(default_factory=list)
    architecture_types: list[str] = field(default_factory=list)
    urgency_types: list[str] = field(default_factory=list)
    quality_types: list[str] = field(default_factory=list)
    agency_types: list[str] = field(default_factory=list)
    tool_directive_types: list[str] = field(default_factory=list)
    terminology_types: list[str] = field(default_factory=list)
    format_types: list[str] = field(default_factory=list)
    mip_types: list[str] = field(default_factory=list)
    scope_types: list[str] = field(default_factory=list)
    redirect_types: list[str] = field(default_factory=list)
    model_cost_types: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)
    tools: list[str] = field(default_factory=list)
    is_pasted_prompt: bool = False

    @property
    def is_correction(self) -> bool: return bool(self.correction_types)
    @property
    def is_preference(self) -> bool: return bool(self.preference_types)
    @property
    def is_persuasion(self) -> bool: return bool(self.persuasion_types)
    @property
    def is_frustration(self) -> bool: return bool(self.frustration_types)
    @property
    def is_urgency(self) -> bool: return bool(self.urgency_types)


def classify_turn(turn: Turn) -> TurnSignal:
    clean = _clean_message(turn.user_message or "")
    msg_lower = clean.lower()
    pasted = _is_pasted_prompt(clean)
    sig = TurnSignal(turn=turn, cleaned_message=clean, is_pasted_prompt=pasted)

    if pasted:
        # Skip signal extraction on pasted external prompts — not user voice
        return sig

    _scan = lambda patterns, bucket: [bucket.append(lbl) for pat, lbl in patterns if pat.search(msg_lower)]
    _scan(CORRECTION_PATTERNS,               sig.correction_types)
    _scan(PREFERENCE_PATTERNS,               sig.preference_types)
    _scan(PERSUASION_PATTERNS,               sig.persuasion_types)
    _scan(METHODOLOGY_CHALLENGE_PATTERNS,    sig.methodology_types)
    _scan(KNOWLEDGE_BOUNDARY_PATTERNS,       sig.knowledge_boundary_types)
    _scan(STAKEHOLDER_CONTEXT_PATTERNS,      sig.stakeholder_types)
    _scan(DECISION_PATTERN_PATTERNS,         sig.decision_types)
    _scan(FRUSTRATION_PATTERNS,              sig.frustration_types)
    _scan(TRUST_CALIBRATION_PATTERNS,        sig.trust_types)
    _scan(ARCHITECTURE_MENTAL_MODEL_PATTERNS, sig.architecture_types)
    _scan(URGENCY_PATTERNS,                  sig.urgency_types)
    _scan(QUALITY_BAR_PATTERNS,              sig.quality_types)
    _scan(AGENCY_PREFERENCE_PATTERNS,        sig.agency_types)
    _scan(TOOL_DIRECTIVE_PATTERNS,           sig.tool_directive_types)
    _scan(TERMINOLOGY_CORRECTION_PATTERNS,   sig.terminology_types)
    _scan(FORMAT_PREFERENCE_PATTERNS,        sig.format_types)
    _scan(MIP_COMPLIANCE_PATTERNS,           sig.mip_types)
    _scan(SCOPE_DISAMBIGUATION_PATTERNS,     sig.scope_types)
    _scan(USER_REDIRECT_PATTERNS,            sig.redirect_types)
    _scan(MODEL_COST_PATTERNS,               sig.model_cost_types)

    for topic, keywords in TOPIC_BUCKETS.items():
        for kw in keywords:
            if kw in msg_lower:
                sig.topics.append(topic)
                break

    for tool, keywords in TOOL_KEYWORDS.items():
        for kw in keywords:
            if kw in msg_lower:
                sig.tools.append(tool)
                break

    return sig


def classify_all(turns: list[Turn]) -> list[TurnSignal]:
    return [classify_turn(t) for t in turns]
