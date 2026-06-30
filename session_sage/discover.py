"""
Statistical intelligence layer — no hardcoded domain knowledge.

Everything is DERIVED from the data:
  - Tools     : extracted from text by syntactic pattern (*.py, command invocations)
  - Paths     : mined from text, ranked by session spread
  - Preferences : session-spread n-grams that recur across many sessions
  - Topics    : TF-IDF per-session → greedy term-cluster merging
  - Corrections : behavioural signal turns → content extraction

Scoring is based on SESSION SPREAD (how many distinct sessions contain a signal),
not raw count — a pattern in 30 sessions is a standing rule; 30 hits in 1 session
is a rant.

No imports beyond stdlib.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .classify import TurnSignal
    from .extract import Session


# ---------------------------------------------------------------------------
# Discovered artefact types
# ---------------------------------------------------------------------------

@dataclass
class DiscoveredTool:
    name: str
    session_count: int
    total_count: int
    first_seen: str = ""
    last_seen: str = ""
    examples: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.session_count * math.log1p(self.total_count)


@dataclass
class DiscoveredPath:
    root: str                  # e.g. ~/Projects, ~/Downloads/excel
    session_count: int
    total_count: int
    examples: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return self.session_count * math.log1p(self.total_count)


@dataclass
class DiscoveredPreference:
    phrase: str
    session_count: int
    total_count: int
    first_seen: str = ""
    last_seen: str = ""
    examples: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        # Heavily weight session spread — that's what makes something a standing rule
        return (self.session_count ** 1.5) * math.log1p(self.total_count)


@dataclass
class DiscoveredTopic:
    id: str
    label: str                 # Top TF-IDF term (title-cased)
    top_terms: list[str]       # Top 5 distinctive terms
    session_count: int
    turn_count: int
    first_seen: str = ""
    last_seen: str = ""


@dataclass
class DiscoveredCorrection:
    subject: str               # Extracted subject of the correction
    count: int
    session_count: int
    first_seen: str = ""
    last_seen: str = ""
    examples: list[str] = field(default_factory=list)


@dataclass
class DiscoveredSkill:
    name: str                  # e.g. "dream", "ce-brainstorm", "org-lookup"
    session_count: int
    total_count: int
    first_seen: str = ""
    last_seen: str = ""
    examples: list[str] = field(default_factory=list)
    skill_type: str = ""       # "ce" | "named" | "slash"  — detected source

    @property
    def score(self) -> float:
        return (self.session_count ** 1.5) * math.log1p(self.total_count)


@dataclass
class DiscoveredKnowledge:
    tools: list[DiscoveredTool]
    paths: list[DiscoveredPath]
    preferences: list[DiscoveredPreference]
    topics: list[DiscoveredTopic]
    corrections: list[DiscoveredCorrection]
    skills: list[DiscoveredSkill]
    total_sessions: int
    total_turns: int


# ---------------------------------------------------------------------------
# Text utilities
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset("""
a about after again all also am an and any are as at be because been being but by
can could did do does doing don done down each even for from get got had has have
he her here him his how i if in into is it its just know let like me more my
no not now of off on one or our out own please put re said same see should so
some than that the their them then there these they this those though through
to too under up us was we well were what when where which while who will with
would you your
""".split())

# Common filler phrases to ignore in bigram extraction
_FILLER_BIGRAMS = frozenset("""
can you could you please do please let me can i would like want to need to going to
i want i need i have i am i'm it is it's is it are you you can you should
this is that is which is what is
""".split(" __SEP__ ".join([""] * 2))[0:0])  # empty — filtered by stopword check below

_TOOL_STOPWORDS = frozenset("""
test setup main utils helper common base config init cli core
""".split())


def _tokenize(text: str) -> list[str]:
    """Lower-case word tokens, stripping noise."""
    return [w for w in re.findall(r"\b[a-z][a-z0-9_]{1,30}\b", text.lower())
            if w not in _STOPWORDS and len(w) > 2]


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _short(text: str, n: int = 80) -> str:
    t = text.strip()[:n]
    return t + "…" if len(text.strip()) > n else t


def _ts_to_dt(ts: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, AttributeError):
        return None


# ---------------------------------------------------------------------------
# Tool discovery — syntactic extraction, no hardcoded tool names
# ---------------------------------------------------------------------------

# Generic patterns that identify executable artefacts
_PY_FILE   = re.compile(r"\b([\w][\w.-]*\.py)\b")
_SH_FILE   = re.compile(r"\b([\w][\w.-]*\.sh)\b")
_UV_RUN    = re.compile(r"\buv\s+run\b(?:\s+--[\w-]+\s+\S+)*\s+([\w./~-]+\.py)")
_CLI_CMD   = re.compile(r"\b(git|gh|curl|docker|brew|npm|pip|conda|poetry|pdm|uv)\s+([\w-]+)")


def discover_tools(signals: list["TurnSignal"]) -> list[DiscoveredTool]:
    """Extract tools by syntactic pattern — no preset tool list."""
    counts: Counter[str] = Counter()
    sessions: dict[str, set[str]] = defaultdict(set)
    first_seen: dict[str, str] = {}
    last_seen: dict[str, str] = {}
    examples: dict[str, list[str]] = defaultdict(list)

    for sig in signals:
        msg = sig.cleaned_message
        sid = sig.turn.session_id
        ts = sig.turn.timestamp

        found: set[str] = set()
        for pat in (_PY_FILE, _SH_FILE):
            for m in pat.finditer(msg):
                name = m.group(1).lower()
                if name not in _TOOL_STOPWORDS and "__" not in name:
                    found.add(name)
        for m in _UV_RUN.finditer(msg):
            name = m.group(1).lower().split("/")[-1]
            if name:
                found.add(name)
        for m in _CLI_CMD.finditer(msg.lower()):
            found.add(m.group(1))  # e.g. "git", "gh", "uv"

        for name in found:
            counts[name] += 1
            sessions[name].add(sid)
            if name not in first_seen or ts < first_seen[name]:
                first_seen[name] = ts
            if name not in last_seen or ts > last_seen[name]:
                last_seen[name] = ts
            if len(examples[name]) < 4:
                examples[name].append(_short(msg))

    result = [
        DiscoveredTool(
            name=name,
            session_count=len(sessions[name]),
            total_count=counts[name],
            first_seen=first_seen.get(name, ""),
            last_seen=last_seen.get(name, ""),
            examples=examples[name],
        )
        for name in counts
        if len(sessions[name]) >= 2  # must appear in ≥2 sessions
    ]
    return sorted(result, key=lambda t: -t.score)


# ---------------------------------------------------------------------------
# Path discovery — mine canonical directories from text
# ---------------------------------------------------------------------------

_PATH_PAT = re.compile(r"(~/[\w/.+@-]{3,}|/(?:Users|home)/\w+/[\w/.+@-]{3,})")


def _path_root(path: str, depth: int = 2) -> str:
    """Return the first `depth` path components after ~ or /Users/name."""
    path = re.sub(r"^/(?:Users|home)/\w+", "~", path)
    parts = [p for p in path.split("/") if p and p != "~"]
    return "~/" + "/".join(parts[:depth]) if parts else path


def discover_paths(signals: list["TurnSignal"]) -> list[DiscoveredPath]:
    """Discover canonical directories from text — no preset list."""
    counts: Counter[str] = Counter()
    sessions: dict[str, set[str]] = defaultdict(set)
    examples: dict[str, list[str]] = defaultdict(list)

    for sig in signals:
        msg = sig.cleaned_message
        sid = sig.turn.session_id
        for m in _PATH_PAT.finditer(msg):
            root = _path_root(m.group(1))
            counts[root] += 1
            sessions[root].add(sid)
            if len(examples[root]) < 3:
                examples[root].append(_short(msg))

    result = [
        DiscoveredPath(
            root=root,
            session_count=len(sessions[root]),
            total_count=counts[root],
            examples=examples[root],
        )
        for root in counts
        if len(sessions[root]) >= 2
    ]
    return sorted(result, key=lambda p: -p.score)


# ---------------------------------------------------------------------------
# Skill discovery — extracts used Copilot skills from session turns
# ---------------------------------------------------------------------------

# Three extraction signals — no hardcoded skill names
# 1. <skill-context name="X"> — the CLI injects this tag when a skill activates
_SKILL_TAG    = re.compile(r'<skill-context\s+name=["\']([^"\']+)["\']', re.I)
# 2. Compound-engineering skill pattern: ce-<word>[-<word>]+  (purely syntactic)
_SKILL_CE     = re.compile(r'\bce-[a-z][a-z0-9]+(?:-[a-z][a-z0-9]+)+\b')
# 3. Slash invocation at start of message or after whitespace: /skill-name
#    Must be hyphenated (multi-word) to filter out /help, /clear, etc.
_SKILL_SLASH  = re.compile(r'(?:^|\s)/([a-z][a-z0-9]+(?:-[a-z][a-z0-9]+)+)\b', re.M)
# 4. "skill: X" or "skill X" invocation pattern in instructions text
_SKILL_INVOKE = re.compile(r'\bskill[:\s]+["\']?([a-z][a-z0-9]+(?:-[a-z][a-z0-9]+)*)["\']?', re.I)

# Single-segment names that are well-known standalone skills (not caught by hyphens)
_SKILL_SINGLE = re.compile(r'\b(dream|graphify|mempalace)\b', re.I)

# Filter noise — generic lowercase words that look like skill names but aren't
_SKILL_STOPWORDS = frozenset("""
and the for with this that from not use call run add do get set has have
when where which what how why did does being been would could should must
""".split())


def discover_skills(signals: list["TurnSignal"]) -> list[DiscoveredSkill]:
    """
    Discover which Copilot skills were used across sessions.

    Three extraction tiers (no hardcoded skill list):
    1. <skill-context name="X"> injection tags  — highest precision
    2. ce-* compound-engineering pattern         — syntactic
    3. /skill-name slash invocations             — contextual
    4. skill: "name" invocation references       — instructional
    5. Known standalone names (dream, graphify)  — short but unambiguous
    """
    counts:     Counter[str] = Counter()
    sessions:   dict[str, set[str]] = defaultdict(set)
    first_seen: dict[str, str] = {}
    last_seen:  dict[str, str] = {}
    examples:   dict[str, list[str]] = defaultdict(list)
    skill_type: dict[str, str] = {}

    for sig in signals:
        raw = sig.turn.user_message or ""
        msg = sig.cleaned_message
        sid = sig.turn.session_id
        ts  = sig.turn.timestamp

        found: dict[str, str] = {}  # name → source type

        # Tier 1: tag extraction (most reliable — CLI injects these automatically)
        for m in _SKILL_TAG.finditer(raw):
            name = m.group(1).strip().lower()
            if name and name not in _SKILL_STOPWORDS and len(name) > 2:
                found[name] = "tag"

        # Tier 2: ce- pattern (syntactic, catches all compound-engineering skills)
        for m in _SKILL_CE.finditer(raw):
            found[m.group(0).lower()] = "ce"

        # Tier 3: slash invocations (at message start or after whitespace)
        for m in _SKILL_SLASH.finditer(raw):
            name = m.group(1).lower()
            if name not in _SKILL_STOPWORDS:
                found[name] = found.get(name, "slash")

        # Tier 4 removed — "skill: X" pattern too noisy (matches prose like "Primary:", "Executes:")

        # Tier 5: well-known single-word standalone skills
        for m in _SKILL_SINGLE.finditer(raw):
            found[m.group(1).lower()] = found.get(m.group(1).lower(), "named")

        for name, stype in found.items():
            counts[name] += 1
            sessions[name].add(sid)
            if name not in first_seen or ts < first_seen[name]:
                first_seen[name] = ts
            if name not in last_seen or ts > last_seen[name]:
                last_seen[name] = ts
            if skill_type.get(name, "named") in ("named", "invoke", "slash") and stype in ("tag", "ce"):
                skill_type[name] = stype  # promote to higher-confidence source
            elif name not in skill_type:
                skill_type[name] = stype
            if len(examples[name]) < 4:
                examples[name].append(_short(raw, 100))

    result = [
        DiscoveredSkill(
            name=name,
            session_count=len(sessions[name]),
            total_count=counts[name],
            first_seen=first_seen.get(name, ""),
            last_seen=last_seen.get(name, ""),
            examples=examples[name],
            skill_type=skill_type.get(name, "named"),
        )
        for name in counts
        if len(sessions[name]) >= 2  # must appear in ≥2 sessions
    ]
    return sorted(result, key=lambda s: -s.score)


# ---------------------------------------------------------------------------
# Preference discovery — session-spread n-grams
# ---------------------------------------------------------------------------

# Bigrams that are too generic to be preferences
_GENERIC_BIGRAMS = frozenset({
    ("can", "you"), ("please", "do"), ("let", "me"), ("need", "to"),
    ("want", "to"), ("going", "to"), ("would", "like"), ("make", "sure"),
    ("sure", "to"), ("able", "to"), ("how", "to"), ("what", "is"),
    ("this", "is"), ("it", "is"), ("that", "is"), ("there", "is"),
    ("should", "be"), ("will", "be"), ("has", "been"), ("have", "been"),
    ("the", "same"), ("the", "following"), ("as", "well"), ("well", "as"),
    ("such", "as"), ("for", "example"), ("based", "on"), ("due", "to"),
})


def discover_preferences(signals: list["TurnSignal"]) -> list[DiscoveredPreference]:
    """
    Discover standing preferences as high-session-spread n-grams.

    A phrase that appears in many distinct sessions is a standing rule —
    the user keeps repeating it because the agent keeps not following it,
    OR because it's how they think about their domain.
    """
    bigram_sessions:    dict[tuple, set[str]] = defaultdict(set)
    bigram_counts:      Counter[tuple] = Counter()
    bigram_timestamps:  dict[tuple, list[str]] = defaultdict(list)
    bigram_examples:    dict[tuple, list[str]] = defaultdict(list)

    trigram_sessions:   dict[tuple, set[str]] = defaultdict(set)
    trigram_counts:     Counter[tuple] = Counter()
    trigram_timestamps: dict[tuple, list[str]] = defaultdict(list)
    trigram_examples:   dict[tuple, list[str]] = defaultdict(list)

    # Patterns that indicate injected system prompts or tool scaffolding, not user preferences
    _SYSTEM_PROMPT_RE = re.compile(
        r"^(?:you are a|you're a|act as|your role is|system:|<system>|resolve the user query|today:\s*\d{4})",
        re.I,
    )
    # Strip URLs and file paths before tokenizing so they don't bleed into n-grams
    _NOISE_RE = re.compile(r"https?://\S+|~/[\w/.+-]+|/(?:Users|home)/\w+/[\w/.+-]+")

    for sig in signals:
        # Skip pasted prompts and injected system messages
        if sig.is_pasted_prompt:
            continue
        raw = (sig.turn.user_message or "").lstrip()
        if _SYSTEM_PROMPT_RE.match(raw):
            continue
        # Also skip very long messages (>600 chars) — likely prompt templates, not direct requests
        if len(raw) > 600:
            continue

        # Strip URL/path noise before n-gram extraction
        clean_for_ngrams = _NOISE_RE.sub(" ", sig.cleaned_message)
        tokens = _tokenize(clean_for_ngrams)
        sid = sig.turn.session_id
        ts = sig.turn.timestamp
        excerpt = _short(sig.cleaned_message)

        for bg in _ngrams(tokens, 2):
            if bg in _GENERIC_BIGRAMS:
                continue
            bigram_sessions[bg].add(sid)
            bigram_counts[bg] += 1
            bigram_timestamps[bg].append(ts)
            if len(bigram_examples[bg]) < 4:
                bigram_examples[bg].append(excerpt)

        for tg in _ngrams(tokens, 3):
            trigram_sessions[tg].add(sid)
            trigram_counts[tg] += 1
            trigram_timestamps[tg].append(ts)
            if len(trigram_examples[tg]) < 4:
                trigram_examples[tg].append(excerpt)

    MIN_SESSIONS = 3  # must appear in ≥3 distinct sessions

    # Collect qualifying bigrams
    candidates: list[DiscoveredPreference] = []
    covered_bigrams: set[tuple] = set()

    # Prefer trigrams that subsume bigrams (longer phrase = more specific preference)
    for tg, sids in trigram_sessions.items():
        if len(sids) < MIN_SESSIONS:
            continue
        phrase = " ".join(tg)
        tss = trigram_timestamps[tg]
        candidates.append(DiscoveredPreference(
            phrase=phrase,
            session_count=len(sids),
            total_count=trigram_counts[tg],
            first_seen=min(tss),
            last_seen=max(tss),
            examples=list(dict.fromkeys(trigram_examples[tg]))[:4],
        ))
        # Mark constituent bigrams as covered
        covered_bigrams.add(tg[:2])
        covered_bigrams.add(tg[1:])

    for bg, sids in bigram_sessions.items():
        if len(sids) < MIN_SESSIONS or bg in covered_bigrams:
            continue
        phrase = " ".join(bg)
        tss = bigram_timestamps[bg]
        candidates.append(DiscoveredPreference(
            phrase=phrase,
            session_count=len(sids),
            total_count=bigram_counts[bg],
            first_seen=min(tss),
            last_seen=max(tss),
            examples=list(dict.fromkeys(bigram_examples[bg]))[:4],
        ))

    return sorted(candidates, key=lambda p: -p.score)[:80]  # top 80


# ---------------------------------------------------------------------------
# Topic discovery — TF-IDF per session → greedy cluster merge
# ---------------------------------------------------------------------------

def _compute_tfidf(
    session_docs: dict[str, list[str]],
) -> dict[str, dict[str, float]]:
    """Return {session_id: {term: tfidf_score}}."""
    N = len(session_docs)
    if N == 0:
        return {}

    # Per-session term frequency
    tf: dict[str, Counter] = {sid: Counter(tokens) for sid, tokens in session_docs.items()}

    # Document frequency
    df: Counter = Counter()
    for counter in tf.values():
        for term in counter:
            df[term] += 1

    # IDF — terms that appear in every session have near-zero IDF
    idf: dict[str, float] = {
        term: math.log((N + 1) / (freq + 1)) + 1.0
        for term, freq in df.items()
        if freq > 1  # ignore hapax legomena
    }

    result: dict[str, dict[str, float]] = {}
    for sid, counter in tf.items():
        total = max(sum(counter.values()), 1)
        result[sid] = {
            term: (count / total) * idf[term]
            for term, count in counter.items()
            if term in idf
        }
    return result


def discover_topics(
    sessions: list["Session"],
    signals: list["TurnSignal"],
) -> list[DiscoveredTopic]:
    """
    Discover topics by TF-IDF per session followed by greedy term-cluster merging.
    No predefined topic names — labels emerge from top distinctive terms.
    """
    # Build per-session document
    session_docs: dict[str, list[str]] = defaultdict(list)
    session_timestamps: dict[str, list[str]] = defaultdict(list)

    for sig in signals:
        session_docs[sig.turn.session_id].extend(_tokenize(sig.cleaned_message))
        session_timestamps[sig.turn.session_id].append(sig.turn.timestamp)

    tfidf = _compute_tfidf(session_docs)
    if not tfidf:
        return []

    # Top-5 TF-IDF terms per session
    session_top: dict[str, list[str]] = {
        sid: sorted(scores, key=lambda t: -scores[t])[:5]
        for sid, scores in tfidf.items()
        if scores
    }

    # Greedy cluster merge: sessions that share ≥2 top terms → same topic
    clusters: list[set[str]] = []   # list of session-id sets
    cluster_terms: list[Counter] = []

    for sid, top_terms in session_top.items():
        best_cluster = -1
        best_overlap = 1  # must share at least 2 terms

        for i, cterms in enumerate(cluster_terms):
            overlap = sum(1 for t in top_terms if t in cterms)
            if overlap > best_overlap:
                best_overlap = overlap
                best_cluster = i

        if best_cluster >= 0:
            clusters[best_cluster].add(sid)
            cluster_terms[best_cluster].update(top_terms)
        else:
            clusters.append({sid})
            cluster_terms.append(Counter(top_terms))

    # Build DiscoveredTopic for clusters with ≥3 sessions
    result: list[DiscoveredTopic] = []
    for i, sids in enumerate(clusters):
        if len(sids) < 3:
            continue
        top5 = [t for t, _ in cluster_terms[i].most_common(5)]
        if not top5:
            continue

        # Timestamps
        all_ts = [ts for sid in sids for ts in session_timestamps.get(sid, [])]
        turn_count = sum(len(session_docs.get(sid, [])) for sid in sids)

        topic_id = re.sub(r"[^a-z0-9]", "_", top5[0])
        result.append(DiscoveredTopic(
            id=f"topic_{topic_id}",
            label=top5[0].replace("_", " ").title(),
            top_terms=top5,
            session_count=len(sids),
            turn_count=turn_count,
            first_seen=min(all_ts) if all_ts else "",
            last_seen=max(all_ts) if all_ts else "",
        ))

    return sorted(result, key=lambda t: -t.session_count)


# ---------------------------------------------------------------------------
# Correction discovery — full statistical scan, independent of classifier
# ---------------------------------------------------------------------------

# Signals that a turn is a user correction (not an agent response)
_CORRECTION_SIGNALS = [
    # Explicit negation/redirection
    re.compile(r"^\s*(?:no[,!. ]|nope[,!. ]|not\s+(?:quite|exactly|right|correct)|wrong\b)", re.I),
    re.compile(r"\b(?:that'?s|this is|that is|you'?re|you are)\s+(?:wrong|incorrect|not right|not what i|not how)", re.I),
    re.compile(r"\b(?:i said|i asked|i told you|i mentioned|i specified|i already told)\b", re.I),
    re.compile(r"\b(?:you (?:should|shouldn'?t|must|mustn'?t|can'?t|don'?t)|don'?t (?:use|do|call|say|write|add|run|apply|send|forget|hardcode))\b", re.I),
    re.compile(r"\b(?:never use|always (?:use|do|apply|remember|include)|stop (?:using|doing|calling|adding))\b", re.I),
    # Frustration + correction context
    re.compile(r"\b(?:again[,!? ]|still\s+(?:wrong|not|broken|missing|the same)|keep(?:s)?\s+(?:wrong|forget|missing|using|doing))\b", re.I),
    re.compile(r"\b(?:why (?:did you|are you|is it|isn'?t it)|you missed|you forgot|you used|you called)\b", re.I),
    # Format / tool corrections
    re.compile(r"\b(?:not (?:xlsx|excel|json|csv)|should be (?:xlsx|excel)|(?:wrong|incorrect) (?:tool|format|file|path|output|query|model))\b", re.I),
]

# Patterns to extract the SUBJECT of what was corrected
_CORRECTION_SUBJ = re.compile(
    r"""
    (?:that'?s?\s+not|you\s+(?:said|wrote|did|missed|forgot|used|called)|
       not\s+(?:what|how|the\s+way)|you'?re\s+(?:wrong|incorrect)|
       still\s+(?:wrong|broken|not\s+working)|
       why\s+(?:is|are|did|didn'?t)|you\s+(?:should|must)\s+not|
       never\s+use|don'?t\s+use|wrong\s+(?:tool|file|path|format|output|query)|
       you\s+(?:missed|forgot|used|called)|i\s+(?:said|asked|told\s+you|mentioned|specified))
    \s+(.{4,70}?)(?:[.!?\n]|$)
    """,
    re.I | re.X,
)

_NEGATION_SUBJECT = re.compile(
    r"\b(?:not|never|don'?t|shouldn'?t|mustn'?t)\s+((?:use|call|say|write|do|run|apply|add|create|send|hardcode|forget|include)\s+\w[\w\s]{2,40}?)(?:[,.\n]|$)",
    re.I,
)

# Minimal subject vocab filter — subjects that are too generic to be useful
_SUBJ_STOPWORDS = frozenset("this that the it a an is was were has have do did be been".split())

_SYSTEM_TURN_RE = re.compile(
    r"^(?:<skill-context|you are a|resolve the user|today:\s*\d{4})", re.I
)


def discover_corrections(signals: list["TurnSignal"]) -> list[DiscoveredCorrection]:
    """
    Discover corrections by statistically scanning ALL turns — no pre-classification gate.
    Uses multiple signal patterns to detect correction context, then extracts subject.
    """
    subj_sessions: dict[str, set[str]] = defaultdict(set)
    subj_counts:   Counter[str] = Counter()
    subj_first:    dict[str, str] = {}
    subj_last:     dict[str, str] = {}
    subj_examples: dict[str, list[str]] = defaultdict(list)

    for sig in signals:
        raw = (sig.turn.user_message or "").lstrip()
        # Skip injected system messages and pasted prompts
        if sig.is_pasted_prompt or _SYSTEM_TURN_RE.match(raw) or len(raw) > 800:
            continue

        msg = sig.cleaned_message

        # Check if this turn has any correction signal
        has_signal = any(pat.search(raw) for pat in _CORRECTION_SIGNALS)
        if not has_signal:
            continue

        sid = sig.turn.session_id
        ts  = sig.turn.timestamp
        subjects: list[str] = []

        # Try to extract specific subjects
        for m in _CORRECTION_SUBJ.finditer(msg):
            raw_subj = m.group(1).strip().lower()
            toks = [w for w in raw_subj.split() if w not in _STOPWORDS and w not in _SUBJ_STOPWORDS]
            if 2 <= len(toks) <= 8:
                subjects.append(" ".join(toks[:6]))

        for m in _NEGATION_SUBJECT.finditer(msg):
            raw_subj = m.group(1).strip().lower()
            toks = [w for w in raw_subj.split() if w not in _STOPWORDS and w not in _SUBJ_STOPWORDS]
            if 2 <= len(toks) <= 8:
                subjects.append(" ".join(toks[:6]))

        # Fallback: for short turns use key n-grams as subject
        if not subjects and len(raw) <= 150:
            toks = [w for w in _tokenize(msg) if w not in _SUBJ_STOPWORDS]
            if len(toks) >= 2:
                subjects.append(" ".join(toks[:5]))

        for subj in subjects:
            subj_sessions[subj].add(sid)
            subj_counts[subj] += 1
            if subj not in subj_first or ts < subj_first[subj]:
                subj_first[subj] = ts
            if subj not in subj_last or ts > subj_last[subj]:
                subj_last[subj] = ts
            if len(subj_examples[subj]) < 4:
                subj_examples[subj].append(_short(raw, 100))

    result = [
        DiscoveredCorrection(
            subject=subj,
            count=subj_counts[subj],
            session_count=len(subj_sessions[subj]),
            first_seen=subj_first.get(subj, ""),
            last_seen=subj_last.get(subj, ""),
            examples=subj_examples[subj],
        )
        for subj in subj_counts
        if len(subj_sessions[subj]) >= 1
    ]
    # Score: session spread is primary signal, count is secondary
    return sorted(result, key=lambda c: -(c.session_count**1.5 * math.log1p(c.count)))[:50]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def discover_all(
    sessions: list["Session"],
    signals: list["TurnSignal"],
) -> DiscoveredKnowledge:
    """Run all discovery passes and return a DiscoveredKnowledge bundle."""
    all_turns = [t for s in sessions for t in s.turns]
    return DiscoveredKnowledge(
        tools=discover_tools(signals),
        paths=discover_paths(signals),
        preferences=discover_preferences(signals),
        topics=discover_topics(sessions, signals),
        corrections=discover_corrections(signals),
        skills=discover_skills(signals),
        total_sessions=len(sessions),
        total_turns=len(all_turns),
    )
