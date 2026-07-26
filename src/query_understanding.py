"""
src/query_understanding.py
--------------------------
Query Understanding Tool for the AML Financial Crime Committee Agent.

Responsibilities:
  - Parse a natural-language user query into a structured intent/filter/entity JSON
  - Primary path  : Groq (llama-3.3-70b-versatile) with JSON-mode system prompt
  - Fallback path : regex/keyword-based rule extractor

Canonical output schema (Section 8.1):
{
  "intent"        : str,   # one of INTENTS
  "filters"       : {"date_range": str|None, "country": str|None,
                      "segment": str|None, "transaction_type": str|None},
  "entities"      : list,  # e.g. ["4521"] — raw ID, no prefix
  "target_pattern": str,   # "structuring" | "smurfing" | "layering" | "none"
  "requires_eda"  : bool,
  "requires_ml"   : bool,
}
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

# ---------------------------------------------------------------------------
# Module-level setup
# ---------------------------------------------------------------------------

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

INTENTS: list[str] = [
    "pattern_detection",
    "aggregation_query",
    "single_entity_lookup",
    "explain_flag",
    "network_query",
    "comparative_query",
    "escalation_query",
    "broad_exploration",
]

PATTERNS: list[str] = ["structuring", "smurfing", "layering", "none"]

_QUERY_PARSER_SYSTEM_PROMPT = """\
You are a financial crime query parser. Extract structured information from the user's query.
Return ONLY valid JSON with these exact fields:
{
  "intent": "<one of: pattern_detection, aggregation_query, single_entity_lookup, explain_flag, network_query, comparative_query, escalation_query, broad_exploration>",
  "filters": {
    "date_range": "<e.g. last_30_days, last_7_days, Q1_2024, or null>",
    "country": "<country name or null>",
    "segment": "<segment name or null>",
    "transaction_type": "<type or null>"
  },
  "entities": ["<the account or transaction ID exactly as it identifies a real record, with no prefix added>"],
  "target_pattern": "<structuring | smurfing | layering | none>",
  "requires_eda": <true | false>,
  "requires_ml": <true | false>
}

IMPORTANT for "entities": real account IDs in this dataset are raw numeric strings (e.g. "1615764138"), not prefixed. If the user says "customer 4521" or "customer ID 1615764138", extract just the number/ID itself (e.g. "4521" or "1615764138") — do NOT prepend "customer_" or "transaction_" to it, since that prefix does not match any real account identifier and would cause the lookup to fail.

Rules:
- pattern_detection: user wants to find a specific AML pattern (structuring/smurfing/layering)
- aggregation_query: user wants a count/sum/filter query, no ML needed
- single_entity_lookup: user asks about a specific customer or transaction ID
- explain_flag: user wants to know WHY something was flagged
- network_query: user asks about relationships/connections between entities
- comparative_query: user wants to compare groups (flagged vs non-flagged, etc.)
- escalation_query: user asks what compliance action to take
- broad_exploration: user wants a full analysis of the dataset

requires_eda = true only for broad_exploration
requires_ml = true for pattern_detection and broad_exploration
"""

# ---------------------------------------------------------------------------
# Helper: safe JSON extraction from a raw LLM text response
# ---------------------------------------------------------------------------


def _extract_json_from_text(text: str) -> Optional[dict]:
    """Attempt to extract a JSON object from a raw text string.

    Handles both pure-JSON responses and responses that wrap JSON inside
    a markdown code fence (```json ... ```).

    Args:
        text: Raw string that may contain a JSON object.

    Returns:
        Parsed dict if successful, else None.
    """
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from markdown code fences
    fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1))
        except json.JSONDecodeError:
            pass

    # Greedy brace-matching to find the outermost JSON object
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(text[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start: i + 1])
                except json.JSONDecodeError:
                    break
    return None


def _none_if_null(value) -> Optional[str]:
    """Convert JSON null / None / 'null' / 'None' strings to Python None.

    Args:
        value: Any value to check.

    Returns:
        None if value is null-like, otherwise the original value.
    """
    if value is None:
        return None
    if isinstance(value, str) and value.lower() in ("null", "none", ""):
        return None
    return value


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class QueryUnderstandingTool:
    """Parse a natural-language AML query into a structured intent/filter JSON.

    Primary path  : Groq (``llama-3.3-70b-versatile``) via the
                    groq SDK with a JSON-mode system prompt.
    Fallback path : Deterministic regex/keyword rule extractor that handles
                    all mandatory test cases and common query forms.

    Usage::

        tool = QueryUnderstandingTool()
        result = tool.parse("Find structuring patterns in the last 30 days")
    """

    _DEFAULT_MODEL = "llama-3.3-70b-versatile"

    def __init__(self) -> None:
        """Initialise the Groq client using GROQ_API_KEY from .env."""
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning(
                "GROQ_API_KEY not found in environment. "
                "Groq path will be unavailable; rule-based fallback will be used."
            )
            self._client = None
            self._model_name = None
        else:
            try:
                self._client = Groq(api_key=api_key)
                self._model_name = os.getenv("GROQ_MODEL", self._DEFAULT_MODEL)
                logger.info("Groq model '%s' initialised successfully.", self._model_name)
            except Exception as exc:
                logger.error("Failed to initialise Groq model: %s", exc)
                self._client = None
                self._model_name = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, query: str) -> dict:
        """Parse the query and return the full structured intent dict.

        Tries the Groq API first; falls back to the rule-based extractor
        if Groq is unavailable or returns an invalid response.

        Args:
            query: Raw natural-language query string from the user.

        Returns:
            A dict with keys: intent, filters, entities, target_pattern,
            requires_eda, requires_ml.
        """
        if not query or not query.strip():
            logger.warning("Empty query received; returning default 'broad_exploration' parse.")
            return self._default_result()

        # Primary: Groq
        if self._client is not None:
            try:
                result = self._parse_with_llm(query)
                if result and self._validate(result):
                    logger.info("Query parsed via LLM. intent=%s", result.get("intent"))
                    return result
                logger.warning("LLM returned invalid/incomplete JSON; switching to rules.")
            except Exception as exc:
                logger.warning("LLM parse failed (%s); switching to rule-based fallback.", exc)

        # Fallback: rules
        result = self._parse_with_rules(query)
        logger.info("Query parsed via rules. intent=%s", result.get("intent"))
        return result

    # ------------------------------------------------------------------
    # Private: LLM path
    # ------------------------------------------------------------------

    def _parse_with_llm(self, query: str) -> dict:
        """Send the query to Groq and parse its JSON response.

        Args:
            query: Raw user query string.

        Returns:
            Parsed and normalised dict from LLM's response, or empty dict
            on parse failure.

        Raises:
            Exception: Propagated from the groq SDK on API errors.
        """
        user_message = f"Parse the following AML query:\n\n{query}"
        response = self._client.chat.completions.create(
            messages=[
                {"role": "system", "content": _QUERY_PARSER_SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            model=self._model_name,
            response_format={"type": "json_object"},
            temperature=0.0,
        )

        raw_text: str = response.choices[0].message.content if response.choices[0].message.content else ""
        parsed = _extract_json_from_text(raw_text)
        if parsed is None:
            logger.debug("LLM raw response could not be JSON-decoded: %r", raw_text[:400])
            return {}

        return self._normalise(parsed)

    # ------------------------------------------------------------------
    # Private: rule-based fallback
    # ------------------------------------------------------------------

    def _parse_with_rules(self, query: str) -> dict:
        """Regex/keyword-based extractor — deterministic fallback.

        Correctly handles all three mandatory test cases:
          1. 'Find structuring patterns in the last 30 days'
             -> intent=pattern_detection, target_pattern=structuring,
                filters.date_range=last_30_days
          2. 'Which customers made 10+ transactions under $10,000?'
             -> intent=aggregation_query, target_pattern=none
          3. 'Is customer ID 4521 suspicious?'
             -> intent=single_entity_lookup, entities=['4521']

        Real SAML-D account IDs are raw numeric strings with no prefix
        (e.g. '1615764138') — entities are extracted as the bare ID, never
        prefixed with 'customer_'/'transaction_', so they match directly
        against the sender_account/receiver_account columns.

        Args:
            query: Raw user query string.

        Returns:
            Fully populated parsed-query dict.
        """
        q = query.lower().strip()

        # ---- 1. Detect entities (customer / transaction IDs) -----------
        entities: list[str] = []
        for m in re.finditer(r"customer[\s_\-]*(?:id[\s:]*)?(\d+)", q):
            entities.append(m.group(1))
        for m in re.finditer(r"transaction[\s_\-]*(?:id[\s:]*)?(\d+)", q):
            entities.append(m.group(1))
        entities = list(dict.fromkeys(entities))  # deduplicate, preserve order

        # ---- 2. Detect target pattern ----------------------------------
        target_pattern = "none"
        if re.search(r"\bstructur", q):
            target_pattern = "structuring"
        elif re.search(r"\bsmurf", q):
            target_pattern = "smurfing"
        elif re.search(r"\blayer", q):
            target_pattern = "layering"

        # ---- 3. Detect date_range filter --------------------------------
        date_range: Optional[str] = None
        date_map: list[tuple] = [
            (r"last\s+30\s+days?",  "last_30_days"),
            (r"last\s+7\s+days?",   "last_7_days"),
            (r"last\s+week",        "last_7_days"),
            (r"last\s+month",       "last_30_days"),
            (r"last\s+90\s+days?",  "last_90_days"),
            (r"last\s+quarter",     "last_90_days"),
            (r"last\s+year",        "last_365_days"),
            (r"last\s+365\s+days?", "last_365_days"),
            (r"\bq([1-4])\s+(\d{4})\b", "__dynamic_qn_year__"),
            (r"\bq([1-4])\b",           "__dynamic_qn__"),
        ]
        for pat, mapped in date_map:
            m = re.search(pat, q, re.IGNORECASE)
            if m:
                if mapped == "__dynamic_qn_year__":
                    date_range = f"Q{m.group(1)}_{m.group(2)}"
                elif mapped == "__dynamic_qn__":
                    date_range = f"Q{m.group(1)}"
                else:
                    date_range = mapped
                break

        # ---- 4. Detect country (use original casing query) -------------
        country: Optional[str] = None
        country_m = re.search(
            r"\b(?:in|for|from|country[:\s]+)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b",
            query,
        )
        if country_m:
            country = country_m.group(1)

        # ---- 5. Detect segment -----------------------------------------
        segment: Optional[str] = None
        seg_m = re.search(r"\b(?:segment|tier|category)[:\s]+([A-Za-z0-9_\-]+)", q)
        if seg_m:
            segment = seg_m.group(1)

        # ---- 6. Detect transaction_type --------------------------------
        transaction_type: Optional[str] = None
        txtype_m = re.search(
            r"\b(wire\s+transfer|cash\s+deposit|cash\s+withdrawal|atm|swift|ach"
            r"|credit|debit|remittance|cryptocurrency|crypto)\b",
            q,
        )
        if txtype_m:
            transaction_type = txtype_m.group(1).replace(" ", "_")

        # ---- 7. Determine intent ----------------------------------------
        intent = self._detect_intent(q, entities, target_pattern)

        # ---- 8. Derive requires_eda / requires_ml ----------------------
        requires_eda = intent == "broad_exploration"
        requires_ml = intent in ("pattern_detection", "broad_exploration")

        return {
            "intent": intent,
            "filters": {
                "date_range": date_range,
                "country": country,
                "segment": segment,
                "transaction_type": transaction_type,
            },
            "entities": entities,
            "target_pattern": target_pattern,
            "requires_eda": requires_eda,
            "requires_ml": requires_ml,
        }

    def _detect_intent(
        self, q: str, entities: list[str], target_pattern: str
    ) -> str:
        """Determine the query intent using keyword and structural heuristics.

        Priority order (first match wins):
          single_entity_lookup -> entities detected AND query is asking about
                                  that specific entity
          explain_flag         -> 'why' / 'explain' / 'flagged' keywords
          escalation_query     -> 'escalate' / 'sar' / 'compliance' keywords
          network_query        -> 'network' / 'connection' / 'related' / 'link'
          pattern_detection    -> pattern keyword (structuring/smurfing/layering)
          comparative_query    -> 'compare' / 'vs' / 'versus' keywords
          broad_exploration    -> 'full analysis' / 'overview' / 'everything'
          aggregation_query    -> count/sum/threshold-style keywords
          single_entity_lookup -> entity present (fallback)
          broad_exploration    -> default

        Args:
            q:              Lower-cased query string.
            entities:       List of detected entity IDs.
            target_pattern: Detected AML pattern or 'none'.

        Returns:
            One of the eight INTENTS strings.
        """
        # network_query (checked first if network keywords present)
        if re.search(
            r"\b(network|connection|connected|related|link|relationship|"
            r"graph|neighbou?r|community|centrality|coordinat(?:e|ing|ion))\b",
            q,
        ):
            return "network_query"

        # explain_flag
        if re.search(r"\b(why|explain|reason|because|what\s+caused|flagged|flag)\b", q):
            return "explain_flag"

        # escalation_query
        if re.search(
            r"\b(escalate|sar\b|suspicious\s+activity\s+report|"
            r"compliance|file\s+a?\s+report|what\s+action|should\s+we)\b",
            q,
        ):
            return "escalation_query"

        # single_entity_lookup – entity present and query is about that entity
        entity_question = bool(entities) and bool(re.search(
            r"\b(suspicious|risk|profile|history|transactions?|activity|"
            r"is\s+customer|check|lookup|detail|alert|investigate)\b",
            q,
        ))
        if entity_question:
            return "single_entity_lookup"

        # pattern_detection – a specific AML pattern was named
        if target_pattern != "none":
            return "pattern_detection"

        # comparative_query
        if re.search(
            r"\b(compare|comparison|versus|vs\.?\b|difference|differ|"
            r"flagged\s+vs|non[-\s]?flagged|between)\b",
            q,
        ):
            return "comparative_query"

        # broad_exploration
        if re.search(
            r"\b(full\s+analysis|overall|overview|everything|all\s+patterns|"
            r"comprehensive|entire\s+dataset|explore|exploration)\b",
            q,
        ):
            return "broad_exploration"

        # aggregation_query
        if re.search(
            r"\b(how\s+many|count|total|sum|average|avg|under\s+\$?"
            r"|above\s+\$?|more\s+than|less\s+than|\d+\+\s*transactions?|"
            r"threshold|aggregate|group\s+by|top\s+\d+|which\s+customers?)\b",
            q,
        ):
            return "aggregation_query"

        # single_entity_lookup – entity present (no specific question detected)
        if entities:
            return "single_entity_lookup"

        # Default fallback
        return "broad_exploration"

    # ------------------------------------------------------------------
    # Private: validation and normalisation helpers
    # ------------------------------------------------------------------

    def _validate(self, parsed: dict) -> bool:
        """Return True if the parsed dict contains all required keys with valid values.

        Args:
            parsed: Dict to validate.

        Returns:
            True if valid, False otherwise.
        """
        required_keys = {
            "intent", "filters", "entities", "target_pattern",
            "requires_eda", "requires_ml",
        }
        if not required_keys.issubset(parsed.keys()):
            missing = required_keys - set(parsed.keys())
            logger.debug("Parsed dict missing keys: %s", missing)
            return False

        if parsed.get("intent") not in INTENTS:
            logger.debug("Invalid intent: %r", parsed.get("intent"))
            return False

        if parsed.get("target_pattern") not in PATTERNS:
            logger.debug("Invalid target_pattern: %r", parsed.get("target_pattern"))
            return False

        if not isinstance(parsed.get("filters"), dict):
            return False

        if not isinstance(parsed.get("entities"), list):
            return False

        return True

    def _normalise(self, parsed: dict) -> dict:
        """Normalise field values returned by LLM to canonical types.

        - intent        : strip and lower, fallback to broad_exploration
        - target_pattern: strip and lower, fallback to none
        - filters       : coerce null-ish strings to None
        - entities      : ensure list of non-empty strings
        - requires_eda / requires_ml: ensure booleans

        Args:
            parsed: Raw dict from LLM.

        Returns:
            Normalised dict.
        """
        intent = str(parsed.get("intent", "broad_exploration")).strip().lower()
        if intent not in INTENTS:
            intent = "broad_exploration"

        tp = str(parsed.get("target_pattern", "none")).strip().lower()
        if tp not in PATTERNS:
            tp = "none"

        raw_filters = parsed.get("filters", {}) or {}
        filters = {
            "date_range":       _none_if_null(raw_filters.get("date_range")),
            "country":          _none_if_null(raw_filters.get("country")),
            "segment":          _none_if_null(raw_filters.get("segment")),
            "transaction_type": _none_if_null(raw_filters.get("transaction_type")),
        }

        raw_entities = parsed.get("entities", []) or []
        entities = [str(e) for e in raw_entities if e]

        req_eda = bool(parsed.get("requires_eda", False))
        req_ml = bool(parsed.get("requires_ml", False))

        return {
            "intent":         intent,
            "filters":        filters,
            "entities":       entities,
            "target_pattern": tp,
            "requires_eda":   req_eda,
            "requires_ml":    req_ml,
        }

    def _default_result(self) -> dict:
        """Return a safe default result for empty or unparseable queries.

        Returns:
            Default parsed dict with broad_exploration intent.
        """
        return {
            "intent": "broad_exploration",
            "filters": {
                "date_range": None,
                "country": None,
                "segment": None,
                "transaction_type": None,
            },
            "entities": [],
            "target_pattern": "none",
            "requires_eda": True,
            "requires_ml": True,
        }
