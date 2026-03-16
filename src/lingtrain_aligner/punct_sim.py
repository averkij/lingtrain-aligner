"""
Punctuation-based similarity for alignment quality improvement.

Punctuation marks are largely preserved across translations. A question in one
language almost always corresponds to a question in the other. This module
provides a language-independent punctuation similarity signal that complements
embedding-based cosine similarity, especially for rare languages where
embedding models are weak.
"""

import re
import numpy as np

# ============================================================
# Unicode punctuation normalization
# ============================================================

# Map every known variant of a punctuation "role" to a canonical symbol.
# This handles CJK fullwidth forms, typographic variants, etc.

_QUOTE_CHARS = set(
    '"'  # ASCII double quote
    "'"  # ASCII single quote
    "\u00AB\u00BB"  # « »  guillemets
    "\u2018\u2019"  # ' '  curly single quotes
    "\u201A"        # ‚    single low-9
    "\u201C\u201D"  # " "  curly double quotes
    "\u201E"        # „    double low-9
    "\u2039\u203A"  # ‹ ›  single guillemets
    "\u300C\u300D"  # 「 」 CJK corner brackets
    "\u300E\u300F"  # 『 』 CJK white corner brackets
    "\uFF02"        # ＂   fullwidth quotation mark
    "\uFF07"        # ＇   fullwidth apostrophe
    "\uFE41\uFE42"  # ﹁ ﹂ presentation form corner brackets
    "\uFE43\uFE44"  # ﹃ ﹄ presentation form white corner brackets
)

_QUESTION_CHARS = set(
    "?"
    "\uFF1F"        # ？   fullwidth question mark
    "\u037E"        # ;    Greek question mark (looks like semicolon)
    "\u061F"        # ؟    Arabic question mark
    "\u2E2E"        # ⸮    reversed question mark
)

_EXCLAMATION_CHARS = set(
    "!"
    "\uFF01"        # ！   fullwidth exclamation
    "\u00A1"        # ¡    inverted exclamation
)

_PERIOD_CHARS = set(
    "."
    "\u3002"        # 。   CJK period
    "\uFF0E"        # ．   fullwidth period
    "\u06D4"        # ۔    Arabic period
    "\u0964"        # ।    Devanagari danda
    "\u0965"        # ॥    Devanagari double danda
)

_COMMA_CHARS = set(
    ","
    "\u3001"        # 、   CJK comma
    "\uFF0C"        # ，   fullwidth comma
    "\u060C"        # ،    Arabic comma
)

_COLON_CHARS = set(
    ":"
    "\uFF1A"        # ：   fullwidth colon
)

_SEMICOLON_CHARS = set(
    ";"
    "\uFF1B"        # ；   fullwidth semicolon
)

_DASH_CHARS = set(
    "-"
    "\u2013"        # –    en dash
    "\u2014"        # —    em dash
    "\u2015"        # ―    horizontal bar
    "\u2012"        # ‒    figure dash
    "\uFF0D"        # －   fullwidth hyphen-minus
    "\u30FC"        # ー   katakana-hiragana prolonged sound mark
)

_ELLIPSIS_CHARS = set(
    "\u2026"        # …    ellipsis
)

_OPEN_PAREN_CHARS = set(
    "("
    "\uFF08"        # （   fullwidth
    "\uFE59"        # ﹙   small form
    "["
    "\uFF3B"        # ［   fullwidth
)

_CLOSE_PAREN_CHARS = set(
    ")"
    "\uFF09"        # ）   fullwidth
    "\uFE5A"        # ﹚   small form
    "]"
    "\uFF3D"        # ］   fullwidth
)


# Build a character -> canonical category lookup.
_CHAR_TO_CATEGORY = {}

def _register(chars, category):
    for ch in chars:
        _CHAR_TO_CATEGORY[ch] = category

_register(_QUOTE_CHARS, "Q")        # quote
_register(_QUESTION_CHARS, "?")     # question
_register(_EXCLAMATION_CHARS, "!")   # exclamation
_register(_PERIOD_CHARS, ".")       # period / full stop
_register(_COMMA_CHARS, ",")        # comma
_register(_COLON_CHARS, ":")        # colon
_register(_SEMICOLON_CHARS, ";")    # semicolon
_register(_DASH_CHARS, "-")         # dash
_register(_ELLIPSIS_CHARS, "…")     # ellipsis
_register(_OPEN_PAREN_CHARS, "(")   # open paren/bracket
_register(_CLOSE_PAREN_CHARS, ")")  # close paren/bracket


# ============================================================
# Feature extraction
# ============================================================

# ============================================================
# Number extraction
# ============================================================

# Fullwidth digit mapping: ０-９ -> 0-9
_FULLWIDTH_DIGIT_MAP = str.maketrans("０１２３４５６７８９", "0123456789")

# CJK numeral characters -> digit value (for basic single-character cases)
_CJK_NUMERAL_MAP = {
    "〇": "0", "零": "0",
    "一": "1", "壱": "1", "壹": "1",
    "二": "2", "弐": "2", "貳": "2",
    "三": "3", "参": "3", "參": "3",
    "四": "4",
    "五": "5",
    "六": "6",
    "七": "7",
    "八": "8",
    "九": "9",
    "十": "10", "拾": "10",
    "百": "100", "佰": "100",
    "千": "1000", "仟": "1000",
    "万": "10000", "萬": "10000",
    "亿": "100000000", "億": "100000000",
}

_NUMBER_PATTERN = re.compile(r"\d+")


def _extract_numbers(text):
    """
    Extract a set of number strings from text.

    Normalizes fullwidth digits to ASCII. Returns a set of digit sequences
    found in the text (e.g., {"612", "1909", "3"}).
    """
    # Normalize fullwidth digits
    normalized = text.translate(_FULLWIDTH_DIGIT_MAP)
    return set(_NUMBER_PATTERN.findall(normalized))


def _extract_cjk_numerals(text):
    """
    Extract standalone CJK numeral characters as number strings.
    Only extracts isolated CJK numerals that represent meaningful standalone
    numbers (e.g., 三 in chapter context), not compound words.
    """
    result = set()
    for ch in text:
        if ch in _CJK_NUMERAL_MAP:
            result.add(_CJK_NUMERAL_MAP[ch])
    return result


def _number_overlap_score(nums_from, nums_to):
    """
    Compute number overlap between two sentence number sets.

    Distinctive numbers (>= 2 digits) get much higher weight because they're
    rare and almost certainly indicate a content match. Single digits are
    common and less informative.

    Returns a score in [0, 1].
    """
    if not nums_from and not nums_to:
        return 0.5  # neutral: neither has numbers

    if not nums_from or not nums_to:
        return 0.3  # one has numbers, other doesn't — mild negative

    # Weight numbers by distinctiveness (digit count)
    def weight(n):
        length = len(n)
        if length >= 4:
            return 4.0   # year-like: 1984, 3251
        elif length >= 3:
            return 3.0   # e.g., 612, 325
        elif length >= 2:
            return 2.0   # e.g., 40, 15
        else:
            return 0.5   # single digit: common, low signal

    # Weighted Jaccard-like overlap
    common = nums_from & nums_to
    all_nums = nums_from | nums_to

    if not all_nums:
        return 0.5

    weighted_common = sum(weight(n) for n in common)
    weighted_all = sum(weight(n) for n in all_nums)

    return weighted_common / weighted_all


# ============================================================
# Feature extraction
# ============================================================

def _get_categories(text):
    """Extract ordered list of canonical punctuation categories from text."""
    return [_CHAR_TO_CATEGORY[ch] for ch in text if ch in _CHAR_TO_CATEGORY]


def _get_ending_category(text):
    """Get the sentence-ending punctuation category."""
    text = text.rstrip()
    # Walk backwards past closing quotes/parens to find the "real" ending
    for ch in reversed(text[-8:] if len(text) >= 8 else text):
        cat = _CHAR_TO_CATEGORY.get(ch)
        if cat in ("?", "!", "…", ":"):
            return cat
        if cat == ".":
            return "."
        if cat in ("Q", ")", None):
            # Skip closing quotes/parens, continue looking
            if cat is None and (ch.isalpha() or ch.isdigit()):
                return "."  # default: statement
            continue
    return "."


def _has_three_dots(text):
    """Check for '...' pattern (separate from single ellipsis char)."""
    return "..." in text


def extract_features(text):
    """
    Extract punctuation features from a sentence.

    Returns a dict with boolean/numeric features used for similarity comparison.
    """
    cats = _get_categories(text)
    cat_set = set(cats)
    ending = _get_ending_category(text)

    # Extract numbers (Arabic digits + CJK numerals)
    numbers = _extract_numbers(text) | _extract_cjk_numerals(text)

    return {
        "ending": ending,
        "has_question": "?" in cat_set,
        "has_exclamation": "!" in cat_set,
        "has_quote": "Q" in cat_set,
        "has_colon": ":" in cat_set,
        "has_semicolon": ";" in cat_set,
        "has_dash": "-" in cat_set,
        "has_ellipsis": "…" in cat_set or _has_three_dots(text),
        "has_paren": "(" in cat_set or ")" in cat_set,
        "cat_set": cat_set,
        "cat_count": len(cats),
        "text_len": len(text),
        "numbers": numbers,
    }


# ============================================================
# Pairwise similarity
# ============================================================

def punct_similarity(feat_from, feat_to):
    """
    Compute punctuation-based similarity between two sentences.

    Parameters
    ----------
    feat_from : dict
        Features from extract_features() for the source sentence.
    feat_to : dict
        Features from extract_features() for the target sentence.

    Returns
    -------
    float
        Similarity score in [0, 1]. Higher = more punctuation agreement.
    """
    score = 0.0
    total_weight = 0.0

    # 1. Sentence-ending match (strongest signal)
    w = 3.0
    total_weight += w
    ef, et = feat_from["ending"], feat_to["ending"]
    if ef == et:
        score += w
    elif ef == "." and et == ".":
        score += w * 0.5  # both plain statements

    # 2. Question mark agreement
    w = 2.5
    total_weight += w
    if feat_from["has_question"] == feat_to["has_question"]:
        score += w

    # 3. Exclamation mark agreement
    w = 2.0
    total_weight += w
    if feat_from["has_exclamation"] == feat_to["has_exclamation"]:
        score += w

    # 4. Quote presence agreement
    w = 1.5
    total_weight += w
    if feat_from["has_quote"] == feat_to["has_quote"]:
        score += w

    # 5. Colon agreement
    w = 1.5
    total_weight += w
    if feat_from["has_colon"] == feat_to["has_colon"]:
        score += w

    # 6. Ellipsis agreement
    w = 1.5
    total_weight += w
    if feat_from["has_ellipsis"] == feat_to["has_ellipsis"]:
        score += w

    # 7. Dash agreement
    w = 1.0
    total_weight += w
    if feat_from["has_dash"] == feat_to["has_dash"]:
        score += w

    # 8. Semicolon agreement
    w = 0.5
    total_weight += w
    if feat_from["has_semicolon"] == feat_to["has_semicolon"]:
        score += w

    # 9. Punctuation set Jaccard overlap
    w = 1.0
    total_weight += w
    s1, s2 = feat_from["cat_set"], feat_to["cat_set"]
    if s1 or s2:
        jaccard = len(s1 & s2) / len(s1 | s2)
        score += w * jaccard
    else:
        score += w * 0.5  # both empty => neutral

    # 10. Punctuation density similarity
    w = 0.5
    total_weight += w
    len1 = max(feat_from["text_len"], 1)
    len2 = max(feat_to["text_len"], 1)
    density1 = feat_from["cat_count"] / len1
    density2 = feat_to["cat_count"] / len2
    density_diff = abs(density1 - density2)
    score += w * max(0.0, 1.0 - density_diff * 20)

    base_score = score / total_weight

    # 11-12. Number matching bonus (conditional — only fires when numbers exist)
    #
    # Numbers are rare but very distinctive. Instead of diluting the punct
    # score for the majority of sentences that have no numbers, we apply an
    # additive bonus/penalty only when at least one side has numbers.
    nums_from = feat_from["numbers"]
    nums_to = feat_to["numbers"]

    if nums_from or nums_to:
        overlap = _number_overlap_score(nums_from, nums_to)
        # overlap: 0.5 = neutral (both empty, shouldn't reach here),
        #          0.3 = one has numbers other doesn't,
        #          0.0-1.0 based on weighted Jaccard when both have numbers
        #
        # Convert to a bonus centered at 0: positive for good match,
        # negative for mismatch. Scale by 0.15 to keep proportional.
        number_bonus = (overlap - 0.5) * 0.30
        base_score = min(1.0, max(0.0, base_score + number_bonus))

    return base_score


def punct_similarity_texts(text_from, text_to):
    """
    Convenience: compute punctuation similarity directly from text strings.
    """
    return punct_similarity(extract_features(text_from), extract_features(text_to))


# ============================================================
# Batch matrix computation
# ============================================================

def compute_punct_matrix(lines_from, lines_to, embedding_sim_matrix):
    """
    Build a punctuation similarity matrix for the same (from, to) pairs
    that have non-zero entries in the embedding similarity matrix.

    Parameters
    ----------
    lines_from : list[str]
        Source sentences (same order as embedding_sim_matrix rows).
    lines_to : list[str]
        Target sentences (same order as embedding_sim_matrix columns).
    embedding_sim_matrix : np.ndarray
        The (N x M) embedding similarity matrix. Used only for the window mask
        (non-zero positions indicate which pairs to compute).

    Returns
    -------
    np.ndarray
        Punctuation similarity matrix of the same shape, with scores in [0, 1].
    """
    n_from = len(lines_from)
    n_to = len(lines_to)
    punct_matrix = np.zeros_like(embedding_sim_matrix)

    # Pre-compute features for all sentences
    feats_from = [extract_features(t) for t in lines_from]
    feats_to = [extract_features(t) for t in lines_to]

    for i in range(n_from):
        for j in range(n_to):
            if embedding_sim_matrix[i, j] > 0:
                punct_matrix[i, j] = punct_similarity(feats_from[i], feats_to[j])

    return punct_matrix


def boost_sim_matrix(sim_matrix, lines_from, lines_to, weight=0.15):
    """
    Combine embedding similarity matrix with punctuation similarity.

    The blended matrix is:
        combined = (1 - weight) * sim_matrix + weight * punct_matrix

    Both matrices are aligned: punct_matrix only has non-zero entries where
    sim_matrix has non-zero entries (same window).

    Parameters
    ----------
    sim_matrix : np.ndarray
        Original embedding-based similarity matrix.
    lines_from : list[str]
        Source sentences.
    lines_to : list[str]
        Target sentences.
    weight : float
        Blending weight for punctuation (default 0.15). Higher values give
        more influence to punctuation. Recommended range: 0.10 - 0.25.

    Returns
    -------
    np.ndarray
        Blended similarity matrix of the same shape.
    """
    if weight <= 0:
        return sim_matrix

    punct_matrix = compute_punct_matrix(lines_from, lines_to, sim_matrix)

    # Scale punct_matrix to roughly match embedding score range.
    # Embedding scores are typically 0.2-0.9; punct scores are 0.4-1.0.
    # We scale punct into the same range as the per-row embedding scores.
    combined = sim_matrix.copy()
    for i in range(sim_matrix.shape[0]):
        row_mask = sim_matrix[i] > 0
        if not np.any(row_mask):
            continue
        # Get the scale of embedding scores in this row
        emb_vals = sim_matrix[i, row_mask]
        emb_mean = np.mean(emb_vals)
        if emb_mean < 0.01:
            continue
        # Scale punct to match embedding range, then blend
        punct_row = punct_matrix[i, row_mask] * emb_mean
        combined[i, row_mask] = (1 - weight) * emb_vals + weight * punct_row

    return combined


def punct_bonus_for_texts(text_from, text_to):
    """
    Compute a small punctuation bonus to add to an embedding similarity score.

    Used in conflict resolution and variant scoring, where we have individual
    text pairs rather than a full matrix.

    Returns a bonus in [0, ~0.1] that should be ADDED to the embedding sim.
    The bonus is highest when punctuation strongly agrees (question<->question,
    exclamation<->exclamation, etc.) and near zero for neutral cases.
    """
    feat_from = extract_features(text_from)
    feat_to = extract_features(text_to)
    sim = punct_similarity(feat_from, feat_to)

    # Convert similarity (0-1) to a bonus.
    # Base similarity (two plain statements with no special punct) is ~0.75.
    # We give a bonus only when sim exceeds the neutral baseline, and a
    # penalty when it's below (e.g., question matched with statement).
    baseline = 0.75
    bonus = (sim - baseline) * 0.15  # range: roughly -0.11 to +0.04
    return bonus
