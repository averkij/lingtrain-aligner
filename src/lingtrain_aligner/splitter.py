"""Texts splitter part of the engine"""

import logging
import re

import razdel
import sentencex

from lingtrain_aligner import preprocessor

logger = logging.getLogger(__name__)

RU_CODE = "ru"
BE_CODE = "bu"
ZH_CODE = "zh"
DE_CODE = "de"
EN_CODE = "en"
FR_CODE = "fr"
IT_CODE = "it"
TR_CODE = "tr"
ES_CODE = "es"
PL_CODE = "pl"
PT_CODE = "pt"
HU_CODE = "hu"
CZ_CODE = "cz"
JP_CODE = "ja"
BA_CODE = "ba"
KO_CODE = "ko"
NL_CODE = "nl"
SW_CODE = "sw"
UK_CODE = "uk"
CV_CODE = "cv"
HY_CODE = "hy"
XX_CODE = "xx"

LANGUAGES = {
    RU_CODE: {"name": "Russian"},
    BE_CODE: {"name": "Belarusian"},
    ZH_CODE: {"name": "Chinese"},
    DE_CODE: {"name": "German"},
    EN_CODE: {"name": "English"},
    FR_CODE: {"name": "French"},
    IT_CODE: {"name": "Italian"},
    TR_CODE: {"name": "Turkish"},
    ES_CODE: {"name": "Spanish"},
    PL_CODE: {"name": "Polish"},
    PT_CODE: {"name": "Portugal"},
    HU_CODE: {"name": "Hungarian"},
    CZ_CODE: {"name": "Czech"},
    JP_CODE: {"name": "Japanese"},
    BA_CODE: {"name": "Bashkir"},
    KO_CODE: {"name": "Korean"},
    SW_CODE: {"name": "Sweden"},
    NL_CODE: {"name": "Dutch"},
    UK_CODE: {"name": "Ukrainian"},
    CV_CODE: {"name": "Chuvash"},
    HY_CODE: {"name": "Armenian"},
    XX_CODE: {"name": "Unknown"},
}


# pattern_ru_orig = re.compile(r'[a-zA-Z\(\)\[\]\/\<\>•\'\n]+')
pattern_ru_orig = re.compile(r"[\/\<\>•\'\n]+")
double_spaces = re.compile(r"[\s]{2,}")
double_commas = re.compile(r"[,]{2,}")
double_dash = re.compile(r"[-—]{2,}")
# Normalize all non-ASCII quote styles that confuse sentencex:
# »« (guillemets), „" (low-9/left), "" (curly double)
fancy_quotes = re.compile(r"[»«\u201e\u201c\u201d]+")
quotes = re.compile(r"[\u201c\u201d\u201e\u201f]+")
pattern_zh = re.compile(
    r"[」「\u201c\u201d\u201e\u201f\x1a⓪①②③④⑤⑥⑦⑧⑨⑩⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽*а-яА-Я\(\)\[\]\s\n\/\-\:•＂＃＄％＆＇＊＋－／＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》【】〔〕〖〗〘〙〜〟〰〾〿–—''‛‧﹏〉]+"
)
pattern_zh_total = re.compile(
    r"[」「\u201c\u201d\u201e\u201f\x1a⓪①②③④⑤⑥⑦⑧⑨⑩⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽*a-zA-Zа-яА-Я\(\)\[\]\s\n\/\-\:•＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》【】〔〕〖〗〘〙〜〟〰〾〿–—''‛‧﹏〉]+"
)
pattern_jp = re.compile(
    r"[\u201c\u201d\u201e\u201f\x1a⓪①②③④⑤⑥⑦⑧⑨⑩⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽*a-zA-Zа-яА-Я\(\)\[\]\s\n\/\-\:•＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》【】〔〕〖〗〘〙〜〟〰〾〿–—''‛‧﹏〉]+"
)
pat_comma = re.compile(r"[\.]+")
first_numbers = re.compile(r"^[0-9,\.]+")
last_punct = re.compile(r"[,\.]+$")
multiple_spaces = re.compile(r"\s+")
pattern_ru = re.compile(r"[a-zA-Z\.\(\)\[\]\/\-\:!?\<\>;•\"\'«»——,]+")
pattern_ru_letters_only = re.compile(r"[^а-яА-Я\s]+")
german_foo = "%@%"
german_months = "Januar|Jänner|Janner|Februar|März|Marz|April|Mai|Juni|Juli|August|September|Oktober|October|November|Dezember"
german_dates = re.compile(rf"(\s)(\d{{1,2}})\.(\s+)({german_months})")
german_bdates = re.compile(rf"(\s)(\d{{1,2}}){german_foo}(\s+)({german_months})")


DEFAULT_PREPROCESSING = [(double_spaces, " "), (double_commas, ","), (double_dash, "—")]


def is_lang_code_valid(langcode):
    """Check if language code is valid"""
    return langcode in LANGUAGES


def split_by_razdel(line):
    """Split line using 'razdel' library (best for Russian and Cyrillic-script languages)"""
    return list(x.text for x in razdel.sentenize(line))


# sentencex uses ISO 639-1 codes; map our custom codes
_SENTENCEX_CODE_MAP = {"bu": "be", "cz": "cs", "sw": "sv"}


def split_by_sentencex(line, langcode="xx"):
    """Split using sentencex (Wikimedia) — multilingual, ~300 languages, fast."""
    sx_code = _SENTENCEX_CODE_MAP.get(langcode, langcode)
    sentences = list(sentencex.segment(sx_code, line))
    return [s for s in sentences if s.strip()]


def split_zh(line):
    """Split line in Chinese"""
    return list(re.findall(r"[^!?。！？\.\!\?]+[!?。！？\.\!\?]?", line, flags=re.U))


def split_jp(line):
    """Split line in Japanese"""
    res = list(re.findall(r"[^!?。！？\.\!\?]+[!?。！？\.\!\?]?", line, flags=re.U))
    for i, x in enumerate(res):
        if x and x[0] == "」":
            res[i - 1] = res[i - 1] + "」"
            res[i] = res[i][1:]
    return res


def split_hy(text):
    """Split string in Armenian"""
    text = re.sub(r"\.{3,}", "…", text)
    res = list(re.findall(r"[^։:…]+[։:…]?", text))
    res = [s.strip() for s in res if s.strip()]
    return res


def split_ko(line):
    """Split line in Korean (handles both full-width and half-width punctuation)"""
    res = list(re.findall(r"[^!?\u3002\uff01\uff1f\.\!\?]+[!?\u3002\uff01\uff1f\.\!\?]?", line, flags=re.U))
    return [s for s in res if s.strip()]


# --- German custom splitter ---
# German abbreviations that end with a period but are NOT sentence boundaries.
_DE_ABBREVIATIONS = {
    # Titles
    "dr", "prof", "hr", "fr", "ing", "dipl", "mag",
    # Common abbreviations
    "bzw", "ca", "evtl", "ggf", "inkl", "nr", "str", "abs", "bd",
    "hrsg", "usw", "usf", "vgl", "sog", "bes", "geb", "gest",
    "tel", "fax", "orig", "hauptstr", "kirchenstr", "bahnhofstr",
    "anm", "aufl", "bearb", "dgl", "ebd", "gem", "kap", "erg",
    "jan", "feb", "m\u00e4r", "apr", "jun", "jul", "aug", "sep",
    "sept", "okt", "nov", "dez",
    # Single letters used in multi-part abbreviations (z.B., d.h., u.a., etc.)
    "z", "d", "u", "o", "s", "m", "i", "v", "n", "a", "b", "e", "h",
}

# Pattern to find candidate sentence-end positions: . ! ? followed by space+uppercase
# or followed by space+quote+uppercase, or at end of text.
_de_split_candidate = re.compile(
    r'([.!?]["\'\u00bb\u00ab\u201c\u201d\u201e]*)\s+'
)


def split_de(line):
    """Split German text into sentences with abbreviation and ordinal awareness."""
    # Normalize German quotes for consistent handling
    line = re.sub(r'[\u00bb\u00ab\u201e\u201c\u201d]+', '"', line)
    # Normalize triple-dot ellipsis to single character
    line = re.sub(r'\.{3,}', '\u2026', line)

    # Find all candidate split positions
    sentences = []
    last = 0
    for m in _de_split_candidate.finditer(line):
        end_pos = m.end()  # position after the space
        punct_start = m.start()

        # Only check abbreviation/ordinal rules for periods (not ! or ?)
        if m.group(1)[0] == '.':
            before = line[last:punct_start]
            token_match = re.search(r'(\S+)$', before)
            if token_match:
                raw_token = token_match.group(1)
                token = raw_token.lower().rstrip('.')

                # Skip known abbreviations
                if token in _DE_ABBREVIATIONS:
                    continue

                # Skip ordinals: bare digits before period (3. Januar)
                if re.match(r'^\d+$', raw_token):
                    continue

                # Skip multi-part abbreviations: x.Y pattern (z.B, d.h, u.a, etc.)
                if re.match(r'^[a-z\u00e4\u00f6\u00fc]\.[a-zA-Z\u00c4\u00d6\u00dc]$', raw_token, re.I):
                    continue

        # This is a real sentence boundary
        sentence = line[last:end_pos].strip()
        if sentence:
            sentences.append(sentence)
        last = end_pos

    # Add remaining text
    remainder = line[last:].strip()
    if remainder:
        sentences.append(remainder)

    return [s for s in sentences if s]


def split_ar(line):
    """Split line in Arabic (handles Arabic question mark U+061F)"""
    res = list(re.findall(r"[^!?\u061f\.\!\?]+[!?\u061f\.\!\?]?", line, flags=re.U))
    return [s.strip() for s in res if s.strip()]


def _make_sentencex_splitter(langcode):
    """Create a closure that splits using sentencex for a specific language."""
    sx_code = _SENTENCEX_CODE_MAP.get(langcode, langcode)
    def splitter(line):
        sentences = list(sentencex.segment(sx_code, line))
        return [s for s in sentences if s.strip()]
    return splitter


def after_fr(lines):
    """Get French orthography into account"""
    for i, x in enumerate(lines):
        if x and x[0] == "»":
            lines[i - 1] = lines[i - 1] + " »"
            lines[i] = lines[i][1:]
    return lines


def after_de(lines):
    """Some wierd German stuff"""
    return preprocess_raw(lines, [(german_bdates, r"\1\2.\3\4")])


def preprocess_raw(lines, re_list):
    """Preprocess raw file lines"""
    for i in range(len(lines)):
        for pat, val in re_list:
            lines[i] = re.sub(pat, val, lines[i])
    return lines


def preprocess(line, re_list, splitter, after_fn):
    """Preprocess general line"""
    for pat, val in re_list:
        line = re.sub(pat, val, line)
    splitted = splitter(line)
    return after_fn(splitted)


def ensure_paragraph_splitting(lines):
    """Split line by the paragraph marks if splitter failed"""
    line_endings = [preprocessor.PARAGRAPH_MARK + x for x in preprocessor.LINE_ENDINGS]
    res = []
    for line in lines:
        ser = []
        get_substrings(line, "", line_endings, ser)
        res.extend(ser)
    return res


def get_substrings(line, sep, endings, res):
    """Get all parts using recursion"""
    match = next((x for x in endings if x in line), False)
    if match:
        parts = line.partition(match)
        get_substrings(parts[0], parts[1], endings, res)
        get_substrings(parts[2], sep, endings, res)
    else:
        if line:
            res.append(line + sep)


def split_by_sentences_wrapper(lines, langcode, clean_text=True):
    """Special wrapper with an additional paragraph splitting"""
    res, acc = [], []
    marks = preprocessor.get_all_meta_marks()
    for line in lines:
        if not line.strip():
            continue
        if any(m in line for m in marks):
            # print("found mark", line)
            if acc:
                sentences = ensure_paragraph_splitting(
                    split_by_sentences(acc, langcode, clean_text)
                )
                res.extend(sentences)
                acc = []
            res.append(line)
        else:
            acc.append(line)
    if acc:
        sentences = ensure_paragraph_splitting(
            split_by_sentences(acc, langcode, clean_text)
        )
        res.extend(sentences)
    return res


# Cyrillic-script language codes — razdel is purpose-built for Russian and works
# well for all languages using Russian-style Cyrillic punctuation.
CYRILLIC_LANG_CODES = {
    "ru",  # Russian
    "bu",  # Belarusian
    "uk",  # Ukrainian
    "ba",  # Bashkir
    "cv",  # Chuvash
    "tt",  # Tatar
    "kk",  # Kazakh
    "ky",  # Kyrgyz
    "uz",  # Uzbek
    "sah", # Yakut
    "kv",  # Komi
    "udm", # Udmurt
    "mhr", # Meadow Mari
    "mrj", # Hill Mari
    "myv", # Erzya
    "mdf", # Moksha
    "os",  # Ossetian
    "inh", # Ingush
    "bua", # Buryat
    "xal", # Kalmyk
    "sr",  # Serbian
    "bg",  # Bulgarian
    "mn",  # Mongolian
    "alt", # Altai
    "kjh", # Khakas
}

# Languages with sentencex + quote normalization preprocessing.
# sentencex is fast (~same as razdel) and handles abbreviations, ordinals, etc.
# Quote normalization is needed because sentencex gets confused by guillemets/low-9 quotes.
SENTENCEX_LANG_CODES = {
    "en", "fr", "es", "it", "pt", "nl", "pl",
    "hu", "cz", "tr", "sw", "da", "el", "sk",
}

splitter_fn = {
    JP_CODE: split_jp,
    ZH_CODE: split_zh,
    HY_CODE: split_hy,
    KO_CODE: split_ko,
    DE_CODE: split_de,
}

# Route Cyrillic-script languages to razdel
for _cc in CYRILLIC_LANG_CODES:
    splitter_fn[_cc] = split_by_razdel

# Route Western European / Latin-script languages to sentencex
for _lc in SENTENCEX_LANG_CODES:
    splitter_fn[_lc] = _make_sentencex_splitter(_lc)

# Preprocessing rules.
# German/French/etc. need quote normalization so sentencex can split correctly
# (sentencex gets confused by »...« guillemets and „..." low-9 quotes).
# Normalize three-dot ellipsis to single character (sentencex handles … but splits on ...)
_ellipsis_dots = re.compile(r"\.{3}")

_SENTENCEX_PREPROCESSING = [(fancy_quotes, '"'), (_ellipsis_dots, "\u2026"), *DEFAULT_PREPROCESSING]

preprocessing_rules = {
    RU_CODE: [(pattern_ru_orig, ""), *DEFAULT_PREPROCESSING],
    ZH_CODE: [(pattern_zh, "")],
    JP_CODE: [(pat_comma, "\u3002"), (pattern_jp, "")],
}

# All sentencex languages get quote normalization preprocessing
for _lc in SENTENCEX_LANG_CODES:
    preprocessing_rules[_lc] = _SENTENCEX_PREPROCESSING

# Postprocessing: not needed — sentencex handles splitting natively.
postprocessing_rules = {}


def split_by_sentences(lines, langcode, clean_text=True):
    """Split line by sentences using language specific rules"""
    line = " ".join(lines)
    if langcode in splitter_fn:
        split_fn = splitter_fn[langcode]
    else:
        # Default fallback: sentencex — supports ~300 languages with
        # script-aware sentence boundary detection
        split_fn = lambda l: split_by_sentencex(l, langcode)
    after_fn = postprocessing_rules.get(langcode, lambda x: x)

    if clean_text:
        pre_rules = preprocessing_rules.get(langcode, [*DEFAULT_PREPROCESSING])
    else:
        pre_rules = [*DEFAULT_PREPROCESSING]

    sentences = preprocess(line, pre_rules, split_fn, after_fn)

    # Filter empty sentences
    sentences = [s for s in sentences if s.strip()]

    return sentences


def split_by_sentences_and_save(
    raw_path, splitted_path, langcode, handle_marks=False, clean_text=True
):
    """Split raw text file by sentences and save"""
    with open(raw_path, mode="r", encoding="utf-8") as input_file, open(
        splitted_path, mode="w", encoding="utf-8"
    ) as out_file:
        if not is_lang_code_valid(langcode):
            logger.warning(
                "Unsupported language code '%s', falling back to '%s' (General)",
                langcode, XX_CODE,
            )
            langcode = XX_CODE
        lines = input_file.readlines()
        lines = preprocess_raw(lines, [(quotes, '"')])
        if handle_marks:
            lines = preprocessor.mark_paragraphs(lines)
            sentences = split_by_sentences_wrapper(lines, langcode, clean_text)
        else:
            sentences = split_by_sentences(lines, langcode, clean_text)

        count = 1
        for x in sentences:
            if count < len(sentences):
                out_file.write(x.strip() + "\n")
            else:
                out_file.write(x.strip())
            count += 1


def get_supported_languages():
    """Get list of supported languages"""
    return LANGUAGES
