"""Texts splitter part of the engine"""

import logging
import re

import razdel

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
# Normalize missing whitespace in patterns like '.— «Next' before handing
# the text to razdel. The inserted space is stripped away by sentence trimming.
sentence_end_before_dialogue_dash = re.compile(
    r'([.!?\u2026]["\')\]\u00bb\u201d\u2019]*)(?=[\u2014\u2013-]\s*(?:[\u00ab\u201e\u201c"\(\[]\s*)?[A-Z\u0410-\u042f\u0401])'
)


DEFAULT_PREPROCESSING = [
    (double_spaces, " "),
    (double_commas, ","),
    (double_dash, "—"),
]


def is_lang_code_valid(langcode):
    """Check if language code is valid"""
    return langcode in LANGUAGES


def split_by_razdel(line):
    """Split line using 'razdel' library (best for Russian and Cyrillic-script languages)"""
    line = re.sub(sentence_end_before_dialogue_dash, r"\1 ", line)
    return list(x.text for x in razdel.sentenize(line))



# Closing quotation / bracket glyphs that may trail a sentence terminator in
# CJK text — e.g. a quotation that ends a paragraph prints as ``。」``. The
# terminator-based split would otherwise peel the lone closer off as its own
# bogus "sentence", which desynchronises 1:1 alignment against a target whose
# closing quote stays attached to the last sentence. We re-attach any leading
# run of these closers to the previous segment.
_CJK_CLOSERS = "」』）》】〕〗〙〛〉”’"


def _reattach_leading_closers(res):
    """Fold a segment's leading run of CJK closing brackets onto the previous
    segment. Generalises split_jp's historic single-``」`` handling so a
    paragraph ending in ``。」`` does not split the ``」`` off as a sentence."""
    out = []
    for seg in res:
        if out and seg and seg[0] in _CJK_CLOSERS:
            i = 0
            while i < len(seg) and seg[i] in _CJK_CLOSERS:
                i += 1
            out[-1] = out[-1] + seg[:i]
            rest = seg[i:].lstrip()
            if rest:
                out.append(rest)
        else:
            out.append(seg)
    return out


def split_zh(line):
    """Split line in Chinese"""
    res = list(re.findall(r"[^!?。！？\.\!\?]+[!?。！？\.\!\?]?", line, flags=re.U))
    return _reattach_leading_closers(res)


def split_jp(line):
    """Split line in Japanese"""
    res = list(re.findall(r"[^!?。！？\.\!\?]+[!?。！？\.\!\?]?", line, flags=re.U))
    return _reattach_leading_closers(res)


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


# --- Abbreviation-aware post-merge (shared by English and Russian wrappers) ---
# razdel is primarily tuned for Russian prose; both English and Russian produce
# spurious splits around certain reference abbreviations. Instead of replacing
# razdel, we run it first and then undo the false boundaries by merging back
# fragments whose trailing token is a known non-boundary abbreviation.

# Plain tail: last non-whitespace token ending in a period ("... Fig." / "...Табл.").
_tail_abbrev_plain = re.compile(r"(\S+)\.\s*$")
# Captioned tail: "<abbrev>. <number>." where <number> may contain internal
# dots. Catches "Рис. 1.", "Fig. 1.5.", "П. 4.3.2." — caption prefixes razdel
# treats as sentence boundaries when the number's period is followed by a
# capital letter.
_tail_abbrev_number = re.compile(r"(\S+)\.\s+\d[\d.]*\.\s*$")


def _trailing_token_is_nonboundary(text, abbrevs):
    """True if `text` ends with `<abbrev>.` or `<abbrev>. <digits>.` where
    the lowercased abbrev is in `abbrevs`."""
    for pat in (_tail_abbrev_number, _tail_abbrev_plain):
        m = pat.search(text)
        if m and m.group(1).lower() in abbrevs:
            return True
    return False


def _merge_abbrev_boundaries(sentences, abbrevs):
    """Merge consecutive razdel fragments when the earlier fragment ends with
    a known non-boundary abbreviation, undoing razdel's false split."""
    if not sentences:
        return sentences
    merged = []
    buffer = None
    for s in sentences:
        buffer = s if buffer is None else buffer + " " + s
        if _trailing_token_is_nonboundary(buffer, abbrevs):
            continue
        merged.append(buffer)
        buffer = None
    if buffer is not None:
        merged.append(buffer)
    return merged


# --- English custom splitter ---
# English abbreviations razdel does not recognize. Matched case-insensitively
# on the non-whitespace chunk immediately before the trailing period, so
# multi-part forms like "e.g" and "U.S" match their dotted prefix.
_EN_NONBOUNDARY_ABBREVS = frozenset({
    # Reference / citation
    "p", "pp", "fig", "figs", "no", "nos", "vol", "vols",
    "ch", "chap", "chaps", "sec", "secs", "par", "pars",
    "col", "cols", "ed", "eds", "ff", "cf", "ibid", "al", "et",
    # Titles and honorifics that razdel misses
    "prof", "rev", "hon", "sr", "jr",
    "gen", "capt", "lt", "sgt", "cpl", "pvt",
    "pres", "gov", "sen", "rep", "atty", "supt",
    # Common abbreviations
    "inc", "ltd", "corp", "dept", "univ", "assn", "co",
    "bros", "mt", "ave", "blvd", "rd", "ln", "apt", "ste", "bldg",
    # Months (when abbreviated mid-sentence)
    "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sep", "sept", "oct", "nov", "dec",
    # Days
    "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri", "sat", "sun",
    # Multi-part dotted abbreviations (matched as their dotted prefix)
    "e.g", "i.e", "u.s", "u.k", "u.s.a",
})


def split_en(line):
    """Split English text using razdel, then merge false boundaries caused by
    razdel not recognizing common English abbreviations ('Fig.', 'p.', 'Vol.',
    etc.)."""
    line = re.sub(sentence_end_before_dialogue_dash, r"\1 ", line)
    raw = [x.text for x in razdel.sentenize(line)]
    return _merge_abbrev_boundaries(raw, _EN_NONBOUNDARY_ABBREVS)


# --- Russian (and related Cyrillic) custom splitter ---
# Razdel already handles many Russian abbreviations (с., т., см., напр.,
# etc.), but it still splits after caption prefixes like "Рис. 1.", "Табл. 3.",
# "Ил. 5.", "Прим. 2." where the number's trailing period is followed by the
# capital letter of the caption text. This set covers the caption/reference
# shapes razdel misses; it is applied to all Cyrillic-script languages that
# currently route through razdel.
_RU_NONBOUNDARY_ABBREVS = frozenset({
    # Caption / figure / table / illustration
    "рис",   # рисунок
    "табл",  # таблица
    "ил",    # иллюстрация
    "прим",  # примечание
    # Structural references
    "гл",    # глава
    "разд",  # раздел
    "кн",    # книга
    "вып",   # выпуск
    "ст",    # статья / стих
    "п",     # пункт / параграф
    "пп",    # подпункт
    "ч",     # часть
    "т",     # том
    # Page / see / compare / e.g.
    "стр",   # страница
    "с",     # страница (short)
    "см",    # см.
    "ср",    # ср.
    "напр",  # например
})


def split_ru(line):
    """Split Russian (and related Cyrillic-script) text using razdel, then
    merge false boundaries razdel produces around caption-style references
    like 'Рис. 1.' or 'Табл. 3.'."""
    line = re.sub(sentence_end_before_dialogue_dash, r"\1 ", line)
    raw = [x.text for x in razdel.sentenize(line)]
    return _merge_abbrev_boundaries(raw, _RU_NONBOUNDARY_ABBREVS)


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



def after_fr(lines):
    """Get French orthography into account"""
    for i, x in enumerate(lines):
        if i > 0 and x and x[0] == "»":
            lines[i - 1] = lines[i - 1] + " »"
            lines[i] = lines[i][1:].lstrip()
    return lines


def after_de(lines):
    """Restore German date punctuation hidden during preprocessing."""
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
        if line.strip():
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

splitter_fn = {
    JP_CODE: split_jp,
    ZH_CODE: split_zh,
    HY_CODE: split_hy,
    KO_CODE: split_ko,
    DE_CODE: split_de,
    EN_CODE: split_en,
}

# Route Cyrillic-script languages to the Russian splitter (razdel + caption merge)
for _cc in CYRILLIC_LANG_CODES:
    splitter_fn[_cc] = split_ru

preprocessing_rules = {
    RU_CODE: [(pattern_ru_orig, ""), *DEFAULT_PREPROCESSING],
    DE_CODE: [
        (german_dates, rf"\1\2{german_foo}\3\4"),
        *DEFAULT_PREPROCESSING,
    ],
    ZH_CODE: [(pattern_zh, "")],
    JP_CODE: [(pat_comma, "\u3002"), (pattern_jp, "")],
}

postprocessing_rules = {FR_CODE: after_fr, DE_CODE: after_de}


def split_by_sentences(lines, langcode, clean_text=True):
    """Split line by sentences using language specific rules"""
    line = " ".join(lines)
    if langcode in splitter_fn:
        split_fn = splitter_fn[langcode]
    else:
        # Default fallback: razdel
        split_fn = split_by_razdel
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
