"""Texts splitter part of the engine"""


import re

import razdel
from lingtrain_aligner import preprocessor

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
JP_CODE = "jp"
BA_CODE = "ba"
KO_CODE = "ko"
NL_CODE = "nl"
SW_CODE = "sw"
UK_CODE = "uk"
CV_CODE = "cv"
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
    XX_CODE: {"name": "Unknown"},
}


# pattern_ru_orig = re.compile(r'[a-zA-Z\(\)\[\]\/\<\>•\'\n]+')
pattern_ru_orig = re.compile(r"[\/\<\>•\'\n]+")
double_spaces = re.compile(r"[\s]{2,}")
double_commas = re.compile(r"[,]{2,}")
double_dash = re.compile(r"[-—]{2,}")
german_quotes = re.compile(r"[»«“„]+")
quotes = re.compile(r"[“”„‟]+")
pattern_zh = re.compile(
    r"[」「“”„‟\x1a⓪①②③④⑤⑥⑦⑧⑨⑩⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽*а-яА-Я\(\)\[\]\s\n\/\-\:•＂＃＄％＆＇＊＋－／＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》【】〔〕〖〗〘〙〜〟〰〾〿–—‘’‛‧﹏〉]+"
)
pattern_zh_total = re.compile(
    r"[」「“”„‟\x1a⓪①②③④⑤⑥⑦⑧⑨⑩⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽*a-zA-Zа-яА-Я\(\)\[\]\s\n\/\-\:•＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》【】〔〕〖〗〘〙〜〟〰〾〿–—‘’‛‧﹏〉]+"
)
pattern_jp = re.compile(
    r"[“”„‟\x1a⓪①②③④⑤⑥⑦⑧⑨⑩⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽*a-zA-Zа-яА-Я\(\)\[\]\s\n\/\-\:•＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》【】〔〕〖〗〘〙〜〟〰〾〿–—‘’‛‧﹏〉]+"
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
    """Split line using 'razdel' library"""
    return list(x.text for x in razdel.sentenize(line))


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
        sentences = ensure_paragraph_splitting(split_by_sentences(acc, langcode, clean_text))
        res.extend(sentences)
    return res


splitter_fn = {
    JP_CODE: split_jp,
    ZH_CODE: split_zh
}

preprocessing_rules = {
    RU_CODE: [(pattern_ru_orig, ""), *DEFAULT_PREPROCESSING],
    DE_CODE: [(german_quotes, '"'), (german_dates, rf"\1\2{german_foo}\3\4"), *DEFAULT_PREPROCESSING,],
    ZH_CODE: [(pattern_zh, "")],
    JP_CODE: [(pat_comma, "。"), (pattern_jp, "")],
}

postprocessing_rules = {
    FR_CODE: after_fr,
    DE_CODE: after_de
}

def split_by_sentences(lines, langcode, clean_text=True):
    """Split line by sentences using language specific rules"""
    line = " ".join(lines)
    split_fn = splitter_fn.get(langcode, split_by_razdel)
    after_fn = postprocessing_rules.get(langcode, lambda x: x)

    if clean_text:
        pre_rules = preprocessing_rules.get(langcode, [*DEFAULT_PREPROCESSING])
    else:        
        pre_rules = [*DEFAULT_PREPROCESSING]

    sentences = preprocess(line, pre_rules, split_fn, after_fn)

    if sentences[-1].strip() == "":
        sentences = sentences[:-1]

    return sentences


def split_by_sentences_and_save(raw_path, splitted_path, langcode, handle_marks=False, clean_text=True):
    """Split raw text file by sentences and save"""
    with open(raw_path, mode="r", encoding="utf-8") as input_file, open(
        splitted_path, mode="w", encoding="utf-8"
    ) as out_file:
        if is_lang_code_valid(langcode):
            lines = input_file.readlines()
            lines = preprocess_raw(lines, [(quotes, '"')])
            if handle_marks:
                lines = preprocessor.mark_paragraphs(lines)
                sentences = split_by_sentences_wrapper(lines, langcode, clean_text)
            else:
                sentences = split_by_sentences(lines, langcode, clean_text)
        else:
            raise Exception("Unknown language code.")

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
