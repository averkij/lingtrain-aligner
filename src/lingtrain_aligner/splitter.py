"""Texts splitter part of the engine"""


import os
import re

# import constants as con
import razdel

RU_CODE = "ru"
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
LANGUAGES = [RU_CODE, ZH_CODE, DE_CODE, EN_CODE, FR_CODE,
             IT_CODE, TR_CODE, ES_CODE, PL_CODE, PT_CODE, HU_CODE, CZ_CODE, JP_CODE]

PARAGRAPH_MARK = "%%%%%"
LINE_ENDINGS = [".","!","?",";",":","。"]

# pattern_ru_orig = re.compile(r'[a-zA-Z\(\)\[\]\/\<\>•\'\n]+')
pattern_ru_orig = re.compile(r'[\(\)\[\]\/\<\>•\'\n]+')
double_spaces = re.compile(r'[\s]+')
double_commas = re.compile(r'[,]+')
double_dash = re.compile(r'[-—]+')
german_quotes = re.compile(r'[»«“„]+')
pattern_zh = re.compile(
    r'[」「“”„‟\x1a⓪①②③④⑤⑥⑦⑧⑨⑩⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽*a-zA-Zа-яА-Я\(\)\[\]\s\n\/\-\:•＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》【】〔〕〖〗〘〙〜〟〰〾〿–—‘’‛‧﹏〉]+')
pattern_jp = re.compile(
    r'[“”„‟\x1a⓪①②③④⑤⑥⑦⑧⑨⑩⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽*a-zA-Zа-яА-Я\(\)\[\]\s\n\/\-\:•＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》【】〔〕〖〗〘〙〜〟〰〾〿–—‘’‛‧﹏〉]+')
pat_comma = re.compile(r'[\.]+')
first_numbers = re.compile(r'^[0-9,\.]+')
last_punct = re.compile(r'[,\.]+$')
multiple_spaces = re.compile(r'\s+')
pattern_ru = re.compile(r'[a-zA-Z\.\(\)\[\]\/\-\:!?\<\>;•\"\'«»——,]+')
pattern_ru_letters_only = re.compile(r'[^а-яА-Я\s]+')

DEFAULT_PREPROCESSING = [
    (double_spaces, ' '),
    (double_commas, ','),
    (double_dash, '—')
]


def is_lang_code_valid(langcode):
    """Check if language code is valid"""
    return langcode in LANGUAGES


def split_by_razdel(line):
    """Split line using 'razdel' library"""
    return list(x.text for x in razdel.sentenize(line))


def split_zh(line):
    """Split line in Chinese"""
    return list(re.findall(r'[^!?。！？\.\!\?]+[!?。！？\.\!\?]?', line, flags=re.U))


def split_jp(line):
    """Split line in Japanese"""
    return list(re.findall(r'[^!?。！？\.\!\?・]+[!?。！？\.\!\?・]?', line, flags=re.U))


def preprocess(line, re_list, splitter):
    """Preprocess general line"""
    for pat, val in re_list:
        line = re.sub(pat, val, line)
    return splitter(line)


def split_by_sentences(lines, langcode, add_paragraph_mark=False):
    """Split line by sentences using language specific rules"""
    if add_paragraph_mark:
        line = detect_paragraphs_and_join(lines, langcode)
    else:
        line = ' '.join(lines)
    if langcode == RU_CODE:
        sentences = preprocess(line, [
            (pattern_ru_orig, ''),
            *DEFAULT_PREPROCESSING
        ],
            split_by_razdel)
        return sentences
    if langcode == DE_CODE:
        sentences = preprocess(line, [
            (german_quotes, ''),
            *DEFAULT_PREPROCESSING
        ],
            split_by_razdel)
        return sentences
    if langcode == ZH_CODE:
        sentences = preprocess(line, [
            (pat_comma, '。'),
            (pattern_zh, '')
        ],
            split_zh)
        return sentences
    if langcode == JP_CODE:
        sentences = preprocess(line, [
            (pat_comma, '。'),
            (pattern_jp, '')
        ],
            split_jp)
        return sentences

    # apply default splitting
    sentences = preprocess(line, [
        *DEFAULT_PREPROCESSING
    ],
        split_by_razdel)

    if sentences[-1].strip() == '':
        return sentences[:-1]

    return sentences


def detect_paragraphs_and_join(lines, langcode):
    line_endings = tuple([x + "\n" for x in LINE_ENDINGS])
    for i, line in enumerate(lines):
        if line.endswith(line_endings):
            lines[i] = line[:-2] + PARAGRAPH_MARK + line[-2:-1]
    return ' '.join(lines)


def split_by_sentences_and_save(raw_path, splitted_path, filename, langcode, username, add_paragraph_mark=False):
    """Split raw text file by sentences and save"""

    with open(raw_path, mode='r', encoding='utf-8') as input_file, open(splitted_path, mode='w', encoding='utf-8') as out_file:
        if is_lang_code_valid(langcode):
            sentences = split_by_sentences(
                input_file.readlines(), langcode, add_paragraph_mark)
        else:
            raise Exception("Unknown language code.")

        count = 1
        for x in sentences:
            if count < len(sentences):
                out_file.write(x.strip() + "\n")
            else:
                out_file.write(x.strip())
            count += 1
