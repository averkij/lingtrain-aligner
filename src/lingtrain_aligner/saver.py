"""Output functions"""

from datetime import datetime
from lingtrain_aligner import helper, preprocessor, reader
import json
import xmltodict
from bs4 import BeautifulSoup


CULTURE_LIST = {
    "en": "en-US",
    "zh": "zh-CN",
    "ru": "ru-RU",
    "de": "de-DE"
}

DEFAULT_CULTURE = "en"

TMX_BEGIN = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<tmx version="1.4">
    <header creationtool="Lingtrain Alignment Stutio" segtype="sentence" adminlang="ru-RU" srclang="ru-RU" datatype="xml" creationdate="20190909T153841Z" creationid="LINGTRAIN"/>
    <body>"""

TMX_END = """
    </body>
</tmx>"""

TMX_BLOCK = """
        <tu creationdate="{timestamp}" creationid="LINGTRAIN">
            <tuv xml:lang="{culture_from}">
                <seg>{{text_from}}</seg>
            </tuv>
            <tuv xml:lang="{culture_to}">
                <seg>{{text_to}}</seg>
            </tuv>
        </tu>"""

JSON_FORMAT_VERSION = "0.1"
XML_FORMAT_VERSION = "0.1"

def save_tmx(db_path, output_path, lang_from, lang_to):
    """Save text document in TMX format"""
    tmx_template = TMX_BLOCK.format(timestamp=datetime.utcnow().strftime('%Y%m%dT%H%M%SZ'),
                                    culture_from=get_culture(lang_from), culture_to=get_culture(lang_to))
    doc_from, doc_to = helper.read_processing(db_path)
    with open(output_path, mode="w", encoding="utf-8") as doc_out:
        doc_out.write(TMX_BEGIN)
        for f, t in zip(doc_from, doc_to):
            doc_out.write(tmx_template.format(
                text_from = f.strip(), text_to = t.strip()))
        doc_out.write(TMX_END)


def save_plain_text(db_path, output_path, direction, batch_ids=[]):
    """Save text document in TXT format"""
    doc_from, doc_to = helper.read_processing(db_path, batch_ids)
    to_save = doc_from if direction=="from" else doc_to
    with open(output_path, mode="w", encoding="utf-8") as doc_out:
        text = "\n".join(to_save)
        doc_out.write(text)


def save_paragraphs(paragraphs, lang_code, output_path):
    """Save aligned paragraphs for a choosen direction"""
    with open(output_path, "w", encoding="utf8") as res_html:
        res = []
        for par in paragraphs[lang_code]:
            res.append(" ".join(par))
        text = "\n".join(res)
        res_html.write(text)


def get_culture(lang_code):
    """Get language culture"""
    if lang_code in CULTURE_LIST:
        return CULTURE_LIST[lang_code]
    return CULTURE_LIST[DEFAULT_CULTURE]


def save_json(db_path, output_path, lang_from, lang_to):
    """Save text document in JSON format"""
    text = export_json(db_path, [lang_from, lang_to])
    with open(output_path, mode="w", encoding="utf-8") as doc_out:
        doc_out.write(text)


def save_xml(db_path, output_path, lang_from, lang_to, direction = "to"):
    """Save text document in XML format"""
    text = export_xml(db_path, [lang_from, lang_to], direction)
    with open(output_path, mode="w", encoding="utf-8") as doc_out:
        doc_out.write(text)


def write_next(next_mark, metas_dict, lang_ordered, add_string=False):
    """Write next header to string or stream"""
    metas = metas_dict["items"]
    main_lang_code = metas_dict["main_lang_code"]
    res = []
    for lang in lang_ordered:
        meta = metas[lang][next_mark]
        if meta:
            val = meta.pop(0)
            res.append((next_mark, val[0], val[2]))

    # pop main lang if not yet popped
    if main_lang_code not in lang_ordered:
        meta = metas[main_lang_code][next_mark]
        if meta:
            meta.pop(0)
    return res


def get_root(db_path, direction="to"):
    """Prepare root element for extracting"""
    paragraphs, delimeters, metas, sent_counter = reader.get_paragraphs(
        db_path, direction)
    reader.sort_meta(metas)
    min_par_len = min([len(paragraphs[x]) for x in paragraphs])
    min_par_len = min(min_par_len, len(delimeters))
    next_mark, next_meta_par_id = reader.get_next_meta_par_id(metas)
    root = {"head":[], "body": []}

    return root, paragraphs, delimeters, metas, sent_counter, min_par_len, next_mark, next_meta_par_id


def export_json(db_path, lang_ordered, direction="to"):
    """Export book in JSON format"""
    root, paragraphs, delimeters, metas, sent_counter, par_len, next_mark, next_meta_par_id = get_root(db_path, direction)

    #head
    root["head"] = {
            "creator": "Lingtrain Alignment Studio",
            "paragraphs": par_len,
            "langs": [lang for lang in lang_ordered],
            "sentences": sent_counter,
            "author": { lang: reader.get_meta(metas["items"][lang], preprocessor.AUTHOR) for lang in lang_ordered},
            "title": { lang: reader.get_meta(metas["items"][lang], preprocessor.TITLE) for lang in lang_ordered},
            "version": JSON_FORMAT_VERSION
        }

    for actual_paragraphs_id in range(par_len):
        real_par_id = delimeters[actual_paragraphs_id]

        #marks
        while next_meta_par_id <= real_par_id:
            mark_item = write_next(next_mark, metas, lang_ordered)
            next_mark, next_meta_par_id = reader.get_next_meta_par_id(metas)
            content = {}
            for i, lang in enumerate(lang_ordered):
                content[lang] = mark_item[i][1]
            item = {
                "t": mark_item[0][0],
                "c": content,
                "p": mark_item[0][2]
            }
            root["body"].append(item)
        
        #sentences
        content = {}
        for i, lang in enumerate(lang_ordered):
            content[lang] = []
            for _, sent in enumerate(paragraphs[lang][actual_paragraphs_id]):
                content[lang].append(sent)
        item = {
            "t": "text",
            "c": content,
            "p": real_par_id
        }
        root["body"].append(item)

    #transform unicode
    root = json.dumps(root, ensure_ascii=False).encode('utf8')
    root = root.decode()

    return root


def export_xml(db_path, lang_ordered, direction="to"):
    """Export book in XML format"""
    root, paragraphs, delimeters, metas, sent_counter, par_len, next_mark, next_meta_par_id = get_root(db_path, direction)

    root["@version"] = XML_FORMAT_VERSION

    #head
    root["head"] = {
            "creationtool": "Lingtrain Alignment Studio",
            "creationid": "LINGTRAIN",
            "creationdate": datetime.utcnow().strftime("%Y%m%dT%H%M%SZ"),
            "paragraphs": par_len,
            "langs": { "lang": [] },
            "author": { "s": [] },
            "title": { "s": [] }
        }
    
    for lang in lang_ordered:
        root["head"]["langs"]["lang"].append({ "@id": lang, "sentences": sent_counter[lang] })
        meta = metas["items"][lang]
        title = reader.get_meta(meta, preprocessor.TITLE)
        author = reader.get_meta(meta, preprocessor.AUTHOR)
        root["head"]["author"]["s"].append({ "@lang": lang, "#text": author })
        root["head"]["title"]["s"].append({ "@lang": lang, "#text": title })

    #body
    root["body"] = { "p": [] }
    for actual_paragraphs_id in range(par_len):
        real_par_id = delimeters[actual_paragraphs_id]

        #marks
        while next_meta_par_id <= real_par_id:
            mark_item = write_next(next_mark, metas, lang_ordered)
            next_mark, next_meta_par_id = reader.get_next_meta_par_id(metas)

            sentences = []
            for i, lang in enumerate(lang_ordered):
                sentences.append(sent_item(lang, mark_item[i][1]))

            root["body"]["p"].append({
                "@type": mark_item[0][0],
                "@id": mark_item[0][2],
                "sentence": [ {"su": sentences } ]
                })

        #sentences
        sentences = []
        for i in range(len(paragraphs[lang_ordered[0]][actual_paragraphs_id])):
            sentence_pair = []
            for lang in lang_ordered:
                sentence_pair.append(sent_item(lang, paragraphs[lang][actual_paragraphs_id][i]))

            item = {
                "su": sentence_pair
            }
            sentences.append(item)

        root["body"]["p"].append({
                "@type": "text",
                "@id": real_par_id,
                "sentence": sentences})

    root = xmltodict.unparse({ "book": root }, )
    bs = BeautifulSoup(root, 'xml')
    root = bs.prettify()

    return root


def sent_item(lang, text):
    """Get xml sentence item"""
    item = {
        "@lang": lang,
        "#text": text
        }
    return item
