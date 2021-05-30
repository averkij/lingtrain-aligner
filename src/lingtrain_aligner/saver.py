"""Output functions"""

from datetime import datetime
from lingtrain_aligner import helper


def save_tmx(output_path, lang_from, lang_to, text_from, text_to):
    """Save text document in TMX format"""
    tmx_template = TMX_BLOCK.format(timestamp=datetime.utcnow().strftime('%Y%m%dT%H%M%SZ'),
                                    culture_from=get_culture(lang_from), culture_to=get_culture(lang_to))
    with open(output_path, mode="w", encoding="utf-8") as doc_out:
        doc_out.write(TMX_BEGIN)
        for f, t in zip(text_from, text_to):
            doc_out.write(tmx_template.format(
                text_from = f.strip(), text_to = t.strip()))
        doc_out.write(TMX_END)


def save_plain_text(output_path, text):
    """Save text document in TXT format"""
    with open(output_path, mode="w", encoding="utf-8") as doc_out:
        text = "\n".join(text)
        doc_out.write(text)


def get_culture(lang_code):
    """Get language culture"""
    if lang_code in CULTURE_LIST:
        return CULTURE_LIST[lang_code]
    return CULTURE_LIST[DEFAULT_CULTURE]


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
