from lingtrain_aligner import splitter as sp

CONTENTS = {
    sp.RU_CODE: "Содержание",
    sp.ZH_CODE: "目录",
    sp.DE_CODE: "Inhalt",
    sp.EN_CODE: "Contents",
    sp.FR_CODE: "Table des matières",
    sp.IT_CODE: "Sommario",
    sp.TR_CODE: "İçindekiler",
    sp.ES_CODE: "Tabla de contenido",
    sp.PL_CODE: "Spis treści",
    sp.PT_CODE: "Conteúdo",
    sp.HU_CODE: "Tartalom",
    sp.CZ_CODE: "Obsah",
    sp.JP_CODE: "目次",
    sp.BA_CODE: "Йөкмәткеһе",
    sp.KO_CODE: "목차",
    sp.NL_CODE: "Inhoudsopgave",
    sp.SW_CODE: "Innehållsförteckning",
    sp.UK_CODE: "Зміст",
    sp.CV_CODE: "Содержани",
    sp.XX_CODE: "Table of contents",
}


def get_contents_name(code):
    if code in CONTENTS:
        return CONTENTS[code]
    return CONTENTS[sp.XX_CODE]