"""Regression tests for splitter language-specific postprocessing."""

from lingtrain_aligner import aligner, preprocessor, splitter


def _split_saved(tmp_path, text, *, preserve=False):
    raw_path = tmp_path / "raw.txt"
    splitted_path = tmp_path / "split.txt"
    raw_path.write_text(text, encoding="utf-8")
    splitter.split_by_sentences_and_save(
        raw_path,
        splitted_path,
        splitter.EN_CODE,
        handle_marks=True,
        preserve_blank_line_paragraphs=preserve,
    )
    return splitted_path.read_text(encoding="utf-8").splitlines()


def test_blank_line_paragraphs_are_opt_in(tmp_path):
    text = "First block without punctuation\n\nSecond block without punctuation"

    assert _split_saved(tmp_path, text) == [
        "First block without punctuation Second block without punctuation"
    ]
    assert _split_saved(tmp_path, text, preserve=True) == [
        "First block without punctuation%%%%%",
        "Second block without punctuation",
    ]


def test_preserved_blank_line_paragraphs_accept_multiple_whitespace_lines(tmp_path):
    text = "First block\n \t\n\nSecond block"

    assert _split_saved(tmp_path, text, preserve=True) == [
        "First block%%%%%",
        "Second block",
    ]


def test_preserved_blank_line_boundary_does_not_add_punctuation(tmp_path):
    lines = _split_saved(tmp_path, "First block\n\nFinal block", preserve=True)

    assert lines == ["First block%%%%%", "Final block"]
    parsed = preprocessor.parse_marked_line(lines[0])
    assert parsed["text"] == "First block"
    assert parsed["pa"] is True

    marks = []
    preprocessor.extract_marks(marks, lines[0], 0)
    assert marks == []

    ingested, _, _ = aligner.handle_marks(lines)
    assert ingested == [
        ("First block", (0, 0, 0, 0, 0, 0, 0, 0)),
        ("Final block", (1, 0, 0, 0, 0, 0, 0, 0)),
    ]


def test_preserve_option_keeps_legacy_punctuated_markers(tmp_path):
    text = "First sentence.\nSecond sentence."

    assert _split_saved(tmp_path, text, preserve=True) == [
        "First sentence%%%%%.",
        "Second sentence%%%%%.",
    ]


def test_paragraph_marker_survives_closing_quote_after_terminator(tmp_path):
    """A paragraph whose text ends with a closing quote/bracket that itself
    follows a sentence terminator ('… se termine. »') must keep its paragraph
    boundary.

    ``mark_paragraphs`` inserts the ``%%%%%`` marker before the trailing ``»``,
    producing ``… se termine. %%%%%»``. The splitter then treats the earlier
    ``.`` as the real boundary and emits ``%%%%%»`` at the HEAD of the next
    segment, where it has no text of its own and used to be dropped — collapsing
    every French-quotes paragraph into a single one. Regression for the
    'Не объединять абзацы' upload bug on guillemet-punctuated text (reproduced
    with EN routing, so the fix must not depend on the French post-pass)."""
    text = (
        "Le premier bloc se termine. »\n\n"
        "« Le deuxième bloc arrive. Encore le deuxième. »\n\n"
        "« Le dernier bloc finit ici. »."
    )

    lines = _split_saved(tmp_path, text, preserve=True)

    assert lines == [
        "Le premier bloc se termine.%%%%%»",
        "« Le deuxième bloc arrive.",
        "Encore le deuxième.%%%%%»",
        "« Le dernier bloc finit ici. »%%%%%.",
    ]
    # Three source paragraphs -> exactly three paragraph-terminated lines.
    para_terminated = [l for l in lines if preprocessor.strip_paragraph_mark(l)[1]]
    assert len(para_terminated) == 3


def test_ensure_paragraph_splitting_reattaches_orphan_marker():
    """A paragraph marker pushed to the head of a segment folds back onto the
    previous segment instead of being discarded."""
    mark = preprocessor.PARAGRAPH_MARK

    sentences = [
        "First paragraph ends here.",
        f"{mark}» Second paragraph starts.",
        f"Second continues here.{mark}»",
    ]

    assert splitter.ensure_paragraph_splitting(sentences) == [
        f"First paragraph ends here.{mark}»",
        "Second paragraph starts.",
        f"Second continues here.{mark}»",
    ]


def test_ensure_paragraph_splitting_keeps_leading_orphan_without_previous():
    """A leading orphan marker with no previous segment to receive it must not
    crash and must not resurrect an empty paragraph — the homeless marker is
    dropped and only the real content survives."""
    mark = preprocessor.PARAGRAPH_MARK

    result = splitter.ensure_paragraph_splitting([f"{mark}» Only segment here."])

    assert [r.strip() for r in result] == ["Only segment here."]
    assert not any(preprocessor.strip_paragraph_mark(r)[1] for r in result)


def test_split_by_sentences_applies_french_postprocessing(monkeypatch):
    """French postprocessing should reattach a leading closing guillemet."""

    monkeypatch.setitem(
        splitter.splitter_fn,
        splitter.FR_CODE,
        lambda line: ["Bonjour.", "\u00bb Salut."],
    )

    sentences = splitter.split_by_sentences(["ignored"], splitter.FR_CODE)

    assert sentences == ["Bonjour. \u00bb", "Salut."]


def test_split_by_sentences_masks_and_restores_german_dates(monkeypatch):
    """German preprocessing should mask ordinals and postprocessing should restore them."""

    seen = {}

    def fake_split(line):
        seen["line"] = line
        return [line]

    monkeypatch.setitem(splitter.splitter_fn, splitter.DE_CODE, fake_split)

    sentences = splitter.split_by_sentences(
        ["Am 3. Januar ging es los."],
        splitter.DE_CODE,
    )

    assert f"3{splitter.german_foo} Januar" in seen["line"]
    assert sentences == ["Am 3. Januar ging es los."]


def test_split_by_sentences_preserves_german_quotes():
    """German splitting should preserve source quote glyphs."""

    sentences = splitter.split_by_sentences(
        ["\u201eHallo.\u201c Dann ging er."],
        splitter.DE_CODE,
    )

    assert sentences == ["\u201eHallo.\u201c", "Dann ging er."]


def test_split_by_sentences_and_save_preserves_german_quotes(tmp_path):
    """The file save path should not normalize German quotes to ASCII quotes."""

    raw_path = tmp_path / "raw.txt"
    splitted_path = tmp_path / "split.txt"
    raw_path.write_text("\u201eHallo.\u201c Dann ging er.", encoding="utf-8")

    splitter.split_by_sentences_and_save(raw_path, splitted_path, splitter.DE_CODE)

    assert splitted_path.read_text(encoding="utf-8").splitlines() == [
        "\u201eHallo.\u201c",
        "Dann ging er.",
    ]


def test_split_by_sentences_handles_adjacent_dialogue_dash_boundaries():
    """Adjacent '.—' dialogue boundaries should still split into sentences."""

    text = (
        "— «Акулиной,— отвечала Лиза, стараясь освободить свои пальцы от руки "
        "Алексеевой; — да пусти ж, барин; мне и домой пора».— «Ну, мой друг "
        "Акулина, непременно буду в гости к твоему батюшке, к Василью-кузнецу»."
        "— «Что ты? — возразила с живостию Лиза,— ради Христа, не приходи."
    )

    sentences = splitter.split_by_sentences([text], splitter.RU_CODE)

    assert sentences == [
        "— «Акулиной,— отвечала Лиза, стараясь освободить свои пальцы от руки Алексеевой; — да пусти ж, барин; мне и домой пора».",
        "— «Ну, мой друг Акулина, непременно буду в гости к твоему батюшке, к Василью-кузнецу».",
        "— «Что ты? — возразила с живостию Лиза,— ради Христа, не приходи.",
    ]


def test_split_en_keeps_page_references_in_one_sentence():
    """English 'p.' / 'pp.' as page references must not terminate a sentence."""

    sentences = splitter.split_by_sentences(
        ["You can see this in p. 1 and in p.2 for details."],
        splitter.EN_CODE,
    )

    assert sentences == ["You can see this in p. 1 and in p.2 for details."]


def test_split_en_keeps_figure_caption_with_number():
    """'Fig. 1. <Caption>' must be kept as a single sentence — the period after
    the figure number is not a sentence boundary."""

    sentences = splitter.split_by_sentences(
        [
            "Fig. 1. Cult Image from the Bakhty Village (Florinsky, 1896, Table VIII). "
            "The next sentence starts here."
        ],
        splitter.EN_CODE,
    )

    assert sentences == [
        "Fig. 1. Cult Image from the Bakhty Village (Florinsky, 1896, Table VIII).",
        "The next sentence starts here.",
    ]


def test_split_en_handles_reference_abbreviations():
    """Reference abbreviations Vol./No./pp. inside a single sentence must not
    cause false boundaries."""

    sentences = splitter.split_by_sentences(
        ["Vol. 3, No. 2 of the journal is on p. 12."],
        splitter.EN_CODE,
    )

    assert sentences == ["Vol. 3, No. 2 of the journal is on p. 12."]


def test_split_en_still_splits_real_sentence_boundaries():
    """The English post-merge must not swallow real sentence boundaries after
    a figure caption — 'Fig. 1. Caption.' must stay intact, but a following
    sentence must still separate."""

    sentences = splitter.split_by_sentences(
        ["See Fig. 1. Cult Image from the village. Another sentence follows."],
        splitter.EN_CODE,
    )

    assert sentences == [
        "See Fig. 1. Cult Image from the village.",
        "Another sentence follows.",
    ]


def test_split_en_preserves_initials():
    """Razdel already handles 'J. K. Rowling'-style initials — make sure
    the English wrapper does not regress this."""

    sentences = splitter.split_by_sentences(
        ["J. K. Rowling wrote a book. It was popular."],
        splitter.EN_CODE,
    )

    assert sentences == ["J. K. Rowling wrote a book.", "It was popular."]


def test_split_en_keeps_messrs_honorific():
    """'Messrs.' (plural of Mr.) is an honorific, not a sentence boundary —
    razdel splits after it. Regression for the Titanic 'built by Messrs. Harland
    & Wolff' paragraph that desynced 1:1 alignment."""

    sentences = splitter.split_by_sentences(
        ["The ship was built by Messrs. Harland & Wolff in Belfast."],
        splitter.EN_CODE,
    )

    assert sentences == ["The ship was built by Messrs. Harland & Wolff in Belfast."]


def test_split_en_keeps_coordinate_refs_in_numeric_context():
    """'Lat.'/'Long.' before a number are coordinate references, not sentence
    boundaries — but only in numeric context (see the non-regression test for
    the adverb 'long')."""

    sentences = splitter.split_by_sentences(
        ["She sank in Lat. 41 N. and Long. 50 W. that night."],
        splitter.EN_CODE,
    )

    assert sentences == ["She sank in Lat. 41 N. and Long. 50 W. that night."]


def test_split_en_still_splits_long_as_ordinary_word():
    """The coordinate handling must NOT merge a sentence that merely ends in the
    adverb 'long' / 'flat' — the digit-context guard keeps these splitting."""

    sentences = splitter.split_by_sentences(
        ["He did not wait long. Then he left."],
        splitter.EN_CODE,
    )

    assert sentences == ["He did not wait long.", "Then he left."]


def test_split_ru_keeps_figure_caption_with_number():
    """'Рис. N. <Caption>' must stay intact — razdel splits after 'Рис. 1.'
    because the number's trailing period looks like a sentence boundary."""

    sentences = splitter.split_by_sentences(
        ["Рис. 1. Название рисунка. Следующее предложение."],
        splitter.RU_CODE,
    )

    assert sentences == [
        "Рис. 1. Название рисунка.",
        "Следующее предложение.",
    ]


def test_split_ru_keeps_bakhty_caption_with_internal_табл():
    """Real caption from the user's corpus — 'Рис. 1. Культовое изображение …
    табл. VIII).' must survive as a single sentence despite the nested
    'табл.' reference razdel splits on."""

    sentences = splitter.split_by_sentences(
        [
            "Рис. 1. Культовое изображение из села Бахты "
            "(Флоринский, 1896, табл. VIII)."
        ],
        splitter.RU_CODE,
    )

    assert sentences == [
        "Рис. 1. Культовое изображение из села Бахты "
        "(Флоринский, 1896, табл. VIII)."
    ]


def test_split_ru_keeps_table_caption_with_number():
    """'Табл. N. <Caption>' follows the same pattern as 'Рис. N.'."""

    sentences = splitter.split_by_sentences(
        ["Табл. 3. Данные эксперимента. Здесь продолжение."],
        splitter.RU_CODE,
    )

    assert sentences == [
        "Табл. 3. Данные эксперимента.",
        "Здесь продолжение.",
    ]


def test_split_ru_preserves_existing_page_references():
    """Razdel-native handling of 'с.' and 'т. е.' must not regress after
    the caption-aware post-merge is added."""

    sentences = splitter.split_by_sentences(
        ["Смотрите с. 1 и с. 2 для подробностей."],
        splitter.RU_CODE,
    )
    assert sentences == ["Смотрите с. 1 и с. 2 для подробностей."]


def test_split_ru_still_splits_plain_sentences():
    """Ordinary prose must still split normally."""

    sentences = splitter.split_by_sentences(
        ["Дом стоял на холме. Ветер дул с моря."],
        splitter.RU_CODE,
    )
    assert sentences == ["Дом стоял на холме.", "Ветер дул с моря."]


def test_split_fr_keeps_honorifics_and_saint():
    """French honorifics ('Mme', 'Mlle') and 'St.' (Saint) followed by a capital
    must not terminate a sentence — French routes through razdel, which has no
    French abbreviation knowledge."""

    assert splitter.split_by_sentences(
        ["Mme Dupont et Mlle Posh sont venues. Quelle joie."], splitter.FR_CODE
    ) == ["Mme Dupont et Mlle Posh sont venues.", "Quelle joie."]

    assert splitter.split_by_sentences(
        ["Nous sommes allés à St. Pancras. Puis rentrés."], splitter.FR_CODE
    ) == ["Nous sommes allés à St. Pancras.", "Puis rentrés."]


def test_split_fr_still_splits_word_colliding_with_abbrev():
    """The French set deliberately omits 'vol'/'art'/'sept' so a sentence merely
    ending in those ordinary words still splits."""

    assert splitter.split_by_sentences(
        ["C’était un vol. Puis la police arriva."], splitter.FR_CODE
    ) == ["C’était un vol.", "Puis la police arriva."]


def test_split_it_keeps_honorifics():
    """Italian honorifics ('Sig.', 'Dott.') followed by a capital must not
    terminate a sentence."""

    assert splitter.split_by_sentences(
        ["Ho visto il Sig. Perkupp ieri. Era raggiante."], splitter.IT_CODE
    ) == ["Ho visto il Sig. Perkupp ieri.", "Era raggiante."]

    assert splitter.split_by_sentences(
        ["Il Dott. Bianchi è arrivato. Tutti erano contenti."], splitter.IT_CODE
    ) == ["Il Dott. Bianchi è arrivato.", "Tutti erano contenti."]


def test_split_it_still_splits_plain_sentences():
    """Ordinary Italian prose still splits on real boundaries."""

    assert splitter.split_by_sentences(
        ["La casa era sul colle. Il vento soffiava."], splitter.IT_CODE
    ) == ["La casa era sul colle.", "Il vento soffiava."]


def test_split_nl_keeps_honorifics_and_sint():
    """Dutch honorifics ('Dhr.', 'Mevr.') and 'St.' (Sint) followed by a capital
    must not terminate a sentence."""

    assert splitter.split_by_sentences(
        ["Dhr. De Vries en Mevr. Jansen kwamen. Leuk."], splitter.NL_CODE
    ) == ["Dhr. De Vries en Mevr. Jansen kwamen.", "Leuk."]

    assert splitter.split_by_sentences(
        ["Op St. Nicolaas kreeg hij cadeaus. Hij was blij."], splitter.NL_CODE
    ) == ["Op St. Nicolaas kreeg hij cadeaus.", "Hij was blij."]


def test_split_zh_reattaches_trailing_closing_bracket():
    """A Chinese paragraph that ends a quotation with ``。」`` must NOT peel the
    lone ``」`` off as its own sentence — it re-attaches to the last sentence so
    the per-paragraph sentence count matches a 1:1 translation (whose closing
    quote stays attached). Regression for the入蜀記 zh→ru alignment build."""

    sentences = splitter.split_by_sentences(
        ["坐中，國器云：「天生此為我用也。其後，石坐罪，竟荷校云。」"],
        splitter.ZH_CODE,
        clean_text=False,
    )

    assert sentences == [
        "坐中，國器云：「天生此為我用也。",
        "其後，石坐罪，竟荷校云。」",
    ]


def test_split_zh_reattaches_mid_paragraph_closer():
    """A closing ``」`` that leads a following clause re-attaches to the prior
    sentence, and the remaining clause stays its own sentence."""

    sentences = splitter.split_by_sentences(
        ["他說：「甲乙丙。」丁戊己。"],
        splitter.ZH_CODE,
        clean_text=False,
    )

    assert sentences == ["他說：「甲乙丙。」", "丁戊己。"]


def test_split_zh_plain_sentences_unaffected():
    """Ordinary Chinese prose without trailing closers still splits on 。 only."""

    sentences = splitter.split_by_sentences(
        ["山益奇怪。江平無波。夜無蚊。"],
        splitter.ZH_CODE,
        clean_text=False,
    )
    assert sentences == ["山益奇怪。", "江平無波。", "夜無蚊。"]


def test_split_jp_still_reattaches_corner_bracket():
    """split_jp's historic ``」`` re-attachment must be preserved by the shared
    helper."""

    sentences = splitter.split_by_sentences(
        ["彼は言った。「そうだ。」次の文。"],
        splitter.JP_CODE,
        clean_text=False,
    )
    assert sentences == ["彼は言った。", "「そうだ。」", "次の文。"]


def test_split_ko_folds_paragraph_final_lone_closing_quote():
    """Korean dialogue uses ASCII straight quotes, so a paragraph-final speech
    printed `...요."` must not peel the closing `"` off as a bogus lone sentence
    (which would desync 1:1 alignment). The lone trailing closer folds back."""

    sentences = splitter.split_by_sentences(
        ['그가 말했다. 다 끝났다고 절 부려 주신답니다요."'],
        splitter.KO_CODE,
    )

    # exactly two sentences — the closing `"` stays attached to `요."`, not a 3rd
    assert len(sentences) == 2
    assert sentences[-1].strip() == '다 끝났다고 절 부려 주신답니다요."'


def test_split_ko_does_not_misfold_opening_quote():
    """An OPENING ASCII quote is always followed by text, so it must stay with
    its own sentence and never be folded onto the previous one."""

    sentences = splitter.split_by_sentences(
        ['그가 말했다. "안녕하세요. 잘 지내요?" 나는 웃었다.'],
        splitter.KO_CODE,
    )

    assert len(sentences) == 4
    assert sentences[0].strip() == '그가 말했다.'
    assert sentences[1].strip() == '"안녕하세요.'


def test_every_splittable_language_is_a_valid_code():
    """Every language the splitter has specific support for must pass
    is_lang_code_valid — otherwise callers that gate on it (e.g. alignment
    import) coerce real language codes to the generic XX_CODE ("xx" was never
    Khakas or any concrete language; it is the "Unknown/General" fallback)."""

    for code in splitter.CYRILLIC_LANG_CODES:
        assert splitter.is_lang_code_valid(code), code
    for code in splitter.splitter_fn:
        assert splitter.is_lang_code_valid(code), code


def test_xx_stays_the_generic_unknown_code():
    assert splitter.XX_CODE == "xx"
    assert splitter.LANGUAGES["xx"]["name"] == "Unknown"
