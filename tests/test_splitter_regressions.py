"""Regression tests for splitter language-specific postprocessing."""

from lingtrain_aligner import splitter


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
