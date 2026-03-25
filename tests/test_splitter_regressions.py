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
