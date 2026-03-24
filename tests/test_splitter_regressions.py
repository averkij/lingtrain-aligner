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
