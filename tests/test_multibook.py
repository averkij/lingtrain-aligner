"""Tests for the multilingual (.ltm) format: trivial_alignment_multi,
add_language and the direct multilingual reader. N strictly-1:1:N editions in one
render-only file — no embeddings, no polybook merge.

Fixtures are derived from the Fall of the House of Usher sample (qtext/qname
epigraph, multi-sentence prose, a four-line poem split into two stanzas) so every
builder branch is exercised on realistic markup.
"""

import sqlite3
from pathlib import Path

import pytest

from lingtrain_aligner import aligner, constants, helper, reader
from lingtrain_aligner.aligner import TrivialAlignmentError


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


# EN is the structural source: author/title (side marks), a qtext/qname epigraph,
# an h2, a 2-sentence prose paragraph, a 1-sentence prose paragraph, then a
# four-line poem broken into two stanzas by a blank line.
EN = [
    "Edgar Allan Poe%%%%%author.",
    "The Fall of the House of Usher%%%%%title.",
    "Son coeur est un luth suspendu.%%%%%qtext.",
    "DE BERANGER.%%%%%qname.",
    "Chapter One%%%%%h2.",
    "First sentence. Second sentence.",
    "A lone paragraph here.",
    "In the greenest of our valleys,%%%%%verse.",
    "By good angels tenanted,%%%%%verse.",
    "",
    "Banners yellow, glorious, golden,%%%%%verse.",
    "On its roof did float and flow.%%%%%verse.",
]

# RU additionally credits a translator the source never had (a legal asymmetric
# side mark — must not consume a body slot).
RU = [
    "Эдгар Аллан По%%%%%author.",
    "Падение дома Ашеров%%%%%title.",
    "Перевод Lingtrain%%%%%translator.",
    "Сердце его лютня.%%%%%qtext.",
    "ДЕ БЕРАНЖЕ.%%%%%qname.",
    "Глава первая%%%%%h2.",
    "Первое предложение. Второе предложение.",
    "Одинокий абзац здесь.",
    "В зеленейшей из наших долин,%%%%%verse.",
    "Где жили добрые ангелы,%%%%%verse.",
    "",
    "Жёлтые знамёна золотые,%%%%%verse.",
    "Реяли над крышей дворца.%%%%%verse.",
]

DE = [
    "Edgar Allan Poe%%%%%author.",
    "Der Untergang des Hauses Usher%%%%%title.",
    "Sein Herz ist eine Laute.%%%%%qtext.",
    "DE BERANGER.%%%%%qname.",
    "Erstes Kapitel%%%%%h2.",
    "Erster Satz. Zweiter Satz.",
    "Ein einsamer Absatz hier.",
    "Im grünsten unserer Täler,%%%%%verse.",
    "Von guten Engeln bewohnt,%%%%%verse.",
    "",
    "Gelbe Banner herrlich golden,%%%%%verse.",
    "Wehten über dem Dach.%%%%%verse.",
]

# Chinese edition (added later). zh needs clean_text=True for the splitter.
ZH = [
    "爱伦·坡%%%%%author.",
    "厄舍府的倒塌%%%%%title.",
    "他的心是一架琴。%%%%%qtext.",
    "德贝朗瑞。%%%%%qname.",
    "第一章%%%%%h2.",
    "第一句。第二句。",
    "这里是孤独的一段。",
    "在我们最翠绿的山谷，%%%%%verse.",
    "善良的天使居住其间，%%%%%verse.",
    "",
    "黄色的旗帜辉煌金色，%%%%%verse.",
    "在屋顶上飘扬。%%%%%verse.",
]


def _build_en_ru_de(tmp_path, **kwargs):
    paths = {
        "en": _write(tmp_path / "usher.en.marked.txt", EN),
        "ru": _write(tmp_path / "usher.ru.marked.txt", RU),
        "de": _write(tmp_path / "usher.de.marked.txt", DE),
    }
    out = str(tmp_path / "usher.ltm")
    report = aligner.trivial_alignment_multi(
        paths, out, source_lang="en", name="Usher", **kwargs
    )
    return out, report


# ---------------------------------------------------------------------------
# Build: happy path
# ---------------------------------------------------------------------------


class TestBuildPerfect:
    @pytest.fixture
    def built(self, tmp_path):
        return _build_en_ru_de(tmp_path)

    def test_report(self, built):
        _, report = built
        assert report["status"] == "perfect"
        assert report["merged_paragraphs"] == 0
        assert report["format"] == "ltm"
        assert report["langs"] == ["en", "ru", "de"]
        assert report["source_lang"] == "en"
        assert report["paragraphs"] == 9  # qtext,qname,h2,2 prose,4 verse
        assert report["structural_marks"] == 3  # qtext, qname, h2
        assert report["verse_lines"] == 4
        assert report["sentences_by_lang"] == {"en": 7, "ru": 7, "de": 7}

    def test_is_ltm_and_version(self, built):
        out, _ = built
        assert helper.is_ltm(out) is True
        with sqlite3.connect(out) as db:
            version = db.execute("select version from version").fetchone()[0]
            fmt = db.execute("select val from info where key='format'").fetchone()[0]
        assert version == constants.LTM_VERSION == "M1.0"
        assert fmt == "ltm"

    def test_languages_table(self, built):
        out, _ = built
        langs = helper.get_ltm_languages(out)
        assert [l["lang"] for l in langs] == ["en", "ru", "de"]
        assert [l["ord"] for l in langs] == [0, 1, 2]
        assert langs[0]["is_source"] is True
        assert langs[1]["is_source"] is False
        assert helper.get_ltm_source_lang(out) == "en"

    def test_structure_table(self, built):
        out, _ = built
        st = helper.get_ltm_structure(out)
        assert [s["paragraph"] for s in st] == list(range(1, 10))
        assert [s["kind"] for s in st] == [
            "qtext", "qname", "h2", "text", "text",
            "verse", "verse", "verse", "verse",
        ]
        # sentence_count: marks 0, prose 2 then 1, verse 1 each.
        assert [s["sentence_count"] for s in st] == [0, 0, 0, 2, 1, 1, 1, 1, 1]
        # verse stanza indices: first two lines stanza 1, last two stanza 2.
        assert [s["verse"] for s in st] == [0, 0, 0, 0, 0, 1, 1, 2, 2]

    def test_splitted_counts(self, built):
        out, _ = built
        with sqlite3.connect(out) as db:
            for lang in ("en", "ru", "de"):
                n = db.execute(
                    "select count(*) from splitted where lang=?", (lang,)
                ).fetchone()[0]
                assert n == 7
            total = db.execute("select count(*) from splitted").fetchone()[0]
        assert total == 21

    def test_meta_bare_keys_and_translator_asymmetry(self, built):
        out, _ = built
        en_meta = helper.get_ltm_meta_for_lang(out, "en")
        ru_meta = helper.get_ltm_meta_for_lang(out, "ru")
        assert set(en_meta) == {"author", "title", "qtext", "qname", "h2"}
        # RU additionally carries the translator mark; EN never did.
        assert "translator" in ru_meta
        assert "translator" not in en_meta
        assert ru_meta["translator"][0][0] == "Перевод Lingtrain"
        # Title is a bare key (not title_from / title_to) and per edition.
        assert en_meta["title"][0][0] == "The Fall of the House of Usher"
        assert ru_meta["title"][0][0] == "Падение дома Ашеров"

    def test_content_version_starts_at_one(self, built):
        out, _ = built
        with sqlite3.connect(out) as db:
            cv = db.execute(
                "select val from info where key='app_content_version'"
            ).fetchone()[0]
        assert cv == "1"

    def test_overwrites_existing(self, tmp_path):
        _build_en_ru_de(tmp_path)
        out, _ = _build_en_ru_de(tmp_path)  # second run on same path
        with sqlite3.connect(out) as db:
            assert db.execute("select count(*) from splitted").fetchone()[0] == 21
            assert db.execute("select count(*) from languages").fetchone()[0] == 3


# ---------------------------------------------------------------------------
# Single edition: a .ltm may hold ONE edition (a monolingual book) and later
# expand to a parallel one without changing the original coordinates.
# ---------------------------------------------------------------------------


class TestSingleEdition:
    @pytest.fixture
    def built(self, tmp_path):
        path = _write(tmp_path / "usher.en.marked.txt", EN)
        out = str(tmp_path / "usher.ltm")
        # build_ltm is the neutral alias for trivial_alignment_multi.
        report = aligner.build_ltm({"en": path}, out, source_lang="en", name="Usher")
        return out, report, tmp_path

    def test_report(self, built):
        _, report, _ = built
        assert report["status"] == "perfect"
        assert report["format"] == "ltm"
        assert report["langs"] == ["en"]
        assert report["source_lang"] == "en"
        assert report["paragraphs"] == 9
        assert report["sentences_by_lang"] == {"en": 7}

    def test_is_ltm_single_language(self, built):
        out, _, _ = built
        assert helper.is_ltm(out) is True
        assert helper.get_ltm_lang_codes(out) == ["en"]
        assert helper.get_ltm_source_lang(out) == "en"
        langs = helper.get_ltm_languages(out)
        assert langs[0]["is_source"] is True

    def test_reader_reads_single_edition(self, built):
        out, _, _ = built
        paragraphs, par_ids, meta_info, sent_counter, verse_map = (
            reader.get_paragraphs_multi(out)
        )
        assert set(paragraphs) == {"en"}
        assert par_ids == [4, 5, 6, 7, 8, 9]
        assert paragraphs["en"][0] == ["First sentence.", "Second sentence."]
        assert sent_counter == {"en": 7}
        assert meta_info["main_lang_code"] == "en"

    def test_add_language_upgrades_to_parallel(self, built):
        out, _, tmp_path = built
        # Capture the source edition's coordinates before expansion.
        with sqlite3.connect(out) as db:
            before = db.execute(
                "select paragraph, sentence, id, text from splitted "
                "where lang='en' order by id"
            ).fetchall()
        ru = _write(tmp_path / "usher.ru.marked.txt", RU)
        report = aligner.add_language(out, ru, "ru")
        assert report["status"] == "ok"
        assert helper.get_ltm_lang_codes(out) == ["en", "ru"]
        with sqlite3.connect(out) as db:
            after = db.execute(
                "select paragraph, sentence, id, text from splitted "
                "where lang='en' order by id"
            ).fetchall()
        # Existing (paragraph, sentence, id, text) rows are byte-identical.
        assert after == before
        paragraphs, _, _, _, _ = reader.get_paragraphs_multi(out)
        assert set(paragraphs) == {"en", "ru"}
        assert paragraphs["ru"][0] == ["Первое предложение.", "Второе предложение."]


# ---------------------------------------------------------------------------
# Single edition built from the editable prepared/splitted artifact
# ---------------------------------------------------------------------------


PREPARED = [
    "Source title%%%%%title.",
    "Source author%%%%%author.",
    "Chapter One%%%%%h2.",
    "Edited first sentence.",
    "Kept second sentence%%%%%.",
    "Joined after a deleted boundary.",
    "Still the same paragraph%%%%%!",
    "First surviving verse line%%%%%verse.",
    "Second surviving verse line%%%%%verse.",
    "A section break%%%%%h3.",
    "Third surviving verse line%%%%%verse.",
]


class TestBuildLtmFromPrepared:
    def _build(self, tmp_path, lines=PREPARED, name="Canonical title"):
        prepared = _write(tmp_path / "edited-preview.txt", lines)
        out = str(tmp_path / "prepared.ltm")
        report = aligner.build_ltm_from_prepared(
            prepared,
            out,
            source_lang="en",
            name=name,
            file_name="uploaded-source.txt",
            guid="document-guid",
        )
        return out, report

    def test_uses_prepared_sentences_and_retained_paragraph_boundaries(self, tmp_path):
        out, report = self._build(tmp_path)
        assert report["sentences_by_lang"] == {"en": 7}
        structure = helper.get_ltm_structure(out)
        assert [(row["kind"], row["sentence_count"]) for row in structure] == [
            ("h2", 0),
            ("text", 2),
            ("text", 2),
            ("verse", 1),
            ("verse", 1),
            ("h3", 0),
            ("verse", 1),
        ]
        with sqlite3.connect(out) as db:
            prose = db.execute(
                "select text from splitted where lang='en' and paragraph=2 order by sentence"
            ).fetchall()
            joined = db.execute(
                "select text from splitted where lang='en' and paragraph=3 order by sentence"
            ).fetchall()
        assert prose == [("Edited first sentence.",), ("Kept second sentence.",)]
        assert joined == [
            ("Joined after a deleted boundary.",),
            ("Still the same paragraph!",),
        ]
        assert all("%%%%%" not in row[0] for row in prose + joined)

    def test_replaces_title_preserves_marks_and_collapses_verse_runs(self, tmp_path):
        out, _ = self._build(tmp_path)
        meta = helper.get_ltm_meta_for_lang(out, "en")
        assert [(row[0], row[2]) for row in meta["title"]] == [("Canonical title", 0)]
        assert [(row[0], row[2]) for row in meta["author"]] == [("Source author", 0)]
        assert [(row[0], row[2]) for row in meta["h2"]] == [("Chapter One", 1)]
        assert [(row[0], row[2]) for row in meta["h3"]] == [("A section break", 6)]
        with sqlite3.connect(out) as db:
            verses = db.execute(
                "select paragraph, text, verse from splitted where verse > 0 order by id"
            ).fetchall()
        assert verses == [
            (4, "First surviving verse line", 1),
            (5, "Second surviving verse line", 1),
            (7, "Third surviving verse line", 2),
        ]

    def test_inserts_canonical_title_when_source_has_none(self, tmp_path):
        out, _ = self._build(
            tmp_path,
            ["Author%%%%%author.", "Only edited sentence.%%%%%."],
            name="Inserted title",
        )
        meta = helper.get_ltm_meta_for_lang(out, "en")
        assert [(row[0], row[2]) for row in meta["title"]] == [("Inserted title", 0)]

    def test_strips_bare_paragraph_marker_after_unpunctuated_text(self, tmp_path):
        out, _ = self._build(
            tmp_path,
            ["First line without punctuation%%%%%", "Second paragraph%%%%%."],
        )
        with sqlite3.connect(out) as db:
            rows = db.execute(
                "select paragraph, text from splitted order by id"
            ).fetchall()
        assert rows == [(1, "First line without punctuation"), (2, "Second paragraph.")]

    def test_classifies_marks_wrapped_by_a_retained_paragraph_boundary(self, tmp_path):
        out, _ = self._build(
            tmp_path,
            [
                "Source title%%%%%title.%%%%%",
                "Source author%%%%%author.%%%%%",
                "Chapter One%%%%%h2.%%%%%",
                "A verse line%%%%%verse.%%%%%",
                "Body sentence%%%%%.",
            ],
            name="Canonical wrapped title",
        )
        with sqlite3.connect(out) as db:
            meta = db.execute(
                "select key, val from meta order by id"
            ).fetchall()
            structure = db.execute(
                "select kind, sentence_count, verse from structure order by paragraph"
            ).fetchall()
            sentence_texts = [
                row[0] for row in db.execute("select text from splitted order by id")
            ]

        assert meta == [
            ("title", "Canonical wrapped title"),
            ("author", "Source author"),
            ("h2", "Chapter One"),
        ]
        assert structure == [("h2", 0, 0), ("verse", 1, 1), ("text", 1, 0)]
        assert sentence_texts == ["A verse line", "Body sentence."]
        assert all("%%%%%" not in text for text in sentence_texts)

    def test_records_provenance_content_version_and_distinct_history(self, tmp_path):
        out, _ = self._build(tmp_path)
        with sqlite3.connect(out) as db:
            assert db.execute(
                "select lang, name, guid from files"
            ).fetchone() == ("en", "uploaded-source.txt", "document-guid")
            assert db.execute(
                "select val from info where key='source_lang'"
            ).fetchone()[0] == "en"
            assert db.execute(
                "select val from info where key='app_content_version'"
            ).fetchone()[0] == "1"
            assert db.execute(
                "select operation from history"
            ).fetchone()[0] == constants.OPERATION_BUILD_LTM_FROM_PREPARED

    def test_rejects_empty_or_metadata_only_prepared_input(self, tmp_path):
        empty = _write(tmp_path / "empty.txt", [])
        with pytest.raises(ValueError, match="no readable content"):
            aligner.build_ltm_from_prepared(
                empty, str(tmp_path / "empty.ltm"), source_lang="en", name="Book"
            )

        metadata_only = _write(tmp_path / "meta.txt", ["Title%%%%%title."])
        with pytest.raises(ValueError, match="no readable body content"):
            aligner.build_ltm_from_prepared(
                metadata_only,
                str(tmp_path / "meta.ltm"),
                source_lang="en",
                name="Book",
            )

    def test_output_remains_compatible_with_add_language(self, tmp_path):
        out, _ = self._build(tmp_path)
        translation = _write(
            tmp_path / "translation.de.txt",
            [
                "Deutscher Titel%%%%%title.",
                "Autor%%%%%author.",
                "Kapitel eins%%%%%h2.",
                "Bearbeiteter erster Satz. Zweiter Satz.",
                "Nach einer gelöschten Grenze. Noch derselbe Absatz!",
                "Erste Verszeile%%%%%verse.",
                "Zweite Verszeile%%%%%verse.",
                "Abschnitt%%%%%h3.",
                "Dritte Verszeile%%%%%verse.",
            ],
        )
        report = aligner.add_language(out, translation, "de")
        assert report["status"] == "ok"
        assert helper.get_ltm_lang_codes(out) == ["en", "de"]


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


class TestReader:
    @pytest.fixture
    def built(self, tmp_path):
        return _build_en_ru_de(tmp_path)

    def test_get_paragraphs_multi_all(self, built):
        out, _ = built
        paragraphs, par_ids, meta_info, sent_counter, verse_map = (
            reader.get_paragraphs_multi(out)
        )
        assert par_ids == [4, 5, 6, 7, 8, 9]
        assert paragraphs["en"][0] == ["First sentence.", "Second sentence."]
        assert paragraphs["en"][1] == ["A lone paragraph here."]
        assert paragraphs["en"][2] == ["In the greenest of our valleys,"]
        assert paragraphs["ru"][0] == ["Первое предложение.", "Второе предложение."]
        assert paragraphs["de"][1] == ["Ein einsamer Absatz hier."]
        assert verse_map == {6: 1, 7: 1, 8: 2, 9: 2}
        assert sent_counter == {"en": 7, "ru": 7, "de": 7}
        assert meta_info["main_lang_code"] == "en"
        assert meta_info["items"]["ru"]["title"][0][0] == "Падение дома Ашеров"

    def test_subset(self, built):
        out, _ = built
        paragraphs, par_ids, _, _, _ = reader.get_paragraphs_multi(out, ["en", "de"])
        assert set(paragraphs) == {"en", "de"}
        assert par_ids == [4, 5, 6, 7, 8, 9]

    def test_as_pair_projection(self, built):
        out, _ = built
        pair, par_ids, meta_info, sent, verse_map = (
            reader.get_paragraphs_multi_as_pair(out, "en", "ru")
        )
        assert set(pair) == {"from", "to"}
        assert pair["from"][0] == ["First sentence.", "Second sentence."]
        assert pair["to"][0] == ["Первое предложение.", "Второе предложение."]
        assert meta_info["main_lang_code"] == "from"
        assert sent == {"from": 7, "to": 7}

    def test_invalid_subset_raises(self, built):
        out, _ = built
        with pytest.raises(ValueError):
            reader.get_paragraphs_multi(out, ["xx"])


# ---------------------------------------------------------------------------
# add_language (the expandability path)
# ---------------------------------------------------------------------------


class TestAddLanguage:
    @pytest.fixture
    def built(self, tmp_path):
        out, _ = _build_en_ru_de(tmp_path)
        return out, tmp_path

    def test_add_zh_is_additive(self, built):
        out, tmp_path = built
        zh = _write(tmp_path / "usher.zh.marked.txt", ZH)
        report = aligner.add_language(out, zh, "zh", clean_text=True)
        assert report["status"] == "ok"
        assert report["replaced"] is False
        assert report["sentences"] == 7
        # languages now en, ru, de, zh; the source rows are untouched.
        assert helper.get_ltm_lang_codes(out) == ["en", "ru", "de", "zh"]
        with sqlite3.connect(out) as db:
            for lang in ("en", "ru", "de"):
                assert db.execute(
                    "select count(*) from splitted where lang=?", (lang,)
                ).fetchone()[0] == 7
            assert db.execute(
                "select count(*) from splitted where lang='zh'"
            ).fetchone()[0] == 7
            cv = db.execute(
                "select val from info where key='app_content_version'"
            ).fetchone()[0]
        # content version bumped from 1 -> 2 (cache invalidation on re-upload).
        assert cv == "2"
        # The new edition reads back through the same structure.
        paragraphs, par_ids, _, _, _ = reader.get_paragraphs_multi(out, ["en", "zh"])
        assert par_ids == [4, 5, 6, 7, 8, 9]
        assert paragraphs["zh"][1] == ["这里是孤独的一段。"]

    def test_duplicate_without_replace_raises(self, built):
        out, tmp_path = built
        ru2 = _write(tmp_path / "usher.ru2.marked.txt", RU)
        with pytest.raises(TrivialAlignmentError, match="already exists"):
            aligner.add_language(out, ru2, "ru")

    def test_replace_is_delete_then_insert(self, built):
        out, tmp_path = built
        ru_fixed = list(RU)
        ru_fixed[1] = "Падение дома Ашеров (испр.)%%%%%title."
        ru2 = _write(tmp_path / "usher.ru.fixed.marked.txt", ru_fixed)
        report = aligner.add_language(out, ru2, "ru", replace=True)
        assert report["replaced"] is True
        # Still exactly one ru edition, with the corrected title.
        assert helper.get_ltm_lang_codes(out) == ["en", "ru", "de"]
        with sqlite3.connect(out) as db:
            assert db.execute(
                "select count(*) from splitted where lang='ru'"
            ).fetchone()[0] == 7
        assert helper.get_ltm_meta_for_lang(out, "ru")["title"][0][0] == (
            "Падение дома Ашеров (испр.)"
        )

    def test_structure_mismatch_rejected(self, built):
        out, tmp_path = built
        # A zh edition whose first prose paragraph splits into 1, not 2.
        bad = list(ZH)
        bad[5] = "第一句第二句没有句号"  # splits into 1
        bad_path = _write(tmp_path / "usher.zh.bad.marked.txt", bad)
        with pytest.raises(TrivialAlignmentError, match="Sentence-count mismatch"):
            aligner.add_language(out, bad_path, "zh", clean_text=True)

    def test_body_count_mismatch_rejected(self, built):
        out, tmp_path = built
        short = list(ZH)
        short.pop()  # drop a verse line -> body count differs
        bad = _write(tmp_path / "usher.zh.short.marked.txt", short)
        with pytest.raises(TrivialAlignmentError, match="Body line count mismatch"):
            aligner.add_language(out, bad, "zh", clean_text=True)

    def test_merge_refused(self, built):
        out, tmp_path = built
        zh = _write(tmp_path / "usher.zh.marked.txt", ZH)
        with pytest.raises(ValueError, match="on_mismatch"):
            aligner.add_language(out, zh, "zh", clean_text=True, on_mismatch="merge")

    def test_not_ltm_rejected(self, tmp_path):
        # A plain .lt must not be accepted by add_language.
        f = _write(tmp_path / "from.txt", ["T%%%%%title.", "A sentence."])
        t = _write(tmp_path / "to.txt", ["Z%%%%%title.", "Предложение."])
        lt = str(tmp_path / "book.lt")
        aligner.trivial_alignment(f, t, "en", "ru", lt)
        extra = _write(tmp_path / "extra.txt", ["X%%%%%title.", "Satz."])
        with pytest.raises(TrivialAlignmentError, match="not a multilingual"):
            aligner.add_language(lt, extra, "de")


# ---------------------------------------------------------------------------
# Sentence-count mismatch across editions
# ---------------------------------------------------------------------------

MM_EN = ["T%%%%%title.", "First sentence. Second sentence.", "Tail paragraph."]
MM_RU = ["Z%%%%%title.", "Первое предложение. Второе предложение.", "Хвостовой абзац."]
MM_DE = ["D%%%%%title.", "Ein Satz ohne Punkt drin", "Schlussabsatz."]  # 1 sentence


class TestMismatch:
    def _paths(self, tmp_path):
        return {
            "en": _write(tmp_path / "mm.en.txt", MM_EN),
            "ru": _write(tmp_path / "mm.ru.txt", MM_RU),
            "de": _write(tmp_path / "mm.de.txt", MM_DE),
        }

    def test_strict_raises(self, tmp_path):
        paths = self._paths(tmp_path)
        out = str(tmp_path / "mm.ltm")
        with pytest.raises(TrivialAlignmentError, match="Sentence-count mismatch"):
            aligner.trivial_alignment_multi(paths, out, source_lang="en")

    def test_merge_collapses_paragraph(self, tmp_path):
        paths = self._paths(tmp_path)
        out = str(tmp_path / "mm.ltm")
        report = aligner.trivial_alignment_multi(
            paths, out, source_lang="en", on_mismatch="merge"
        )
        assert report["status"] == "merged"
        assert report["merged_paragraphs"] == 1
        st = helper.get_ltm_structure(out)
        # Two prose paragraphs; the first was merged to a single unit (sc=1).
        text_rows = [s for s in st if s["kind"] == "text"]
        assert text_rows[0]["sentence_count"] == 1
        assert text_rows[1]["sentence_count"] == 1
        with sqlite3.connect(out) as db:
            # en merged paragraph stored as one joined row.
            rows = db.execute(
                "select text from splitted where lang='en' and paragraph=?",
                (text_rows[0]["paragraph"],),
            ).fetchall()
        assert rows == [("First sentence. Second sentence.",)]


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------


class TestStructuralValidation:
    def test_body_count_mismatch(self, tmp_path):
        paths = {
            "en": _write(tmp_path / "a.en.txt", ["One. Two.", "Three."]),
            "ru": _write(tmp_path / "a.ru.txt", ["Один."]),
        }
        out = str(tmp_path / "a.ltm")
        with pytest.raises(TrivialAlignmentError, match="Body line count mismatch"):
            aligner.trivial_alignment_multi(paths, out, source_lang="en")

    def test_mark_mismatch(self, tmp_path):
        paths = {
            "en": _write(tmp_path / "b.en.txt", ["Heading%%%%%h2.", "Body."]),
            "ru": _write(tmp_path / "b.ru.txt", ["Заголовок%%%%%h3.", "Тело."]),
        }
        out = str(tmp_path / "b.ltm")
        with pytest.raises(TrivialAlignmentError, match="Markup mismatch"):
            aligner.trivial_alignment_multi(paths, out, source_lang="en")

    def test_zero_editions_rejected(self, tmp_path):
        out = str(tmp_path / "c.ltm")
        with pytest.raises(ValueError, match="at least 1 edition"):
            aligner.trivial_alignment_multi({}, out, source_lang="en")

    def test_unknown_source_lang(self, tmp_path):
        paths = {
            "en": _write(tmp_path / "d.en.txt", ["A."]),
            "ru": _write(tmp_path / "d.ru.txt", ["Б."]),
        }
        out = str(tmp_path / "d.ltm")
        with pytest.raises(ValueError, match="source_lang"):
            aligner.trivial_alignment_multi(paths, out, source_lang="zz")

    def test_invalid_on_mismatch(self, tmp_path):
        paths = {
            "en": _write(tmp_path / "e.en.txt", ["A."]),
            "ru": _write(tmp_path / "e.ru.txt", ["Б."]),
        }
        out = str(tmp_path / "e.ltm")
        with pytest.raises(ValueError, match="on_mismatch"):
            aligner.trivial_alignment_multi(paths, out, source_lang="en", on_mismatch="nope")


# ---------------------------------------------------------------------------
# .lt regression: the bilingual path is untouched
# ---------------------------------------------------------------------------


class TestLtUntouched:
    def test_lt_build_and_read_unchanged(self, tmp_path):
        frm = _write(tmp_path / "from.txt", ["T%%%%%title.", "First. Second.", "Tail."])
        to = _write(tmp_path / "to.txt", ["Z%%%%%title.", "Первое. Второе.", "Хвост."])
        out = str(tmp_path / "book.lt")
        report = aligner.trivial_alignment(frm, to, "en", "ru", out)
        assert report["status"] == "perfect"
        # A .lt is NOT detected as a multibook.
        assert helper.is_ltm(out) is False
        # The plain reader still works.
        paragraphs, par_ids, meta_info, sent_counter = reader.get_paragraphs(out)
        assert sent_counter["from"] == sent_counter["to"]
