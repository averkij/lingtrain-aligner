"""Tests for the `verse` poetry mark — atomic, one-row-per-line body units with a
stanza index, aligned 1:1 by `aligner.trivial_alignment` and never fused by the
splitter."""

import sqlite3

import pytest

from lingtrain_aligner import aligner, helper, preprocessor, reader, splitter
from lingtrain_aligner.aligner import TrivialAlignmentError


def _write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


# A JP poem: title/author, one chapter (h2), two stanzas of two verse lines each
# (blank line = stanza break), then a normal two-sentence prose paragraph.
POEM_FROM = [
    "海の歌%%%%%title.",
    "宮沢賢治%%%%%author.",
    "",
    "第一%%%%%h2.",
    "",
    "うみは　ひろいな%%%%%verse.",
    "おおきいな%%%%%verse.",
    "",
    "つきが　のぼるし%%%%%verse.",
    "ひがしずむ%%%%%verse.",
    "",
    "これは普通の散文です。二つ目の文です。",
]
POEM_TO = [
    "Sea Song%%%%%title.",
    "Kenji%%%%%author.",
    "",
    "One%%%%%h2.",
    "",
    "the sea is wide%%%%%verse.",
    "and very big%%%%%verse.",
    "",
    "the moon climbs up%%%%%verse.",
    "the sun goes down%%%%%verse.",
    "",
    "This is prose. Second sentence.",
]


def test_verse_aligns_1to1_with_stanza_index(tmp_path):
    sp = _write(tmp_path / "s.txt", POEM_FROM)
    tp = _write(tmp_path / "t.txt", POEM_TO)
    out = str(tmp_path / "o.lt")
    rep = aligner.trivial_alignment(sp, tp, "ja", "en", out, on_mismatch="error")

    assert rep["status"] == "perfect"
    assert rep["merged_paragraphs"] == 0
    # 4 verse rows + 2 prose sentences = 6 aligned rows on each side.
    assert rep["from_sentences"] == 6
    assert rep["to_sentences"] == 6

    with sqlite3.connect(out) as db:
        rows_f = db.execute(
            "select paragraph, verse, text from splitted_from order by id"
        ).fetchall()
        n_to = db.execute("select count(*) from splitted_to").fetchone()[0]

    assert n_to == 6
    # Each verse line is its own paragraph; verse column = stanza index.
    assert [r[1] for r in rows_f] == [1, 1, 2, 2, 0, 0]
    # Every verse line is a distinct paragraph id (never merged with a neighbour).
    verse_pars = [r[0] for r in rows_f if r[1]]
    assert len(set(verse_pars)) == 4
    # The %%%%%verse. mark is stripped from the stored text.
    assert rows_f[0][2] == "うみは　ひろいな"
    assert "%%%%%" not in rows_f[0][2]


def test_fresh_db_has_verse_column_and_version_74(tmp_path):
    sp = _write(tmp_path / "s.txt", POEM_FROM)
    tp = _write(tmp_path / "t.txt", POEM_TO)
    out = str(tmp_path / "o.lt")
    aligner.trivial_alignment(sp, tp, "ja", "en", out)
    with sqlite3.connect(out) as db:
        assert db.execute("select version from version").fetchone()[0] == "7.4"
        for table in ("splitted_from", "splitted_to"):
            cols = [c[1] for c in db.execute(f"PRAGMA table_info({table})")]
            assert "verse" in cols


def test_get_verse_map(tmp_path):
    sp = _write(tmp_path / "s.txt", POEM_FROM)
    tp = _write(tmp_path / "t.txt", POEM_TO)
    out = str(tmp_path / "o.lt")
    aligner.trivial_alignment(sp, tp, "ja", "en", out)
    vm = reader.get_verse_map(out, "from")
    # Four verse paragraphs across two stanzas.
    assert sorted(vm.values()) == [1, 1, 2, 2]
    # Prose paragraphs never appear in the verse map.
    assert len(vm) == 4


def test_reader_renders_verse_cells_and_stanza_break(tmp_path):
    sp = _write(tmp_path / "s.txt", POEM_FROM)
    tp = _write(tmp_path / "t.txt", POEM_TO)
    out = str(tmp_path / "o.lt")
    html = str(tmp_path / "book.html")
    aligner.trivial_alignment(sp, tp, "ja", "en", out)
    vm = reader.get_verse_map(out, "from")
    paragraphs, delimeters, metas, sent_counter = reader.get_paragraphs(out, "from")
    reader.create_book(["from", "to"], paragraphs, delimeters, metas,
                       sent_counter, html, template="", verse_map=vm)
    h = open(html, encoding="utf-8").read()
    assert "dt-cell verse" in h            # verse cells tagged
    assert "dt-row stanza-break" in h      # exactly one stanza gap rendered
    assert h.count("dt-row stanza-break") == 1


def test_splitter_does_not_fuse_verse_lines():
    # Without the verse mark these lines (no terminal punctuation) would join into
    # ONE segment; with it each verse line stays its own segment.
    lines = ["うみは%%%%%verse.", "つきが%%%%%verse.", "ひが%%%%%verse."]
    work = preprocessor.mark_paragraphs(list(lines))
    segs = splitter.split_by_sentences_wrapper(work, "ja")
    assert len(segs) == 3


def test_handle_marks_strips_verse_and_sets_stanza():
    lines = [
        "a%%%%%verse.",
        "b%%%%%verse.",
        "",
        "c%%%%%verse.",
    ]
    work = preprocessor.mark_paragraphs(list(lines))
    res, meta, meta_par_ids = aligner.handle_marks(work)
    # three content rows, verse mark stripped, stanza index in the 8th marks slot
    assert [r[0] for r in res] == ["a", "b", "c"]
    assert [r[1][7] for r in res] == [1, 1, 2]
    # verse never lifted into the meta table
    assert "verse" not in meta


def test_mixed_prose_and_verse(tmp_path):
    src = [
        "T%%%%%title.",
        "A normal prose paragraph here. It has two sentences.",
        "a verse line%%%%%verse.",
        "another verse line%%%%%verse.",
        "Back to prose now.",
    ]
    tgt = [
        "T%%%%%title.",
        "Обычный прозаический абзац здесь. В нём два предложения.",
        "строка стиха%%%%%verse.",
        "ещё строка стиха%%%%%verse.",
        "Снова проза.",
    ]
    sp = _write(tmp_path / "s.txt", src)
    tp = _write(tmp_path / "t.txt", tgt)
    out = str(tmp_path / "o.lt")
    rep = aligner.trivial_alignment(sp, tp, "en", "ru", out, on_mismatch="error")
    assert rep["status"] == "perfect"
    with sqlite3.connect(out) as db:
        rows = db.execute("select verse, text from splitted_from order by id").fetchall()
    # 2 prose sentences (verse=0), 2 verse lines (verse>0), 1 prose sentence (0)
    assert [r[0] for r in rows] == [0, 0, 1, 1, 0]


def test_verse_vs_prose_mismatch_raises(tmp_path):
    sp = _write(tmp_path / "s.txt", ["x%%%%%verse."])
    tp = _write(tmp_path / "t.txt", ["x"])  # prose on the other side
    out = str(tmp_path / "o.lt")
    with pytest.raises(TrivialAlignmentError):
        aligner.trivial_alignment(sp, tp, "en", "en", out)


def test_migration_adds_verse_column_to_old_db(tmp_path):
    sp = _write(tmp_path / "s.txt", POEM_FROM)
    tp = _write(tmp_path / "t.txt", POEM_TO)
    out = str(tmp_path / "o.lt")
    aligner.trivial_alignment(sp, tp, "ja", "en", out)

    # Simulate a pre-7.4 DB: rebuild splitted_* without the verse column and set
    # the version back to 7.3 (recreate-table works on every SQLite version).
    with sqlite3.connect(out) as db:
        for t in ("splitted_from", "splitted_to"):
            cols = [c[1] for c in db.execute(f"PRAGMA table_info({t})") if c[1] != "verse"]
            collist = ",".join(cols)
            db.execute(f"create table {t}_old as select {collist} from {t}")
            db.execute(f"drop table {t}")
            db.execute(f"alter table {t}_old rename to {t}")
        db.execute("update version set version='7.3'")
        assert "verse" not in [c[1] for c in db.execute("PRAGMA table_info(splitted_from)")]

    helper.migrate_document_db(out)

    with sqlite3.connect(out) as db:
        assert "verse" in [c[1] for c in db.execute("PRAGMA table_info(splitted_from)")]
        assert "verse" in [c[1] for c in db.execute("PRAGMA table_info(splitted_to)")]
        assert db.execute("select version from version").fetchone()[0] == "7.4"
        # migrated rows default to 0 (prose) — safe, never crashes a reader.
        assert reader.get_verse_map(out, "from") == {}
