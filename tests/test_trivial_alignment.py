"""Tests for aligner.trivial_alignment — embedding-free 1:1 alignment of two
structurally parallel marked texts."""

import json
import sqlite3
from pathlib import Path

import pytest

from lingtrain_aligner import aligner, helper, reader
from lingtrain_aligner.aligner import TrivialAlignmentError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


# Constants mirrored from the web app (be/app/config.py + AlignmentState) so we
# can assert an exported .lt imports as a *completed* alignment.
APP_BATCH_SIZE = 200
APP_MAX_BATCHES = 2000
STATE_INIT, STATE_IN_PROGRESS_DONE, STATE_DONE = 0, 2, 3


def _simulate_import_state(db_path, batch_size=APP_BATCH_SIZE, max_batches=APP_MAX_BATCHES):
    """Replicate be.alignment_service.upload_alignment's batch/state inference
    for a file with no app-metadata info keys (a fresh library export)."""
    with sqlite3.connect(db_path) as db:
        len_from = db.execute("select count(*) from splitted_from").fetchone()[0]
        batch_ids = [r[0] for r in db.execute("select batch_id from batches").fetchall()]

    is_last = len_from % batch_size > 0
    inferred_total = len_from // batch_size + (1 if is_last else 0)
    if max_batches > 0:
        inferred_total = min(max_batches, inferred_total)
    inferred_curr = len(batch_ids)

    curr = min(inferred_curr, inferred_curr)
    total = max(inferred_total, inferred_curr)
    if total > 0:
        curr = max(0, min(curr, total))

    if curr > 0 and total > 0 and curr >= total:
        state = STATE_DONE
    elif curr > 0:
        state = STATE_IN_PROGRESS_DONE
    else:
        state = STATE_INIT
    return {"curr": curr, "total": total, "state": state, "batch_ids": sorted(batch_ids)}


# A minimal pair of structurally identical marked texts (perfect 1:1 split).
PERFECT_FROM = [
    "Some Author%%%%%author.",
    "Some Title%%%%%title.",
    "Chapter One%%%%%h2.",
    "First sentence. Second sentence.",
    "A lone paragraph here.",
    "Chapter Two%%%%%h2.",
    "Third sentence. Fourth sentence. Fifth one.",
]
PERFECT_TO = [
    "Некий Автор%%%%%author.",
    "Некое Название%%%%%title.",
    "Глава Один%%%%%h2.",
    "Первое предложение. Второе предложение.",
    "Одинокий абзац здесь.",
    "Глава Два%%%%%h2.",
    "Третье предложение. Четвёртое предложение. Пятое.",
]


# ---------------------------------------------------------------------------
# Happy path: perfect 1:1
# ---------------------------------------------------------------------------


class TestPerfectAlignment:
    @pytest.fixture
    def aligned(self, tmp_path):
        f = _write(tmp_path / "from.txt", PERFECT_FROM)
        t = _write(tmp_path / "to.txt", PERFECT_TO)
        out = str(tmp_path / "book.lt")
        report = aligner.trivial_alignment(f, t, "en", "ru", out, name="Demo")
        return out, report

    def test_status_perfect(self, aligned):
        _, report = aligned
        assert report["status"] == "perfect"
        assert report["merged_paragraphs"] == 0
        assert report["lines"] == len(PERFECT_FROM)
        assert report["from_sentences"] == report["to_sentences"]
        # 2 + 1 + 3 regular sentences; meta lines are extracted separately
        assert report["from_sentences"] == 6
        assert report["units"] == 6
        assert report["one_to_one"] == 6
        assert report["meta"] == {"author": 1, "title": 1, "h2": 2}

    def test_db_tables(self, aligned):
        out, _ = aligned
        with sqlite3.connect(out) as db:
            sf = db.execute("select count(*) from splitted_from").fetchone()[0]
            st = db.execute("select count(*) from splitted_to").fetchone()[0]
            pf = db.execute("select count(*) from processing_from").fetchone()[0]
            pt = db.execute("select count(*) from processing_to").fetchone()[0]
            nb = db.execute("select count(*) from batches").fetchone()[0]
            nh = db.execute("select count(*) from history").fetchone()[0]
            lang_from = db.execute(
                "select val from languages where key='from'"
            ).fetchone()[0]
            lang_to = db.execute(
                "select val from languages where key='to'"
            ).fetchone()[0]
            name = helper.get_name(out)
        assert sf == st == pf == pt == 6
        assert nb == 1
        assert nh == 1  # one history row for the single trivial batch
        assert lang_from == "en"
        assert lang_to == "ru"
        assert name == "Demo"

    def test_doc_index_is_one_to_one(self, aligned):
        out, _ = aligned
        index = helper.get_doc_index_original(out)
        assert len(index) == 1  # single batch
        batch = index[0]
        assert len(batch) == 6
        for item in batch:
            from_ids = json.loads(item[1])
            to_ids = json.loads(item[3])
            assert len(from_ids) == 1 and len(to_ids) == 1

    def test_read_processing_aligned(self, aligned):
        out, _ = aligned
        pf, pt = helper.read_processing(out)
        assert len(pf) == len(pt) == 6
        assert pf[0] == "First sentence."
        assert pt[0] == "Первое предложение."

    def test_meta_par_ids_match(self, aligned):
        """Structurally identical texts must produce identical meta paragraph
        ids on both sides (titles/authors/headings line up)."""
        out, _ = aligned
        meta = helper.get_meta_dict(out)
        for key in ("author", "title", "h2"):
            from_pars = sorted(x[2] for x in meta[f"{key}_from"])
            to_pars = sorted(x[2] for x in meta[f"{key}_to"])
            assert from_pars == to_pars

    def test_no_empty_cells(self, aligned):
        out, _ = aligned
        assert reader.is_empty_cells(out) is False

    def test_overwrites_existing(self, tmp_path):
        f = _write(tmp_path / "from.txt", PERFECT_FROM)
        t = _write(tmp_path / "to.txt", PERFECT_TO)
        out = str(tmp_path / "book.lt")
        aligner.trivial_alignment(f, t, "en", "ru", out)
        # Running again on the same path must not append/duplicate.
        aligner.trivial_alignment(f, t, "en", "ru", out)
        with sqlite3.connect(out) as db:
            assert db.execute("select count(*) from splitted_from").fetchone()[0] == 6


# ---------------------------------------------------------------------------
# Per-paragraph sentence mismatch
# ---------------------------------------------------------------------------

MISMATCH_FROM = [
    "T%%%%%title.",
    "First sentence. Second sentence.",  # 2 sentences
    "Plain tail paragraph.",
]
MISMATCH_TO = [
    "Z%%%%%title.",
    "Только одно предложение без точек внутри",  # 1 sentence
    "Простой хвостовой абзац.",
]


class TestMismatchMerge:
    def test_merge_default(self, tmp_path):
        f = _write(tmp_path / "from.txt", MISMATCH_FROM)
        t = _write(tmp_path / "to.txt", MISMATCH_TO)
        out = str(tmp_path / "book.lt")
        report = aligner.trivial_alignment(f, t, "en", "ru", out)  # on_mismatch="merge"

        assert report["status"] == "merged"
        assert report["merged_paragraphs"] == 1
        assert report["merged_details"][0]["from_count"] == 2
        assert report["merged_details"][0]["to_count"] == 1

        index = helper.get_doc_index_original(out)
        batch = index[0]
        # One merged unit (2 from-ids -> 1 to-id) + one clean 1:1 tail unit.
        merged_units = [
            it for it in batch if len(json.loads(it[1])) != 1 or len(json.loads(it[3])) != 1
        ]
        assert len(merged_units) == 1
        merged = merged_units[0]
        assert len(json.loads(merged[1])) == 2
        assert len(json.loads(merged[3])) == 1

    def test_strict_raises(self, tmp_path):
        f = _write(tmp_path / "from.txt", MISMATCH_FROM)
        t = _write(tmp_path / "to.txt", MISMATCH_TO)
        out = str(tmp_path / "book.lt")
        with pytest.raises(TrivialAlignmentError, match="Sentence-count mismatch"):
            aligner.trivial_alignment(f, t, "en", "ru", out, on_mismatch="error")


# ---------------------------------------------------------------------------
# Batch layout / web-app import readiness
# ---------------------------------------------------------------------------


class TestBatchingForImport:
    def _make(self, tmp_path, n_paragraphs, batch_size=APP_BATCH_SIZE):
        # n single-sentence paragraphs on each side (title mark on top).
        frm = ["T%%%%%title."] + [f"Sentence number {i}." for i in range(n_paragraphs)]
        to = ["Z%%%%%title."] + [f"Предложение номер {i}." for i in range(n_paragraphs)]
        f = _write(tmp_path / "from.txt", frm)
        t = _write(tmp_path / "to.txt", to)
        out = str(tmp_path / "book.lt")
        report = aligner.trivial_alignment(
            f, t, "en", "ru", out, batch_size=batch_size
        )
        return out, report

    def test_multiple_dense_batches(self, tmp_path):
        out, report = self._make(tmp_path, 450)  # -> ceil(450/200) = 3 batches
        assert report["from_sentences"] == 450
        assert report["batches"] == 3
        index = helper.get_doc_index_original(out)
        assert len(index) == 3
        # Dense batch ids 0..2, every batch non-empty.
        with sqlite3.connect(out) as db:
            batch_ids = sorted(r[0] for r in db.execute("select batch_id from batches"))
        assert batch_ids == [0, 1, 2]
        assert all(len(b) > 0 for b in index)

    def test_imports_as_done(self, tmp_path):
        out, _ = self._make(tmp_path, 450)
        sim = _simulate_import_state(out)
        assert sim["curr"] == sim["total"] == 3
        assert sim["state"] == STATE_DONE
        assert sim["batch_ids"] == [0, 1, 2]

    def test_single_batch_small(self, tmp_path):
        out, report = self._make(tmp_path, 10)
        assert report["batches"] == 1
        sim = _simulate_import_state(out)
        assert sim["state"] == STATE_DONE  # curr == total == 1

    def test_processing_pairing_across_batches(self, tmp_path):
        """doc_index from/to ids must stay correctly paired across batch joins."""
        out, _ = self._make(tmp_path, 450)
        pf, pt = helper.read_processing(out)
        assert len(pf) == len(pt) == 450
        # Each from sentence i pairs with to sentence i (same index value).
        for i in (0, 199, 200, 201, 449):
            assert pf[i] == f"Sentence number {i}."
            assert pt[i] == f"Предложение номер {i}."


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------


class TestStructuralValidation:
    def test_line_count_mismatch_raises(self, tmp_path):
        f = _write(tmp_path / "from.txt", ["One paragraph.", "Two paragraph."])
        t = _write(tmp_path / "to.txt", ["Один абзац."])
        out = str(tmp_path / "book.lt")
        with pytest.raises(TrivialAlignmentError, match="count mismatch"):
            aligner.trivial_alignment(f, t, "en", "ru", out)

    def test_mark_mismatch_raises(self, tmp_path):
        # Same number of lines, but a mark differs (h2 vs h3) on line 1.
        f = _write(tmp_path / "from.txt", ["Heading%%%%%h2.", "Body sentence."])
        t = _write(tmp_path / "to.txt", ["Заголовок%%%%%h3.", "Тело предложение."])
        out = str(tmp_path / "book.lt")
        with pytest.raises(TrivialAlignmentError, match="Markup mismatch"):
            aligner.trivial_alignment(f, t, "en", "ru", out)

    def test_invalid_on_mismatch(self, tmp_path):
        f = _write(tmp_path / "from.txt", ["A."])
        t = _write(tmp_path / "to.txt", ["Б."])
        out = str(tmp_path / "book.lt")
        with pytest.raises(ValueError, match="on_mismatch"):
            aligner.trivial_alignment(f, t, "en", "ru", out, on_mismatch="nope")


# ---------------------------------------------------------------------------
# Integration with the real Sea-Wolf samples (if available)
# ---------------------------------------------------------------------------

_SUBMODULE_ROOT = Path(__file__).resolve().parents[1]
_PARENT_REPO = _SUBMODULE_ROOT.parent
_SAMPLE_DIR = _PARENT_REPO / "samples" / "texts" / "unmarked"
EN_SAMPLE = _SAMPLE_DIR / "london_en.marked.txt"
RU_SAMPLE = _SAMPLE_DIR / "london_en.marked.txt.ru.marked.txt"


@pytest.mark.skipif(
    not (EN_SAMPLE.exists() and RU_SAMPLE.exists()),
    reason="Sea-Wolf samples not available",
)
class TestRealSamples:
    def test_perfect_trivial_alignment(self, tmp_path):
        out = str(tmp_path / "london.lt")
        report = aligner.trivial_alignment(
            str(EN_SAMPLE), str(RU_SAMPLE), "en", "ru", out, name="Sea-Wolf"
        )
        assert report["lines"] == 2173
        assert report["meta"] == {"author": 1, "title": 1, "h2": 39, "divider": 1}
        # Self-translated, structure-preserving text aligns perfectly 1:1.
        assert report["status"] == "perfect"
        assert report["from_sentences"] == report["to_sentences"]
        assert reader.is_empty_cells(out) is False

        pf, pt = helper.read_processing(out)
        assert len(pf) == len(pt) == report["units"]

        # The exported .lt must import into the web app as a completed alignment.
        sim = _simulate_import_state(out)
        assert sim["state"] == STATE_DONE
        assert sim["curr"] == sim["total"]
        assert sim["batch_ids"] == list(range(report["batches"]))
