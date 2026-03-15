"""Regression tests for batch and SQLite row ID consistency."""

import tempfile
from unittest.mock import patch

import numpy as np

from lingtrain_aligner import aligner, helper


def _fake_vectors(lines, *args, **kwargs):
    """Return deterministic vectors without loading a real embedding model."""
    base = np.array([1.0, 0.5], dtype=np.float32)
    return [base.copy() for _ in lines]


def test_get_batch_intersected_covers_last_db_row():
    batches = list(
        aligner.get_batch_intersected(
            ["from"] * 5,
            ["to"] * 5,
            n=2,
            window=1,
            batch_ids=[0, 1, 2],
        )
    )

    covered_from = sorted({line_id for *_, line_ids_from, _, _ in batches for line_id in line_ids_from})
    covered_to = sorted({line_id for *_, _, line_ids_to, _ in batches for line_id in line_ids_to})

    assert covered_from == [1, 2, 3, 4, 5]
    assert covered_to == [1, 2, 3, 4, 5]


def test_process_batch_keeps_db_ids_in_processing_rows():
    with patch(
        "lingtrain_aligner.aligner._process_batch_with_cache",
        return_value=(
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        ),
    ):
        texts_from, texts_to = aligner.process_batch(
            db_path="unused.db",
            lines_from_batch=["from one", "from two"],
            lines_to_batch=["to one", "to two"],
            line_ids_from=[1, 2],
            line_ids_to=[1, 2],
            batch_number=0,
            model_name="fake-model",
            window=10,
            embed_batch_size=2,
            normalize_embeddings=True,
            show_progress_bar=False,
            embedding_cache={"from": {}, "to": {}},
        )

    assert texts_from == [
        ("[1]", 1, "from one"),
        ("[2]", 2, "from two"),
    ]
    assert texts_to == [
        ("[1]", 1, "to one"),
        ("[2]", 2, "to two"),
    ]


def test_align_db_stores_embeddings_for_final_sentence():
    lines_from = [f"from {idx}" for idx in range(1, 6)]
    lines_to = [f"to {idx}" for idx in range(1, 6)]

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = f"{tmpdir}/batch_regression.db"
        aligner.fill_db(
            db_path,
            "en",
            "ru",
            splitted_from=lines_from,
            splitted_to=lines_to,
        )

        with patch("lingtrain_aligner.aligner.get_line_vectors", side_effect=_fake_vectors):
            aligner.align_db(
                db_path,
                model_name="fake-model",
                batch_size=2,
                window=1,
                store_embeddings=True,
            )

        from_embeddings = dict(helper.get_embeddings(db_path, "from", [1, 2, 3, 4, 5]))
        to_embeddings = dict(helper.get_embeddings(db_path, "to", [1, 2, 3, 4, 5]))

        assert sorted(from_embeddings) == [1, 2, 3, 4, 5]
        assert sorted(to_embeddings) == [1, 2, 3, 4, 5]
        assert all(embedding is not None for embedding in from_embeddings.values())
        assert all(embedding is not None for embedding in to_embeddings.values())
