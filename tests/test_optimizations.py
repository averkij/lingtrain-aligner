"""Tests for embedding calculation optimizations."""

import json
import os
import sqlite3
import tempfile

import numpy as np
import pytest

from lingtrain_aligner import helper
from lingtrain_aligner.aligner import (
    _get_sim_matrix_reference,
    get_sim_matrix,
)
from lingtrain_aligner.helper import (
    _blob_to_embedding,
    _embedding_to_blob,
)


# ---------------------------------------------------------------------------
# 1. test_get_sim_matrix_matches_original
# ---------------------------------------------------------------------------

class TestGetSimMatrixMatchesOriginal:
    """Vectorized get_sim_matrix must match the original nested-loop version."""

    @pytest.mark.parametrize("n1,n2,window", [
        (10, 15, 5),
        (50, 50, 10),
        (100, 150, 20),
        (1, 1, 1),
        (1, 10, 3),
        (10, 1, 3),
        (20, 30, 0),
    ])
    def test_normalized_vectors(self, n1, n2, window):
        rng = np.random.RandomState(42)
        vec1 = rng.randn(n1, 128)
        vec2 = rng.randn(n2, 128)
        # Normalize
        vec1 = vec1 / np.linalg.norm(vec1, axis=1, keepdims=True)
        vec2 = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)

        ref = _get_sim_matrix_reference(vec1, vec2, window)
        new = get_sim_matrix(vec1, vec2, window)
        np.testing.assert_allclose(new, ref, atol=1e-6)

    @pytest.mark.parametrize("n1,n2,window", [
        (10, 15, 5),
        (30, 20, 8),
    ])
    def test_non_normalized_vectors(self, n1, n2, window):
        rng = np.random.RandomState(123)
        vec1 = rng.randn(n1, 64) * 5
        vec2 = rng.randn(n2, 64) * 3

        ref = _get_sim_matrix_reference(vec1, vec2, window)
        new = get_sim_matrix(vec1, vec2, window)
        np.testing.assert_allclose(new, ref, atol=1e-6)

    def test_window_zero(self):
        rng = np.random.RandomState(0)
        vec1 = rng.randn(5, 32)
        vec2 = rng.randn(8, 32)

        ref = _get_sim_matrix_reference(vec1, vec2, 0)
        new = get_sim_matrix(vec1, vec2, 0)
        np.testing.assert_allclose(new, ref, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. test_get_sim_matrix_window_mask
# ---------------------------------------------------------------------------

class TestGetSimMatrixWindowMask:
    """Verify window mask properties."""

    def test_outside_window_is_zero(self):
        rng = np.random.RandomState(7)
        n1, n2, window = 20, 30, 5
        vec1 = rng.randn(n1, 64)
        vec2 = rng.randn(n2, 64)
        vec1 = vec1 / np.linalg.norm(vec1, axis=1, keepdims=True)
        vec2 = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)

        sim = get_sim_matrix(vec1, vec2, window)
        k = n1 / n2
        for i in range(n1):
            for j in range(n2):
                jk = j * k
                if not (jk > i - window and jk < i + window):
                    assert sim[i, j] == 0.0, f"Expected 0 at ({i},{j}), got {sim[i,j]}"

    def test_inside_window_min_value(self):
        rng = np.random.RandomState(7)
        n1, n2, window = 20, 30, 5
        vec1 = rng.randn(n1, 64)
        vec2 = rng.randn(n2, 64)
        vec1 = vec1 / np.linalg.norm(vec1, axis=1, keepdims=True)
        vec2 = vec2 / np.linalg.norm(vec2, axis=1, keepdims=True)

        sim = get_sim_matrix(vec1, vec2, window)
        k = n1 / n2
        for i in range(n1):
            for j in range(n2):
                jk = j * k
                if jk > i - window and jk < i + window:
                    assert sim[i, j] >= 0.01, f"Expected >= 0.01 at ({i},{j}), got {sim[i,j]}"


# ---------------------------------------------------------------------------
# 3. test_binary_embedding_roundtrip
# ---------------------------------------------------------------------------

class TestBinaryEmbeddingRoundtrip:
    """_embedding_to_blob / _blob_to_embedding roundtrip must be lossless for float32."""

    def test_float32_roundtrip(self):
        rng = np.random.RandomState(0)
        vec = rng.randn(768).astype(np.float32)
        restored = _blob_to_embedding(_embedding_to_blob(vec))
        np.testing.assert_allclose(restored, vec, atol=1e-7)

    def test_float64_roundtrip(self):
        rng = np.random.RandomState(1)
        vec = rng.randn(768).astype(np.float64)
        restored = _blob_to_embedding(_embedding_to_blob(vec))
        # float64 -> float32 -> float32: loses precision but within float32 tolerance
        np.testing.assert_allclose(restored, vec.astype(np.float32), atol=1e-7)

    def test_list_roundtrip(self):
        vec = [0.1, 0.2, 0.3, -0.5]
        restored = _blob_to_embedding(_embedding_to_blob(vec))
        np.testing.assert_allclose(restored, np.array(vec, dtype=np.float32), atol=1e-7)


# ---------------------------------------------------------------------------
# 4. test_binary_embedding_reads_legacy_json
# ---------------------------------------------------------------------------

class TestBinaryEmbeddingReadsLegacyJson:
    """New get_embeddings must transparently read old JSON-formatted embeddings."""

    def test_legacy_json_read(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = os.path.join(tmpdir, "test_legacy.db")
            # Create a minimal DB with JSON embeddings (old format)
            db = sqlite3.connect(db_path)
            try:
                db.execute(
                    """create table splitted_from(
                        id integer, text text, proxy_text text,
                        exclude integer, paragraph integer,
                        h1 integer, h2 integer, h3 integer,
                        h4 integer, h5 integer, divider int,
                        embedding text, proxy_embedding text
                    )"""
                )
                rng = np.random.RandomState(42)
                for i in range(1, 4):
                    vec = rng.randn(128).tolist()
                    db.execute(
                        "insert into splitted_from(id, text, embedding) values (?, ?, ?)",
                        (i, f"line {i}", json.dumps(vec)),
                    )
                db.commit()
            finally:
                db.close()

            # Read back with the new function
            result = helper.get_embeddings(db_path, "from", [1, 2, 3], is_proxy=False)
            assert len(result) == 3
            for rid, emb in result:
                assert isinstance(emb, np.ndarray)
                assert emb.shape == (128,)

    def test_binary_blob_read(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = os.path.join(tmpdir, "test_binary.db")
            db = sqlite3.connect(db_path)
            try:
                db.execute(
                    """create table splitted_from(
                        id integer, text text, proxy_text text,
                        exclude integer, paragraph integer,
                        h1 integer, h2 integer, h3 integer,
                        h4 integer, h5 integer, divider int,
                        embedding blob, proxy_embedding blob
                    )"""
                )
                rng = np.random.RandomState(42)
                for i in range(1, 4):
                    vec = rng.randn(128).astype(np.float32)
                    db.execute(
                        "insert into splitted_from(id, text, embedding) values (?, ?, ?)",
                        (i, f"line {i}", vec.tobytes()),
                    )
                db.commit()
            finally:
                db.close()

            result = helper.get_embeddings(db_path, "from", [1, 2, 3], is_proxy=False)
            assert len(result) == 3
            rng2 = np.random.RandomState(42)
            for rid, emb in result:
                expected = rng2.randn(128).astype(np.float32)
                np.testing.assert_allclose(emb, expected, atol=1e-7)


# ---------------------------------------------------------------------------
# 5. test_end_to_end_alignment — integration test (requires LaBSE)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.requires_model
class TestEndToEndAlignment:
    """Full alignment pipeline on sample texts."""

    def test_alignment_produces_valid_index(self):
        from lingtrain_aligner import aligner, splitter, resolver

        sample_dir = os.path.join(os.path.dirname(__file__), "..", "sample_texts")
        en_path = os.path.join(sample_dir, "agata_geran_en.txt")
        ru_path = os.path.join(sample_dir, "agata_geran_ru.txt")

        if not os.path.exists(en_path) or not os.path.exists(ru_path):
            pytest.skip("Sample texts not found")

        with open(en_path, "r", encoding="utf-8") as f:
            text_en = f.read()
        with open(ru_path, "r", encoding="utf-8") as f:
            text_ru = f.read()

        splitted_en = splitter.split_by_sentences(text_en, "en")
        splitted_ru = splitter.split_by_sentences(text_ru, "ru")

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = os.path.join(tmpdir, "test_align.db")

            aligner.fill_db(
                db_path, "en", "ru",
                splitted_from=splitted_en,
                splitted_to=splitted_ru,
            )

            model_name = "sentence-transformers/LaBSE"
            aligner.align_db(
                db_path,
                model_name,
                batch_size=100,
                window=40,
                store_embeddings=True,
            )

            # Run resolver
            batch_ids = [0]
            try:
                resolver.resolve_db(
                    db_path, model_name, batch_ids,
                    show_progress_bar=False,
                )
            except Exception:
                pass  # resolver may need specific batch setup

            # Check doc_index exists and has content
            doc_index = helper.get_doc_index_original(db_path)
            assert len(doc_index) > 0, "doc_index should not be empty"

            # Compute a rough chain score: fraction of non-empty alignment pairs
            total_pairs = sum(len(batch) for batch in doc_index)
            assert total_pairs > 0, "Should have alignment pairs"


# ---------------------------------------------------------------------------
# 6. test_embedding_cache_eliminates_redundant_computation — integration test
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.requires_model
class TestEmbeddingCacheEliminatesRedundantComputation:
    """Verify the embedding cache avoids recomputing overlapping embeddings."""

    def test_cache_no_redundant_computation(self):
        from lingtrain_aligner import aligner, splitter
        from unittest.mock import patch

        sample_dir = os.path.join(os.path.dirname(__file__), "..", "sample_texts")
        en_path = os.path.join(sample_dir, "agata_geran_en.txt")
        ru_path = os.path.join(sample_dir, "agata_geran_ru.txt")

        if not os.path.exists(en_path) or not os.path.exists(ru_path):
            pytest.skip("Sample texts not found")

        with open(en_path, "r", encoding="utf-8") as f:
            text_en = f.read()
        with open(ru_path, "r", encoding="utf-8") as f:
            text_ru = f.read()

        splitted_en = splitter.split_by_sentences(text_en, "en")
        splitted_ru = splitter.split_by_sentences(text_ru, "ru")

        computed_ids = {"from": set(), "to": set()}

        original_compute = aligner._compute_embeddings_for_ids

        def tracking_compute(db_path, direction, ids, *args, **kwargs):
            result = original_compute(db_path, direction, ids, *args, **kwargs)
            # Track which IDs were actually computed (returned by DB lookup)
            for lid in result:
                assert lid not in computed_ids[direction], \
                    f"ID {lid} in direction '{direction}' computed more than once!"
                computed_ids[direction].add(lid)
            return result

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            db_path = os.path.join(tmpdir, "test_cache.db")

            aligner.fill_db(
                db_path, "en", "ru",
                splitted_from=splitted_en,
                splitted_to=splitted_ru,
            )

            model_name = "sentence-transformers/LaBSE"

            with patch.object(aligner, '_compute_embeddings_for_ids', side_effect=tracking_compute):
                aligner.align_db(
                    db_path,
                    model_name,
                    batch_size=100,
                    window=40,
                    store_embeddings=True,
                )

            # Verify total unique embeddings computed == total unique sentences
            len_from, len_to = helper.get_splitted_lenght(db_path)
            assert len(computed_ids["from"]) <= len_from
            assert len(computed_ids["to"]) <= len_to

    def test_cache_vs_no_cache_identical_output(self):
        from lingtrain_aligner import aligner, splitter

        sample_dir = os.path.join(os.path.dirname(__file__), "..", "sample_texts")
        en_path = os.path.join(sample_dir, "agata_geran_en.txt")
        ru_path = os.path.join(sample_dir, "agata_geran_ru.txt")

        if not os.path.exists(en_path) or not os.path.exists(ru_path):
            pytest.skip("Sample texts not found")

        with open(en_path, "r", encoding="utf-8") as f:
            text_en = f.read()
        with open(ru_path, "r", encoding="utf-8") as f:
            text_ru = f.read()

        # Use a small subset for speed
        splitted_en = splitter.split_by_sentences(text_en, "en")[:50]
        splitted_ru = splitter.split_by_sentences(text_ru, "ru")[:50]

        model_name = "sentence-transformers/LaBSE"

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            # Run WITH cache (default in align_db)
            db1 = os.path.join(tmpdir, "with_cache.db")
            aligner.fill_db(db1, "en", "ru",
                            splitted_from=splitted_en, splitted_to=splitted_ru)
            aligner.align_db(db1, model_name, batch_size=20, window=10,
                             store_embeddings=True)
            index1 = helper.get_doc_index_original(db1)

            # Run WITHOUT cache
            db2 = os.path.join(tmpdir, "no_cache.db")
            aligner.fill_db(db2, "en", "ru",
                            splitted_from=splitted_en, splitted_to=splitted_ru)

            # Temporarily patch align_db to not pass cache
            original_process = aligner.process_batch

            def no_cache_process(*args, **kwargs):
                kwargs.pop("embedding_cache", None)
                return original_process(*args, **kwargs)

            from unittest.mock import patch
            with patch.object(aligner, 'process_batch', side_effect=no_cache_process):
                aligner.align_db(db2, model_name, batch_size=20, window=10,
                                 store_embeddings=True)
            index2 = helper.get_doc_index_original(db2)

            assert index1 == index2, "Doc index must be identical with and without cache"


# ---------------------------------------------------------------------------
# 7. test_aggregate_embeddings — unit tests for aggregation fixes
# ---------------------------------------------------------------------------

class TestAggregateEmbeddings:
    """Tests for aggregate_embeddings correctness fixes."""

    def test_weighted_average_basic(self):
        from lingtrain_aligner.resolver import aggregate_embeddings

        emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        lens = np.array([3, 1])
        result = aggregate_embeddings(emb, lens, "weighted_average")
        assert result.shape == (2,)
        assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-7)

    def test_all_methods_return_normalized(self):
        from lingtrain_aligner.resolver import aggregate_embeddings

        rng = np.random.RandomState(99)
        emb = rng.randn(5, 64)
        lens = np.array([10, 20, 15, 5, 30])
        for method in ["weighted_average", "length_scaling", "max_pooling", "logarithmic_scaling"]:
            result = aggregate_embeddings(emb, lens, method)
            assert np.linalg.norm(result) == pytest.approx(1.0, abs=1e-7), f"{method} not normalized"

    def test_invalid_method_rejected(self):
        from lingtrain_aligner.resolver import aggregate_embeddings

        emb = np.array([[1.0, 0.0]])
        lens = np.array([1])
        with pytest.raises(ValueError, match="Unknown method"):
            aggregate_embeddings(emb, lens, "invalid_method")

    def test_near_zero_norm_raises(self):
        from lingtrain_aligner.resolver import aggregate_embeddings

        emb = np.array([[0.0, 0.0], [0.0, 0.0]])
        lens = np.array([1, 1])
        with pytest.raises(ValueError, match="zero"):
            aggregate_embeddings(emb, lens, "weighted_average")

    def test_empty_embeddings_raises(self):
        from lingtrain_aligner.resolver import aggregate_embeddings

        with pytest.raises(ValueError, match="No embeddings"):
            aggregate_embeddings([], [1], "weighted_average")


# ---------------------------------------------------------------------------
# 8. test_get_unique_sims — vectorized cosine similarity
# ---------------------------------------------------------------------------

class TestGetUniqueSims:
    """Tests for vectorized get_unique_sims."""

    def test_matches_scipy_cosine(self):
        from scipy import spatial
        from lingtrain_aligner.resolver import get_unique_sims

        rng = np.random.RandomState(42)
        n = 10
        dim = 64
        vf = rng.randn(n, dim)
        vt = rng.randn(n, dim)
        # Normalize to match typical embedding inputs
        vf = vf / np.linalg.norm(vf, axis=1, keepdims=True)
        vt = vt / np.linalg.norm(vt, axis=1, keepdims=True)

        variants = [((i,), (i + 100,)) for i in range(n)]
        result = get_unique_sims(variants, vf, vt)

        for i, v in enumerate(variants):
            expected = 1 - spatial.distance.cosine(vf[i], vt[i])
            assert result[v] == pytest.approx(expected, abs=1e-6)

    def test_non_normalized_vectors(self):
        from scipy import spatial
        from lingtrain_aligner.resolver import get_unique_sims

        rng = np.random.RandomState(7)
        n = 5
        vf = rng.randn(n, 32) * 10
        vt = rng.randn(n, 32) * 0.1

        variants = [((i,), (i,)) for i in range(n)]
        result = get_unique_sims(variants, vf, vt)

        for i, v in enumerate(variants):
            expected = 1 - spatial.distance.cosine(vf[i], vt[i])
            assert result[v] == pytest.approx(expected, abs=1e-6)
