"""Tests for conflict splitting via Penalized DP."""

import numpy as np
import pytest
from pathlib import Path

from lingtrain_aligner.resolver import (
    penalized_dp_anchors,
    diagonal_peaks_anchors,
    _run_penalized_dp,
    dp_path_to_solution,
    MAX_DIRECT_RESOLVE_SIZE,
)


# ---------------------------------------------------------------------------
# penalized_dp_anchors
# ---------------------------------------------------------------------------


class TestPenalizedDPAnchors:
    def test_identity_matrix(self):
        """Perfect 1:1 diagonal should yield anchors at every position (filtered by min_gap)."""
        n = 6
        sim = np.eye(n, dtype=np.float32)
        anchors = penalized_dp_anchors(sim, anchor_threshold=0.5, min_gap=2)
        # Should find anchors along the diagonal, spaced by min_gap
        assert len(anchors) >= 2
        for fi, ti, conf in anchors:
            assert fi == ti, "Anchors should lie on diagonal for identity matrix"
            assert conf >= 0.5

    def test_strong_diagonal_with_noise(self):
        """Strong diagonal signal with low off-diagonal noise."""
        n, m = 8, 8
        sim = np.random.uniform(0.1, 0.3, (n, m)).astype(np.float32)
        for i in range(n):
            sim[i, i] = 0.9
        anchors = penalized_dp_anchors(sim, anchor_threshold=0.5, min_gap=2)
        assert len(anchors) >= 2
        for fi, ti, conf in anchors:
            assert fi == ti

    def test_shifted_diagonal(self):
        """Peak at (i, i+1) — anchors should follow the shifted diagonal."""
        n, m = 6, 7
        sim = np.zeros((n, m), dtype=np.float32)
        for i in range(n):
            sim[i, i + 1] = 0.85
        anchors = penalized_dp_anchors(sim, anchor_threshold=0.5, min_gap=2)
        assert len(anchors) >= 1
        for fi, ti, conf in anchors:
            assert ti == fi + 1

    def test_low_similarity_no_anchors(self):
        """All similarities below threshold → no anchors."""
        sim = np.full((5, 5), 0.2, dtype=np.float32)
        anchors = penalized_dp_anchors(sim, anchor_threshold=0.5)
        assert len(anchors) == 0

    def test_single_cell(self):
        """1x1 matrix — (0,0) is the start cell, not a DP transition, so no anchor."""
        sim = np.array([[0.9]], dtype=np.float32)
        anchors = penalized_dp_anchors(sim, anchor_threshold=0.5)
        # (0,0) has no parent transition, so move_type stays -1 → not an anchor
        assert len(anchors) == 0

    def test_two_by_two(self):
        """2x2 matrix — only (1,1) is a diagonal transition from (0,0)."""
        sim = np.array([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32)
        anchors = penalized_dp_anchors(sim, anchor_threshold=0.5, min_gap=1)
        assert len(anchors) == 1
        assert anchors[0][:2] == (1, 1)

    def test_empty_matrix(self):
        """Empty matrix returns no anchors."""
        sim = np.zeros((0, 5), dtype=np.float32)
        assert penalized_dp_anchors(sim) == []

    def test_min_gap_filtering(self):
        """Min gap of 3 should remove closely-spaced anchors."""
        n = 10
        sim = np.eye(n, dtype=np.float32) * 0.9
        anchors_gap2 = penalized_dp_anchors(sim, anchor_threshold=0.5, min_gap=2)
        anchors_gap3 = penalized_dp_anchors(sim, anchor_threshold=0.5, min_gap=3)
        assert len(anchors_gap3) <= len(anchors_gap2)

    def test_merge_scenario(self):
        """Scenario where 2 from-sentences map to 1 to-sentence (2:1 merge).

        The DP should still find anchors around the merge point.
        """
        # 5 from, 4 to: lines 0-1 merge to 0, then 1:1 for the rest
        sim = np.zeros((5, 4), dtype=np.float32)
        sim[0, 0] = 0.7
        sim[1, 0] = 0.7
        sim[2, 1] = 0.9
        sim[3, 2] = 0.9
        sim[4, 3] = 0.9
        anchors = penalized_dp_anchors(sim, anchor_threshold=0.5, min_gap=2)
        # Should find at least some anchors in the 1:1 region
        assert len(anchors) >= 1


# ---------------------------------------------------------------------------
# diagonal_peaks_anchors
# ---------------------------------------------------------------------------


class TestDiagonalPeaksAnchors:
    def test_perfect_diagonal(self):
        """Perfect diagonal should yield mutual best matches."""
        n = 6
        sim = np.eye(n, dtype=np.float32)
        anchors = diagonal_peaks_anchors(sim, threshold=0.5, min_gap=2)
        assert len(anchors) >= 2
        for fi, ti, conf in anchors:
            assert fi == ti

    def test_no_mutual_matches(self):
        """When no mutual best matches exist above threshold."""
        sim = np.full((5, 5), 0.3, dtype=np.float32)
        anchors = diagonal_peaks_anchors(sim, threshold=0.5)
        assert len(anchors) == 0

    def test_off_diagonal_rejected(self):
        """Strong matches far from diagonal should be rejected."""
        sim = np.zeros((6, 6), dtype=np.float32)
        # Put strong match far from diagonal
        sim[0, 5] = 0.99
        sim[5, 0] = 0.99
        anchors = diagonal_peaks_anchors(sim, threshold=0.5)
        # These are mutual best matches but far from diagonal
        assert len(anchors) == 0

    def test_empty_matrix(self):
        sim = np.zeros((0, 3), dtype=np.float32)
        assert diagonal_peaks_anchors(sim) == []


# ---------------------------------------------------------------------------
# Integration: MAX_DIRECT_RESOLVE_SIZE constant
# ---------------------------------------------------------------------------


class TestDPPathToSolution:
    def test_diagonal_path(self):
        """Perfect diagonal path should produce 1:1 pairs."""
        sim = np.eye(4, dtype=np.float32) * 0.9
        from_ids = [10, 11, 12, 13]
        to_ids = [20, 21, 22, 23]
        path = _run_penalized_dp(sim)
        solution = dp_path_to_solution(path, from_ids, to_ids)
        # Should have pairs, and all IDs should be covered
        all_from = set()
        all_to = set()
        for sf, st in solution:
            all_from.update(sf)
            all_to.update(st)
        assert all_from == set(from_ids)
        assert all_to == set(to_ids)

    def test_merge_path_covers_all(self):
        """With merges, all IDs must still be accounted for."""
        # 5 from, 3 to — forces merges
        sim = np.zeros((5, 3), dtype=np.float32)
        sim[0, 0] = 0.8
        sim[1, 0] = 0.7
        sim[2, 1] = 0.9
        sim[3, 2] = 0.8
        sim[4, 2] = 0.7
        from_ids = [1, 2, 3, 4, 5]
        to_ids = [10, 11, 12]
        path = _run_penalized_dp(sim)
        solution = dp_path_to_solution(path, from_ids, to_ids)
        all_from = set()
        all_to = set()
        for sf, st in solution:
            all_from.update(sf)
            all_to.update(st)
        assert all_from == set(from_ids), f"Missing from IDs: {set(from_ids) - all_from}"
        assert all_to == set(to_ids), f"Missing to IDs: {set(to_ids) - all_to}"

    def test_empty_path(self):
        """Empty matrix produces a single merged pair."""
        solution = dp_path_to_solution([], [1, 2], [10, 11])
        assert len(solution) == 1
        assert solution[0] == ((1, 2), (10, 11))


def test_max_direct_resolve_size():
    """The threshold should be 12 (matching the plan)."""
    assert MAX_DIRECT_RESOLVE_SIZE == 12


# ---------------------------------------------------------------------------
# Integration test with real research sample
# ---------------------------------------------------------------------------

# Sample may be in the parent repo (lingtrain) rather than the submodule
_SUBMODULE_ROOT = Path(__file__).resolve().parents[1]
_PARENT_REPO = _SUBMODULE_ROOT.parent
SAMPLE_PATH = _PARENT_REPO / "samples" / "conflicts" / "conflict_sample_1.json"
if not SAMPLE_PATH.exists():
    SAMPLE_PATH = _SUBMODULE_ROOT / "samples" / "conflicts" / "conflict_sample_1.json"


@pytest.mark.skipif(not SAMPLE_PATH.exists(), reason="Research sample not available")
class TestWithResearchSample:
    """Integration tests using the 41:44 Little Prince (kv-ru) conflict sample."""

    @pytest.fixture(autouse=True)
    def load_sample(self):
        import json as _json
        with open(SAMPLE_PATH, "r", encoding="utf-8") as f:
            data = _json.load(f)

        def _parse_side(items):
            ids, embeddings = [], []
            for item in items:
                idx_str = list(item.keys())[0]
                ids.append(int(idx_str))
                embeddings.append(np.array(item[idx_str]["embedding"], dtype=np.float32))
            return ids, np.array(embeddings)

        self.from_ids, emb_from = _parse_side(data["lang_from"])
        self.to_ids, emb_to = _parse_side(data["lang_to"])

        # L2 normalize
        norms_f = np.linalg.norm(emb_from, axis=1, keepdims=True)
        norms_t = np.linalg.norm(emb_to, axis=1, keepdims=True)
        norms_f = np.where(norms_f < 1e-10, 1.0, norms_f)
        norms_t = np.where(norms_t < 1e-10, 1.0, norms_t)
        emb_from = emb_from / norms_f
        emb_to = emb_to / norms_t

        self.sim_matrix = emb_from @ emb_to.T

    def test_dp_finds_anchors(self):
        """Penalized DP should find multiple anchors in the 41:44 conflict."""
        anchors = penalized_dp_anchors(self.sim_matrix, anchor_threshold=0.45)
        assert len(anchors) >= 10, f"Expected >=10 anchors, got {len(anchors)}"

    def test_dp_anchors_are_monotonic(self):
        """All anchors should be monotonically increasing on both axes."""
        anchors = penalized_dp_anchors(self.sim_matrix, anchor_threshold=0.45)
        for i in range(1, len(anchors)):
            assert anchors[i][0] > anchors[i - 1][0], "from_idx not monotonic"
            assert anchors[i][1] > anchors[i - 1][1], "to_idx not monotonic"

    def test_variant_reduction(self):
        """Splitting should massively reduce variant count vs brute-force."""
        from math import comb

        n, m = self.sim_matrix.shape
        # Brute-force variant count (from get_variants formula)
        brute_force = sum(
            comb(n - 1, k - 1) * comb(m - 1, k - 1)
            for k in range(1, min(n, m) + 1)
        )

        anchors = penalized_dp_anchors(self.sim_matrix, anchor_threshold=0.45)
        # Compute variant count for sub-conflicts between anchors
        boundaries = [(0, 0)] + [(a[0], a[1]) for a in anchors] + [(n, m)]
        total_variants = 0
        for i in range(1, len(boundaries)):
            sub_n = boundaries[i][0] - boundaries[i - 1][0]
            sub_m = boundaries[i][1] - boundaries[i - 1][1]
            if sub_n > 0 and sub_m > 0:
                sub_variants = sum(
                    comb(sub_n - 1, k - 1) * comb(sub_m - 1, k - 1)
                    for k in range(1, min(sub_n, sub_m) + 1)
                )
                total_variants += sub_variants

        reduction = brute_force / max(total_variants, 1)
        assert reduction > 100, f"Expected >100x reduction, got {reduction:.1f}x"

    def test_diagonal_peaks_finds_anchors(self):
        """Diagonal peaks should also find anchors (fewer than DP)."""
        anchors = diagonal_peaks_anchors(self.sim_matrix, threshold=0.55)
        assert len(anchors) >= 3
