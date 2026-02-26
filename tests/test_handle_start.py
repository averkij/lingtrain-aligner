"""Tests for handle_start / handle_finish edge-conflict detection in resolver."""

import pytest

from lingtrain_aligner.resolver import get_good_chains, get_conflicts


def _ix(from_id, to_id, batch_id=0, sub_id=None):
    """Helper to build a prepared-index entry."""
    if sub_id is None:
        sub_id = to_id - 1
    return {
        "from": [from_id],
        "to": to_id,
        "batch_id": batch_id,
        "sub_id": sub_id,
        "from_was_edited": False,
        "to_was_edited": False,
    }


# ---------------------------------------------------------------------------
# Original handle_start behaviour (first to != 1)
# ---------------------------------------------------------------------------


class TestHandleStartOriginal:
    """When the first alignment doesn't start at to=1, handle_start should
    detect the gap — this is the original behaviour that must not regress."""

    def test_first_to_not_1_creates_conflict(self):
        # Alignment starts at to=3, skipping to lines 1-2
        ix = [_ix(1, 3), _ix(2, 4), _ix(3, 5), _ix(4, 6)]
        chains_from, chains_to = get_good_chains(
            ix, min_len=2, handle_start=True, handle_finish=False
        )
        conflicts, _ = get_conflicts(chains_from, chains_to, max_len=20)
        assert len(conflicts) >= 1
        # The first conflict should start at to=1
        assert conflicts[0]["to"]["start"][0] == 1


# ---------------------------------------------------------------------------
# NEW: handle_start when first pair is (1,1) but chain breaks immediately
# ---------------------------------------------------------------------------


class TestHandleStartFirstPairOk:
    """When the first pair is (1,1) but the second pair jumps (not 2-2),
    handle_start should still detect the gap at the document start."""

    def test_first_pair_ok_second_jumps(self):
        # Pair 0: (1,1), Pair 1: (2,5), then consecutive
        ix = [_ix(1, 1), _ix(2, 5), _ix(3, 6), _ix(4, 7), _ix(5, 8)]
        chains_from, chains_to = get_good_chains(
            ix, min_len=2, handle_start=True, handle_finish=False
        )
        conflicts, _ = get_conflicts(chains_from, chains_to, max_len=20)

        assert len(conflicts) >= 1, (
            "Should detect a start conflict when (1,1) chain is too short"
        )
        # Anchor should be at to=1
        assert conflicts[0]["to"]["start"][0] == 1
        # Next good chain starts at to=5
        assert conflicts[0]["to"]["end"][0] == 5

    def test_no_conflict_without_flag(self):
        # Same data but handle_start=False — no start conflict expected
        ix = [_ix(1, 1), _ix(2, 5), _ix(3, 6), _ix(4, 7), _ix(5, 8)]
        chains_from, chains_to = get_good_chains(
            ix, min_len=2, handle_start=False, handle_finish=False
        )
        conflicts, _ = get_conflicts(chains_from, chains_to, max_len=20)
        # Without handle_start, the short first chain is discarded and no
        # start boundary exists
        start_conflicts = [
            c for c in conflicts if c["to"]["start"][0] == 1
        ]
        assert len(start_conflicts) == 0

    def test_perfectly_aligned_start_no_false_conflict(self):
        # Consecutive from line 1 — no conflict should be generated
        ix = [_ix(1, 1), _ix(2, 2), _ix(3, 3), _ix(4, 4)]
        chains_from, chains_to = get_good_chains(
            ix, min_len=2, handle_start=True, handle_finish=False
        )
        conflicts, _ = get_conflicts(chains_from, chains_to, max_len=20)
        assert len(conflicts) == 0, (
            "Perfectly aligned start should not produce a false conflict"
        )

    def test_multiple_short_chains_before_good_one(self):
        # (1,1) breaks, (2,5) breaks, then (3,8)-(4,9)-(5,10) is good
        ix = [_ix(1, 1), _ix(2, 5), _ix(3, 8), _ix(4, 9), _ix(5, 10)]
        chains_from, chains_to = get_good_chains(
            ix, min_len=2, handle_start=True, handle_finish=False
        )
        conflicts, _ = get_conflicts(chains_from, chains_to, max_len=20)

        assert len(conflicts) >= 1
        # Anchor at (1, 1)
        assert conflicts[0]["to"]["start"][0] == 1
        assert conflicts[0]["from"]["start"][0] == 1
        # Next good chain starts at to=8
        assert conflicts[0]["to"]["end"][0] == 8


# ---------------------------------------------------------------------------
# handle_finish symmetry
# ---------------------------------------------------------------------------


class TestHandleFinish:
    """Verify handle_finish still works correctly after the refactoring."""

    def test_short_last_chain_with_finish(self):
        # Good chain, then a single trailing pair
        ix = [_ix(1, 1), _ix(2, 2), _ix(3, 3), _ix(4, 10)]
        chains_from, chains_to = get_good_chains(
            ix, min_len=2, handle_start=False, handle_finish=True,
            len_from=5, len_to=12,
        )
        conflicts, _ = get_conflicts(chains_from, chains_to, max_len=20)
        assert len(conflicts) >= 1
        # End anchor should be at len_to
        assert conflicts[-1]["to"]["end"][0] == 12

    def test_both_flags_short_everywhere(self):
        # Only two pairs, neither forms a chain of length 2
        ix = [_ix(1, 1), _ix(2, 5)]
        chains_from, chains_to = get_good_chains(
            ix, min_len=2, handle_start=True, handle_finish=True,
            len_from=10, len_to=10,
        )
        # Should have start anchor + finish anchor → 1 conflict
        conflicts, _ = get_conflicts(chains_from, chains_to, max_len=20)
        assert len(conflicts) >= 1
