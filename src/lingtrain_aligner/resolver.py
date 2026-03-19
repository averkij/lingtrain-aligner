"""Conflicts resolver part of the engine"""

import json
import sqlite3
from collections import defaultdict

import more_itertools as mit
from lingtrain_aligner import aligner, helper, punct_sim
from tqdm import tqdm
import logging
import copy

import numpy as np


def prepare_index(db_path, batch_id=-1, index=None):
    """Get totally flattened index ids"""
    res = []
    if batch_id >= 0:
        if not index:
            index = helper.get_doc_index_original(db_path)
        total_batches = len(index)
        for i, ix in enumerate(index[batch_id]):
            from_ids = json.loads(ix[1])
            to_ids = json.loads(ix[3])
            if not from_ids or not to_ids:
                continue
            for t_id in to_ids:
                res.append(
                    {
                        "from": from_ids,
                        "to": t_id,
                        "batch_id": batch_id,
                        "sub_id": i,
                        "from_was_edited": len(from_ids) > 1,
                        "to_was_edited": len(to_ids) > 1,
                    }
                )
    else:
        index = helper.get_flatten_doc_index_with_batch_id(db_path, index=index)
        total_batches = 1
        for ix, sub_id, batch_id in index:
            from_ids = json.loads(ix[1])
            to_ids = json.loads(ix[3])
            if not from_ids or not to_ids:
                continue
            for t_id in to_ids:
                res.append(
                    {
                        "from": from_ids,
                        "to": t_id,
                        "batch_id": batch_id,
                        "sub_id": sub_id,
                        "from_was_edited": len(from_ids) > 1,
                        "to_was_edited": len(to_ids) > 1,
                    }
                )
    return res, total_batches


def get_good_chains(
    ix, min_len=2, handle_start=False, handle_finish=False, len_from=-1, len_to=-1
):
    """Calculate valid alignment chains"""

    curr_from = ix[0]["from"][0]
    curr_to = ix[0]["to"]

    chains_from = []
    chains_to = []

    chain_from = [(curr_from, ix[0]["batch_id"], ix[0]["sub_id"])]
    chain_to = [(curr_to, ix[0]["batch_id"], ix[0]["sub_id"])]

    start = 1

    if handle_start and curr_to != 1:
        chains_from.append(chain_from)
        chains_to.append([(1, ix[0]["batch_id"], ix[0]["sub_id"])])
        chain_from = [(ix[1]["from"][0], ix[1]["batch_id"], ix[1]["sub_id"])]
        chain_to = [(ix[1]["to"], ix[1]["batch_id"], ix[1]["sub_id"])]
        curr_from = ix[1]["from"][0]
        curr_to = ix[1]["to"]
        start = 2

    for i in range(start, len(ix)):
        val_from = ix[i]["from"][0]
        val_to = ix[i]["to"]

        # continue chain
        if val_to == curr_to + 1:
            chain_from.append((val_from, ix[i]["batch_id"], ix[i]["sub_id"]))
            chain_to.append((val_to, ix[i]["batch_id"], ix[i]["sub_id"]))
            curr_from = val_from
            curr_to = val_to

        # add chain and start new
        elif len(chain_to) >= min_len:
            chains_from.append(chain_from)
            chains_to.append(chain_to)
            chain_from = [(val_from, ix[i]["batch_id"], ix[i]["sub_id"])]
            chain_to = [(val_to, ix[i]["batch_id"], ix[i]["sub_id"])]
            curr_from = val_from
            curr_to = val_to

        # start new chain
        else:
            # First chain too short — save as start anchor so the gap is detected
            if handle_start and not chains_from:
                chains_from.append(chain_from)
                chains_to.append(chain_to)
            chain_from = [(val_from, ix[i]["batch_id"], ix[i]["sub_id"])]
            chain_to = [(val_to, ix[i]["batch_id"], ix[i]["sub_id"])]
            curr_from = val_from
            curr_to = val_to

    if len(chain_to) >= min_len:
        chains_from.append(chain_from)
        chains_to.append(chain_to)
    else:
        # Last chain too short — save as start anchor if nothing saved yet
        if handle_start and not chains_from:
            chains_from.append(chain_from)
            chains_to.append(chain_to)
        if handle_finish:
            chains_from.append([(len_from, ix[-1]["batch_id"], ix[-1]["sub_id"])])
            chains_to.append([(len_to, ix[-1]["batch_id"], ix[-1]["sub_id"])])

    # print("handle_finish", handle_finish)
    # print("chains_from", chains_from)
    # print("chains_to", chains_to)
    # print("chain_from", chain_from)
    # print("chain_to", chain_to)

    # print("curr_from", curr_from)
    # print("curr_to", curr_to)

    # print("len_from", len_from)
    # print("len_to", len_to)

    return chains_from, chains_to


def get_conflicts(chains_from, chains_to, max_len=6):
    """Calculate conflicts between the chains"""
    conflicts_to_solve = []
    conflicts_rest = []
    for i in range(1, len(chains_to)):
        conflict = {
            "from": {"start": chains_from[i - 1][-1], "end": chains_from[i][0]},
            "to": {"start": chains_to[i - 1][-1], "end": chains_to[i][0]},
        }
        conflict_len_from = conflict["from"]["end"][0] - conflict["from"]["start"][0]
        conflict_len_to = conflict["to"]["end"][0] - conflict["to"]["start"][0]
        if (
            conflict_len_to < max_len
            and conflict_len_from < max_len
            and conflict_len_to >= 0
            and conflict_len_from >= 0
        ):
            conflicts_to_solve.append(conflict)
        else:
            conflicts_rest.append(conflict)
        # print("get conflict:", conflict, "len:", conflict_len_to)
    print("conflicts to solve:", len(conflicts_to_solve))
    print("total conflicts:", len(conflicts_to_solve) + len(conflicts_rest))
    return conflicts_to_solve, conflicts_rest


def _contiguous_partitions(items, k):
    """Yield all ways to split *items* into *k* contiguous groups.

    Uses C(n-1, k-1) split-point combinations — lightweight and bounded.
    """
    n = len(items)
    if k == 1:
        yield (tuple(items),)
        return
    if k == n:
        yield tuple((x,) for x in items)
        return
    if k > n or k < 1:
        return
    from itertools import combinations
    for splits in combinations(range(1, n), k - 1):
        groups = []
        prev = 0
        for s in splits:
            groups.append(tuple(items[prev:s]))
            prev = s
        groups.append(tuple(items[prev:]))
        yield tuple(groups)


_MAX_EXTENDED_VARIANTS = 5000


def get_variants(conflict, show_logs=False):
    """Get resolving variants.

    Generates contiguous partition variants for ALL group counts (1 to min(N,M))
    and partitions BOTH sides. This allows finding solutions that require merging
    on both the source and target sides.

    For large conflicts where the extended cross-product exceeds
    ``_MAX_EXTENDED_VARIANTS``, falls back to the original single-side
    partitioning at k = min(N, M) to stay within safe bounds.
    """
    ids_from = [
        x for x in range(conflict["from"]["start"][0], conflict["from"]["end"][0] + 1)
    ]
    ids_to = [
        x for x in range(conflict["to"]["start"][0], conflict["to"]["end"][0] + 1)
    ]

    if show_logs:
        print("ids_from", ids_from)
        print("ids_to", ids_to)
        print("\n")

    n, m = len(ids_from), len(ids_to)
    max_groups = min(n, m)

    # Check expected variant count before generating.
    from math import comb
    expected = sum(
        comb(n - 1, k - 1) * comb(m - 1, k - 1)
        for k in range(1, max_groups + 1)
    )

    if expected <= _MAX_EXTENDED_VARIANTS:
        # Extended: all group counts, both sides partitioned
        res = []
        for k in range(1, max_groups + 1):
            from_parts = list(_contiguous_partitions(ids_from, k))
            to_parts = list(_contiguous_partitions(ids_to, k))
            for pf in from_parts:
                for pt in to_parts:
                    res.append([(tuple(a), tuple(b)) for a, b in zip(pf, pt)])
        return res

    # Fallback for large conflicts: partition longer side only at k = max_groups
    res = []
    if n < m:
        for pt in _contiguous_partitions(ids_to, max_groups):
            res.append([((a,), tuple(b)) for a, b in zip(ids_from, pt)])
    elif n > m:
        for pf in _contiguous_partitions(ids_from, max_groups):
            res.append([(tuple(a), (b,)) for a, b in zip(pf, ids_to)])
    else:
        res.append([((a,), (b,)) for a, b in zip(ids_from, ids_to)])
    return res


# ---------------------------------------------------------------------------
# Conflict splitting via Penalized DP
# ---------------------------------------------------------------------------

MAX_DIRECT_RESOLVE_SIZE = 12


def _run_penalized_dp(sim_matrix, skip_penalty=-0.2, merge_penalty=-0.15):
    """Run the penalized DP and return the full backtracked path.

    Returns list of (from_idx, to_idx, similarity, move_type) tuples.
    Move types: 0=1:1 diagonal, 1=2:1 merge, 2=1:2 merge, 3=skip_from, 4=skip_to.
    """
    n, m = sim_matrix.shape
    if n == 0 or m == 0:
        return []

    dp = np.full((n, m), -np.inf)
    parent = np.full((n, m, 2), -1, dtype=np.int32)
    move_type = np.full((n, m), -1, dtype=np.int32)

    dp[0, 0] = float(sim_matrix[0, 0])

    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                continue
            best_val = -np.inf
            best_parent = (-1, -1)
            best_move = -1

            # 1:1 diagonal
            if i > 0 and j > 0 and dp[i - 1, j - 1] > -np.inf:
                val = dp[i - 1, j - 1] + sim_matrix[i, j]
                if val > best_val:
                    best_val, best_parent, best_move = val, (i - 1, j - 1), 0

            # 2:1 merge (two from-sentences → one to)
            if i > 1 and j > 0 and dp[i - 2, j - 1] > -np.inf:
                avg_sim = (sim_matrix[i - 1, j] + sim_matrix[i, j]) / 2
                val = dp[i - 2, j - 1] + avg_sim + merge_penalty
                if val > best_val:
                    best_val, best_parent, best_move = val, (i - 2, j - 1), 1

            # 1:2 merge (one from → two to-sentences)
            if i > 0 and j > 1 and dp[i - 1, j - 2] > -np.inf:
                avg_sim = (sim_matrix[i, j - 1] + sim_matrix[i, j]) / 2
                val = dp[i - 1, j - 2] + avg_sim + merge_penalty
                if val > best_val:
                    best_val, best_parent, best_move = val, (i - 1, j - 2), 2

            # Skip from
            if i > 0 and dp[i - 1, j] > -np.inf:
                val = dp[i - 1, j] + skip_penalty
                if val > best_val:
                    best_val, best_parent, best_move = val, (i - 1, j), 3

            # Skip to
            if j > 0 and dp[i, j - 1] > -np.inf:
                val = dp[i, j - 1] + skip_penalty
                if val > best_val:
                    best_val, best_parent, best_move = val, (i, j - 1), 4

            dp[i, j] = best_val
            parent[i, j] = [best_parent[0], best_parent[1]]
            move_type[i, j] = best_move

    # Backtrack
    path = []
    i, j = n - 1, m - 1
    while i >= 0 and j >= 0:
        path.append((i, j, float(sim_matrix[i, j]), int(move_type[i, j])))
        pi, pj = int(parent[i, j, 0]), int(parent[i, j, 1])
        if pi == -1:
            break
        i, j = pi, pj
    path.reverse()

    return path


def penalized_dp_anchors(
    sim_matrix,
    skip_penalty=-0.2,
    merge_penalty=-0.15,
    anchor_threshold=0.5,
    min_gap=2,
):
    """Find anchor points in a similarity matrix using penalized DP.

    Returns list of (from_idx, to_idx, confidence) where indices are
    0-based positions within the matrix.
    """
    path = _run_penalized_dp(sim_matrix, skip_penalty, merge_penalty)

    # Extract 1:1 diagonal matches as anchors
    anchors = []
    for (i, j, s, mt) in path:
        if mt == 0 and s >= anchor_threshold:
            anchors.append((i, j, s))

    # Filter minimum gap
    filtered = []
    for a in anchors:
        if not filtered or (
            a[0] - filtered[-1][0] >= min_gap
            and a[1] - filtered[-1][1] >= min_gap
        ):
            filtered.append(a)

    return filtered


def dp_path_to_solution(path, from_ids, to_ids):
    """Convert a DP path into a solution (list of (from_tuple, to_tuple) pairs).

    Used as a fallback when a large conflict has no usable anchors —
    the DP path itself IS the best alignment we can produce.
    """
    solution = []
    consumed_from = set()
    consumed_to = set()

    for i, j, sim, mt in path:
        if mt == 0:
            # 1:1 diagonal
            solution.append(((from_ids[i],), (to_ids[j],)))
            consumed_from.add(i)
            consumed_to.add(j)
        elif mt == 1:
            # 2:1 merge: from[i-1]+from[i] → to[j]
            solution.append(((from_ids[i - 1], from_ids[i]), (to_ids[j],)))
            consumed_from.update({i - 1, i})
            consumed_to.add(j)
        elif mt == 2:
            # 1:2 merge: from[i] → to[j-1]+to[j]
            solution.append(((from_ids[i],), (to_ids[j - 1], to_ids[j])))
            consumed_from.add(i)
            consumed_to.update({j - 1, j})
        # skip_from (3) and skip_to (4) produce no alignment pair

    # Handle any remaining unconsumed lines by merging into nearest pair
    unconsumed_from = [k for k in range(len(from_ids)) if k not in consumed_from]
    unconsumed_to = [k for k in range(len(to_ids)) if k not in consumed_to]

    if unconsumed_from and solution:
        # Merge unconsumed from-lines into their nearest existing pair
        for k in unconsumed_from:
            # Find the pair whose from-ids are closest
            best_idx = 0
            best_dist = abs(from_ids[k] - solution[0][0][0])
            for si, (sf, st) in enumerate(solution):
                d = min(abs(from_ids[k] - fid) for fid in sf)
                if d < best_dist:
                    best_dist = d
                    best_idx = si
            sf, st = solution[best_idx]
            solution[best_idx] = (tuple(sorted(set(sf + (from_ids[k],)))), st)

    if unconsumed_to and solution:
        for k in unconsumed_to:
            best_idx = 0
            best_dist = abs(to_ids[k] - solution[0][1][0])
            for si, (sf, st) in enumerate(solution):
                d = min(abs(to_ids[k] - tid) for tid in st)
                if d < best_dist:
                    best_dist = d
                    best_idx = si
            sf, st = solution[best_idx]
            solution[best_idx] = (sf, tuple(sorted(set(st + (to_ids[k],)))))

    # If no path produced any pairs (very degenerate), create one big merge
    if not solution:
        solution = [(tuple(from_ids), tuple(to_ids))]

    return solution


def diagonal_peaks_anchors(sim_matrix, threshold=0.55, min_gap=2):
    """Find mutual best-match anchors near the expected diagonal.

    Returns list of (from_idx, to_idx, confidence) in 0-based matrix indices.
    """
    n, m = sim_matrix.shape
    if n == 0 or m == 0:
        return []

    ratio = m / n
    best_to_for_from = np.argmax(sim_matrix, axis=1)
    best_from_for_to = np.argmax(sim_matrix, axis=0)
    best_sim_for_from = np.max(sim_matrix, axis=1)

    anchors = []
    for i in range(n):
        j = int(best_to_for_from[i])
        if best_from_for_to[j] == i:
            sim_val = float(best_sim_for_from[i])
            expected_j = i * ratio
            diagonal_dist = abs(j - expected_j) / max(m, 1)
            if sim_val >= threshold and diagonal_dist < 0.15:
                anchors.append((i, j, sim_val))

    filtered = []
    for a in anchors:
        if not filtered or (
            a[0] - filtered[-1][0] >= min_gap
            and a[1] - filtered[-1][1] >= min_gap
        ):
            filtered.append(a)

    return filtered


def compute_conflict_sim_matrix(db_path, conflict, use_proxy_from=False, use_proxy_to=False):
    """Compute cosine similarity matrix for a conflict's line embeddings.

    Returns (sim_matrix, from_ids, to_ids) where from_ids/to_ids are lists
    of document-level IDs corresponding to matrix rows/columns.
    """
    from_start = conflict["from"]["start"][0]
    from_end = conflict["from"]["end"][0]
    to_start = conflict["to"]["start"][0]
    to_end = conflict["to"]["end"][0]

    from_ids = list(range(from_start, from_end + 1))
    to_ids = list(range(to_start, to_end + 1))

    if not from_ids or not to_ids:
        return np.zeros((0, 0)), from_ids, to_ids

    emb_from_raw = dict(
        helper.get_embeddings(db_path, "from", from_ids, is_proxy=use_proxy_from)
    )
    emb_to_raw = dict(
        helper.get_embeddings(db_path, "to", to_ids, is_proxy=use_proxy_to)
    )

    # Check for missing embeddings
    missing_from = [fid for fid in from_ids if emb_from_raw.get(fid) is None]
    missing_to = [tid for tid in to_ids if emb_to_raw.get(tid) is None]
    if missing_from or missing_to:
        logging.warning(
            "Missing embeddings for conflict splitting: from=%s, to=%s",
            missing_from, missing_to,
        )
        # Remove IDs with missing embeddings from the matrix
        from_ids = [fid for fid in from_ids if emb_from_raw.get(fid) is not None]
        to_ids = [tid for tid in to_ids if emb_to_raw.get(tid) is not None]
        if not from_ids or not to_ids:
            return np.zeros((0, 0)), from_ids, to_ids

    emb_from = np.array([emb_from_raw[fid] for fid in from_ids], dtype=np.float32)
    emb_to = np.array([emb_to_raw[tid] for tid in to_ids], dtype=np.float32)

    # L2 normalize
    norms_from = np.linalg.norm(emb_from, axis=1, keepdims=True)
    norms_to = np.linalg.norm(emb_to, axis=1, keepdims=True)
    norms_from = np.where(norms_from < 1e-10, 1.0, norms_from)
    norms_to = np.where(norms_to < 1e-10, 1.0, norms_to)
    emb_from = emb_from / norms_from
    emb_to = emb_to / norms_to

    sim_matrix = emb_from @ emb_to.T
    return sim_matrix, from_ids, to_ids


def find_conflict_anchors(
    db_path,
    conflict,
    use_proxy_from=False,
    use_proxy_to=False,
    use_cross_validation=False,
    dp_threshold=0.5,
    peak_threshold=0.55,
    high_confidence_threshold=0.8,
):
    """Find anchor points within a large conflict using Penalized DP.

    Returns list of (from_doc_id, to_doc_id, confidence).
    """
    sim_matrix, from_ids, to_ids = compute_conflict_sim_matrix(
        db_path, conflict, use_proxy_from, use_proxy_to,
    )
    if sim_matrix.size == 0:
        return [], from_ids, to_ids

    dp_anchors = penalized_dp_anchors(
        sim_matrix, anchor_threshold=dp_threshold,
    )

    if use_cross_validation and dp_anchors:
        peak_anchors = diagonal_peaks_anchors(
            sim_matrix, threshold=peak_threshold,
        )
        peak_set = set((a[0], a[1]) for a in peak_anchors)
        confirmed = []
        for fi, ti, conf in dp_anchors:
            near_peak = any(
                abs(fi - pi) <= 1 and abs(ti - pj) <= 1
                for pi, pj in peak_set
            )
            if near_peak or conf >= high_confidence_threshold:
                confirmed.append((fi, ti, conf))
        dp_anchors = confirmed

    # Map matrix indices back to document IDs
    doc_anchors = [
        (from_ids[fi], to_ids[ti], conf)
        for fi, ti, conf in dp_anchors
    ]

    return doc_anchors, from_ids, to_ids


def _build_sub_conflict(conflict, from_start_id, from_end_id, to_start_id, to_end_id):
    """Build a sub-conflict dict using the parent conflict's batch/sub coordinates."""
    parent_batch_id = conflict["from"]["start"][1]
    parent_sub_id = conflict["from"]["start"][2]
    return {
        "from": {
            "start": (from_start_id, parent_batch_id, parent_sub_id),
            "end": (from_end_id, parent_batch_id, parent_sub_id),
        },
        "to": {
            "start": (to_start_id, parent_batch_id, parent_sub_id),
            "end": (to_end_id, parent_batch_id, parent_sub_id),
        },
    }


def squash_conflict_with_splitting(
    db_path,
    conflict,
    model_name,
    show_logs=False,
    model=None,
    use_proxy_from=False,
    use_proxy_to=False,
    lang_emb_from="ell_Grek",
    lang_emb_to="ell_Grek",
    use_aggregation=False,
    aggregation_method="weighted_average",
    power_scoring_exponent=0.6,
    _depth=0,
):
    """Find the best solution for a large conflict by splitting via DP anchors.

    Same return signature as squash_conflict: (solution, lines_from, lines_to).
    """
    from_start = conflict["from"]["start"][0]
    from_end = conflict["from"]["end"][0]
    to_start = conflict["to"]["start"][0]
    to_end = conflict["to"]["end"][0]

    splitted_from, proxy_from = helper.get_splitted_from_by_id_range(
        db_path, from_start, from_end,
    )
    splitted_to, proxy_to = helper.get_splitted_to_by_id_range(
        db_path, to_start, to_end,
    )

    # Find anchors
    doc_anchors, from_ids, to_ids = find_conflict_anchors(
        db_path, conflict, use_proxy_from, use_proxy_to,
    )

    if not doc_anchors:
        n = from_end - from_start + 1
        m = to_end - to_start + 1
        if n <= MAX_DIRECT_RESOLVE_SIZE and m <= MAX_DIRECT_RESOLVE_SIZE:
            logging.info(
                "No anchors for conflict %d-%d : %d-%d (small enough for direct resolve)",
                from_start, from_end, to_start, to_end,
            )
            return _squash_conflict_direct(
                db_path, conflict, model_name, show_logs, model,
                use_proxy_from, use_proxy_to, lang_emb_from, lang_emb_to,
                use_aggregation, aggregation_method, power_scoring_exponent,
            )

        # Conflict is too large for brute-force — try DP with lower threshold
        sim_matrix, sim_from_ids, sim_to_ids = compute_conflict_sim_matrix(
            db_path, conflict, use_proxy_from, use_proxy_to,
        )
        if sim_matrix.size > 0:
            # Retry with progressively lower thresholds
            for retry_threshold in [0.35, 0.2]:
                retry_anchors = penalized_dp_anchors(
                    sim_matrix, anchor_threshold=retry_threshold,
                )
                if retry_anchors:
                    doc_anchors = [
                        (sim_from_ids[fi], sim_to_ids[ti], conf)
                        for fi, ti, conf in retry_anchors
                    ]
                    logging.info(
                        "Retry with threshold=%.2f found %d anchor(s) for conflict %d-%d : %d-%d",
                        retry_threshold, len(doc_anchors), from_start, from_end, to_start, to_end,
                    )
                    break

        if not doc_anchors:
            # Last resort: use the DP path itself as the alignment solution
            logging.info(
                "No anchors even at low threshold for conflict %d-%d : %d-%d, "
                "using DP path as solution",
                from_start, from_end, to_start, to_end,
            )
            if sim_matrix.size > 0:
                path = _run_penalized_dp(sim_matrix)
                solution = dp_path_to_solution(path, sim_from_ids, sim_to_ids)
            else:
                # Degenerate: no embeddings — merge everything
                solution = [(
                    tuple(range(from_start, from_end + 1)),
                    tuple(range(to_start, to_end + 1)),
                )]
            return solution, splitted_from, splitted_to

    logging.info(
        "Splitting conflict %d-%d : %d-%d with %d anchor(s)",
        from_start, from_end, to_start, to_end, len(doc_anchors),
    )

    # Build sub-regions: gaps between anchors + anchor pairs
    combined_solution = []
    prev_f = from_start
    prev_t = to_start

    resolve_kwargs = dict(
        model_name=model_name, show_logs=show_logs, model=model,
        use_proxy_from=use_proxy_from, use_proxy_to=use_proxy_to,
        lang_emb_from=lang_emb_from, lang_emb_to=lang_emb_to,
        use_aggregation=use_aggregation, aggregation_method=aggregation_method,
        power_scoring_exponent=power_scoring_exponent,
    )

    for anchor_f, anchor_t, anchor_conf in doc_anchors:
        # Gap before this anchor
        gap_f_start, gap_f_end = prev_f, anchor_f - 1
        gap_t_start, gap_t_end = prev_t, anchor_t - 1
        gap_n = gap_f_end - gap_f_start + 1
        gap_m = gap_t_end - gap_t_start + 1

        if gap_n > 0 and gap_m > 0:
            gap_conflict = _build_sub_conflict(
                conflict, gap_f_start, gap_f_end, gap_t_start, gap_t_end,
            )
            gap_solution = _resolve_sub_conflict(
                db_path, gap_conflict, _depth=_depth, **resolve_kwargs,
            )
            combined_solution.extend(gap_solution)
        elif gap_n > 0:
            # Extra from-lines with no to-lines: merge them into the anchor
            combined_solution.append(
                (tuple(range(gap_f_start, anchor_f + 1)), (anchor_t,))
            )
            prev_f = anchor_f + 1
            prev_t = anchor_t + 1
            continue
        elif gap_m > 0:
            # Extra to-lines with no from-lines: merge them into the anchor
            combined_solution.append(
                ((anchor_f,), tuple(range(gap_t_start, anchor_t + 1)))
            )
            prev_f = anchor_f + 1
            prev_t = anchor_t + 1
            continue

        # The anchor itself: 1:1 pair
        combined_solution.append(((anchor_f,), (anchor_t,)))
        prev_f = anchor_f + 1
        prev_t = anchor_t + 1

    # Trailing gap after last anchor
    if prev_f <= from_end and prev_t <= to_end:
        tail_conflict = _build_sub_conflict(
            conflict, prev_f, from_end, prev_t, to_end,
        )
        tail_solution = _resolve_sub_conflict(
            db_path, tail_conflict, _depth=_depth, **resolve_kwargs,
        )
        combined_solution.extend(tail_solution)
    elif prev_f <= from_end:
        # Extra trailing from-lines: merge into last pair
        if combined_solution:
            last_from, last_to = combined_solution[-1]
            combined_solution[-1] = (
                last_from + tuple(range(prev_f, from_end + 1)),
                last_to,
            )
        else:
            combined_solution.append(
                (tuple(range(prev_f, from_end + 1)), (to_end,))
            )
    elif prev_t <= to_end:
        # Extra trailing to-lines: merge into last pair
        if combined_solution:
            last_from, last_to = combined_solution[-1]
            combined_solution[-1] = (
                last_from,
                last_to + tuple(range(prev_t, to_end + 1)),
            )
        else:
            combined_solution.append(
                ((from_end,), tuple(range(prev_t, to_end + 1)))
            )

    return combined_solution, splitted_from, splitted_to


def _resolve_sub_conflict(
    db_path,
    sub_conflict,
    model_name,
    show_logs=False,
    model=None,
    use_proxy_from=False,
    use_proxy_to=False,
    lang_emb_from="ell_Grek",
    lang_emb_to="ell_Grek",
    use_aggregation=False,
    aggregation_method="weighted_average",
    power_scoring_exponent=0.6,
    _depth=0,
):
    """Resolve a sub-conflict, recursing into splitting if still too large."""
    n = sub_conflict["from"]["end"][0] - sub_conflict["from"]["start"][0] + 1
    m = sub_conflict["to"]["end"][0] - sub_conflict["to"]["start"][0] + 1

    if n <= 0 or m <= 0:
        return []

    # Trivial 1:1
    if n == 1 and m == 1:
        return [(
            (sub_conflict["from"]["start"][0],),
            (sub_conflict["to"]["start"][0],),
        )]

    if n > MAX_DIRECT_RESOLVE_SIZE or m > MAX_DIRECT_RESOLVE_SIZE:
        if _depth < 3:
            solution, _, _ = squash_conflict_with_splitting(
                db_path, sub_conflict, model_name, show_logs, model,
                use_proxy_from, use_proxy_to, lang_emb_from, lang_emb_to,
                use_aggregation, aggregation_method, power_scoring_exponent,
                _depth=_depth + 1,
            )
            return solution

        # Max recursion depth — use DP path directly instead of brute-force
        logging.info(
            "Max split depth reached for sub-conflict %d-%d : %d-%d, using DP path",
            sub_conflict["from"]["start"][0], sub_conflict["from"]["end"][0],
            sub_conflict["to"]["start"][0], sub_conflict["to"]["end"][0],
        )
        sim_matrix, from_ids, to_ids = compute_conflict_sim_matrix(
            db_path, sub_conflict, use_proxy_from, use_proxy_to,
        )
        if sim_matrix.size > 0:
            path = _run_penalized_dp(sim_matrix)
            return dp_path_to_solution(path, from_ids, to_ids)
        # Degenerate fallback
        return [(
            tuple(range(sub_conflict["from"]["start"][0], sub_conflict["from"]["end"][0] + 1)),
            tuple(range(sub_conflict["to"]["start"][0], sub_conflict["to"]["end"][0] + 1)),
        )]

    # Small enough: use direct variant enumeration
    solution, _, _ = _squash_conflict_direct(
        db_path, sub_conflict, model_name, show_logs, model,
        use_proxy_from, use_proxy_to, lang_emb_from, lang_emb_to,
        use_aggregation, aggregation_method, power_scoring_exponent,
    )
    return solution


def _squash_conflict_direct(
    db_path,
    conflict,
    model_name,
    show_logs=False,
    model=None,
    use_proxy_from=False,
    use_proxy_to=False,
    lang_emb_from="ell_Grek",
    lang_emb_to="ell_Grek",
    use_aggregation=False,
    aggregation_method="weighted_average",
    power_scoring_exponent=0.6,
):
    """Original brute-force variant enumeration for small conflicts."""
    splitted_from, proxy_from = helper.get_splitted_from_by_id_range(
        db_path, conflict["from"]["start"][0], conflict["from"]["end"][0]
    )
    splitted_to, proxy_to = helper.get_splitted_to_by_id_range(
        db_path, conflict["to"]["start"][0], conflict["to"]["end"][0]
    )

    variants_ids = get_variants(conflict, show_logs)
    unique_variants = helper.get_unique_variants(variants_ids)

    vec_lines_from = proxy_from if use_proxy_from else splitted_from
    vec_lines_to = proxy_to if use_proxy_to else splitted_to

    vecs_from, vecs_to = get_vectors(
        db_path, unique_variants, vec_lines_from, vec_lines_to,
        model_name, model, lang_emb_from, lang_emb_to,
        use_proxy_from=use_proxy_from, use_proxy_to=use_proxy_to,
        use_aggregation=use_aggregation, aggregation_method=aggregation_method,
    )

    unique_sims = get_unique_sims(unique_variants, vecs_from, vecs_to)

    for key in unique_sims:
        from_ids, to_ids = key
        text_from = helper.get_string(splitted_from, from_ids)
        text_to = helper.get_string(splitted_to, to_ids)
        unique_sims[key] += punct_sim.punct_bonus_for_texts(text_from, text_to)

    variant_sims = [
        sum(unique_sims[id] for id in ids) / (len(ids) ** power_scoring_exponent)
        for ids in variants_ids
    ]
    best_var_index = int(np.argmax(variant_sims))

    return variants_ids[best_var_index], splitted_from, splitted_to


# ---------------------------------------------------------------------------
# Negative-length conflict expansion
# ---------------------------------------------------------------------------


def fix_negative_conflicts(db_path, conflicts, batch_id=-1):
    """Fix conflicts with negative length by expanding their boundaries.

    Instead of the old +1 increment hack, this expands overlapping regions
    into valid positive-length conflicts by swapping boundaries and adjusting
    the doc_index to remove confused entries in the overlap zone.

    Returns the number of fixed conflicts.
    """
    negative_conflicts = []
    for c in conflicts:
        len_from = c["from"]["end"][0] - c["from"]["start"][0]
        len_to = c["to"]["end"][0] - c["to"]["start"][0]
        if len_from < 0 or len_to < 0:
            negative_conflicts.append(c)

    if not negative_conflicts:
        return 0

    logging.info("Fixing %d negative-length conflict(s) by boundary expansion", len(negative_conflicts))

    with sqlite3.connect(db_path) as db:
        index = aligner.get_doc_index(db)

    fixed = 0
    for c in negative_conflicts:
        f_start = c["from"]["start"][0]
        f_end = c["from"]["end"][0]
        t_start = c["to"]["start"][0]
        t_end = c["to"]["end"][0]

        # Get the batch_id and sub_id coordinates
        start_batch = c["from"]["start"][1]
        start_sub = c["from"]["start"][2]
        end_batch = c["from"]["end"][1]
        end_sub = c["from"]["end"][2]

        # Expand: ensure start <= end on both sides
        new_f_start = min(f_start, f_end)
        new_f_end = max(f_start, f_end)
        new_t_start = min(t_start, t_end)
        new_t_end = max(t_start, t_end)

        # Fix the index entries at the conflict boundaries.
        # The start boundary entry has to_ids pointing backwards — adjust it
        # to point forward by setting its to_ids to [new_t_start].
        if start_batch < len(index) and start_sub < len(index[start_batch]):
            entry = list(index[start_batch][start_sub])
            to_ids = json.loads(entry[3])
            # Replace with the corrected minimum to_id
            corrected_to = [new_t_start] + [x for x in to_ids if x > new_t_start]
            if not corrected_to:
                corrected_to = [new_t_start]
            entry[3] = json.dumps(corrected_to)
            index[start_batch][start_sub] = entry
            fixed += 1

    if fixed > 0:
        logging.info("Expanded %d negative conflict(s), updating index", fixed)
        with sqlite3.connect(db_path) as db:
            aligner.update_doc_index(db, index)

    return fixed


def get_conflict_coordinates(conflict):
    """Get conflict coordinates"""
    return (conflict["from"]["start"][1], conflict["from"]["start"][2]), (
        conflict["from"]["end"][1],
        conflict["from"]["end"][2],
    )


def squash_conflict(
    db_path,
    conflict,
    model_name,
    show_logs=False,
    model=None,
    use_proxy_from=False,
    use_proxy_to=False,
    lang_emb_from="ell_Grek",
    lang_emb_to="ell_Grek",
    use_aggregation=False,
    aggregation_method="weighted_average",
    power_scoring_exponent=0.6,
):
    """Find the best solution (auto-dispatches to splitting for large conflicts)."""
    n = conflict["from"]["end"][0] - conflict["from"]["start"][0] + 1
    m = conflict["to"]["end"][0] - conflict["to"]["start"][0] + 1

    if n > MAX_DIRECT_RESOLVE_SIZE or m > MAX_DIRECT_RESOLVE_SIZE:
        return squash_conflict_with_splitting(
            db_path, conflict, model_name, show_logs, model,
            use_proxy_from, use_proxy_to, lang_emb_from, lang_emb_to,
            use_aggregation, aggregation_method, power_scoring_exponent,
        )

    return _squash_conflict_direct(
        db_path, conflict, model_name, show_logs, model,
        use_proxy_from, use_proxy_to, lang_emb_from, lang_emb_to,
        use_aggregation, aggregation_method, power_scoring_exponent,
    )


def resolve_conflict(
    db_path, conflict, solution, lines_from, lines_to, show_logs=False
):
    """Apply the solution to the database"""
    # (batch_id, sub_id)
    start, end = get_conflict_coordinates(conflict)
    if show_logs:
        print("start, end", start, end, "\n")

    index_solution = []

    with sqlite3.connect(db_path) as db:
        index_for_update = aligner.get_doc_index(db)

        for line in solution:
            from_id, to_id = helper.add_resolved_processing_line(
                db,
                start[0],
                helper.get_string(lines_from, line[0]),
                helper.get_string(lines_to, line[1]),
            )
            index_solution.append(
                (from_id, json.dumps(line[0]), to_id, json.dumps(line[1]))
            )

        if show_logs:
            print("\n---------")
            print("index_solution", index_solution)
            print("\n========================================================\n")

        # detect if solution is between the batches
        if start[0] == end[0]:
            index_for_update[start[0]][start[1] : end[1] + 1] = index_solution
        else:
            index_for_update[start[0]][start[1] :] = index_solution
            index_for_update[end[0]][: end[1] + 1] = []

        aligner.update_doc_index(db, index_for_update)


def show_conflict(db_path, conflict, print_conf=True):
    """Print the conflict information"""
    splitted_from, _ = helper.get_splitted_from_by_id_range(
        db_path, conflict["from"]["start"][0], conflict["from"]["end"][0]
    )
    splitted_to, _ = helper.get_splitted_to_by_id_range(
        db_path, conflict["to"]["start"][0], conflict["to"]["end"][0]
    )
    if print_conf:
        for i, id in enumerate(splitted_from):
            print(id, splitted_from[id])
        print("\n")
        for i, id in enumerate(splitted_to):
            print(id, splitted_to[id])
        print("-----------------------------------------------")
    return splitted_from, splitted_to


def get_statistics(conflicts, print_stat=True):
    """Print the conflicts statistics"""
    statistics = defaultdict(int)
    for i, c in enumerate(conflicts):
        len_from = c["from"]["end"][0] - c["from"]["start"][0] + 1
        len_to = c["to"]["end"][0] - c["to"]["start"][0] + 1
        conflict_type = f"{len_from}:{len_to}"
        statistics[conflict_type] += 1
    table = sorted(statistics.items(), key=lambda x: x[1], reverse=True)
    if print_stat:
        for x in table:
            print(x)
    return statistics


def get_all_conflicts(
    db_path,
    min_chain_length=3,
    max_conflicts_len=6,
    batch_id=-1,
    handle_start=False,
    handle_finish=False,
    index=None,
):
    """Get conflicts to solve and other"""
    splitted_from_len = len(aligner.get_splitted_from(db_path))
    splitted_to_len = len(aligner.get_splitted_to(db_path))
    prepared_index, total_batches = prepare_index(db_path, batch_id, index=index)
    if not prepared_index:
        return [], []

    if total_batches != 1:
        if batch_id > 0:
            handle_start = False
        if batch_id < total_batches - 1:
            handle_finish = False

    # print(
    #     "get_all_conflicts, handle_start:",
    #     handle_start,
    #     "handle_finish:",
    #     handle_finish,
    #     "batch_id",
    #     batch_id,
    # )

    chains_from, chains_to = get_good_chains(
        prepared_index,
        min_len=min_chain_length,
        handle_start=handle_start,
        handle_finish=handle_finish,
        len_from=splitted_from_len,
        len_to=splitted_to_len,
    )
    conflicts_to_solve, conflicts_rest = get_conflicts(
        chains_from, chains_to, max_len=max_conflicts_len
    )
    return conflicts_to_solve, conflicts_rest


def calculate_conflicts_amount_by_index(
    db_path,
    index,
    batch_id=-1,
    min_chain_length=2,
    max_conflicts_len=26,
    handle_start=False,
    handle_finish=False,
):
    """Calculate unused conflicts amount using index only"""
    _, rest = get_all_conflicts(
        db_path,
        min_chain_length,
        max_conflicts_len,
        batch_id,
        handle_start,
        handle_finish,
        index,
    )
    return len(rest)


def resolve_all_conflicts(
    db_path,
    conflicts,
    model_name,
    show_logs=False,
    model=None,
    use_proxy_from=False,
    use_proxy_to=False,
    lang_emb_from="ell_Grek",
    lang_emb_to="ell_Grek",
    use_aggregation=False,
    aggregation_method="weighted_average",
):
    """Apply all the solutions to the database"""
    for _, conflict in enumerate(tqdm(conflicts[::-1])):
        solution, lines_from, lines_to = squash_conflict(
            db_path,
            conflict,
            model_name,
            show_logs,
            model,
            use_proxy_from,
            use_proxy_to,
            lang_emb_from,
            lang_emb_to,
            use_aggregation,
            aggregation_method,
        )
        resolve_conflict(db_path, conflict, solution, lines_from, lines_to, show_logs)


def fix_start(
    db_path,
    model_name,
    max_conflicts_len=6,
    show_logs=False,
    model=None,
    use_proxy_from=False,
    use_proxy_to=False,
    lang_emb_from="ell_Grek",
    lang_emb_to="ell_Grek",
    use_aggregation=False,
    aggregation_method="weighted_average",
):
    """Find the first conflict and resolve"""
    splitted_from_len = len(aligner.get_splitted_from(db_path))
    splitted_to_len = len(aligner.get_splitted_to(db_path))
    prepared_index, _ = prepare_index(db_path, 0)
    chains_from, chains_to = get_good_chains(
        prepared_index,
        min_len=2,
        handle_start=True,
        len_from=splitted_from_len,
        len_to=splitted_to_len,
    )
    conflicts_to_solve, _ = get_conflicts(
        chains_from, chains_to, max_len=max_conflicts_len
    )
    resolve_all_conflicts(
        db_path,
        conflicts_to_solve,
        model_name,
        show_logs,
        model,
        use_proxy_from,
        use_proxy_to,
        lang_emb_from,
        lang_emb_to,
        use_aggregation,
        aggregation_method,
    )


def correct_conflicts(
    db_path,
    conflicts,
    batch_id=-1,
    min_chain_length=2,
    max_conflicts_len=26,
    handle_start=False,
    handle_finish=False,
):
    """Handle case with negative conflict's length"""

    logging.info(
        "Trying to decrease a number of unused conflicts. Fixing negative lenghts."
    )

    negative_conflicts_from, negative_conflicts_to = [], []
    for c in conflicts:
        len_from = c["from"]["end"][0] - c["from"]["start"][0] + 1
        len_to = c["to"]["end"][0] - c["to"]["start"][0] + 1
        if len_from < 0:
            negative_conflicts_from.append(c)
        if len_to < 0:
            negative_conflicts_to.append(c)

    logging.info(f"Found {len(negative_conflicts_to)} conflicts with negative length.")

    with sqlite3.connect(db_path) as db:
        index = aligner.get_doc_index(db)

    curr_conf_len = calculate_conflicts_amount_by_index(
        db_path,
        index,
        batch_id,
        min_chain_length,
        max_conflicts_len,
        handle_start,
        handle_finish,
    )

    fixed_conflicts = 0
    for n_conf in negative_conflicts_to:
        start, end = get_conflict_coordinates(n_conf)

        index_copy = try_fix_conflict_ending(index, start)
        conf_len = calculate_conflicts_amount_by_index(
            db_path,
            index_copy,
            batch_id,
            min_chain_length,
            max_conflicts_len,
            handle_start,
            handle_finish,
        )
        if conf_len != curr_conf_len:
            index = index_copy
            curr_conf_len = conf_len
            fixed_conflicts += 1
            continue

        index_copy = try_fix_conflict_ending(index, end)
        conf_len = calculate_conflicts_amount_by_index(
            db_path,
            index_copy,
            batch_id,
            min_chain_length,
            max_conflicts_len,
            handle_start,
            handle_finish,
        )
        if conf_len != curr_conf_len:
            index = index_copy
            fixed_conflicts += 1

    logging.info(f"{fixed_conflicts} was fixed.")
    if fixed_conflicts > 0:
        logging.info("Updating index.")
        with sqlite3.connect(db_path) as db:
            aligner.update_doc_index(db, index)

    return fixed_conflicts


def try_fix_conflict_ending(index, ending_coordinate):
    """Try to fix the conflict ending"""
    index_copy = copy.deepcopy(index)
    conf_start = index_copy[ending_coordinate[0]][ending_coordinate[1]]
    conf_start_to = json.loads(conf_start[3])  # [159, '[160]', 159, '[243]']
    candidate = json.dumps([conf_start_to[0] + 1] + conf_start_to[1:])

    print("conf_start_to, candidate", conf_start_to, candidate)
    conf_start[3] = candidate

    index_copy[ending_coordinate[0]][ending_coordinate[1]] = conf_start

    return index_copy


def get_vectors(
    db_path,
    unique_variants,
    splitted_from,
    splitted_to,
    model_name,
    model=None,
    lang_emb_from="ell_Grek",
    lang_emb_to="ell_Grek",
    use_proxy_from=False,
    use_proxy_to=False,
    use_aggregation=False,
    aggregation_method="weighted_average",
):
    """Get embeddings for unique variants"""

    strings_from = []
    strings_to = []
    for x in unique_variants:
        strings_from.append(helper.get_string(splitted_from, x[0]))
        strings_to.append(helper.get_string(splitted_to, x[1]))

    # print("strings_from", len(strings_from), strings_from)
    # print("strings_to", len(strings_from), strings_to)

    if not use_aggregation:
        # print("Generating embeddings for unique variants")
        return (
            aligner.get_line_vectors(
                strings_from, model_name, model=model, lang=lang_emb_from
            ),
            aligner.get_line_vectors(
                strings_to, model_name, model=model, lang=lang_emb_to
            ),
        )
    else:
        # print("Aggregating embeddings for unique variants")
        embeddings_from, embeddings_to = [], []
        sent_lens_from, sent_lens_to = [], []

        # Batch: collect all needed IDs across variants, query once per direction
        all_from_ids = set()
        all_to_ids = set()
        for line_ids in unique_variants:
            all_from_ids.update(line_ids[0])
            all_to_ids.update(line_ids[1])

        all_emb_from = dict(helper.get_embeddings(
            db_path, direction="from", line_ids=list(all_from_ids), is_proxy=use_proxy_from
        ))
        all_emb_to = dict(helper.get_embeddings(
            db_path, direction="to", line_ids=list(all_to_ids), is_proxy=use_proxy_to
        ))

        # If any embedding is missing, fall back to computing fresh embeddings
        has_missing = (
            any(v is None for v in all_emb_from.values())
            or any(v is None for v in all_emb_to.values())
        )
        if has_missing:
            return (
                aligner.get_line_vectors(
                    strings_from, model_name, model=model, lang=lang_emb_from
                ),
                aligner.get_line_vectors(
                    strings_to, model_name, model=model, lang=lang_emb_to
                ),
            )

        for line_ids in unique_variants:
            sent_lens_from.append(helper.get_string_lens(splitted_from, line_ids[0]))
            sent_lens_to.append(helper.get_string_lens(splitted_to, line_ids[1]))
            embeddings_from.append([all_emb_from[id] for id in line_ids[0]])
            embeddings_to.append([all_emb_to[id] for id in line_ids[1]])

        # print("embeddings_from", len(embeddings_from), [len(x) for x in embeddings_from])
        # print("embeddings_to", len(embeddings_to), [len(x) for x in embeddings_to])

        aggregated_from = []
        for i, x in enumerate(embeddings_from):
            aggregated_from.append(
                aggregate_embeddings(x, sent_lens_from[i], aggregation_method)
            )
        aggregated_to = []
        for i, x in enumerate(embeddings_to):
            aggregated_to.append(aggregate_embeddings(x, sent_lens_to[i], aggregation_method))

        return (aggregated_from, aggregated_to)


def get_unique_sims(unique_variants, vecs_from, vecs_to):
    """Calculate unique similarities (vectorized cosine similarity)"""
    vf = np.array(vecs_from)
    vt = np.array(vecs_to)
    dots = np.sum(vf * vt, axis=1)
    norms = np.linalg.norm(vf, axis=1) * np.linalg.norm(vt, axis=1)
    norms = np.where(norms < 1e-10, 1.0, norms)
    sims = dots / norms
    return {x: sims[i] for i, x in enumerate(unique_variants)}


def aggregate_embeddings(embeddings, sentence_lengths, method, **kwargs):
    """
    Embedding aggregation function with multiple methods.

    Parameters
    ----------
    embeddings : list or np.ndarray
        A list (or array) of sentence embeddings. Shape: (n_sentences, embedding_dim).
    sentence_lengths : list or np.ndarray
        Lengths (e.g., token counts) corresponding to each sentence. Shape: (n_sentences,).
    method : str
        Aggregation method. One of:
        ["weighted_average", "length_scaling", "max_pooling", "logarithmic_scaling"].
    kwargs : dict
        Additional parameters for certain methods.
        - For "logarithmic_scaling", you can pass {"offset": float} to modify the log offset.
    
    Returns
    -------
    np.ndarray
        The aggregated embedding of shape (embedding_dim,).
    """
    if len(embeddings) == 0:
        raise ValueError("No embeddings provided.")
    if method not in {
        "weighted_average",
        "length_scaling",
        "max_pooling",
        "logarithmic_scaling",
    }:
        raise ValueError(
            "Unknown method. Choose one of: "
            "'weighted_average', 'length_scaling', 'max_pooling', 'logarithmic_scaling'."
        )

    embeddings = np.array(embeddings)
    sentence_lengths = np.array(sentence_lengths)

    # Handle edge cases
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be 2-dimensional (n_sentences x embedding_dim).")
    if len(sentence_lengths) != embeddings.shape[0]:
        raise ValueError("sentence_lengths must match the number of embeddings.")

    if method == "weighted_average":
        # Weighted by sentence length, then average
        weights = sentence_lengths / np.sum(sentence_lengths)
        aggregated_embedding = np.average(embeddings, axis=0, weights=weights)

    elif method == "length_scaling":
        # Multiply each embedding by its corresponding length, then mean
        scaled_embeddings = embeddings * sentence_lengths[:, None]
        aggregated_embedding = np.mean(scaled_embeddings, axis=0)

    elif method == "max_pooling":
        # Take the component-wise maximum across all embeddings
        aggregated_embedding = np.max(embeddings, axis=0)

    elif method == "logarithmic_scaling":
        # Weight embeddings by log(1 + length), or a custom offset
        offset = kwargs.get("offset", 1.0)  # default offset=1
        log_lengths = np.log(offset + sentence_lengths)
        scaled_embeddings = embeddings * log_lengths[:, None]
        aggregated_embedding = np.mean(scaled_embeddings, axis=0)

    # Normalize the final embedding
    norm = np.linalg.norm(aggregated_embedding)
    if norm < 1e-10:
        raise ValueError("Norm of aggregated embedding is zero.")

    return aggregated_embedding / norm
