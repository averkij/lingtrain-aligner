"""Conflicts resolver part of the engine"""

import json
import sqlite3
from collections import defaultdict

import more_itertools as mit
from lingtrain_aligner import aligner, helper
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


def get_variants(conflict, show_logs=False):
    """Get resolving variants"""
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

    groups = min(len(ids_from), len(ids_to))
    res = []
    if len(ids_from) < len(ids_to):
        grouped_subs = [x for x in mit.partitions(ids_to) if len(x) == groups]
        for _, sub in enumerate(grouped_subs):
            res.append([((a,), tuple(b)) for a, b in zip(ids_from, sub)])
    else:
        grouped_subs = [x for x in mit.partitions(ids_from) if len(x) == groups]
        for _, sub in enumerate(grouped_subs):
            res.append([(tuple(a), (b,)) for a, b in zip(sub, ids_to)])
    return res


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
):
    """Find the best solution"""
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
        db_path,
        unique_variants,
        vec_lines_from,
        vec_lines_to,
        model_name,
        model,
        lang_emb_from,
        lang_emb_to,
        use_proxy_from=use_proxy_from,
        use_proxy_to=use_proxy_to,
        use_aggregation=use_aggregation,
        aggregation_method=aggregation_method,
    )

    unique_sims = get_unique_sims(unique_variants, vecs_from, vecs_to)

    # for key in unique_sims:
    #     print(get_string(splitted_from, key[0]), "<->", get_string(splitted_to, key[1]), "->", unique_sims[key],"\n")

    variant_sims = [sum(unique_sims[id] for id in ids) for ids in variants_ids]
    best_var_index = int(np.argmax(variant_sims))

    if show_logs:
        print("best variant:")
        print(variants_ids[best_var_index])
        print("\n---------------------------------------\n")
        for ids in variants_ids[best_var_index]:
            print(
                helper.get_string(splitted_from, ids[0]),
                "<=>",
                helper.get_string(splitted_to, ids[1]),
                "\n",
            )

    return variants_ids[best_var_index], splitted_from, splitted_to


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
