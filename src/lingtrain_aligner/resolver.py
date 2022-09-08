"""Conflicts resolver part of the engine"""


import json
import sqlite3
from collections import defaultdict

import more_itertools as mit
from lingtrain_aligner import aligner, helper
from scipy import spatial
from tqdm import tqdm


def prepare_index(db_path, batch_id=-1):
    """Get totally flattened index ids"""
    res = []
    if batch_id >= 0:
        index_orig = helper.get_doc_index_original(db_path)
        total_batches = len(index_orig)
        for i, ix in enumerate(index_orig[batch_id]):
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
        index = helper.get_flatten_doc_index_with_batch_id(db_path)
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
    res, min_len=2, handle_start=False, handle_finish=False, len_from=-1, len_to=-1
):
    """Calculate valid alignment chains"""

    curr_from = res[0]["from"][0]
    curr_to = res[0]["to"]

    chains_from = []
    chains_to = []

    chain_from = [(curr_from, res[0]["batch_id"], res[0]["sub_id"])]
    chain_to = [(curr_to, res[0]["batch_id"], res[0]["sub_id"])]

    start = 1

    if handle_start and curr_to != 1:
        chains_from.append(chain_from)
        chains_to.append([(1, res[0]["batch_id"], res[0]["sub_id"])])
        chain_from = [(res[1]["from"][0], res[1]["batch_id"], res[1]["sub_id"])]
        chain_to = [(res[1]["to"], res[1]["batch_id"], res[1]["sub_id"])]
        curr_from = res[1]["from"][0]
        curr_to = res[1]["to"]
        start = 2

    for i in range(start, len(res)):
        val_from = res[i]["from"][0]
        val_to = res[i]["to"]

        # continue chain
        if val_to == curr_to + 1:
            chain_from.append((val_from, res[i]["batch_id"], res[i]["sub_id"]))
            chain_to.append((val_to, res[i]["batch_id"], res[i]["sub_id"]))
            curr_from = val_from
            curr_to = val_to

        # add chain and start new
        elif len(chain_to) >= min_len:
            chains_from.append(chain_from)
            chains_to.append(chain_to)
            chain_from = [(val_from, res[i]["batch_id"], res[i]["sub_id"])]
            chain_to = [(val_to, res[i]["batch_id"], res[i]["sub_id"])]
            curr_from = val_from
            curr_to = val_to

        # start new chain
        else:
            chain_from = [(val_from, res[i]["batch_id"], res[i]["sub_id"])]
            chain_to = [(val_to, res[i]["batch_id"], res[i]["sub_id"])]
            curr_from = val_from
            curr_to = val_to

    if len(chain_to) >= min_len:
        chains_from.append(chain_from)
        chains_to.append(chain_to)
    elif handle_finish:
        chains_from.append([(len_from, res[-1]["batch_id"], res[-1]["sub_id"])])
        chains_to.append([(len_to, res[-1]["batch_id"], res[-1]["sub_id"])])

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
        unique_variants, vec_lines_from, vec_lines_to, model_name, model
    )

    unique_sims = get_unique_sims(unique_variants, vecs_from, vecs_to)

    # for key in unique_sims:
    #     print(get_string(splitted_from, key[0]), "<->", get_string(splitted_to, key[1]), "->", unique_sims[key],"\n")

    max_sims, best_var_index = 0, 0
    for i, ids in enumerate(variants_ids):
        sum = 0
        for id in ids:
            sum += unique_sims[id]
        if sum > max_sims:
            max_sims = sum
            best_var_index = i

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
):
    """Get conflicts to solve and other"""
    splitted_from_len = len(aligner.get_splitted_from(db_path))
    splitted_to_len = len(aligner.get_splitted_to(db_path))
    prepared_index, total_batches = prepare_index(db_path, batch_id)
    if not prepared_index:
        return [], []

    if total_batches != 1:
        if batch_id > 0:
            handle_start = False
        if batch_id < total_batches - 1:
            handle_finish = False

    print(
        "get_all_conflicts, handle_start:",
        handle_start,
        "handle_finish:",
        handle_finish,
        "batch_id",
        batch_id,
    )

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


def resolve_all_conflicts(
    db_path,
    conflicts,
    model_name,
    show_logs=False,
    model=None,
    use_proxy_from=False,
    use_proxy_to=False,
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
):
    """Find the first conflict and resolve"""
    splitted_from_len = len(aligner.get_splitted_from(db_path))
    splitted_to_len = len(aligner.get_splitted_to(db_path))
    prepared_index, total_batches = prepare_index(db_path, 0)
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
    )


def get_vectors(unique_variants, splitted_from, splitted_to, model_name, model=None):
    strings_from = []
    strings_to = []
    for x in unique_variants:
        strings_from.append(helper.get_string(splitted_from, x[0]))
        strings_to.append(helper.get_string(splitted_to, x[1]))

    return (
        aligner.get_line_vectors(strings_from, model_name, model=model),
        aligner.get_line_vectors(strings_to, model_name, model=model),
    )


def get_unique_sims(unique_variants, vecs_from, vecs_to):
    res = dict()
    for i, x in enumerate(unique_variants):
        res[x] = 1 - spatial.distance.cosine(vecs_from[i], vecs_to[i])
    return res
