"""Alignment corrector module"""

import json
import statistics
from copy import deepcopy as copy

import matplotlib.pyplot as plt
import seaborn as sns
from lingtrain_aligner import (
    aligner,
    helper,
    resolver,
)
from scipy import spatial
from tqdm import tqdm


def get_index_data(db_path):
    """Get index data from the database."""
    index = helper.get_flatten_doc_index(db_path)
    page = list(zip(index, range(len(index))))
    data, _, _, splitted_from_dict, splitted_to_dict = (
        helper.get_doc_items_with_splitted(page, db_path)
    )
    for item in data:
        item["lines_from"] = [
            splitted_from_dict[x] for x in json.loads(item["line_id_from"])
        ]
        item["lines_to"] = [splitted_to_dict[x] for x in json.loads(item["line_id_to"])]
    return data


def calculate_outlier_bounds(data, n_std=1.5):
    """Calculates bounds for character and word ratio outliers.
    
    Args:
        data: list of items
        n_std: number of standard deviations to consider as outlier
    """
    for item in data:
        text_from = str(item["text_from"])
        text_to = str(item["text_to"])

        item["len_en_char"] = len(text_from)
        item["len_ru_char"] = len(text_to)
        item["len_en_words"] = len(text_from.split())
        item["len_ru_words"] = len(text_to.split())

        max_char = max(item["len_en_char"], item["len_ru_char"], 1)
        max_word = max(item["len_en_words"], item["len_ru_words"], 1)
        item["char_ratio"] = min(item["len_en_char"], item["len_ru_char"]) / max_char
        item["word_ratio"] = min(item["len_en_words"], item["len_ru_words"]) / max_word

        item["len_from"] = len(text_from)
        item["len_to"] = len(text_to)

    char_ratios = [item["char_ratio"] for item in data]
    word_ratios = [item["word_ratio"] for item in data]

    char_mean = statistics.mean(char_ratios)
    char_std = statistics.stdev(char_ratios)
    word_mean = statistics.mean(word_ratios)
    word_std = statistics.stdev(word_ratios)

    # log
    # print(f"Character Ratio - Mean: {char_mean:.4f}, Std Dev: {char_std:.4f}")
    # print(f"Word Ratio - Mean: {word_mean:.4f}, Std Dev: {word_std:.4f}")

    char_lower_bound = char_mean - n_std * char_std
    char_upper_bound = char_mean + n_std * char_std
    word_lower_bound = word_mean - n_std * word_std
    word_upper_bound = word_mean + n_std * word_std

    return (
        char_lower_bound,
        char_upper_bound,
        word_lower_bound,
        word_upper_bound,
        char_mean,
        word_mean,
    )


def calculate_ratios(text_from, text_to):
    """Calculates character and word ratio between two texts."""
    len_en_char = len(text_from)
    len_ru_char = len(text_to)
    len_en_words = len(text_from.split())
    len_ru_words = len(text_to.split())

    max_char = max(len_en_char, len_ru_char, 1)
    max_word = max(len_en_words, len_ru_words, 1)
    char_ratio = min(len_en_char, len_ru_char) / max_char
    word_ratio = min(len_en_words, len_ru_words) / max_word

    return char_ratio, word_ratio


def is_outlier_by_length(item, bounds):
    """Determines if an item is an outlier based on character and word ratio bounds."""
    (
        char_lower_bound,
        char_upper_bound,
        word_lower_bound,
        word_upper_bound,
        _,
        _,
    ) = bounds
    is_outlier_char = not (char_lower_bound <= item["char_ratio"] <= char_upper_bound)
    is_outlier_word = not (word_lower_bound <= item["word_ratio"] <= word_upper_bound)
    return is_outlier_char or is_outlier_word


def calculate_outliers_by_length_1(data, bounds, min_text_len=10, show_plot=True):
    """Calculates outliers based on character and word ratio bounds."""
    (
        char_lower_bound,
        char_upper_bound,
        word_lower_bound,
        word_upper_bound,
        _,
        _,
    ) = bounds
    cnt = 0
    for item in data:
        item["is_outlier"] = (
            is_outlier_by_length(item, bounds)
            and item["len_from"] > min_text_len
            and item["len_to"] > min_text_len
        )
        if item["is_outlier"]:
            cnt += 1

    # log
    print("outliers detected:", cnt, "total:", len(data))

    # plot
    if show_plot:
        figsize = (14, 8)
        sns.set_theme(rc={"figure.figsize": figsize})
        sns.histplot([item["char_ratio"] for item in data], bins=200)
        sns.histplot([item["word_ratio"] for item in data], bins=200)

        plt.axvline(x=char_lower_bound, color="r", linestyle="--")
        plt.axvline(x=char_upper_bound, color="r", linestyle="--")
        plt.axvline(x=word_lower_bound, color="g", linestyle="--")
        plt.axvline(x=word_upper_bound, color="g", linestyle="--")

        plt.legend(
            [
                "char_lower_bound",
                "char_upper_bound",
                "word_lower_bound",
                "word_upper_bound",
                "char_ratio",
                "word_ratio",
            ]
        )
    return data


def calculate_correction_tasks(data):
    """Get correction tasks from the data."""
    correction_candidates = []
    for i, item in enumerate(data):
        if item["is_outlier"]:
            cand = []
            if i > 0:
                cand.append(data[i - 1])
            cand.append(data[i])
            if i < len(data) - 1:
                cand.append(data[i + 1])
            correction_candidates.append(cand)

    merged_candidates = []
    acc = []
    for i, cand in enumerate(correction_candidates):
        if i == 0:
            acc = cand
            continue

        existed_ids = [x["line_id_from"] for x in acc]
        cand_ids = [x["line_id_from"] for x in cand]

        if set(existed_ids).intersection(cand_ids):
            for c in cand:
                if c["line_id_from"] not in existed_ids:
                    acc.append(c)
        else:
            merged_candidates.append(acc)
            acc = cand

    merged_candidates.append(acc)

    # log
    # print(len(merged_candidates))

    tasks = {"items": [], "coordinates": []}
    for i, cand in enumerate(merged_candidates):
        task = []
        coordinates = []
        for item in cand:
            line_id_from = json.loads(item["line_id_from"])
            lines_from = item["lines_from"]
            items_from = list(zip(line_id_from, lines_from))

            line_id_to = json.loads(item["line_id_to"])
            lines_to = item["lines_to"]
            items_to = list(zip(line_id_to, lines_to))

            task.append([items_from, items_to])
            # batch_id, sub_batch_id
            coordinates.append(((item["batch_id"]), (item["batch_index_id"])))

        tasks["items"].append(task)
        tasks["coordinates"].append({"start": coordinates[0], "end": coordinates[-1]})

    return tasks


def check_variant(variant):
    """Check if a variant is valid."""
    for pairs in variant:
        for pair in pairs:
            if not pair:
                return False
    return True


def remove_empty_pairs(variant):
    """Remove empty pairs from a variant."""
    res = []
    for pairs in variant:
        if not pairs[0] and not pairs[1]:
            continue
        else:
            res.append(pairs)
    return res


def do_step(curr_pair, rest_pairs, task_variants, acc, visited, added):
    """Recursive function to generate correction tasks."""
    if not rest_pairs:
        res = acc + [curr_pair]
        res = remove_empty_pairs(res)

        if check_variant(res):
            full_variant_id = get_full_variant_id(res)
            if full_variant_id not in added:
                task_variants.append(res)
                added.add(full_variant_id)
        return

    pair_2 = rest_pairs[0]
    next_rest = rest_pairs[1:]

    # Initial state
    do_step(pair_2, next_rest, task_variants, acc + [curr_pair], visited, added)

    # Generate states with different combinations
    states = [
        # State 1: append first element from pair_2[0]
        ([curr_pair[0] + [pair_2[0][0]], curr_pair[1]], [pair_2[0][1:], pair_2[1]]),
        # State 2: append first element from pair_2[1]
        ([curr_pair[0], curr_pair[1] + [pair_2[1][0]]], [pair_2[0], pair_2[1][1:]]),
        # State 3: append first elements from both
        (
            [curr_pair[0] + [pair_2[0][0]], curr_pair[1] + [pair_2[1][0]]],
            [pair_2[0][1:], pair_2[1][1:]],
        ),
    ]

    for pair_1_new, pair_2_new in states:
        visited_id = get_variant_id(pair_1_new, pair_2_new)
        if visited_id not in visited:
            visited.add(visited_id)
            do_step(
                pair_2_new, next_rest, task_variants, acc + [pair_1_new], visited, added
            )

    return task_variants


def get_variant_id(pair1, pair2):
    """Get variant id from two pairs."""
    return (
        ",".join([str(x[0]) for x in pair1[0]])
        + "-"
        + ",".join([str(x[0]) for x in pair1[1]])
        + "-"
        + ",".join([str(x[0]) for x in pair2[0]])
        + "-"
        + ",".join([str(x[0]) for x in pair2[1]])
    )


def get_full_variant_id(variant):
    """Get full variant id from a variant."""
    res = []
    for pair in variant:
        res.append(
            ",".join([str(x[0]) for x in pair[0]])
            + "-"
            + ",".join([str(x[0]) for x in pair[1]])
        )
    return "|".join(res)


def score_variants_by_len(variants, char_mean, word_mean):
    """Score variants by character and word ratio means."""
    res = []
    for variant in variants:
        score = 0
        for pair in variant:
            text_from = " ".join([x[1] for x in pair[0]])
            text_to = " ".join([x[1] for x in pair[1]])
            char_ratio, word_ratio = calculate_ratios(text_from, text_to)
            char_diff = abs(char_mean - char_ratio)
            word_diff = abs(word_mean - word_ratio)
            score += char_diff + word_diff
        res.append((score, variant))
    return res


def get_embeddings_dict_from_splitted(db_path, variant, is_proxy=False):
    """Get embeddings dictionary from splitted data."""
    res = {"from": {}, "to": {}}
    ids_from = [x[0] for pair in variant for x in pair[0]]
    ids_to = [x[0] for pair in variant for x in pair[1]]

    embeddings_from = helper.get_embeddings(
        db_path, "from", ids_from, is_proxy=is_proxy
    )
    embeddings_from_dict = {x[0]: x[1] for x in embeddings_from}
    embeddings_to = helper.get_embeddings(db_path, "to", ids_to, is_proxy=is_proxy)
    embeddings_to_dict = {x[0]: x[1] for x in embeddings_to}

    for pair in variant:
        for sent in pair[0]:
            res["from"][sent[0]] = {
                "text": sent[1],
                "embeddings": embeddings_from_dict[sent[0]],
            }
        for sent in pair[1]:
            res["to"][sent[0]] = {
                "text": sent[1],
                "embeddings": embeddings_to_dict[sent[0]],
            }
    return res


def get_solution_from_variant(variant):
    """Get solution from a variant."""
    res = []
    lines_dict_from = {}
    lines_dict_to = {}
    for pair in variant:
        res.append([[x[0] for x in pair[0]], [x[0] for x in pair[1]]])
        for item in pair[0]:
            lines_dict_from[item[0]] = item[1]
        for item in pair[1]:
            lines_dict_to[item[0]] = item[1]
    return res, lines_dict_from, lines_dict_to


def get_corections_and_score(
    db_path,
    task,
    score_by="similarity",
    is_proxy=False,
    aggregate_embeddings=False,
    char_mean=0,
    word_mean=0,
):
    """Get corrections and score."""
    visited, added = set(), set()
    variants = do_step(task[0], task[1:], [], [], visited, added)
    if score_by == "similarity":
        scored = score_variants_by_similarity(
            db_path,
            variants,
            is_proxy=is_proxy,
            aggregate_embeddings=aggregate_embeddings,
        )
        sorted_scored = sorted(scored, key=lambda x: x[0], reverse=True)
    elif score_by == "len":
        scored = score_variants_by_len(variants, char_mean, word_mean)
        sorted_scored = sorted(scored, key=lambda x: x[0])
    return sorted_scored


def score_variants_by_similarity(
    db_path, variants, is_proxy=False, aggregate_embeddings=True
):
    """Score variants by similarity."""
    res = []
    for variant in variants:
        score = 0
        if aggregate_embeddings:
            # get embeddings from db and aggregate (once per variant)
            embeddings_dict = get_embeddings_dict_from_splitted(
                db_path, variant, is_proxy=is_proxy
            )
        for pair in variant:
            if aggregate_embeddings:
                lens_from = [len(x[1]) for x in pair[0]]
                lens_to = [len(x[1]) for x in pair[1]]

                text_from = " ".join([x[1] for x in pair[0]])
                text_to = " ".join([x[1] for x in pair[1]])

                # ids_from = [x[0] for x in pair[0]]
                # ids_to = [x[0] for x in pair[1]]

                embeddings_from = [
                    embeddings_dict["from"][x[0]]["embeddings"] for x in pair[0]
                ]
                embeddings_to = [
                    embeddings_dict["to"][x[0]]["embeddings"] for x in pair[1]
                ]

                aggregated_from = resolver.aggregate_embeddings(
                    embeddings_from, lens_from, method="logarithmic_scaling"
                )
                aggregated_to = resolver.aggregate_embeddings(
                    embeddings_to, lens_to, method="logarithmic_scaling"
                )

                # calculate similarity
                sim = 1 - spatial.distance.cosine(aggregated_from, aggregated_to)

                if sim < -1 or sim > 1:
                    raise ValueError(f"Similarity score out of bounds: {sim}")

                score += sim

                # penalty by distant text lenghts
                len_weight = abs(len(text_from) - len(text_to)) / 100
                score -= len_weight * 0.1
            else:
                # calculate embeddings
                text_from = " ".join([x[1] for x in pair[0]])
                text_to = " ".join([x[1] for x in pair[1]])

                test_emb_from = aligner.get_line_vectors(
                    [text_from], model_name="sentence_transformer_multilingual_labse"
                )
                test_emb_to = aligner.get_line_vectors(
                    [text_to], model_name="sentence_transformer_multilingual_labse"
                )

                sim = 1 - spatial.distance.cosine(test_emb_from[0], test_emb_to[0])
                if sim < -1 or sim > 1:
                    raise ValueError(f"Similarity score out of bounds: {sim}")
                score += sim

                # penalty by distant text lenghts
                len_weight = abs(len(text_from) - len(text_to)) / 100
                score -= len_weight * 0.1

        avg_score = score / len(variant)

        # promote long variants
        variant_lenght_weight = len(variant) * 0.05
        avg_score += variant_lenght_weight

        res.append((avg_score, variant))
    return res


def get_correction_tasks(db_path):
    """Get correction tasks from the index."""
    data = get_index_data(db_path)

    # calculate mean and std for char and word ratios for anomaly detection
    bounds = calculate_outlier_bounds(data, n_std=1.5)
    data = calculate_outliers_by_length_1(data, bounds)
    correction_tasks = calculate_correction_tasks(data)

    return correction_tasks, bounds


def resolve_correction_tasks(
    db_path, correction_tasks, score_by, aggregate_embeddings, bounds
):
    """Resolve correction tasks.

    Args:
        db_path: str, path to the database.
        correction_tasks: dict, correction tasks.
        score_by: str, scoring method.
        aggregate_embeddings: bool, aggregate embedding, means embeddings are stored in the database.
        bounds: tuple, outlier bounds.
    """
    _, _, _, _, char_mean, word_mean = bounds
    same_counter = 0

    for i in tqdm(range(len(correction_tasks["items"]))[::-1]):
        scored_variants = get_corections_and_score(
            db_path,
            correction_tasks["items"][i],
            score_by=score_by,
            is_proxy=False,
            aggregate_embeddings=aggregate_embeddings,
            char_mean=char_mean,
            word_mean=word_mean,
        )

        corr_var_id = get_full_variant_id(scored_variants[0][1])

        if get_full_variant_id(correction_tasks["items"][i]) == corr_var_id:
            same_counter += 1
            print("Corrected variant is the same as initial. Skip resolving.")
            continue

        task_coordinates = correction_tasks["coordinates"][i]

        conflict_mock = {
            "from": {
                "start": [
                    -1,
                    task_coordinates["start"][0],
                    task_coordinates["start"][1],
                ],
                "end": [-1, task_coordinates["end"][0], task_coordinates["end"][1]],
            }
        }

        solution, lines_from, lines_to = get_solution_from_variant(
            scored_variants[0][1]
        )

        resolver.resolve_conflict(
            db_path, conflict_mock, solution, lines_from, lines_to, show_logs=True
        )

    print("same variants (skipped):", same_counter)