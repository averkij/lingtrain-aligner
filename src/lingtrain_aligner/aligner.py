"""Texts aligner part of the engine"""

import json
import logging
import os
import random
import re
import sqlite3
from collections import defaultdict

import numpy as np
from lingtrain_aligner import constants as con
from lingtrain_aligner import (helper, model_dispatcher, preprocessor,
                               vis_helper)
from scipy import spatial
from sentence_transformers import SentenceTransformer
import subprocess

to_delete = re.compile(
    r'[」「@#$%^&»«“”„‟"\x1a⓪①②③④⑤⑥⑦⑧⑨⑩⑴⑵⑶⑷⑸⑹⑺⑻⑼⑽*\(\)\[\]\n\/\-\:•＂＃＄％＆＇（）＊＋－／：；＜＝＞＠［＼］＾＿｀｛｜｝～｟｠｢｣､、〃》【】〔〕〖〗〘〙〜〟〰〾〿–—‘’‛‧﹏〉]+'
)

custom_model_name, custom_model = "", None


def get_line_vectors(
    lines,
    model_name,
    embed_batch_size=10,
    normalize_embeddings=True,
    show_progress_bar=False,
    model=None,
    lang="ell_Grek",
):
    """Calculate embedding of the string"""
    global custom_model_name, custom_model
    if model_name not in model_dispatcher.models and not model:
        if custom_model_name != model_name:
            logging.info(f"Model name is provided. model_name={model_name}.")
            logging.info(f"Trying to load as a SentenceTransformers model.")
            custom_model = SentenceTransformer(
                model_name, cache_folder="./models_cache"
            )
            custom_model_name = model_name
        model = custom_model

    if model:
        return model.encode(
            lines,
            batch_size=embed_batch_size,
            normalize_embeddings=normalize_embeddings,
            show_progress_bar=show_progress_bar,
        )
    else:
        return model_dispatcher.models[model_name].embed(
            lines, embed_batch_size, normalize_embeddings, show_progress_bar, lang=lang
        )


def get_line_vectors_by_api(
    lines,
    line_ids,
    tasks_path,
    result_path,
    api="openai",
    model="text-embedding-3-small",
    remove_after=False,
    max_len=None,
):
    """Calculate embeddings of the strings using API"""
    # make tasks
    jobs = [
        {"model": model, "input": line, "metadata": {"row_id": id}} for id, line in zip(line_ids, lines)
    ]
    with open(tasks_path, "w", encoding="utf8") as f:
        for job in jobs:
            json_string = json.dumps(job, ensure_ascii=False)
            f.write(json_string + "\n")
    
    #run api_request_parallel_processor.py
    #Example command to call script:
    #```
    #python examples/api_request_parallel_processor.py \
    #--requests_filepath examples/data/example_requests_to_parallel_process.jsonl \
    #--save_filepath examples/data/example_requests_to_parallel_process_results.jsonl \
    #--request_url https://api.openai.com/v1/embeddings \
    #--max_requests_per_minute 1500 \
    #--max_tokens_per_minute 6250000 \
    #--token_encoding_name cl100k_base \
    #--max_attempts 5 \
    #--logging_level 20
    #```

    if not os.path.exists(tasks_path):
        raise FileNotFoundError(f"Tasks file {tasks_path} does not exist")
    
    print("Starting to process embeddings using API")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    api_processor_path = os.path.join(current_dir, "api_request_parallel_processor.py")
    command = [
        "python",
        api_processor_path,
        "--requests_filepath", tasks_path,
        "--save_filepath", result_path,
        "--request_url", "https://api.openai.com/v1/embeddings",
        "--max_requests_per_minute", "1500",
        "--max_tokens_per_minute", "6250000",
        "--token_encoding_name", "cl100k_base",
        "--max_attempts", "5",
        "--logging_level", "10"
    ]
    # print("Running the following command: ", " ".join(command))
    subprocess.run(command, check=True)

    # read result from file
    embeddings = []
    with open(result_path, "r", encoding="utf8") as f:
        for line in f:
            result = json.loads(line)
            #line = [{"model": "text-embedding-3-small", "input": "The other, a broad-shouldered young man with tousled reddish hair, his checkered cap cocked back on his head, was wearing a cowboy shirt, wrinkled white trousers and black sneakers."}, {"object": "list", "data": [{"object": "embedding", "index": 0, "embedding": [ 0.008154794, -0.015893724]}], "model": "text-embedding-3-small", "usage": {"prompt_tokens": 42, "total_tokens": 42}}, {"row_id": 4}]
            embeddings.append({"embedding": result[1]["data"][0]["embedding"], "row_id": result[2]["row_id"]})

    #sort array by row_id
    embeddings = sorted(embeddings, key=lambda x: x["row_id"])

    if remove_after:
        try:
            os.remove(tasks_path)
            os.remove(result_path)
        except:
            pass

    if max_len:
        return embeddings[:max_len]
    
    embeddings = [x["embedding"] for x in embeddings]

    return embeddings


def normalize_l2(x):
    x = np.array(x)
    if x.ndim == 1:
        norm = np.linalg.norm(x)
        if norm == 0:
            return x
        return x / norm
    else:
        norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
        return np.where(norm == 0, x, x / norm)


def clean_lines(lines):
    """Clean line"""
    return [re.sub(to_delete, "", line) for line in lines]


def update_embeddings(
    db_path,
    direction,
    ids,
    is_proxy,
    model_name,
    embed_batch_size,
    normalize_embeddings,
    show_progress_bar,
    model,
    lang_emb_from,
    use_api=False,
    api="openai",
    model_api="text-embedding-3-small",
    force=False,
    store_embeddings=False,
):
    """Update embeddings in the database"""
    if not force or not store_embeddings:
        ids_to_update = helper.get_splitted_ids_without_embeddings(
            db_path, direction, ids, is_proxy
        )
        # print(f"Missing embeddings (total: {len(ids_to_update)}):", ids_to_update)
    else:
        ids_to_update = ids
        # print(f"Force update. Line IDs (total: {len(ids_to_update)}):", ids_to_update)

    if ids_to_update:
        if direction == "from":
            lines = helper.get_splitted_from_by_id(db_path, ids_to_update)
        else:
            lines = helper.get_splitted_to_by_id(db_path, ids_to_update)

        if not is_proxy:
            lines = [x[1] for x in lines]
        else:
            lines = [x[2] for x in lines]

        if not use_api:
            embeddings = list(
                get_line_vectors(
                    lines,
                    model_name,
                    embed_batch_size,
                    normalize_embeddings,
                    show_progress_bar,
                    model,
                    lang_emb_from,
                )
            )
        else:
            n = random.randint(0,100000)
            tasks_path = db_path.replace(".db", f"_emb_tasks_{n}.jsonl")
            result_path = db_path.replace(".db", f"_emb_result_{n}.jsonl")
            embeddings = get_line_vectors_by_api(
                lines, ids_to_update, tasks_path, result_path, api, model_api
            )

        if store_embeddings:
            helper.set_embeddings(
                db_path, direction, ids_to_update, embeddings, is_proxy
            )

        return embeddings


def process_batch(
    db_path,
    lines_from_batch,
    lines_to_batch,
    line_ids_from,
    line_ids_to,
    batch_number,
    model_name,
    window,
    embed_batch_size,
    normalize_embeddings,
    show_progress_bar,
    save_pic=False,
    lang_name_from="",
    lang_name_to="",
    img_path="",
    show_info=False,
    show_regression=False,
    model=None,
    use_proxy_from=False,
    use_proxy_to=False,
    lang_emb_from="ell_Grek",
    lang_emb_to="ell_Grek",
    store_embeddings=False,
    use_api=False,
):
    """Do the actual alignment process logic"""
    # try:
    logging.info(f"Batch {batch_number}. Calculating vectors.")

    # vectors1 = [*get_line_vectors(clean_lines(lines_from_batch), model_name, embed_batch_size, normalize_embeddings, show_progress_bar)]
    # vectors2 = [*get_line_vectors(clean_lines(lines_to_batch), model_name, embed_batch_size, normalize_embeddings, show_progress_bar)]

    vectors1 = update_embeddings(
        db_path,
        direction="from",
        ids=line_ids_from,
        is_proxy=use_proxy_from,
        model_name=model_name,
        embed_batch_size=embed_batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=show_progress_bar,
        model=model,
        lang_emb_from=lang_emb_from,
        store_embeddings=store_embeddings,
        use_api=use_api,
    )
    
    vectors2 = update_embeddings(
        db_path,
        direction="to",
        ids=line_ids_to,
        is_proxy=use_proxy_to,
        model_name=model_name,
        embed_batch_size=embed_batch_size,
        normalize_embeddings=normalize_embeddings,
        show_progress_bar=show_progress_bar,
        model=model,
        lang_emb_from=lang_emb_to,
        store_embeddings=store_embeddings,
        use_api=use_api,
    )

    if store_embeddings:
        print("Get embeddings from the database")
        vectors1 = helper.get_embeddings(db_path, "from", line_ids_from, use_proxy_from)
        vectors1 = [x[1] for x in vectors1]

        vectors2 = helper.get_embeddings(db_path, "to", line_ids_to, use_proxy_to)
        vectors2 = [x[1] for x in vectors2]

    logging.debug(
        f"Batch {batch_number}. Vectors calculated. len(vectors1)={len(vectors1)}. len(vectors2)={len(vectors2)}."
    )

    # Similarity matrix
    logging.debug(f"Calculating similarity matrix.")

    sim_matrix = get_sim_matrix(vectors1, vectors2, window)
    sim_matrix_best = best_per_row_with_ones(sim_matrix)

    x_min, y_min = min(line_ids_from), min(line_ids_to)
    x_max, y_max = max(line_ids_from), max(line_ids_to)

    # save picture
    if save_pic:
        vis_helper.save_pic(
            sim_matrix_best,
            lang_name_to,
            lang_name_from,
            img_path,
            batch_number,
            (x_min, x_max),
            (y_min, y_max),
            transparent=True,
            show_info=show_info,
            show_regression=show_regression,
        )

    best_sim_ind = sim_matrix_best.argmax(1)
    texts_from = []
    texts_to = []

    for line_from_id in range(sim_matrix.shape[0]):
        id_from = line_ids_from[line_from_id]
        text_from = lines_from_batch[line_from_id]
        id_to = line_ids_to[best_sim_ind[line_from_id]]
        text_to = lines_to_batch[best_sim_ind[line_from_id]]

        texts_from.append((f"[{id_from+1}]", id_from + 1, text_from.strip()))
        texts_to.append((f"[{id_to+1}]", id_to + 1, text_to.strip()))

    return texts_from, texts_to

    # except Exception as e:
    #     logging.error(e, exc_info=True)
    #     return [], []


# ---------------------- DEPRECATED ----------------------
# def align_texts(
#     splitted_from,
#     splitted_to,
#     model_name,
#     batch_size,
#     window,
#     batch_ids=[],
#     save_pic=False,
#     lang_from="",
#     lang_to="",
#     img_path="",
#     embed_batch_size=10,
#     normalize_embeddings=True,
#     show_progress_bar=False,
#     shift=0,
#     show_info=False,
#     show_regression=False,
#     proxy_from=[],
#     proxy_to=[],
#     use_proxy_from=False,
#     use_proxy_to=False,
#     lang_emb_from="ell_Grek",
#     lang_emb_to="ell_Grek",
# ):
#     result = []
#     task_list = [
#         (
#             lines_from_batch,
#             lines_to_batch,
#             proxy_from_batch,
#             proxy_to_batch,
#             line_ids_from,
#             line_ids_to,
#             batch_id,
#         )
#         for lines_from_batch, lines_to_batch, proxy_from_batch, proxy_to_batch, line_ids_from, line_ids_to, batch_id in get_batch_intersected(
#             splitted_from,
#             splitted_to,
#             batch_size,
#             window,
#             batch_ids,
#             batch_shift=shift,
#             iter3=proxy_from,
#             iter4=proxy_to,
#         )
#     ]

#     for (
#         lines_from_batch,
#         lines_to_batch,
#         proxy_from_batch,
#         proxy_to_batch,
#         line_ids_from,
#         line_ids_to,
#         batch_id,
#     ) in task_list:
#         texts_from, texts_to = process_batch(
#             lines_from_batch,
#             lines_to_batch,
#             proxy_from_batch,
#             proxy_to_batch,
#             line_ids_from,
#             line_ids_to,
#             batch_id,
#             model_name,
#             window,
#             embed_batch_size,
#             normalize_embeddings,
#             show_progress_bar,
#             save_pic,
#             lang_from,
#             lang_to,
#             img_path,
#             show_info=show_info,
#             show_regression=show_regression,
#             use_proxy_from=use_proxy_from,
#             use_proxy_to=use_proxy_to,
#             lang_emb_from=lang_emb_from,
#             lang_emb_to=lang_emb_to,
#         )
#         result.append((batch_id, texts_from, texts_to))

#     # sort by batch_id (will be useful with parallel processing)
#     result.sort()

#     return result


def align_db(
    db_path,
    model_name,
    batch_size,
    window,
    batch_ids=[],
    save_pic=False,
    lang_from="",
    lang_to="",
    img_path="",
    embed_batch_size=10,
    normalize_embeddings=True,
    show_progress_bar=False,
    shift=0,
    show_info=False,
    show_regression=False,
    model=None,
    use_proxy_from=False,
    use_proxy_to=False,
    use_segments=False,
    segmentation_marks=[preprocessor.H2],
    lang_emb_from="ell_Grek",
    lang_emb_to="ell_Grek",
    store_embeddings=False,
    use_api=False,
):
    result = []
    if use_segments:
        print("Aligning using segments.")

        left_segments, right_segments = calculate_segments(db_path, segmentation_marks)
        task_list = [
            (
                lines_from_batch,
                lines_to_batch,
                proxy_from_batch,
                proxy_to_batch,
                line_ids_from,
                line_ids_to,
                batch_id,
            )
            for lines_from_batch, lines_to_batch, proxy_from_batch, proxy_to_batch, line_ids_from, line_ids_to, batch_id in get_batch_intersected_for_segments(
                db_path=db_path,
                left_segments=left_segments,
                right_segments=right_segments,
                batch_size=batch_size,
                window=window,
                batch_ids=batch_ids,
                batch_shift=shift,
            )
        ]
    else:
        print("Aligning without segments.")

        splitted_from = get_splitted_from(db_path)
        splitted_to = get_splitted_to(db_path)
        proxy_from = get_proxy_from(db_path)
        proxy_to = get_proxy_to(db_path)
        task_list = [
            (
                lines_from_batch,
                lines_to_batch,
                proxy_from_batch,
                proxy_to_batch,
                line_ids_from,
                line_ids_to,
                batch_id,
            )
            for lines_from_batch, lines_to_batch, proxy_from_batch, proxy_to_batch, line_ids_from, line_ids_to, batch_id in get_batch_intersected(
                splitted_from,
                splitted_to,
                batch_size,
                window,
                batch_ids,
                batch_shift=shift,
                iter3=proxy_from,
                iter4=proxy_to,
            )
        ]

    print("tasks amount:", len(task_list))

    count = 0
    for (
        lines_from_batch,
        lines_to_batch,
        _,  # proxy_from_batch,
        _,  # proxy_to_batch,
        line_ids_from,
        line_ids_to,
        batch_id,
    ) in task_list:
        print(f"batch: {count} ({batch_id})")
        texts_from, texts_to = process_batch(
            db_path,
            lines_from_batch,
            lines_to_batch,
            line_ids_from,
            line_ids_to,
            batch_id,
            model_name,
            window,
            embed_batch_size,
            normalize_embeddings,
            show_progress_bar,
            save_pic,
            lang_from,
            lang_to,
            img_path,
            show_info=show_info,
            show_regression=show_regression,
            model=model,
            use_proxy_from=use_proxy_from,
            use_proxy_to=use_proxy_to,
            lang_emb_from=lang_emb_from,
            lang_emb_to=lang_emb_to,
            store_embeddings=store_embeddings,
            use_api=use_api,
        )
        result.append((batch_id, texts_from, texts_to, shift, window))
        count += 1

    if not result:
        print("There are nothing to process")
        return

    # sort by batch_id (will be useful with parallel processing)
    result.sort()
    save_db(db_path, result)


# CONTENT HELPERS


def get_splitted_from(db_path):
    """Get lines from splitted_from"""
    with sqlite3.connect(db_path) as db:
        res = db.execute(f"select f.text from splitted_from f order by f.id").fetchall()
    return [x[0] for x in res]


def get_splitted_to(db_path):
    """Get lines from splitted_to"""
    with sqlite3.connect(db_path) as db:
        res = db.execute(f"select t.text from splitted_to t order by t.id").fetchall()
    return [x[0] for x in res]


def get_splitted_from_by_par_with_line_id(db_path, par_id_start, par_id_end):
    """Get lines from splitted_from by paragraphs"""
    with sqlite3.connect(db_path) as db:
        res = db.execute(
            f"""select f.id, f.text from splitted_from f
                                where paragraph > ? and paragraph <= ?
                                order by f.id""",
            (par_id_start, par_id_end),
        ).fetchall()
    return [(x[0], x[1]) for x in res]


def get_splitted_to_by_par_with_line_id(db_path, par_id_start, par_id_end):
    """Get lines from splitted_from by paragraphs"""
    with sqlite3.connect(db_path) as db:
        res = db.execute(
            f"""select f.id, f.text from splitted_to f
                                where paragraph > ? and paragraph <= ?
                                order by f.id""",
            (par_id_start, par_id_end),
        ).fetchall()
    return [(x[0], x[1]) for x in res]


def get_proxy_from(db_path):
    """Get lines from proxy_from"""
    with sqlite3.connect(db_path) as db:
        res = db.execute(
            f"select f.proxy_text from splitted_from f order by f.id"
        ).fetchall()
    return [x[0] for x in res]


def get_proxy_to(db_path):
    """Get lines from proxy_to"""
    with sqlite3.connect(db_path) as db:
        res = db.execute(
            f"select t.proxy_text from splitted_to t order by t.id"
        ).fetchall()
    return [x[0] for x in res]


def get_proxy_from_by_par_with_line_id(db_path, par_id_start, par_id_end):
    """Get proxy lines from splitted_from by paragraphs"""
    with sqlite3.connect(db_path) as db:
        res = db.execute(
            f"""select f.id, f.proxy_text from splitted_from f
                                where paragraph > ? and paragraph < ?
                                order by f.id""",
            (par_id_start, par_id_end),
        ).fetchall()
    return [(x[0], x[1]) for x in res]


def get_proxy_to_by_par_with_line_id(db_path, par_id_start, par_id_end):
    """Get proxy lines from splitted_from by paragraphs"""
    with sqlite3.connect(db_path) as db:
        res = db.execute(
            f"""select f.id, f.proxy_text from splitted_to f
                                where paragraph > ? and paragraph < ?
                                order by f.id""",
            (par_id_start, par_id_end),
        ).fetchall()
    return [(x[0], x[1]) for x in res]


def best_per_row_with_ones(sim_matrix):
    """Transfor matrix by leaving only best match"""
    sim_matrix_best = np.zeros_like(sim_matrix)
    max_sim = sim_matrix.argmax(1)
    sim_matrix_best[range(sim_matrix.shape[0]), max_sim] = 1
    return sim_matrix_best


def get_batch_intersected(
    iter1,
    iter2,
    n,
    window,
    batch_ids=[],
    batch_shift=0,
    iter3=[],
    iter4=[],
    start_batch_id=0,
    batch_start_line_id_from=0,
    batch_start_line_id_to=0,
):
    """Get batch with an additional window"""
    l1 = len(iter1)
    l2 = len(iter2)

    k = int(round(n * l2 / l1))
    kdx = 0 - k

    if not iter3:
        iter3 = ["" for _ in range(l1)]
    if not iter4:
        iter4 = ["" for _ in range(l2)]

    if k < window * 2:
        # subbatches will be intersected
        logging.warning(
            f"Batch for the second language is too small. k = {k}, window = {window}"
        )

    counter = start_batch_id
    for ndx in range(0, l1, n):
        kdx += k
        if counter in batch_ids or len(batch_ids) == 0:
            yield iter1[ndx : min(ndx + n, l1)], iter2[
                max(0, kdx - window + batch_shift) : min(
                    kdx + k + window + batch_shift, l2
                )
            ], iter3[ndx : min(ndx + n, l1)], iter4[
                max(0, kdx - window + batch_shift) : min(
                    kdx + k + window + batch_shift, l2
                )
            ], list(
                range(
                    ndx + batch_start_line_id_from,
                    min(
                        ndx + batch_start_line_id_from + n,
                        batch_start_line_id_from + l1,
                    ),
                )
            ), list(
                range(
                    max(
                        batch_start_line_id_to,
                        batch_start_line_id_to + kdx - window + batch_shift,
                    ),
                    min(
                        batch_start_line_id_to + kdx + k + window + batch_shift,
                        batch_start_line_id_to + l2,
                    ),
                )
            ), counter
        counter += 1


def get_batch_intersected_for_segments(
    db_path,
    left_segments,
    right_segments,
    batch_size,
    window,
    batch_ids=[],
    batch_shift=0,
):
    """Get batches based on segments structure."""
    start_batch_id = 0

    for left, right in zip(left_segments, right_segments):
        iter1 = get_splitted_from_by_par_with_line_id(db_path, left[0], left[1])
        iter2 = get_splitted_to_by_par_with_line_id(db_path, right[0], right[1])
        iter3 = get_proxy_from_by_par_with_line_id(db_path, left[0], left[1])
        iter4 = get_proxy_to_by_par_with_line_id(db_path, right[0], right[1])

        if not iter1:
            print("Empty segment occured. Probably no text between segment delimeters")
            continue

        for (
            lines_from_batch,
            lines_to_batch,
            proxy_from_batch,
            proxy_to_batch,
            line_ids_from,
            line_ids_to,
            batch_id,
        ) in get_batch_intersected(
            [x[1] for x in iter1],
            [x[1] for x in iter2],
            batch_size,
            window,
            batch_ids=[],  # we need to return all batches to estimate needed [batch_ids]
            batch_shift=batch_shift,
            iter3=[x[1] for x in iter3],
            iter4=[x[1] for x in iter4],
            start_batch_id=start_batch_id,
            batch_start_line_id_from=iter1[0][0],
            batch_start_line_id_to=iter2[0][0],
        ):
            if batch_id in batch_ids:
                yield lines_from_batch, lines_to_batch, proxy_from_batch, proxy_to_batch, line_ids_from, line_ids_to, batch_id

            start_batch_id += 1


def calculate_segments(db_path, segmentation_marks=[preprocessor.H2]):
    """Calculate segments based on metadata"""
    left_nails, right_nails = [], []
    meta = helper.get_meta_dict(db_path)
    for mark in meta:
        if mark.split("_")[0] in segmentation_marks:
            # print(mark)
            for segment_mark in meta[mark]:
                if mark.split("_")[-1] == "from":
                    # print(segment_mark)
                    left_nails.append(segment_mark[2])  # par_id
                else:
                    right_nails.append(segment_mark[2])  # par_id

    # remove duplicates
    left_nails = sorted(list(set(left_nails)))
    right_nails = sorted(list(set(right_nails)))

    assert len(left_nails) == len(
        right_nails
    ), f"Error. Different amount of segmentation marks in your texts ({', '.join(segmentation_marks)})"

    left_nails.sort()
    right_nails.sort()

    left_segments, right_segments = [], []
    left_len, right_len = helper.get_splitted_lenght(db_path)

    for i in range(1, len(left_nails)):
        left_segments.append((left_nails[i - 1], left_nails[i]))
        right_segments.append((right_nails[i - 1], right_nails[i]))

    # insert last or the only segment
    if len(left_nails) == 0:
        left_segments.append((0, left_len))
        right_segments.append((0, right_len))
    else:
        left_segments.append((left_nails[-1], left_len))
        right_segments.append((right_nails[-1], right_len))

    return left_segments, right_segments


def get_batch_intersected_for_segments_list(
    db_path, left_segments, right_segments, batch_size
):
    """Get batche structure based on segments."""
    res = []
    start_batch_id = 0

    for left, right in zip(left_segments, right_segments):
        segment_batches = []
        iter1 = get_splitted_from_by_par_with_line_id(db_path, left[0], left[1])
        iter2 = get_splitted_to_by_par_with_line_id(db_path, right[0], right[1])

        if not iter1:
            print("Empty segment occured. Probably no text between segment delimeters")
            continue

        for (
            _,
            _,
            _,
            _,
            _,
            _,
            batch_id,
        ) in get_batch_intersected(
            [x[1] for x in iter1],
            [x[1] for x in iter2],
            batch_size,
            window=0,
            batch_ids=[],  # we need to return all batches to estimate needed [batch_ids]
            start_batch_id=start_batch_id,
            batch_start_line_id_from=iter1[0][0],
            batch_start_line_id_to=iter2[0][0],
        ):
            segment_batches.append(batch_id)
            start_batch_id += 1

        res.append(segment_batches)

    return res


def get_sim_matrix(vec1, vec2, window):
    """Calculate similarity matrix"""
    sim_matrix = np.zeros((len(vec1), len(vec2)))
    k = len(vec1) / len(vec2)
    for i, vector1 in enumerate(vec1):
        for j, vector2 in enumerate(vec2):
            if (j * k > i - window) & (j * k < i + window):
                sim = 1 - spatial.distance.cosine(vector1, vector2)
                sim_matrix[i, j] = max(sim, 0.01)
    return sim_matrix


# DATABASE HELPERS


def save_db(db_path, data):
    """Save data to a SQLite database."""
    with sqlite3.connect(db_path) as db:
        write_processing_batches(db, data)
        create_doc_index(db, data)


def create_doc_index(db, data):
    """Create document index in database"""
    batch_ids = [batch_id for batch_id, _, _, _, _ in data]

    max_batch_id = max(batch_ids)
    doc_index = get_doc_index(db)

    if not doc_index:
        doc_index = [[] for _ in range(max_batch_id + 1)]
    else:
        while len(doc_index) < max_batch_id + 1:
            doc_index.append([])

    for batch_id in batch_ids:
        doc_index[batch_id] = []
        for batch_id, a, b, c, d in db.execute(
            "SELECT f.batch_id, f.id, f.text_ids, t.id, t.text_ids FROM processing_from f join processing_to t on f.id=t.id where f.batch_id = :batch_id order by f.id",
            {"batch_id": batch_id},
        ):
            doc_index[batch_id].append((a, b, c, d))

    update_doc_index(db, doc_index)


def update_doc_index(db, index):
    """Insert or update document index"""
    index = json.dumps(index)
    db.execute(
        "insert or replace into doc_index (id, contents) values ((select id from doc_index limit 1),?)",
        (index,),
    )


def get_doc_index(db):
    """Get document index"""
    res = []
    try:
        cur = db.execute("SELECT contents FROM doc_index")
        res = json.loads(cur.fetchone()[0])
    except:
        logging.warning("can not fetch index db")
    return res


def write_processing_batches(db, data):
    """Insert or rewrite batched data"""
    for batch_id, texts_from, texts_to, shift, window in data:
        db.execute(
            "delete from processing_from where batch_id=:batch_id",
            {"batch_id": batch_id},
        )
        db.executemany(
            "insert into processing_from(batch_id, text_ids, initial_id, text) values (?,?,?,?)",
            [(batch_id, a, b, c) for a, b, c in texts_from],
        )
        db.execute(
            "delete from processing_to where batch_id=:batch_id", {"batch_id": batch_id}
        )
        db.executemany(
            "insert into processing_to(batch_id, text_ids, initial_id, text) values (?,?,?,?)",
            [(batch_id, a, b, c) for a, b, c in texts_to],
        )
        db.execute(
            "insert or replace into batches (batch_id, insert_ts, shift, window) values (?, datetime('now'), ?, ?)",
            (batch_id, shift, window),
        )


def update_history(db_path, batch_ids, operation, parameters):
    """Update batches table with already processed batches IDs"""
    parameters = json.dumps(parameters)
    with sqlite3.connect(db_path) as db:
        db.executemany(
            "insert into history(operation, batch_id, parameters, insert_ts) values (?,?,?, datetime('now'))",
            [(operation, batch_id, parameters) for batch_id in batch_ids],
        )


def fill_db_from_files(
    db_path,
    lang_from,
    lang_to,
    splitted_from_path,
    splitted_to_path,
    proxy_from_path,
    proxy_to_path,
    file_from,
    id_from,
    file_to,
    id_to,
    name="",
):
    """Fill document database (alignment) with prepared document lines"""
    if not os.path.isfile(db_path):
        logging.info(f"Initializing database {db_path}")
        helper.init_document_db(db_path)
    lines = []
    if os.path.isfile(splitted_from_path):
        with open(splitted_from_path, mode="r", encoding="utf-8") as input_path:
            lines = input_path.readlines()
        lines, meta, meta_par_ids = handle_marks(lines)
        lines_proxy = []
        if os.path.isfile(proxy_from_path):
            with open(proxy_from_path, mode="r", encoding="utf-8") as input_path:
                lines_proxy = input_path.readlines()
        if len(lines) == len(lines_proxy):
            data = zip(lines, lines_proxy)
        else:
            data = zip(lines, ["" for _ in range(len(lines))])
        with sqlite3.connect(db_path) as db:
            db.executemany(
                "insert into splitted_from(id, text, proxy_text, exclude, paragraph, h1, h2, h3, h4, h5, divider) values (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        i + 1,
                        text[0].strip(),
                        proxy.strip(),
                        0,
                        text[1][0],
                        text[1][1],
                        text[1][2],
                        text[1][3],
                        text[1][4],
                        text[1][5],
                        text[1][6],
                    )
                    for i, (text, proxy) in enumerate(data)
                ],
            )
            db.executemany(
                "insert into meta(key, val, occurence, par_id) values(?,?,?,?)",
                flatten_meta(meta, meta_par_ids, "from"),
            )
    if os.path.isfile(splitted_to_path):
        with open(splitted_to_path, mode="r", encoding="utf-8") as input_path:
            lines = input_path.readlines()
        lines, meta, meta_par_ids = handle_marks(lines)
        lines_proxy = []
        if os.path.isfile(proxy_to_path):
            with open(proxy_to_path, mode="r", encoding="utf-8") as input_path:
                lines_proxy = input_path.readlines()
        if len(lines) == len(lines_proxy):
            data = zip(lines, lines_proxy)
        else:
            data = zip(lines, ["" for _ in range(len(lines))])
        with sqlite3.connect(db_path) as db:
            db.executemany(
                "insert into splitted_to(id, text, proxy_text, exclude, paragraph, h1, h2, h3, h4, h5, divider) values (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        i + 1,
                        text[0].strip(),
                        proxy.strip(),
                        0,
                        text[1][0],
                        text[1][1],
                        text[1][2],
                        text[1][3],
                        text[1][4],
                        text[1][5],
                        text[1][6],
                    )
                    for i, (text, proxy) in enumerate(data)
                ],
            )
            db.executemany(
                "insert into meta(key, val, occurence, par_id) values(?,?,?,?)",
                flatten_meta(meta, meta_par_ids, "to"),
            )
    with sqlite3.connect(db_path) as db:
        db.executemany(
            "insert into languages(key, val) values(?,?)",
            [("from", lang_from), ("to", lang_to)],
        )
        db.executemany(
            "insert into files(direction, name, guid) values(?,?,?)",
            [("from", file_from, id_from), ("to", file_to, id_to)],
        )
    helper.set_name(db_path, name)


def fill_db(
    db_path,
    lang_from,
    lang_to,
    splitted_from=[],
    splitted_to=[],
    proxy_from=[],
    proxy_to=[],
    file_from="",
    id_from="",
    file_to="",
    id_to="",
    name="",
):
    """Fill document database (alignment) with prepared document lines"""
    if not os.path.isfile(db_path):
        logging.info(f"Initializing database {db_path}")
        helper.init_document_db(db_path)
    if len(splitted_from) > 0:
        splitted_from, meta, meta_par_ids = handle_marks(splitted_from)
        if len(splitted_from) == len(proxy_from):
            data = zip(splitted_from, proxy_from)
        else:
            data = zip(splitted_from, ["" for _ in range(len(splitted_from))])
        with sqlite3.connect(db_path) as db:
            db.executemany(
                "insert into splitted_from(id, text, proxy_text, exclude, paragraph, h1, h2, h3, h4, h5, divider) values (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        i + 1,
                        text[0].strip(),
                        proxy.strip(),
                        0,
                        text[1][0],
                        text[1][1],
                        text[1][2],
                        text[1][3],
                        text[1][4],
                        text[1][5],
                        text[1][6],
                    )
                    for i, (text, proxy) in enumerate(data)
                ],
            )
            db.executemany(
                "insert into meta(key, val, occurence, par_id) values(?,?,?,?)",
                flatten_meta(meta, meta_par_ids, "from"),
            )
    if len(splitted_to) > 0:
        splitted_to, meta, meta_par_ids = handle_marks(splitted_to)
        if len(splitted_to) == len(proxy_to):
            data = zip(splitted_to, proxy_to)
        else:
            data = zip(splitted_to, ["" for _ in range(len(splitted_to))])
        with sqlite3.connect(db_path) as db:
            db.executemany(
                "insert into splitted_to(id, text, proxy_text, exclude, paragraph, h1, h2, h3, h4, h5, divider) values (?,?,?,?,?,?,?,?,?,?,?)",
                [
                    (
                        i + 1,
                        text[0].strip(),
                        proxy.strip(),
                        0,
                        text[1][0],
                        text[1][1],
                        text[1][2],
                        text[1][3],
                        text[1][4],
                        text[1][5],
                        text[1][6],
                    )
                    for i, (text, proxy) in enumerate(data)
                ],
            )
            db.executemany(
                "insert into meta(key, val, occurence, par_id) values(?,?,?,?)",
                flatten_meta(meta, meta_par_ids, "to"),
            )
    with sqlite3.connect(db_path) as db:
        db.executemany(
            "insert into languages(key, val) values(?,?)",
            [("from", lang_from), ("to", lang_to)],
        )
        db.executemany(
            "insert into files(direction, name, guid) values(?,?,?)",
            [("from", file_from, id_from), ("to", file_to, id_to)],
        )
    helper.set_name(db_path, name)


def load_proxy(db_path, filepath, direction):
    lines_proxy = []
    if os.path.isfile(filepath):
        with open(filepath, mode="r", encoding="utf-8") as input_path:
            lines_proxy = input_path.readlines()
    ids = [x for x in range(1, len(lines_proxy) + 1)]
    with sqlite3.connect(db_path) as db:
        if direction == "from":
            db.executemany(
                "update splitted_from set proxy_text=(?) where id=(?)",
                [(proxy, id) for id, proxy in zip(ids, lines_proxy)],
            )
        else:
            db.executemany(
                "update splitted_to set proxy_text=(?) where id=(?)",
                [(proxy, id) for id, proxy in zip(ids, lines_proxy)],
            )


def update_proxy_text(db_path, proxy_texts, ids, direction):
    """Update proxy text"""
    if not ids:
        # try to write proxy_texts for all ids
        ids = [x for x in range(1, len(proxy_texts) + 1)]
    if len(ids) != len(proxy_texts):
        print("proxy_text and ids lengths are not equal. Provide correct ids.")
        return
    with sqlite3.connect(db_path) as db:
        for id, text in zip(ids, proxy_texts):
            if direction == "from":
                db.execute(
                    "update splitted_from set proxy_text=(?) where id=(?)", (text, id)
                )
            else:
                db.execute(
                    "update splitted_to set proxy_text=(?) where id=(?)", (text, id)
                )


def update_proxy_text_from(db_path, proxy_texts, ids=[]):
    """Update proxy text from"""
    update_proxy_text(db_path, proxy_texts, ids, direction="from")


def update_proxy_text_to(db_path, proxy_texts, ids=[]):
    """Update proxy text to"""
    update_proxy_text(db_path, proxy_texts, ids, direction="to")


def handle_marks(lines):
    """Handle markup. Write counters."""
    res = []
    marks_counter = defaultdict(int)
    meta = defaultdict(list)
    meta_par_ids = defaultdict(list)
    marks = (0, 0, 0, 0, 0, 0)
    p_ending = tuple(
        [preprocessor.PARAGRAPH_MARK + x for x in preprocessor.LINE_ENDINGS]
    )

    for line in lines:
        next_par = False
        line = line.strip()

        if line.endswith(p_ending):
            # remove last occurence of PARAGRAPH_MARK
            line = "".join(line.rsplit(preprocessor.PARAGRAPH_MARK, 1))
            next_par = True

        for mark in preprocessor.MARK_COUNTERS:
            update_mark_counter(marks_counter, line, mark)

        update_meta(meta, line, meta_par_ids, marks_counter[preprocessor.PARAGRAPH])

        if not line.endswith(get_all_extraction_endings()):
            marks = (
                marks_counter[preprocessor.PARAGRAPH],
                marks_counter[preprocessor.H1],
                marks_counter[preprocessor.H2],
                marks_counter[preprocessor.H3],
                marks_counter[preprocessor.H4],
                marks_counter[preprocessor.H5],
                marks_counter[preprocessor.DIVIDER],
            )
            res.append((line, marks))

            if next_par:
                marks_counter[preprocessor.PARAGRAPH] += 1
        else:
            marks_counter[preprocessor.PARAGRAPH] += 1

    return res, meta, meta_par_ids


def update_mark_counter(marks_counter, line, mark):
    ending = f"{preprocessor.PARAGRAPH_MARK}{mark}."
    if line.endswith(ending):
        marks_counter[mark] += 1


def get_mark_value(line, mark):
    res = ""
    ending = f"{preprocessor.PARAGRAPH_MARK}{mark}."
    if line.endswith(ending):
        if mark == preprocessor.DIVIDER:
            return "* * *"
        res = line[: len(line) - len(ending)]
    return res


def get_all_extraction_endings():
    return tuple([f"{preprocessor.PARAGRAPH_MARK}{m}." for m in preprocessor.MARK_META])


def update_meta(meta, line, meta_par_ids, par_id):
    for mark in preprocessor.MARK_META:
        val = get_mark_value(line, mark)
        if val:
            meta[mark].append(val)
            meta_par_ids[mark].append(par_id)


def flatten_meta(meta, meta_par_ids, direction):
    res = []
    for key in meta:
        for i, (val, par_id) in enumerate(zip(meta[key], meta_par_ids[key])):
            res.append((f"{key}_{direction}", val, i, par_id))
    return res


def update_index_mapping(db_path, direction, line_id):
    """Update mapping in index"""
    with sqlite3.connect(db_path) as db:
        index = get_doc_index(db)
        for i, index_batch in enumerate(index):
            for j, item in enumerate(index_batch):
                if direction == "from":
                    direction_id = 1
                else:
                    direction_id = 3
                text_ids = json.loads(item[direction_id])
                if any(x > line_id for x in text_ids):
                    new_values = [x if x <= line_id else x + 1 for x in text_ids]
                    index[i][j][direction_id] = json.dumps(new_values)
        update_doc_index(db, index)
