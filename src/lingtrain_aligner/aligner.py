"""Texts aligner part of the engine"""

import gc
import json
import logging
import os
import random
import re
import sqlite3
import uuid
from collections import defaultdict

import numpy as np
from lingtrain_aligner import constants as con
from lingtrain_aligner import (helper, model_dispatcher, preprocessor,
                               punct_sim, splitter, vis_helper)
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


def _openrouter_proxy_url():
    """Return the optional explicit proxy for OpenRouter embedding calls."""
    proxy_url = os.getenv("OPENROUTER_PROXY_URL")
    if proxy_url:
        proxy_url = proxy_url.strip()
    return proxy_url or None


def _openrouter_post_json(url, headers, payload, timeout=120, proxy_url=None):
    """POST JSON to OpenRouter, optionally through an explicit forward proxy."""
    import requests

    if proxy_url:
        proxies = {"http": proxy_url, "https": proxy_url}
        with requests.Session() as session:
            session.trust_env = False
            return session.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
                proxies=proxies,
            )

    return requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=timeout,
    )


def _openrouter_embed_chunk(lines, model, headers, url, timeout=120, proxy_url=None):
    """Send a single batch request to OpenRouter. Returns list of embeddings or None on failure."""
    resp = _openrouter_post_json(
        url,
        headers=headers,
        payload={"model": model, "input": lines},
        timeout=timeout,
        proxy_url=proxy_url,
    )
    data = resp.json()
    if "data" in data and len(data["data"]) == len(lines):
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]
    error_info = data.get("error", data)
    logging.warning(f"OpenRouter batch failed for {model} (size {len(lines)}): {error_info}")
    return None


def _openrouter_embed_batched(lines, model, api_key, proxy_url=None):
    """Call OpenRouter embeddings API with tiered fallback: 300 -> 50 -> 1."""
    url = "https://openrouter.ai/api/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    proxy_url = proxy_url if proxy_url is not None else _openrouter_proxy_url()
    batch_tiers = [300, 50, 1]
    all_embeddings = []
    remaining = list(lines)

    for tier in batch_tiers:
        if not remaining:
            break
        total_chunks = (len(remaining) + tier - 1) // tier
        logging.info(f"OpenRouter: trying batch size {tier} for {len(remaining)} texts ({total_chunks} request(s))")

        failed = []
        for chunk_idx in range(0, len(remaining), tier):
            chunk = remaining[chunk_idx : chunk_idx + tier]
            result = _openrouter_embed_chunk(
                chunk, model, headers, url, proxy_url=proxy_url
            )
            if result is not None:
                all_embeddings.extend(result)
            else:
                failed.extend(chunk)

        remaining = failed
        if remaining:
            logging.info(f"OpenRouter: {len(remaining)} texts failed at batch size {tier}, falling back to {batch_tiers[batch_tiers.index(tier) + 1] if tier != 1 else 'error'}")

    if remaining:
        raise ValueError(
            f"OpenRouter: {len(remaining)} texts failed for model {model} at all batch tiers"
        )

    return all_embeddings


def get_line_vectors_by_api(
    lines,
    line_ids,
    tasks_path,
    result_path,
    api="openai",
    model="text-embedding-3-small",
    remove_after=False,
    max_len=None,
    api_key=None,
    max_input_len=None,
):
    """Calculate embeddings of the strings using API.

    Supported providers:
    - "hf-inference": HuggingFace Inference API via huggingface_hub.InferenceClient.
      Direct synchronous Python call — does NOT use the OpenAI parallel processor subprocess.
      Embeddings are L2-normalized to match local path behavior.
    - "openai": OpenAI Embeddings API via api_request_parallel_processor.py subprocess.
    - "openrouter": OpenRouter Embeddings API via direct requests.post call.
      Sends all texts as a batch in one request. L2-normalized to match local path behavior.

    Args:
        lines: list of text strings to embed
        line_ids: list of integer row IDs (1-to-1 with lines)
        tasks_path: path for JSONL task file (OpenAI path only)
        result_path: path for JSONL result file (OpenAI path only)
        api: provider name — "hf-inference", "openai", or "openrouter"
        model: model identifier for the provider
        remove_after: whether to remove task/result files after (OpenAI path only)
        max_len: if set, truncate result to this length
        api_key: API key for the provider (passed in by caller; not read from env here)
        max_input_len: if set, crop each text string to this many characters before sending
    """
    if max_input_len is not None and max_input_len > 0 and len(lines) > 0:
        cropped = 0
        cropped_lines = []
        for line in lines:
            if len(line) > max_input_len:
                cropped_lines.append(line[:max_input_len])
                cropped += 1
            else:
                cropped_lines.append(line)
        if cropped:
            logging.info(f"Cropped {cropped}/{len(lines)} lines to max_input_len={max_input_len}")
        lines = cropped_lines
    if api == "hf-inference":
        from huggingface_hub import InferenceClient
        client = InferenceClient(api_key=api_key, provider="hf-inference")
        # Batch call — send all texts at once
        result = client.feature_extraction(lines, model=model)
        vecs = np.array(result)
        if vecs.ndim == 3:
            # Some models return (batch, seq_len, dim); mean-pool over seq_len
            vecs = vecs.mean(axis=1)
        # L2-normalize to match local path behavior (API-03)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vecs = vecs / norms
        result_embeddings = vecs.tolist()
        if max_len:
            return result_embeddings[:max_len]
        return result_embeddings
    elif api == "openrouter":
        import requests

        logging.info(f"OpenRouter embeddings: model={model}, {len(lines)} texts")

        all_embeddings = _openrouter_embed_batched(lines, model, api_key)

        # L2-normalize to match local path behavior
        vecs = np.array(all_embeddings)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        vecs = vecs / norms

        result_embeddings = vecs.tolist()
        if max_len:
            return result_embeddings[:max_len]
        return result_embeddings
    elif api == "openai":
        # make tasks
        jobs = [
            {"model": model, "input": line, "metadata": {"row_id": id}} for id, line in zip(line_ids, lines)
        ]
        with open(tasks_path, "w", encoding="utf8") as f:
            for job in jobs:
                json_string = json.dumps(job, ensure_ascii=False)
                f.write(json_string + "\n")

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
        subprocess.run(command, check=True)

        # read result from file
        embeddings = []
        with open(result_path, "r", encoding="utf8") as f:
            for line in f:
                result = json.loads(line)
                embeddings.append({"embedding": result[1]["data"][0]["embedding"], "row_id": result[2]["row_id"]})

        # sort array by row_id
        embeddings = sorted(embeddings, key=lambda x: x["row_id"])

        if remove_after:
            try:
                os.remove(tasks_path)
                os.remove(result_path)
            except Exception:
                pass

        if max_len:
            return embeddings[:max_len]

        embeddings = [x["embedding"] for x in embeddings]
        return embeddings
    else:
        raise ValueError(f"Unknown API provider: {api}")


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
    api_key=None,
    force=False,
    store_embeddings=False,
    provenance_model=None,
    provenance_inference=None,
    max_input_len=None,
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
            logging.debug(f"update_embeddings [{direction}]: local model={model_name}, {len(ids_to_update)} lines")
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
            logging.debug(f"update_embeddings [{direction}]: api={api}, model={model_api}, {len(ids_to_update)} lines")
            n = random.randint(0,100000)
            tasks_path = db_path.replace(".db", f"_emb_tasks_{n}.jsonl")
            result_path = db_path.replace(".db", f"_emb_result_{n}.jsonl")
            embeddings = get_line_vectors_by_api(
                lines, ids_to_update, tasks_path, result_path, api, model_api,
                api_key=api_key,
                max_input_len=max_input_len,
            )

        if store_embeddings:
            helper.set_embeddings(
                db_path, direction, ids_to_update, embeddings, is_proxy
            )
            if provenance_model is not None and not is_proxy:
                helper.set_provenance(
                    db_path, direction, ids_to_update, provenance_model, provenance_inference
                )

        return embeddings

    # Nothing to update — return existing embeddings from DB or empty list
    if store_embeddings:
        existing = helper.get_embeddings(db_path, direction, ids, is_proxy)
        return [x[1] for x in existing]
    return []


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
    embedding_cache=None,
    api=None,
    model_api=None,
    api_key=None,
    provenance_model=None,
    provenance_inference=None,
    max_input_len=None,
):
    """Do the actual alignment process logic"""
    # try:
    import time as _time
    _batch_start = _time.monotonic()
    _inf = f"api={api}, model_api={model_api}" if use_api else f"local, model_name={model_name}"
    logging.info(f"Batch {batch_number}. Calculating vectors. inference=[{_inf}]. store_embeddings={store_embeddings}.")

    if embedding_cache is not None:
        vectors1, vectors2 = _process_batch_with_cache(
            db_path, line_ids_from, line_ids_to, lines_from_batch, lines_to_batch,
            use_proxy_from, use_proxy_to, model_name, embed_batch_size,
            normalize_embeddings, show_progress_bar, model,
            lang_emb_from, lang_emb_to, store_embeddings, use_api, embedding_cache,
            api=api, model_api=model_api, api_key=api_key, max_input_len=max_input_len,
        )
    else:
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
            api=api,
            model_api=model_api,
            api_key=api_key,
            provenance_model=provenance_model,
            provenance_inference=provenance_inference,
            max_input_len=max_input_len,
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
            api=api,
            model_api=model_api,
            api_key=api_key,
            provenance_model=provenance_model,
            provenance_inference=provenance_inference,
            max_input_len=max_input_len,
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

    # Boost with punctuation similarity (language-independent signal)
    sim_matrix = punct_sim.boost_sim_matrix(
        sim_matrix, lines_from_batch, lines_to_batch, weight=0.15
    )

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

        texts_from.append((f"[{id_from}]", id_from, text_from.strip()))
        texts_to.append((f"[{id_to}]", id_to, text_to.strip()))

    _elapsed = _time.monotonic() - _batch_start
    _mins, _secs = divmod(int(_elapsed), 60)
    logging.info(f"Batch {batch_number}. Finished. Time: {_mins}m {_secs}s.")

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
    api=None,
    model_api=None,
    api_key=None,
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

    embedding_cache = {"from": {}, "to": {}}
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
            embedding_cache=embedding_cache,
            api=api,
            model_api=model_api,
            api_key=api_key,
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
    batch_start_line_id_from=1,
    batch_start_line_id_to=1,
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


def _compute_embeddings_for_ids(
    db_path, direction, ids, lines, is_proxy, model_name, embed_batch_size,
    normalize_embeddings, show_progress_bar, model, lang_emb, store_embeddings, use_api,
    api=None, model_api=None, api_key=None, max_input_len=None,
):
    """Compute embeddings for a subset of IDs and return as a dict {id: embedding}."""
    if not ids:
        return {}
    # Get corresponding lines for the ids we need to compute
    if direction == "from":
        rows = helper.get_splitted_from_by_id(db_path, ids)
    else:
        rows = helper.get_splitted_to_by_id(db_path, ids)

    if not is_proxy:
        texts = [x[1] for x in rows]
    else:
        texts = [x[2] for x in rows]
    row_ids = [x[0] for x in rows]

    if not use_api:
        embeddings = list(
            get_line_vectors(
                texts, model_name, embed_batch_size,
                normalize_embeddings, show_progress_bar, model, lang_emb,
            )
        )
    else:
        n = random.randint(0, 100000)
        tasks_path = db_path.replace(".db", f"_emb_tasks_{n}.jsonl")
        result_path = db_path.replace(".db", f"_emb_result_{n}.jsonl")
        embeddings = get_line_vectors_by_api(
            texts, row_ids, tasks_path, result_path,
            api or "openai", model_api or "text-embedding-3-small",
            api_key=api_key,
            max_input_len=max_input_len,
        )

    if store_embeddings:
        helper.set_embeddings(db_path, direction, row_ids, embeddings, is_proxy)

    return dict(zip(row_ids, embeddings))


def _process_batch_with_cache(
    db_path, line_ids_from, line_ids_to, lines_from_batch, lines_to_batch,
    use_proxy_from, use_proxy_to, model_name, embed_batch_size,
    normalize_embeddings, show_progress_bar, model,
    lang_emb_from, lang_emb_to, store_embeddings, use_api, embedding_cache,
    api=None, model_api=None, api_key=None, max_input_len=None,
):
    """Process a batch using the in-memory embedding cache to skip redundant computation."""
    cache_from = embedding_cache["from"]
    cache_to = embedding_cache["to"]

    # Find which IDs are missing from the cache
    missing_from = [lid for lid in line_ids_from if lid not in cache_from]
    missing_to = [lid for lid in line_ids_to if lid not in cache_to]

    # Compute only the missing embeddings
    if missing_from:
        new_from = _compute_embeddings_for_ids(
            db_path, "from", missing_from, lines_from_batch, use_proxy_from,
            model_name, embed_batch_size, normalize_embeddings, show_progress_bar,
            model, lang_emb_from, store_embeddings, use_api,
            api=api, model_api=model_api, api_key=api_key, max_input_len=max_input_len,
        )
        cache_from.update(new_from)

    if missing_to:
        new_to = _compute_embeddings_for_ids(
            db_path, "to", missing_to, lines_to_batch, use_proxy_to,
            model_name, embed_batch_size, normalize_embeddings, show_progress_bar,
            model, lang_emb_to, store_embeddings, use_api,
            api=api, model_api=model_api, api_key=api_key, max_input_len=max_input_len,
        )
        cache_to.update(new_to)

    # Assemble vectors in correct order from cache.
    # Some IDs may not exist in DB (0-based line_ids vs 1-based DB IDs),
    # matching the original update_embeddings behavior that silently skips them.
    vectors1 = [cache_from[lid] for lid in line_ids_from if lid in cache_from]
    vectors2 = [cache_to[lid] for lid in line_ids_to if lid in cache_to]

    return vectors1, vectors2


def _get_sim_matrix_reference(vec1, vec2, window):
    """Original (slow) similarity matrix implementation kept for testing."""
    sim_matrix = np.zeros((len(vec1), len(vec2)))
    k = len(vec1) / len(vec2)
    for i, vector1 in enumerate(vec1):
        for j, vector2 in enumerate(vec2):
            if (j * k > i - window) & (j * k < i + window):
                sim = 1 - spatial.distance.cosine(vector1, vector2)
                sim_matrix[i, j] = max(sim, 0.01)
    return sim_matrix


def get_sim_matrix(vec1, vec2, window):
    """Calculate similarity matrix (vectorized)"""
    vec1 = np.asarray(vec1, dtype=np.float64)
    vec2 = np.asarray(vec2, dtype=np.float64)

    # Normalize rows
    norms1 = np.linalg.norm(vec1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(vec2, axis=1, keepdims=True)
    norms1 = np.where(norms1 == 0, 1, norms1)
    norms2 = np.where(norms2 == 0, 1, norms2)
    normed1 = vec1 / norms1
    normed2 = vec2 / norms2

    # Cosine similarity via single BLAS matmul
    sim_matrix = normed1 @ normed2.T

    # Build window mask: j * k > i - window and j * k < i + window
    k = len(vec1) / len(vec2)
    i_indices = np.arange(len(vec1))[:, None]  # (N, 1)
    j_indices = np.arange(len(vec2))[None, :]  # (1, M)
    jk = j_indices * k
    mask = (jk > i_indices - window) & (jk < i_indices + window)

    # Apply mask and floor
    sim_matrix = np.where(mask, np.maximum(sim_matrix, 0.01), 0.0)
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
    helper.touch_last_edited_conn(db)


def get_doc_index(db):
    """Get document index"""
    res = []
    try:
        cur = db.execute("SELECT contents FROM doc_index")
        res = json.loads(cur.fetchone()[0])
    except:
        # logging.warning("can not fetch index db")
        pass
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
        helper.touch_last_edited_conn(db)


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
                "insert into splitted_from(id, text, proxy_text, exclude, paragraph, h1, h2, h3, h4, h5, divider, verse) values (?,?,?,?,?,?,?,?,?,?,?,?)",
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
                        text[1][7] if len(text[1]) > 7 else 0,
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
                "insert into splitted_to(id, text, proxy_text, exclude, paragraph, h1, h2, h3, h4, h5, divider, verse) values (?,?,?,?,?,?,?,?,?,?,?,?)",
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
                        text[1][7] if len(text[1]) > 7 else 0,
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
                "insert into splitted_from(id, text, proxy_text, exclude, paragraph, h1, h2, h3, h4, h5, divider, verse) values (?,?,?,?,?,?,?,?,?,?,?,?)",
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
                        text[1][7] if len(text[1]) > 7 else 0,
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
                "insert into splitted_to(id, text, proxy_text, exclude, paragraph, h1, h2, h3, h4, h5, divider, verse) values (?,?,?,?,?,?,?,?,?,?,?,?)",
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
                        text[1][7] if len(text[1]) > 7 else 0,
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


class TrivialAlignmentError(Exception):
    """Raised when two texts cannot be aligned trivially because their
    structure does not match (different number of paragraphs/marks, mismatched
    markup, or — in strict mode — a differing per-paragraph sentence split)."""


def _detect_meta_mark(line):
    """Return the meta mark a raw marked line ends with (e.g. 'h2', 'title',
    'divider'), or None for a regular paragraph line."""
    for mark in preprocessor.MARK_META:
        if line.endswith(f"{preprocessor.PARAGRAPH_MARK}{mark}."):
            return mark
    return None


# Metadata marks describe the whole document rather than a position in its body.
# Unlike structural marks (h1-h5/divider/qtext/qname/image) they may legitimately
# appear on one side without the other — e.g. a machine translation credits a
# ``translator`` the original text never had. ``trivial_alignment`` therefore
# matches them per side (recording each into ``meta`` independently) rather than
# positionally, and they never consume a 1:1 body-alignment slot.
META_SIDE_MARKS = (preprocessor.TITLE, preprocessor.AUTHOR, preprocessor.TRANSLATOR)


def _read_nonempty_lines(path):
    """Read a marked text file, returning stripped non-empty lines."""
    with open(path, mode="r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def _read_body_with_verse_stanzas(path):
    """Read a marked file and return ``(lines, stanzas)``.

    ``lines`` is the stripped non-empty line list (the anchor sequence
    ``trivial_alignment`` pairs on — identical to ``_read_nonempty_lines``).

    ``stanzas`` is a parallel list: for a ``%%%%%verse.`` line it holds the poem
    stanza index — a monotonic counter that increments at the start of each poem
    (first verse line after non-verse) and at every blank line *within* a verse
    run; for any non-verse line it is 0. Blank lines are still dropped from the
    body (so they never desync alignment), they only advance this counter so the
    stanza structure survives into the ``verse`` column and the rendered poem.
    The counter is computed per side, so a stray blank-line difference between
    source and translation degrades stanza rendering but never breaks alignment.
    """
    lines, stanzas = [], []
    stanza = 0
    prev_was_verse = False
    pending_break = False
    with open(path, mode="r", encoding="utf-8") as f:
        for raw in f.readlines():
            s = raw.strip()
            if not s:
                if prev_was_verse:
                    pending_break = True
                continue
            if _detect_meta_mark(s) == preprocessor.VERSE:
                if not prev_was_verse or pending_break:
                    stanza += 1
                stanzas.append(stanza)
                prev_was_verse = True
            else:
                stanzas.append(0)
                prev_was_verse = False
            pending_break = False
            lines.append(s)
    return lines, stanzas


def _resolve_split_langcode(langcode):
    """Validate a language code for the splitter, falling back to the generic
    code (with a warning) when unsupported — mirrors split_by_sentences_and_save."""
    if splitter.is_lang_code_valid(langcode):
        return langcode
    logging.warning(
        "Unsupported language code '%s', falling back to '%s' (General) for splitting",
        langcode,
        splitter.XX_CODE,
    )
    return splitter.XX_CODE


def trivial_alignment(
    from_path,
    to_path,
    lang_from,
    lang_to,
    output_path,
    name="",
    clean_text=False,
    on_mismatch="merge",
    batch_size=200,
    file_from=None,
    file_to=None,
    id_from=None,
    id_to=None,
):
    """Build a ready-to-use alignment database from two structurally parallel
    marked texts WITHOUT embeddings or the similarity-based alignment algorithm.

    This is meant for the case where the translation is produced under full
    control (for example by ``smart_translator``) so that both texts share the
    exact same paragraph structure and Lingtrain markup. In that case the
    alignment is trivial: sentences line up 1:1 inside every paragraph, so there
    is no need to compute embeddings or run the resolver.

    The texts are anchored on raw lines (one paragraph or one ``%%%%%`` mark per
    line). This anchor is robust: language-specific sentence splitters can
    legitimately disagree on a few boundaries (quotes, abbreviations, dashes),
    which would desynchronise a naive global 1:1 mapping, but the paragraph
    structure stays aligned by construction.

    Validation (always enforced, raises ``TrivialAlignmentError`` on failure):
      * both files must have the same number of *body* lines — paragraphs plus
        structural marks (h1-h5/divider/qtext/qname/image);
      * the ``%%%%%`` mark on every body line must match between the two texts.

    ``title``/``author``/``translator`` are side-independent metadata: they are
    recorded into the ``meta`` table per side and need NOT match (a machine
    translation may carry a ``translator`` the source lacks). They are expected
    at the top of the document and never participate in body anchoring.

    Per-paragraph sentence split handling (``on_mismatch``):
      * ``"merge"`` (default): when a paragraph splits into a different number of
        sentences on each side, emit it as a single N:M merged pair and continue.
        The returned report lists every merged paragraph so nothing is hidden.
      * ``"error"``: raise ``TrivialAlignmentError`` listing the offending
        paragraphs (strict 1:1, matching "check that sentence counts are equal").

    Args:
        from_path: path to the source marked text file.
        to_path: path to the target (translation) marked text file.
        lang_from: language code of the source text (e.g. "en").
        lang_to: language code of the target text (e.g. "ru").
        output_path: path of the alignment database to create (e.g. "book.lt").
            An existing file at this path is overwritten.
        name: human-readable alignment name stored in the database.
        clean_text: apply language-specific cleaning during splitting (matches
            the ``clean_text`` flag of ``split_by_sentences_and_save``).
        on_mismatch: "merge" (default) or "error" — see above.
        batch_size: number of source lines per batch in the output (default 200).
            Keep this equal to the consuming web app's ``ALIGNER_BATCH_SIZE`` so
            the import recognises the file as fully aligned (DONE) rather than
            partially processed.
        file_from, file_to: original file names stored in the ``files`` table
            (default: basenames of the input paths).
        id_from, id_to: GUIDs stored in the ``files`` table (default: random).

    Returns:
        A report dict with line/paragraph/sentence counts, the meta marks found,
        the number of 1:1 units, and details of any merged paragraphs.
    """
    if on_mismatch not in ("merge", "error"):
        raise ValueError("on_mismatch must be 'merge' or 'error'")

    lang_from_split = _resolve_split_langcode(lang_from)
    lang_to_split = _resolve_split_langcode(lang_to)

    raw_from, verse_stanza_from = _read_body_with_verse_stanzas(from_path)
    raw_to, verse_stanza_to = _read_body_with_verse_stanzas(to_path)

    # ``title``/``author``/``translator`` are side-independent metadata
    # (``META_SIDE_MARKS``): they are drained per side and recorded into ``meta``
    # without consuming a body-alignment slot, so a translation may carry a
    # ``translator`` mark the source lacks. Only the *body* — paragraphs and
    # structural marks (h1-h5/divider/qtext/qname/image) — must line up 1:1.
    n_body_from = sum(1 for ln in raw_from if _detect_meta_mark(ln) not in META_SIDE_MARKS)
    n_body_to = sum(1 for ln in raw_to if _detect_meta_mark(ln) not in META_SIDE_MARKS)
    if n_body_from != n_body_to:
        raise TrivialAlignmentError(
            f"Paragraph/line count mismatch: '{from_path}' has {n_body_from} "
            f"body line(s), '{to_path}' has {n_body_to} (excluding title/author/"
            f"translator metadata). Trivial alignment requires identical paragraph "
            f"structure (one paragraph or one mark per line, in the same order)."
        )

    # Per-side splitted lines: each entry is (text, marks_tuple) where
    # marks_tuple = (paragraph, h1, h2, h3, h4, h5, divider) cumulative counters.
    splitted_from, splitted_to = [], []
    meta_from, meta_par_from = defaultdict(list), defaultdict(list)
    meta_to, meta_par_to = defaultdict(list), defaultdict(list)
    counters_from, counters_to = defaultdict(int), defaultdict(int)

    # Aligned units driving processing tables / doc_index. Each unit is
    # (from_ids, from_text, to_ids, to_text) and becomes one paired row.
    units = []
    merged = []  # diagnostics for paragraphs that were not a clean 1:1 split
    fid = tid = 0  # running splitted ids (1-based)

    def marks_tuple(c):
        return (
            c[preprocessor.PARAGRAPH],
            c[preprocessor.H1],
            c[preprocessor.H2],
            c[preprocessor.H3],
            c[preprocessor.H4],
            c[preprocessor.H5],
            c[preprocessor.DIVIDER],
        )

    def consume_meta_mark(idx, raw, meta, meta_par, counters):
        """If ``raw[idx]`` is a side-independent metadata mark, record it with the
        current paragraph id and return ``idx + 1``; otherwise return ``None``.

        The paragraph counter is bumped exactly as for any other line, so a pair
        of structurally identical texts yields byte-identical paragraph ids — the
        side-independent handling only changes behaviour when the two sides carry
        a *different* set of metadata marks."""
        if idx >= len(raw):
            return None
        mark = _detect_meta_mark(raw[idx])
        if mark not in META_SIDE_MARKS:
            return None
        meta[mark].append(get_mark_value(raw[idx], mark))
        meta_par[mark].append(counters[preprocessor.PARAGRAPH])
        counters[preprocessor.PARAGRAPH] += 1
        return idx + 1

    # Two cursors: drain leading (or otherwise unmatched) metadata marks one side
    # at a time, then pair the next body line on each side. Metadata sits at the
    # top in practice, so it is consumed before any body line and the body stays
    # aligned; a metadata mark elsewhere is still absorbed without desync.
    i = j = 0
    body_index = 0
    while i < len(raw_from) or j < len(raw_to):
        adv = consume_meta_mark(i, raw_from, meta_from, meta_par_from, counters_from)
        if adv is not None:
            i = adv
            continue
        adv = consume_meta_mark(j, raw_to, meta_to, meta_par_to, counters_to)
        if adv is not None:
            j = adv
            continue

        # Both cursors now sit on a body line (the body-count check guarantees
        # the two sides run out of body lines together).
        line_from, line_to = raw_from[i], raw_to[j]
        body_index += 1

        mark_from = _detect_meta_mark(line_from)
        mark_to = _detect_meta_mark(line_to)

        if mark_from != mark_to:
            raise TrivialAlignmentError(
                f"Markup mismatch at body line {body_index} (from line {i + 1}, "
                f"to line {j + 1}): 'from' mark={mark_from!r}, 'to' mark={mark_to!r}.\n"
                f"  from: {line_from[:120]}\n  to:   {line_to[:120]}"
            )

        if mark_from == preprocessor.VERSE:
            # Verse line: a content-bearing ATOMIC body unit. Exactly one aligned
            # row per side (the whole line — never sentence-split, never joined),
            # paired 1:1. The `verse` column carries the poem stanza index so the
            # reader can render the poem with stanza breaks. Paragraph counter is
            # bumped like a one-sentence prose paragraph.
            text_from = get_mark_value(line_from, preprocessor.VERSE)
            text_to = get_mark_value(line_to, preprocessor.VERSE)
            mt_from = marks_tuple(counters_from)
            mt_to = marks_tuple(counters_to)
            fid += 1
            splitted_from.append((text_from, mt_from, verse_stanza_from[i]))
            tid += 1
            splitted_to.append((text_to, mt_to, verse_stanza_to[j]))
            units.append(([fid], text_from, [tid], text_to))
            counters_from[preprocessor.PARAGRAPH] += 1
            counters_to[preprocessor.PARAGRAPH] += 1
            i += 1
            j += 1
            continue

        if mark_from is not None:
            # Structural mark on both sides. Bump structural counters first, then
            # record meta with the current paragraph id, then bump the paragraph
            # counter — mirroring aligner.handle_marks.
            for mark in preprocessor.MARK_COUNTERS:
                ending = f"{preprocessor.PARAGRAPH_MARK}{mark}."
                if line_from.endswith(ending):
                    counters_from[mark] += 1
                if line_to.endswith(ending):
                    counters_to[mark] += 1
            meta_from[mark_from].append(get_mark_value(line_from, mark_from))
            meta_par_from[mark_from].append(counters_from[preprocessor.PARAGRAPH])
            meta_to[mark_to].append(get_mark_value(line_to, mark_to))
            meta_par_to[mark_to].append(counters_to[preprocessor.PARAGRAPH])
            counters_from[preprocessor.PARAGRAPH] += 1
            counters_to[preprocessor.PARAGRAPH] += 1
            i += 1
            j += 1
            continue

        # Regular paragraph on both sides — split into sentences independently.
        sents_from = [
            s.strip()
            for s in splitter.split_by_sentences([line_from], lang_from_split, clean_text)
            if s.strip()
        ]
        sents_to = [
            s.strip()
            for s in splitter.split_by_sentences([line_to], lang_to_split, clean_text)
            if s.strip()
        ]

        mt_from = marks_tuple(counters_from)
        mt_to = marks_tuple(counters_to)

        para_from_ids = []
        for s in sents_from:
            fid += 1
            splitted_from.append((s, mt_from, 0))
            para_from_ids.append(fid)
        para_to_ids = []
        for s in sents_to:
            tid += 1
            splitted_to.append((s, mt_to, 0))
            para_to_ids.append(tid)

        if sents_from and len(sents_from) == len(sents_to):
            # Clean 1:1 split — one paired row per sentence.
            for k in range(len(sents_from)):
                units.append(
                    ([para_from_ids[k]], sents_from[k], [para_to_ids[k]], sents_to[k])
                )
        elif sents_from or sents_to:
            # Differing (or one-sided) split — keep the paragraph as one merged
            # unit so the result stays complete and paragraph-aligned.
            merged.append(
                {
                    "line": i + 1,
                    "paragraph": counters_from[preprocessor.PARAGRAPH],
                    "from_count": len(sents_from),
                    "to_count": len(sents_to),
                    "from_text": " ".join(sents_from)[:160],
                    "to_text": " ".join(sents_to)[:160],
                }
            )
            units.append(
                (
                    para_from_ids,
                    " ".join(sents_from),
                    para_to_ids,
                    " ".join(sents_to),
                )
            )
        # else: both sides empty after splitting — nothing to emit.

        counters_from[preprocessor.PARAGRAPH] += 1
        counters_to[preprocessor.PARAGRAPH] += 1
        i += 1
        j += 1

    if on_mismatch == "error" and merged:
        preview = "\n".join(
            f"  line {m['line']}: from={m['from_count']} sentence(s), "
            f"to={m['to_count']} sentence(s)"
            for m in merged[:20]
        )
        more = "\n  ..." if len(merged) > 20 else ""
        raise TrivialAlignmentError(
            f"Sentence-count mismatch in {len(merged)} paragraph(s); cannot align "
            f"strictly 1:1. Use on_mismatch='merge' to merge them, or fix the "
            f"texts so each paragraph splits into the same number of sentences.\n"
            f"{preview}{more}"
        )

    # Defensive: structural (body) marks were matched per line, so their meta
    # counts must already agree. Side-independent metadata marks (title/author/
    # translator) may legitimately differ between the two sides — skip them here.
    for key in set(meta_from) | set(meta_to):
        if key in META_SIDE_MARKS:
            continue
        if len(meta_from.get(key, [])) != len(meta_to.get(key, [])):
            raise TrivialAlignmentError(
                f"Meta mark count mismatch for '{key}': "
                f"from={len(meta_from.get(key, []))}, to={len(meta_to.get(key, []))}"
            )

    # ---- Build the alignment database ----
    # A prior run may leave sqlite connections held by GC cycles (Python's
    # `with sqlite3.connect()` manages the transaction, not the connection, so
    # the handle is not closed deterministically). On Windows those linger as
    # file locks; collect them so init_document_db can overwrite the file.
    if os.path.isfile(output_path):
        gc.collect()
    helper.init_document_db(output_path)

    file_from = file_from or os.path.basename(from_path)
    file_to = file_to or os.path.basename(to_path)
    id_from = id_from or uuid.uuid4().hex
    id_to = id_to or uuid.uuid4().hex

    # NB: sqlite3's `with` block only manages the transaction, it does NOT
    # close the connection. Close explicitly so a re-run can overwrite the file
    # on Windows (otherwise init_document_db's os.remove hits a lingering lock).
    db = sqlite3.connect(output_path)
    try:
        db.executemany(
            "insert into splitted_from(id, text, proxy_text, exclude, paragraph, h1, h2, h3, h4, h5, divider, verse) values (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (idx + 1, text, "", 0, m[0], m[1], m[2], m[3], m[4], m[5], m[6], v)
                for idx, (text, m, v) in enumerate(splitted_from)
            ],
        )
        db.executemany(
            "insert into splitted_to(id, text, proxy_text, exclude, paragraph, h1, h2, h3, h4, h5, divider, verse) values (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (idx + 1, text, "", 0, m[0], m[1], m[2], m[3], m[4], m[5], m[6], v)
                for idx, (text, m, v) in enumerate(splitted_to)
            ],
        )
        db.executemany(
            "insert into meta(key, val, occurence, par_id) values(?,?,?,?)",
            flatten_meta(meta_from, meta_par_from, "from"),
        )
        db.executemany(
            "insert into meta(key, val, occurence, par_id) values(?,?,?,?)",
            flatten_meta(meta_to, meta_par_to, "to"),
        )
        db.executemany(
            "insert into languages(key, val) values(?,?)",
            [("from", lang_from), ("to", lang_to)],
        )
        db.executemany(
            "insert into files(direction, name, guid) values(?,?,?)",
            [("from", file_from, id_from), ("to", file_to, id_to)],
        )
        db.commit()
    finally:
        db.close()
    helper.set_name(output_path, name)

    # Processing tables + document index. Units are laid out into batches of
    # `batch_size` source lines, mirroring the real aligner. This matters for
    # the web app: on import it infers total_batches = ceil(len_from/batch_size)
    # and curr_batches = number of batches present, marking the alignment DONE
    # only when they match. A single giant batch would otherwise import as
    # partially aligned. Keep batch_size aligned with the app's ALIGNER_BATCH_SIZE.
    batched = defaultdict(list)
    for u in units:
        anchor = u[0][0] if u[0] else (u[2][0] if u[2] else 1)
        batched[(anchor - 1) // batch_size].append(u)

    data = []
    for batch_id in sorted(batched):
        batch_units = batched[batch_id]
        texts_from = [
            (json.dumps(u[0]), (u[0][0] if u[0] else None), u[1]) for u in batch_units
        ]
        texts_to = [
            (json.dumps(u[2]), (u[2][0] if u[2] else None), u[3]) for u in batch_units
        ]
        data.append((batch_id, texts_from, texts_to, 0, 0))

    save_db(output_path, data)
    batch_ids = sorted(batched)
    update_history(
        output_path,
        batch_ids,
        con.OPERATION_TRIVIAL,
        {"on_mismatch": on_mismatch, "merged_paragraphs": len(merged)},
    )
    helper.migrate_document_db(output_path)

    report = {
        "output": output_path,
        "lines": len(raw_from),
        "paragraphs": counters_from[preprocessor.PARAGRAPH],
        "from_sentences": len(splitted_from),
        "to_sentences": len(splitted_to),
        "units": len(units),
        "batches": len(batch_ids),
        "one_to_one": len(units) - len(merged),
        "merged_paragraphs": len(merged),
        "merged_details": merged,
        "meta": {key: len(vals) for key, vals in meta_from.items()},
        "status": "perfect" if not merged else "merged",
    }
    logging.info(
        "trivial_alignment: %s lines, %s units (%s merged) -> %s",
        report["lines"],
        report["units"],
        report["merged_paragraphs"],
        output_path,
    )
    return report


# ---------------------------------------------------------------------------
# Multilingual (.ltm) trivial alignment — N strictly-1:1:N editions in one file
# ---------------------------------------------------------------------------
# This is the multilingual analogue of trivial_alignment: it builds a render-only
# .ltm directly from N structurally-parallel marked texts. It does NOT reuse the
# legacy reader.get_paragraphs_polybook / create_polybook merge (those reconcile
# DISAGREEING embedding-aligned segmentations at read time — impossible and lossy
# under controlled strict-1:1:N translation). The builder generalises
# trivial_alignment's two cursors to N over a single ``source_lang`` reference and
# records the shared body skeleton into the ``structure`` table.


def flatten_meta_multi(meta, meta_par_ids, lang):
    """Flatten one edition's meta into ``(lang, bare_key, val, occurence, par_id)``
    rows for the ``.ltm`` ``meta`` table. Unlike ``flatten_meta`` (which emits
    ``'<mark>_from'``/``'<mark>_to'`` keys) the key stays BARE."""
    res = []
    for key in meta:
        for i, (val, par_id) in enumerate(zip(meta[key], meta_par_ids[key])):
            res.append((lang, key, val, i, par_id))
    return res


def _ltm_mark_mismatch(lang, paragraph, expected_kind, got_mark, line):
    return (
        f"Markup mismatch adding '{lang}' at paragraph {paragraph}: the stored "
        f"structure expects kind={expected_kind!r} but this edition's line is "
        f"mark={got_mark!r}.\n  {line[:160]}"
    )


def trivial_alignment_multi(
    marked_paths_by_lang,
    output_path,
    *,
    source_lang,
    name="",
    clean_text_by_lang=None,
    on_mismatch="error",
    file_names=None,
    guids=None,
):
    """Build an ``.ltm`` book from N>=1 structurally-parallel marked texts.

    A single edition is a valid monolingual book; two or more make it parallel.
    Every edition is produced under full control (smart_translator), so all share
    the exact same paragraph structure and Lingtrain markup and the alignment is
    trivial: sentences line up 1:1:...:1 inside every paragraph. One ``source_lang``
    edition is the immutable structural reference (``is_source``); future
    :func:`add_language` calls validate against the ``structure`` it defines (and
    turn a monolingual book into a parallel one without changing coordinates).

    Args:
        marked_paths_by_lang: ``{langcode: path}`` for every edition.
        output_path: path of the ``.ltm`` to create (overwritten if present).
        source_lang: which edition defines the canonical structure (required;
            must be a key of ``marked_paths_by_lang``).
        name: human-readable book name stored in ``info``.
        clean_text_by_lang: per-edition ``clean_text`` flag for the splitter
            (zh/CJK must be ``True`` or the splitter desyncs sentence counts).
        on_mismatch: ``"error"`` (default, strict) raises listing offending
            paragraphs; ``"merge"`` emits a single merged row per edition for a
            paragraph whose editions disagree on sentence count.
        file_names, guids: optional per-edition ``files`` provenance.

    Returns:
        A report dict (langs, source_lang, paragraphs, per-lang sentence counts,
        structural mark/verse counts, merged paragraph details, status).
    """
    if on_mismatch not in ("merge", "error"):
        raise ValueError("on_mismatch must be 'merge' or 'error'")
    if not marked_paths_by_lang:
        raise ValueError("need at least 1 edition to build a .ltm")
    if source_lang not in marked_paths_by_lang:
        raise ValueError(
            f"source_lang '{source_lang}' is not among the editions "
            f"{list(marked_paths_by_lang)}"
        )

    # Ordered langs: source first, then the rest in input order.
    langs = [source_lang] + [l for l in marked_paths_by_lang if l != source_lang]
    clean_text = {
        l: bool((clean_text_by_lang or {}).get(l, False)) for l in langs
    }
    split_lang = {l: _resolve_split_langcode(l) for l in langs}

    raw, stanzas = {}, {}
    for lang in langs:
        raw[lang], stanzas[lang] = _read_body_with_verse_stanzas(
            marked_paths_by_lang[lang]
        )

    # Body-count invariant: title/author/translator (META_SIDE_MARKS) are
    # side-independent and excluded; everything else must match across editions.
    def _body_count(lines):
        return sum(1 for ln in lines if _detect_meta_mark(ln) not in META_SIDE_MARKS)

    n_body_src = _body_count(raw[source_lang])
    for lang in langs:
        n = _body_count(raw[lang])
        if n != n_body_src:
            raise TrivialAlignmentError(
                f"Body line count mismatch: source '{source_lang}' has {n_body_src} "
                f"body line(s), '{lang}' has {n} (excluding title/author/translator "
                f"metadata). Multilingual trivial alignment requires identical "
                f"paragraph structure (one paragraph or one mark per line)."
            )

    structure = []  # (paragraph, kind, sentence_count, verse)
    splitted = {l: [] for l in langs}  # (paragraph, sentence, id, text, verse)
    meta = {l: defaultdict(list) for l in langs}
    meta_par = {l: defaultdict(list) for l in langs}
    sent_id = {l: 0 for l in langs}
    cursors = {l: 0 for l in langs}
    merged = []
    paragraph = 0

    def _side_mark(lang):
        idx = cursors[lang]
        if idx >= len(raw[lang]):
            return None
        mark = _detect_meta_mark(raw[lang][idx])
        return mark if mark in META_SIDE_MARKS else None

    while True:
        # Drain a side-independent metadata mark from any edition sitting on one.
        # Like trivial_alignment, metadata is consumed before body lines so the
        # body stays aligned even when editions carry a different metadata set.
        drained = False
        for lang in langs:
            mark = _side_mark(lang)
            if mark is not None:
                meta[lang][mark].append(get_mark_value(raw[lang][cursors[lang]], mark))
                meta_par[lang][mark].append(paragraph)
                cursors[lang] += 1
                drained = True
                break
        if drained:
            continue
        if all(cursors[l] >= len(raw[l]) for l in langs):
            break

        lines = {l: raw[l][cursors[l]] for l in langs}
        marks = {l: _detect_meta_mark(lines[l]) for l in langs}
        src_mark = marks[source_lang]
        for lang in langs:
            if marks[lang] != src_mark:
                raise TrivialAlignmentError(
                    f"Markup mismatch at body paragraph {paragraph + 1}: source "
                    f"'{source_lang}' mark={src_mark!r}, '{lang}' mark={marks[lang]!r}.\n"
                    f"  {source_lang}: {lines[source_lang][:120]}\n"
                    f"  {lang}: {lines[lang][:120]}"
                )

        paragraph += 1

        if src_mark == preprocessor.VERSE:
            # Atomic verse line: one row per edition, stanza index from source.
            structure.append(
                (paragraph, preprocessor.VERSE, 1, stanzas[source_lang][cursors[source_lang]])
            )
            for lang in langs:
                text = get_mark_value(lines[lang], preprocessor.VERSE)
                sent_id[lang] += 1
                splitted[lang].append(
                    (paragraph, 1, sent_id[lang], text, stanzas[lang][cursors[lang]])
                )
                cursors[lang] += 1
            continue

        if src_mark is not None:
            # Structural mark (h1-h5/divider/qtext/qname/image): content -> meta,
            # not splitted; one structure row with sentence_count 0.
            structure.append((paragraph, src_mark, 0, 0))
            for lang in langs:
                meta[lang][src_mark].append(get_mark_value(lines[lang], src_mark))
                meta_par[lang][src_mark].append(paragraph)
                cursors[lang] += 1
            continue

        # Regular prose paragraph: split each edition independently.
        sents = {
            l: [
                s.strip()
                for s in splitter.split_by_sentences([lines[l]], split_lang[l], clean_text[l])
                if s.strip()
            ]
            for l in langs
        }
        counts = {l: len(sents[l]) for l in langs}
        src_count = counts[source_lang]

        if src_count == 0 and all(c == 0 for c in counts.values()):
            # Paragraph empty on every edition after splitting — emit nothing.
            paragraph -= 1
            for lang in langs:
                cursors[lang] += 1
            continue

        if src_count > 0 and all(counts[l] == src_count for l in langs):
            sc = src_count
        else:
            # Differing (or one-sided) split: keep the paragraph as one merged
            # unit per edition so the book stays complete and paragraph-aligned.
            merged.append({"paragraph": paragraph, "counts": dict(counts)})
            for lang in langs:
                sents[lang] = [" ".join(sents[lang])] if sents[lang] else [""]
            sc = 1

        structure.append((paragraph, "text", sc, 0))
        for lang in langs:
            for k, s in enumerate(sents[lang], start=1):
                sent_id[lang] += 1
                splitted[lang].append((paragraph, k, sent_id[lang], s, 0))
            cursors[lang] += 1

    if on_mismatch == "error" and merged:
        preview = "\n".join(
            f"  paragraph {m['paragraph']}: "
            + ", ".join(f"{l}={m['counts'][l]}" for l in langs)
            for m in merged[:20]
        )
        more = "\n  ..." if len(merged) > 20 else ""
        raise TrivialAlignmentError(
            f"Sentence-count mismatch in {len(merged)} paragraph(s) across editions; "
            f"cannot align strictly 1:1:N. Use on_mismatch='merge' to merge them, or "
            f"fix the texts so each paragraph splits into the same number of sentences "
            f"in every language.\n{preview}{more}"
        )

    # ---- write the .ltm ----
    if os.path.isfile(output_path):
        gc.collect()
    helper.init_multi_db(output_path)

    file_names = file_names or {}
    guids = guids or {}
    now = helper._utc_now_iso()

    db = sqlite3.connect(output_path)
    try:
        db.executemany(
            "insert into languages(lang, ord, is_source, added_at) values (?,?,?,?)",
            [(l, i, 1 if l == source_lang else 0, now) for i, l in enumerate(langs)],
        )
        db.executemany(
            "insert into structure(paragraph, kind, sentence_count, verse) values (?,?,?,?)",
            structure,
        )
        for lang in langs:
            db.executemany(
                "insert into splitted(lang, paragraph, sentence, id, text, verse) "
                "values (?,?,?,?,?,?)",
                [(lang, p, s, i, t, v) for (p, s, i, t, v) in splitted[lang]],
            )
            db.executemany(
                "insert into meta(lang, key, val, occurence, par_id) values (?,?,?,?,?)",
                flatten_meta_multi(meta[lang], meta_par[lang], lang),
            )
            db.execute(
                "insert into files(lang, name, guid, added_at) values (?,?,?,?)",
                (
                    lang,
                    file_names.get(lang) or os.path.basename(marked_paths_by_lang[lang]),
                    guids.get(lang) or uuid.uuid4().hex,
                    now,
                ),
            )
        helper.set_info_value_conn(db, helper.INFO_KEY_SOURCE_LANG, source_lang)
        helper.set_info_value_conn(db, helper.INFO_KEY_NAME, name)
        db.execute(
            "insert into history(operation, lang, insert_ts, parameters) values (?,?,?,?)",
            (
                con.OPERATION_TRIVIAL_MULTI,
                source_lang,
                now,
                json.dumps(
                    {"on_mismatch": on_mismatch, "langs": langs, "merged_paragraphs": len(merged)}
                ),
            ),
        )
        db.commit()
    finally:
        db.close()

    report = {
        "output": output_path,
        "format": "ltm",
        "langs": langs,
        "source_lang": source_lang,
        "paragraphs": len(structure),
        "sentences_by_lang": {l: len(splitted[l]) for l in langs},
        "structural_marks": sum(1 for s in structure if s[1] in preprocessor.MARK_COUNTERS),
        "verse_lines": sum(1 for s in structure if s[1] == preprocessor.VERSE),
        "merged_paragraphs": len(merged),
        "merged_details": merged,
        "status": "perfect" if not merged else "merged",
    }
    logging.info(
        "trivial_alignment_multi: %s editions, %s paragraphs (%s merged) -> %s",
        len(langs),
        len(structure),
        len(merged),
        output_path,
    )
    return report


# Neutral public name: an .ltm may hold one edition (a single marked text, a
# monolingual book) or many. ``trivial_alignment_multi`` is kept as the original
# alias; new callers should prefer ``build_ltm``.
build_ltm = trivial_alignment_multi


def build_ltm_from_prepared(
    prepared_path,
    output_path,
    *,
    source_lang,
    name,
    file_name=None,
    guid=None,
):
    """Build a single-edition ``.ltm`` from an already prepared document.

    ``prepared_path`` is the editable ``splitted`` artifact produced by
    :func:`splitter.split_by_sentences_and_save`. Its physical lines are already
    sentence units and are therefore copied directly instead of being split a
    second time. Retained ``%%%%%`` paragraph endings group prose lines into the
    shared structure; deleting one of those endings in the preview intentionally
    joins the surrounding sentence groups.

    Empty lines are not present in prepared artifacts, so lost gaps inside verse
    cannot be reconstructed. Every consecutive run of surviving ``verse`` lines
    is stored as one stanza.
    """
    canonical_name = str(name or "").strip()
    if not canonical_name:
        raise ValueError("Book name must not be empty")
    if not source_lang or not str(source_lang).strip():
        raise ValueError("Source language must not be empty")
    if not os.path.isfile(prepared_path):
        raise FileNotFoundError(prepared_path)

    with open(prepared_path, mode="r", encoding="utf-8") as prepared:
        raw_lines = [line.strip() for line in prepared if line.strip()]
    if not raw_lines:
        raise ValueError("Prepared document has no readable content")

    structure = []  # (paragraph, kind, sentence_count, verse)
    splitted_rows = []  # (lang, paragraph, sentence, id, text, verse)
    meta_rows = []  # (lang, key, val, occurrence, par_id)
    meta_occurrences = defaultdict(int)
    pending_prose = []
    paragraph = 0
    sentence_id = 0
    stanza = 0
    previous_was_verse = False
    title_seen = False

    def add_meta(key, value, par_id):
        nonlocal title_seen
        if key == preprocessor.TITLE:
            if title_seen:
                return
            value = canonical_name
            title_seen = True
        occurrence = meta_occurrences[key]
        meta_occurrences[key] += 1
        meta_rows.append(
            (source_lang, key, value, occurrence, par_id)
        )

    def flush_prose():
        nonlocal paragraph, sentence_id, pending_prose
        if not pending_prose:
            return
        paragraph += 1
        structure.append((paragraph, "text", len(pending_prose), 0))
        for sentence, text in enumerate(pending_prose, start=1):
            sentence_id += 1
            splitted_rows.append(
                (source_lang, paragraph, sentence, sentence_id, text, 0)
            )
        pending_prose = []

    for raw_line in raw_lines:
        mark = _detect_meta_mark(raw_line)
        paragraph_end = False
        text = raw_line
        if mark is None:
            # Edited prepared artifacts can retain a paragraph boundary after a
            # structural marker (for example ``Chapter%%%%%h2.%%%%%``). Strip
            # that outer boundary first, then classify the remaining marker so
            # it cannot leak into ``splitted.text`` as ordinary reader prose.
            text, paragraph_end = preprocessor.strip_paragraph_mark(raw_line)
            text = text.strip()
            mark = _detect_meta_mark(text)
            if mark is not None:
                raw_line = text
        if mark is not None:
            flush_prose()
            value = get_mark_value(raw_line, mark).strip()
            if mark in META_SIDE_MARKS:
                add_meta(mark, value, paragraph)
                previous_was_verse = False
                continue

            paragraph += 1
            if mark == preprocessor.VERSE:
                if not previous_was_verse:
                    stanza += 1
                sentence_id += 1
                structure.append((paragraph, preprocessor.VERSE, 1, stanza))
                splitted_rows.append(
                    (source_lang, paragraph, 1, sentence_id, value, stanza)
                )
                previous_was_verse = True
                continue

            # Headings, dividers, quotes and images are positional metadata.
            # Their zero-sentence structure row preserves that position for the
            # reader and for a later add_language compatibility check.
            structure.append((paragraph, mark, 0, 0))
            add_meta(mark, value, paragraph)
            previous_was_verse = False
            continue

        previous_was_verse = False
        if preprocessor.PARAGRAPH_MARK in text:
            raise ValueError(
                "Prepared document contains a Lingtrain marker in an unsupported position"
            )
        if text:
            pending_prose.append(text)
        if paragraph_end:
            flush_prose()

    flush_prose()
    if not splitted_rows:
        raise ValueError("Prepared document has no readable body content")
    if not title_seen:
        # The submitted name is canonical for both the application row and the
        # embedded source edition, even when the uploaded document had no title.
        meta_rows.insert(0, (source_lang, preprocessor.TITLE, canonical_name, 0, 0))

    if os.path.isfile(output_path):
        gc.collect()
    helper.init_multi_db(output_path)
    now = helper._utc_now_iso()
    db = sqlite3.connect(output_path)
    try:
        db.execute(
            "insert into languages(lang, ord, is_source, added_at) values (?,?,?,?)",
            (source_lang, 0, 1, now),
        )
        db.executemany(
            "insert into structure(paragraph, kind, sentence_count, verse) values (?,?,?,?)",
            structure,
        )
        db.executemany(
            "insert into splitted(lang, paragraph, sentence, id, text, verse) "
            "values (?,?,?,?,?,?)",
            splitted_rows,
        )
        db.executemany(
            "insert into meta(lang, key, val, occurence, par_id) values (?,?,?,?,?)",
            meta_rows,
        )
        db.execute(
            "insert into files(lang, name, guid, added_at) values (?,?,?,?)",
            (
                source_lang,
                file_name or os.path.basename(prepared_path),
                guid or uuid.uuid4().hex,
                now,
            ),
        )
        helper.set_info_value_conn(db, helper.INFO_KEY_SOURCE_LANG, source_lang)
        helper.set_info_value_conn(db, helper.INFO_KEY_NAME, canonical_name)
        db.execute(
            "insert into history(operation, lang, insert_ts, parameters) values (?,?,?,?)",
            (
                con.OPERATION_BUILD_LTM_FROM_PREPARED,
                source_lang,
                now,
                json.dumps(
                    {
                        "source": "prepared_document",
                        "file_name": file_name or os.path.basename(prepared_path),
                        "guid": guid,
                    }
                ),
            ),
        )
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    return {
        "output": output_path,
        "format": "ltm",
        "langs": [source_lang],
        "source_lang": source_lang,
        "paragraphs": len(structure),
        "sentences_by_lang": {source_lang: len(splitted_rows)},
        "structural_marks": sum(
            1 for row in structure if row[1] in preprocessor.MARK_COUNTERS
        ),
        "verse_lines": sum(
            1 for row in structure if row[1] == preprocessor.VERSE
        ),
        "status": "perfect",
    }


def add_language(
    ltm_path,
    marked_path,
    lang,
    *,
    clean_text=False,
    on_mismatch="error",
    file_name=None,
    guid=None,
    replace=False,
):
    """Add (or, with ``replace=True``, overwrite) one edition in an existing
    ``.ltm``, validating it against the stored ``structure``.

    Purely additive: only this language's ``splitted``/``meta``/``files``/
    ``languages`` rows are written; NO other edition is read or modified, so every
    existing reader position is preserved. ``on_mismatch='merge'`` is refused here
    because merging would mutate the shared ``structure.sentence_count`` that the
    other editions already satisfy — a structurally divergent edition must be
    fixed (re-punctuated) or the book rebuilt.
    """
    if on_mismatch != "error":
        raise ValueError(
            "add_language only supports on_mismatch='error' (merging would mutate "
            "the shared structure other editions already satisfy)"
        )
    if not helper.is_ltm(ltm_path):
        raise TrivialAlignmentError(f"{ltm_path} is not a multilingual (.ltm) file")

    existing = {d["lang"] for d in helper.get_ltm_languages(ltm_path)}
    if lang in existing and not replace:
        raise TrivialAlignmentError(
            f"Edition '{lang}' already exists in {ltm_path}; pass replace=True to overwrite it."
        )

    structure = helper.get_ltm_structure(ltm_path)  # ordered by paragraph
    split_lang = _resolve_split_langcode(lang)
    raw, stanzas = _read_body_with_verse_stanzas(marked_path)

    body_idx_of = [
        i for i, ln in enumerate(raw) if _detect_meta_mark(ln) not in META_SIDE_MARKS
    ]
    if len(body_idx_of) != len(structure):
        raise TrivialAlignmentError(
            f"Body line count mismatch adding '{lang}': edition has "
            f"{len(body_idx_of)} body line(s), the stored structure has "
            f"{len(structure)}. The edition must match the source structure exactly."
        )

    new_meta = defaultdict(list)
    new_meta_par = defaultdict(list)
    splitted_rows = []  # (paragraph, sentence, id, text, verse)
    sid = 0
    body_idx = 0

    for idx, ln in enumerate(raw):
        mark = _detect_meta_mark(ln)
        if mark in META_SIDE_MARKS:
            # Render before the next unconsumed body paragraph (par_id 0 at top).
            par_id = structure[body_idx]["paragraph"] - 1 if body_idx < len(structure) else 0
            new_meta[mark].append(get_mark_value(ln, mark))
            new_meta_par[mark].append(max(par_id, 0))
            continue

        srow = structure[body_idx]
        paragraph, kind = srow["paragraph"], srow["kind"]

        if kind == preprocessor.VERSE:
            if mark != preprocessor.VERSE:
                raise TrivialAlignmentError(_ltm_mark_mismatch(lang, paragraph, "verse", mark, ln))
            sid += 1
            splitted_rows.append(
                (paragraph, 1, sid, get_mark_value(ln, preprocessor.VERSE), stanzas[idx])
            )
        elif kind == "text":
            if mark is not None:
                raise TrivialAlignmentError(_ltm_mark_mismatch(lang, paragraph, "text", mark, ln))
            sents = [
                s.strip()
                for s in splitter.split_by_sentences([ln], split_lang, clean_text)
                if s.strip()
            ]
            sc = srow["sentence_count"]
            if sc == 1:
                # The paragraph is stored as a single unit (single sentence, or a
                # merged paragraph) — join whatever this edition split into.
                sents = [" ".join(sents)] if sents else [""]
            elif len(sents) != sc:
                raise TrivialAlignmentError(
                    f"Sentence-count mismatch adding '{lang}' at paragraph {paragraph}: "
                    f"this edition splits into {len(sents)} sentence(s) but the stored "
                    f"structure expects {sc}. Re-punctuate the translation to match, or "
                    f"rebuild the book.\n  {ln[:160]}"
                )
            for k, s in enumerate(sents, start=1):
                sid += 1
                splitted_rows.append((paragraph, k, sid, s, 0))
        else:
            # Structural mark (h1-h5/divider/qtext/qname/image): content -> meta.
            if mark != kind:
                raise TrivialAlignmentError(_ltm_mark_mismatch(lang, paragraph, kind, mark, ln))
            new_meta[kind].append(get_mark_value(ln, kind))
            new_meta_par[kind].append(paragraph)
        body_idx += 1

    # ---- additive write ----
    if os.path.isfile(ltm_path):
        gc.collect()
    now = helper._utc_now_iso()
    replaced = replace and lang in existing
    db = sqlite3.connect(ltm_path)
    try:
        # Replacing an edition preserves its position (ord) and is_source flag so
        # a corrected translation never reorders the editions or demotes a source.
        keep_ord, keep_is_source = None, 0
        if replaced:
            row = db.execute(
                "select ord, is_source from languages where lang=?", (lang,)
            ).fetchone()
            if row:
                keep_ord, keep_is_source = row[0], row[1]
            db.execute("delete from splitted where lang=?", (lang,))
            db.execute("delete from meta where lang=?", (lang,))
            db.execute("delete from files where lang=?", (lang,))
            db.execute("delete from languages where lang=?", (lang,))
        if keep_ord is None:
            keep_ord = (
                db.execute("select coalesce(max(ord), -1) from languages").fetchone()[0] + 1
            )
        db.execute(
            "insert into languages(lang, ord, is_source, added_at) values (?,?,?,?)",
            (lang, keep_ord, keep_is_source, now),
        )
        db.executemany(
            "insert into splitted(lang, paragraph, sentence, id, text, verse) values (?,?,?,?,?,?)",
            [(lang, p, s, i, t, v) for (p, s, i, t, v) in splitted_rows],
        )
        db.executemany(
            "insert into meta(lang, key, val, occurence, par_id) values (?,?,?,?,?)",
            flatten_meta_multi(new_meta, new_meta_par, lang),
        )
        db.execute(
            "insert into files(lang, name, guid, added_at) values (?,?,?,?)",
            (lang, file_name or os.path.basename(marked_path), guid or uuid.uuid4().hex, now),
        )
        db.execute(
            "insert into history(operation, lang, insert_ts, parameters) values (?,?,?,?)",
            (con.OPERATION_ADD_LANGUAGE, lang, now, json.dumps({"replace": replaced})),
        )
        helper.touch_ltm_change_conn(db)
        db.commit()
    finally:
        db.close()

    report = {
        "output": ltm_path,
        "lang": lang,
        "replaced": replaced,
        "paragraphs": len(structure),
        "sentences": len(splitted_rows),
        "status": "ok",
    }
    logging.info(
        "add_language: '%s' (%s, +%s rows) -> %s",
        lang,
        "replaced" if replaced else "added",
        len(splitted_rows),
        ltm_path,
    )
    return report


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
        helper.touch_last_edited_conn(db)


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
        helper.touch_last_edited_conn(db)


def update_proxy_text_from(db_path, proxy_texts, ids=[]):
    """Update proxy text from"""
    update_proxy_text(db_path, proxy_texts, ids, direction="from")


def update_proxy_text_to(db_path, proxy_texts, ids=[]):
    """Update proxy text to"""
    update_proxy_text(db_path, proxy_texts, ids, direction="to")


def handle_marks(lines):
    """Handle markup. Write counters.

    Each emitted row is ``(text, marks)`` where ``marks`` is the 8-tuple
    ``(paragraph, h1, h2, h3, h4, h5, divider, verse)``. ``verse`` is the poetry
    stanza index (0 for prose); a ``%%%%%verse.`` line is stripped to its bare
    text and emitted as a normal content row (one stanza-indexed body unit),
    never lifted into ``meta``. Blank lines are not emitted but advance the
    stanza counter so a stanza break inside a poem survives.
    """
    res = []
    marks_counter = defaultdict(int)
    meta = defaultdict(list)
    meta_par_ids = defaultdict(list)
    marks = (0, 0, 0, 0, 0, 0, 0, 0)
    verse_ending = f"{preprocessor.PARAGRAPH_MARK}{preprocessor.VERSE}."
    stanza = 0
    prev_was_verse = False
    pending_break = False

    for line in lines:
        next_par = False
        line = line.strip()

        # Blank line: drop it, but remember a stanza break inside a verse run.
        if not line:
            if prev_was_verse:
                pending_break = True
            continue

        line, next_par = preprocessor.strip_paragraph_mark(line)

        # Verse line: strip the mark, treat as a one-unit content paragraph and
        # assign its poem stanza index.
        verse_idx = 0
        if line.endswith(verse_ending):
            line = line[: -len(verse_ending)]
            if not prev_was_verse or pending_break:
                stanza += 1
            verse_idx = stanza
            next_par = True
            prev_was_verse = True
            pending_break = False
        else:
            prev_was_verse = False
            pending_break = False

        for mark in preprocessor.MARK_COUNTERS:
            update_mark_counter(marks_counter, line, mark)

        update_meta(meta, line, meta_par_ids, marks_counter[preprocessor.PARAGRAPH])

        if not line.endswith(get_all_extraction_endings()):
            if not line:
                # Skip empty lines that may slip through from splitting
                continue
            marks = (
                marks_counter[preprocessor.PARAGRAPH],
                marks_counter[preprocessor.H1],
                marks_counter[preprocessor.H2],
                marks_counter[preprocessor.H3],
                marks_counter[preprocessor.H4],
                marks_counter[preprocessor.H5],
                marks_counter[preprocessor.DIVIDER],
                verse_idx,
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
    # MARK_META_EXTRACT excludes `verse`: a verse line is emitted as a content
    # row by handle_marks, not diverted into the meta table.
    return tuple([f"{preprocessor.PARAGRAPH_MARK}{m}." for m in preprocessor.MARK_META_EXTRACT])


def update_meta(meta, line, meta_par_ids, par_id):
    for mark in preprocessor.MARK_META_EXTRACT:
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
