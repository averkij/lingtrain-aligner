from collections import defaultdict
from lingtrain_aligner import helper, preprocessor, resolver, aligner
import json
import pathlib
import copy
from operator import itemgetter

H1_MARK = preprocessor.PARAGRAPH_MARK + preprocessor.H1
H2_MARK = preprocessor.PARAGRAPH_MARK + preprocessor.H2
H3_MARK = preprocessor.PARAGRAPH_MARK + preprocessor.H3
H4_MARK = preprocessor.PARAGRAPH_MARK + preprocessor.H4
H5_MARK = preprocessor.PARAGRAPH_MARK + preprocessor.H5
DIVIDER_MARK = preprocessor.PARAGRAPH_MARK + preprocessor.DIVIDER
MARKS = [
    preprocessor.H1,
    preprocessor.H2,
    preprocessor.H3,
    preprocessor.H4,
    preprocessor.H5,
    preprocessor.DIVIDER,
    preprocessor.QUOTE_TEXT,
    preprocessor.QUOTE_NAME,
    preprocessor.IMAGE,
]


def get_aligned_pair_chains(db_path, min_len=2):
    """Get aligned pairs. Can be used on uncomplete alignment to extract pairs."""
    res = []
    seen = set()

    prepared_index, _ = resolver.prepare_index(db_path)
    chains_from, chains_to = resolver.get_good_chains(prepared_index, min_len=min_len)
    doc_index = helper.get_doc_index_original(db_path)
    splitted_from = aligner.get_splitted_from(db_path)
    splitted_to = aligner.get_splitted_to(db_path)

    for chain_from, chain_to in zip(chains_from, chains_to):
        for x, y in zip(chain_from, chain_to):
            item = (x[1], x[2])
            if item in seen:
                continue
            doc_item = doc_index[x[1]][x[2]]

            text_from = " ".join(
                [splitted_from[x - 1] for x in json.loads(doc_item[1])]
            )
            text_to = " ".join([splitted_to[x - 1] for x in json.loads(doc_item[3])])

            res.append([text_from, text_to])
            seen.add(item)

    return res


def is_empty_cells(db_path):
    index = helper.get_flatten_doc_index(db_path)
    for item in index:
        from_ids = json.loads(item[0][1])
        to_ids = json.loads(item[0][3])
        if len(from_ids) == 0 or len(to_ids) == 0:
            return True
    return False


def get_paragraphs(db_path, direction="from", par_amount=0):
    """Read all paragraphs with marks from database"""
    # default direction is 'from'
    if direction != "to":
        direction = "from"

    index = helper.get_flatten_doc_index(db_path)
    page = list(zip(index, range(len(index))))

    data, _, __ = helper.get_doc_items(page, db_path)
    lang_from, lang_to = ["from", "to"]

    # extract paragraph info
    from_ids, to_ids = set(), set()
    sent_counter_dict = defaultdict(int)
    for item in index:
        from_json = json.loads(item[0][1])
        to_json = json.loads(item[0][3])
        from_ids.update(from_json)
        to_ids.update(to_json)
        sent_counter_dict[lang_from] += len(from_json)
        sent_counter_dict[lang_to] += len(to_json)

    splitted_from = helper.get_splitted_from_by_id(db_path, from_ids)
    splitted_to = helper.get_splitted_to_by_id(db_path, to_ids)

    paragraphs_from_dict = helper.get_paragraph_dict(splitted_from)
    paragraphs_to_dict = helper.get_paragraph_dict(splitted_to)

    meta = helper.get_meta_dict(db_path)
    meta_dict, paragraphs_dict = dict(), dict()
    meta_dict[lang_from] = prepare_meta(meta, "from")
    meta_dict[lang_to] = prepare_meta(meta, "to")

    gen_main = get_next_paragraph(
        index, data, paragraphs_from_dict, paragraphs_to_dict, direction
    )

    par_info, count = [], 0
    for par_from, par_to, par_id, _, _ in gen_main:
        par_info.append((par_from, par_to, par_id))
        count += 1
        if par_amount != 0 and count == par_amount:
            break
    par_info_list = list(zip(*par_info))

    paragraphs_from, paragraphs_to, par_ids = (
        par_info_list[0],
        par_info_list[1],
        par_info_list[2],
    )

    paragraphs_dict[lang_from] = paragraphs_from
    paragraphs_dict[lang_to] = paragraphs_to

    meta_info = {"items": meta_dict, "main_lang_code": direction}

    return paragraphs_dict, par_ids, meta_info, sent_counter_dict


def get_paragraphs_polybook(db_paths, direction="to", par_amount=0):
    """Read all paragraphs with marks from database"""
    # default direction is 'to'
    if direction != "to":
        direction = "from"
    direction_main_lang = "to" if direction == "from" else "from"

    main_language_side = 1 if direction == "to" else 0
    target_language_side = 0 if direction == "to" else 1

    indexes, datas, paragraphs_from_dicts, paragraphs_to_dicts, lang_codes = (
        [],
        [],
        [],
        [],
        [],
    )
    splitted_dict, meta_dict = dict(), dict()
    add_main_language = True
    for db_path in db_paths:
        index = helper.get_flatten_doc_index(db_path)
        indexes.append(index)

        # get ordered data
        page = list(zip(index, range(len(index))))
        data, _, __ = helper.get_doc_items(page, db_path)
        datas.append(data)

        # extract paragraph info
        from_ids, to_ids = set(), set()
        for item in index:
            from_ids.update(json.loads(item[0][1]))
            to_ids.update(json.loads(item[0][3]))
        splitted_from = helper.get_splitted_from_by_id(db_path, from_ids)
        splitted_to = helper.get_splitted_to_by_id(db_path, to_ids)
        paragraphs_from_dict = helper.get_paragraph_dict(splitted_from)
        paragraphs_to_dict = helper.get_paragraph_dict(splitted_to)
        paragraphs_from_dicts.append(paragraphs_from_dict)
        paragraphs_to_dicts.append(paragraphs_to_dict)

        langs = helper.get_lang_codes(db_path)
        lang_codes.append(langs)

        # get metas and splitted dicts
        splitted_target = (
            helper.get_splitted_from(db_path)
            if direction == "to"
            else helper.get_splitted_to(db_path)
        )
        meta = helper.get_meta_dict(db_path)
        meta_dict[langs[target_language_side]] = prepare_meta(meta, direction_main_lang)
        splitted_dict[langs[target_language_side]] = splitted_target
        if add_main_language:
            meta_dict[langs[main_language_side]] = prepare_meta(meta, direction)
            splitted_main = (
                helper.get_splitted_to(db_path)
                if direction == "to"
                else helper.get_splitted_from(db_path)
            )
            splitted_dict[langs[main_language_side]] = splitted_main
            main_lang_code = langs[main_language_side]
            add_main_language = False

    paragraphs_dict = dict()
    sent_mapping_dict = dict()

    gen_0 = get_next_paragraph(
        indexes[0],
        datas[0],
        paragraphs_from_dicts[0],
        paragraphs_to_dicts[0],
        direction,
    )
    par_info = [
        (par_id, par_sent_ids_from, par_sent_ids_to)
        for _, _, par_id, par_sent_ids_from, par_sent_ids_to in gen_0
    ]

    # sentences mapping in pragraphs
    sent_mapping_dict[lang_codes[0][target_language_side]] = {}
    sent_mapping_dict[lang_codes[0][target_language_side]][
        lang_codes[0][target_language_side]
    ] = [(par_id, par_sent_ids_from) for par_id, par_sent_ids_from, _ in par_info]
    sent_mapping_dict[lang_codes[0][target_language_side]][
        lang_codes[0][main_language_side]
    ] = [(par_id, par_sent_ids_to) for par_id, _, par_sent_ids_to in par_info]

    par_ids = [par_id for par_id, _, _ in par_info]
    for a, b, c, d, lang_code in zip(
        indexes[1:],
        datas[1:],
        paragraphs_from_dicts[1:],
        paragraphs_to_dicts[1:],
        lang_codes[1:],
    ):
        gen_curr = get_next_paragraph(a, b, c, d, direction)
        par_info_curr = [
            (par_id, sent_ids_from, sent_ids_to)
            for _, _, par_id, sent_ids_from, sent_ids_to in gen_curr
        ]
        sent_mapping_dict[lang_code[target_language_side]] = {}
        sent_mapping_dict[lang_code[target_language_side]][
            lang_code[target_language_side]
        ] = [
            (par_id, par_sent_ids_from)
            for par_id, par_sent_ids_from, _ in par_info_curr
        ]
        sent_mapping_dict[lang_code[target_language_side]][
            lang_code[main_language_side]
        ] = [(par_id, par_sent_ids_to) for par_id, _, par_sent_ids_to in par_info_curr]
        par_ids = merge_par_ids(par_ids, [par_id for par_id, _, _ in par_info_curr])

    # merge magic
    merge_sentences_mapping(sent_mapping_dict, par_ids)
    merged_mapping = merge_sent_mappings(
        sent_mapping_dict, lang_codes, target_language_side, main_language_side
    )
    aux_mappings = get_auxiliary_mapping_dict(
        sent_mapping_dict, lang_codes, target_language_side, main_language_side
    )
    aligned_mappings = get_aligned_sentence_mappings(
        merged_mapping,
        aux_mappings,
        lang_codes,
        target_language_side,
        main_language_side,
    )

    for lang_code in aligned_mappings:
        count = 0
        paragraphs_dict[lang_code] = [[] for _ in range(len(par_ids))]
        splitted = splitted_dict[lang_code]
        mapping = aligned_mappings[lang_code]
        curr_par = []

        print(len(par_ids), "==", len(mapping))

        for i, par in enumerate(mapping):
            if par_amount == 0 or count < par_amount:
                for sent_ids in par:
                    curr_par.append(" ".join(splitted[id] for id in sent_ids))
                paragraphs_dict[lang_code][i] = curr_par
                curr_par = []
            count += 1

    meta_info = {"items": meta_dict, "main_lang_code": main_lang_code}

    return paragraphs_dict, par_ids, meta_info


def get_aligned_sentence_mappings(
    mapping_merged, aux_mappings, lang_codes, target_language_side, main_language_side
):
    """Get aligned sentence mappings per paragraph"""
    # init
    new_sent_mappings = {}
    for langs in lang_codes:
        new_sent_mappings[langs[target_language_side]] = [
            [] for _ in range(len(mapping_merged))
        ]
    new_sent_mappings[lang_codes[0][main_language_side]] = [
        [] for _ in range(len(mapping_merged))
    ]

    add_main_language = True
    for langs in lang_codes:
        target_lang = langs[target_language_side]
        main_lang = langs[main_language_side]
        for i, par in enumerate(mapping_merged):
            for sent_block_ids in par:
                merge_key, new_sents, new_sents_main_lang = [], [], []
                for id in sent_block_ids:
                    merge_key.append(id)
                    key = tuple(merge_key)
                    if key in aux_mappings[target_lang][main_lang]:
                        target_id = aux_mappings[target_lang][main_lang][key]
                        target_sent = aux_mappings[target_lang][target_lang][target_id]
                        new_sents.extend(target_sent)
                        if add_main_language:
                            new_sents_main_lang.extend(merge_key)
                        merge_key = []

                new_sent_mappings[target_lang][i].append(new_sents)
                if add_main_language:
                    new_sent_mappings[main_lang][i].append(new_sents_main_lang)

        add_main_language = False

    return new_sent_mappings


def get_auxiliary_mapping_dict(
    sent_mapping_dict, lang_codes, target_language_side, main_language_side
):
    """Calculate auxiliary mappings which we will use for back mapping"""
    sent_aux_mapping_dict = {}
    for langs in lang_codes:
        sent_aux_mapping_dict[langs[target_language_side]] = {}
        sent_aux_mapping_dict[langs[target_language_side]][
            langs[target_language_side]
        ] = []
        sent_aux_mapping_dict[langs[target_language_side]][
            langs[main_language_side]
        ] = {}

    for langs in lang_codes:
        counter = 0
        target_lang = langs[target_language_side]
        main_lang = langs[main_language_side]
        for i in range(len(sent_mapping_dict[target_lang][target_lang])):
            for a, b in zip(
                sent_mapping_dict[target_lang][target_lang][i],
                sent_mapping_dict[target_lang][main_lang][i],
            ):
                sent_aux_mapping_dict[target_lang][target_lang].append(a)
                sent_aux_mapping_dict[target_lang][main_lang][(tuple(b))] = counter
                counter += 1
    return sent_aux_mapping_dict


def merge_sent_mappings(
    sent_mapping_dict, lang_codes, target_language_side, main_language_side
):
    """Calculate one merge sentence mapping"""
    res, curr, merged_res = [], [], []
    mapping_dict = copy.deepcopy(sent_mapping_dict)
    main_lang_code = lang_codes[0][main_language_side]

    min_len = min(
        [
            len(mapping_dict[langs[target_language_side]][main_lang_code])
            for langs in lang_codes
        ]
    )
    print("min_len", min_len)

    mapping_merged = mapping_dict[lang_codes[0][target_language_side]][main_lang_code]
    for langs in lang_codes[1:]:
        mapping_curr = mapping_dict[langs[target_language_side]][main_lang_code]
        for i in range(min_len):
            left, right = mapping_merged[i].pop(0), mapping_curr[i].pop(0)

            # print("merge_sub_arrays------------ left, right", left, right)
            # print(mapping_merged[i], mapping_curr[i])

            merge_sub_arrays(
                res,
                curr,
                mapping_merged[i],
                mapping_curr[i],
                left,
                right,
                len(left),
                len(right),
                left,
            )
            merged_res.append(res)
            res, curr = [], []
        mapping_merged, merged_res = merged_res, []
    return mapping_merged


def merge_sentences_mapping(sent_mapping_dict, par_ids):
    """Merge sentences mapping according to the weakest paragraph mapping (par_ids)"""
    for target_lang_code in sent_mapping_dict:
        for lang in sent_mapping_dict[target_lang_code]:
            res = [[] for _ in range(len(par_ids))]

            curr_par_id = 1
            res[0].extend(sent_mapping_dict[target_lang_code][lang][0][1])

            for i, par_id in enumerate(par_ids[1:]):
                while (
                    len(sent_mapping_dict[target_lang_code][lang]) > curr_par_id
                    and sent_mapping_dict[target_lang_code][lang][curr_par_id][0]
                    <= par_id
                ):
                    if (
                        sent_mapping_dict[target_lang_code][lang][curr_par_id][0]
                        < par_id
                    ):
                        res[i + 1].extend(
                            sent_mapping_dict[target_lang_code][lang][curr_par_id][1]
                        )
                    else:
                        res[i + 1].extend(
                            sent_mapping_dict[target_lang_code][lang][curr_par_id][1]
                        )

                    curr_par_id += 1

            # if target_lang_code == "uk" and lang == "ru":
            #     print(res[:100])

            sent_mapping_dict[target_lang_code][lang] = copy.deepcopy(res)


def merge_sub_arrays(res, curr, a, b, left, right, len1, len2, to_append):
    """Merge subarrays"""
    # a =  [[6,  7], [8], [9,  10, 11, 12],[13,  14,15],[16]]
    # b =  [[6],[7], [8], [9],[10, 11, 12,  13],[14,15,  16]]
    # res = [[6, 7], [8], [9, 10, 11, 12, 13, 14, 15, 16]]
    if len1 == len2:
        curr.extend(to_append)
        res.append(curr)
        if len(a) > 0 and len(b) > 0:
            left, right = a.pop(0), b.pop(0)
            merge_sub_arrays(res, [], a, b, left, right, len(left), len(right), left)
    elif len1 > len2:
        curr.extend(right)
        right_next = b.pop(0)
        merge_sub_arrays(
            res,
            curr,
            a,
            b,
            [x for x in left if x not in right],
            right_next,
            len1,
            len2 + len(right_next),
            right_next,
        )
    else:
        curr.extend(left)
        left_next = a.pop(0)
        merge_sub_arrays(
            res,
            curr,
            a,
            b,
            left_next,
            [x for x in right if x not in left],
            len1 + len(left_next),
            len2,
            left_next,
        )


# TODO OPTIMIZE
def merge_par_ids(par_ids1, par_ids2):
    """Merge two integer arrays with 'take minimum' strategy"""
    res = []
    curr_1 = par_ids1.pop(0)
    curr_2 = par_ids2.pop(0)
    while par_ids1 and par_ids2:
        if curr_1 == curr_2:
            res.append(curr_1)
            curr_1 = par_ids1.pop(0)
            curr_2 = par_ids2.pop(0)
        elif curr_1 < curr_2:
            while curr_1 < curr_2:
                curr_1 = par_ids1.pop(0)
        else:
            curr_2 = par_ids2.pop(0)

    if curr_1 == curr_2:
        res.append(curr_1)
    elif curr_1 < curr_2:
        if par_ids1:
            while curr_1 < curr_2:
                curr_1 = par_ids1.pop(0)
        elif par_ids2:
            res.append(par_ids2[-1])
        else:
            res.append(curr_2)
    else:
        if par_ids2:
            curr_2 = par_ids2.pop(0)
        elif par_ids1:
            res.append(par_ids1[-1])
        else:
            res.append(curr_1)

    return res


def get_next_paragraph(
    index, data, paragraphs_from_dict, paragraphs_to_dict, direction="from"
):
    """Next paragraph generator"""
    if direction == "from":
        prev_meta_max = paragraphs_from_dict[json.loads(index[0][0][1])[-1]]
    else:
        prev_meta_max = paragraphs_to_dict[json.loads(index[0][0][3])[-1]]

    prev_paragraph_max = prev_meta_max[0]

    curr_from, curr_to = [data[0]["text_from"]], [data[0]["text_to"]]
    curr_splitted_ids_from, curr_splitted_ids_to = [json.loads(index[0][0][1])], [
        json.loads(index[0][0][3])
    ]

    for item, texts in zip(index[1:], data[1:]):
        fid_min, fid_max = min(json.loads(item[0][1])), max(json.loads(item[0][1]))
        tid_min, tid_max = min(json.loads(item[0][3])), max(json.loads(item[0][3]))

        if direction == "from":
            curr_paragraph_min = paragraphs_from_dict[fid_min][0]
            curr_paragraph_max = paragraphs_from_dict[fid_max][0]
        else:
            curr_paragraph_min = paragraphs_to_dict[tid_min][0]
            curr_paragraph_max = paragraphs_to_dict[tid_max][0]

        if curr_paragraph_min == prev_paragraph_max:
            curr_from.append(texts["text_from"])
            curr_to.append(texts["text_to"])
            curr_splitted_ids_from.append(json.loads(item[0][1]))
            curr_splitted_ids_to.append(json.loads(item[0][3]))

            prev_paragraph_max = curr_paragraph_max

        else:
            yield curr_from, curr_to, prev_paragraph_max, curr_splitted_ids_from, curr_splitted_ids_to

            prev_paragraph_max = curr_paragraph_max
            curr_from, curr_to = [texts["text_from"]], [texts["text_to"]]
            curr_splitted_ids_from, curr_splitted_ids_to = [json.loads(item[0][1])], [
                json.loads(item[0][3])
            ]

    yield curr_from, curr_to, prev_paragraph_max, curr_splitted_ids_from, curr_splitted_ids_to


def sort_meta(metas):
    for lang in metas["items"]:
        for mark in metas["items"][lang]:
            metas["items"][lang][mark].sort(key=lambda x: x[2])


def create_book(
    lang_ordered,
    paragraphs,
    delimeters,
    metas,
    sent_counter,
    output_path,
    template,
    styles=[],
    highlight="through",
):
    """Generate html"""
    # ensure path is existed
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if template in STYLES:
        css = generate_css(STYLES[template])
        sent_cycle = len(STYLES[template])
    elif template == "custom" and styles:
        css = generate_css(styles)
        sent_cycle = len(styles)
    else:
        css = generate_css([])
        sent_cycle = 2

    meta = sort_meta(metas)

    min_par_len = min([len(paragraphs[x]) for x in paragraphs])
    min_par_len = min(min_par_len, len(delimeters))

    header_text = ""
    for i, lang in enumerate(lang_ordered):
        header_text += f"{sent_counter[lang]} sent. [{lang}] "
        if i == 0:
            header_text += "• "

    with open(output_path, "w", encoding="utf8") as res_html:
        # --------------------HEAD
        res_html.write(
            f"""
<html><head>
    <link rel="preconnect" href="https://fonts.gstatic.com">
    <link href="https://fonts.googleapis.com/css2?family=Noto+Serif:wght@400&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Oswald:wght@300;400&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Josefin+Sans&display=swap" rel="stylesheet">
    <title>Lingtrain Magic Book</title>
    <meta charset="UTF-8">
    {css}
</head>
<body>
<div class="lt-header">🚀 lingtrain parallel book 🡒 {header_text} 🡒 {min_par_len} paragpaphs</div>"""
        )

        # --------------------BOOK
        res_html.write("<div class='dt cont'>")

        # --------------------DIVIDER
        res_html.write("<div class='dt-row header'>")
        for _ in range(len(lang_ordered)):
            res_html.write(
                f"<div class='dt-cell divider'><img class='divider-img' src='{DIVIDER_URL}'/></div>"
            )
        res_html.write("</div>")

        # --------------------TITLE and AUTHOR
        res_html.write("<div class='dt-row header title-cell'>")
        for lang in lang_ordered:
            res_html.write("<div class='title-cell dt-cell'>")
            meta = metas["items"][lang]
            title = get_meta(meta, preprocessor.TITLE)
            if title:
                res_html.write("<h1 class='lt-title'>" + title + "</h1>")
            res_html.write("</div>")
        res_html.write("</div>")
        res_html.write("<div class='dt-row header'>")
        for lang in lang_ordered:
            res_html.write("<div class='author-cell dt-cell'>")
            meta = metas["items"][lang]
            author = get_meta(meta, preprocessor.AUTHOR)
            if author:
                res_html.write("<h1 class='lt-author'>" + author + "</h1>")
            res_html.write("</div>")
        res_html.write("</div>")

        next_mark, next_meta_par_id = get_next_meta_par_id(metas)

        j = 0
        for actual_paragraphs_id in range(min_par_len):
            real_par_id = delimeters[actual_paragraphs_id]

            while next_meta_par_id <= real_par_id:
                _ = write_next_polyheader(res_html, next_mark, metas, lang_ordered)
                next_mark, next_meta_par_id = get_next_meta_par_id(metas)

            res_html.write("<div class='dt-row'>")
            for lang in lang_ordered:
                res_html.write(
                    f"<div class='par dt-cell'><div class='book-par-id'>{real_par_id + 1}</div>"
                )
                for k, sent in enumerate(paragraphs[lang][actual_paragraphs_id]):
                    sent_cycle_index = (
                        (j + k) % sent_cycle
                        if highlight == "through"
                        else k % sent_cycle
                    )
                    res_html.write(
                        f"<span class='s s{sent_cycle_index%sent_cycle}'>{sent}</span>"
                    )
                res_html.write("</div>")

            j += len(paragraphs[lang][actual_paragraphs_id])
            res_html.write("</div>")

        while next_mark:
            _ = write_next_polyheader(res_html, next_mark, metas, lang_ordered)
            next_mark, next_meta_par_id = get_next_meta_par_id(metas)

        # --------------------END BOOK
        res_html.write(f"</div>{HTML_FOOTER}")
        res_html.write("</body></html>")


def create_polybook(
    lang_ordered,
    paragraphs,
    delimeters,
    metas,
    output_path,
    template,
    styles=[],
    highlight="through",
):
    """Generate multilingual html"""
    # ensure path is existed
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    langs_count = len(lang_ordered)

    if template in STYLES:
        css = generate_css(STYLES[template], cols=langs_count)
        sent_cycle = len(STYLES[template])
    elif template == "custom" and styles:
        css = generate_css(styles, cols=langs_count)
        sent_cycle = len(styles)
    else:
        css = generate_css([], cols=langs_count)
        sent_cycle = 2

    meta = sort_meta(metas)

    with open(output_path, "w", encoding="utf8") as res_html:
        # --------------------HEAD
        res_html.write(
            f"""
<html><head>
    <meta charset="utf-8">
    <link rel="preconnect" href="https://fonts.gstatic.com">
    <link href="https://fonts.googleapis.com/css2?family=Noto+Serif:wght@400&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Oswald:wght@300;400&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Josefin+Sans&display=swap" rel="stylesheet">
    <title>Lingtrain Magic Book</title>
    <meta charset="UTF-8">
    {css}
</head>
<body>
<div class="lt-header">🚀 created in lingtrain alignment studio</div>"""
        )

        # --------------------BOOK
        res_html.write("<div class='dt cont'>")

        # --------------------DIVIDER
        res_html.write("<div class='dt-row header'>")
        for _ in range(langs_count):
            res_html.write(
                f"<div class='dt-cell divider'><img class='divider-img' src='{DIVIDER_URL}'/></div>"
            )
        res_html.write("</div>")

        # --------------------TITLE and AUTHOR
        res_html.write("<div class='dt-row header title-cell'>")
        for lang in lang_ordered:
            res_html.write("<div class='title-cell dt-cell'>")
            meta = metas["items"][lang]
            title = get_meta(meta, preprocessor.TITLE)
            if title:
                res_html.write("<h1 class='lt-title'>" + title + "</h1>")
            res_html.write("</div>")
        res_html.write("</div>")
        res_html.write("<div class='dt-row header'>")
        for lang in lang_ordered:
            res_html.write("<div class='author-cell dt-cell'>")
            meta = metas["items"][lang]
            author = get_meta(meta, preprocessor.AUTHOR)
            if author:
                res_html.write("<h1 class='lt-author'>" + author + "</h1>")
            res_html.write("</div>")
        res_html.write("</div>")

        # --------------------PARAGRAPHS
        next_mark, next_meta_par_id = get_next_meta_par_id(metas)
        min_par_len = min([len(paragraphs[x]) for x in paragraphs])

        j = 0
        for actual_paragraphs_id in range(min_par_len):
            real_par_id = delimeters[actual_paragraphs_id]

            while next_meta_par_id <= real_par_id:
                _ = write_next_polyheader(res_html, next_mark, metas, lang_ordered)
                next_mark, next_meta_par_id = get_next_meta_par_id(metas)

            res_html.write("<div class='dt-row'>")
            for lang in lang_ordered:
                res_html.write(f"<div class='par dt-cell'>  ")

                for k, sent in enumerate(paragraphs[lang][actual_paragraphs_id]):
                    sent_cycle_index = (
                        (j + k) % sent_cycle
                        if highlight == "through"
                        else k % sent_cycle
                    )
                    res_html.write(
                        f"<span class='s s{sent_cycle_index%sent_cycle}'>{sent}</span>"
                    )
                res_html.write("</div>")

            j += len(paragraphs[lang][actual_paragraphs_id])
            res_html.write("</div>")

        while next_mark:
            _ = write_next_polyheader(res_html, next_mark, metas, lang_ordered)
            next_mark, next_meta_par_id = get_next_meta_par_id(metas)

        # --------------------END BOOK
        res_html.write(f"</div>{HTML_FOOTER}")
        res_html.write("</body></html>")


def create_polybook_preview(
    lang_ordered,
    paragraphs,
    delimeters,
    metas,
    template,
    styles=[],
    par_amount=0,
    highlight="through",
):
    """Generate multiligual html preview"""
    langs_count = len(lang_ordered)

    if template in STYLES:
        css = generate_css(STYLES[template], cols=langs_count)
        sent_cycle = len(STYLES[template])
    elif template == "custom" and styles:
        css = generate_css(styles, cols=langs_count)
        sent_cycle = len(styles)
    else:
        css = generate_css([], cols=langs_count)
        template = "simple"
        sent_cycle = 4

    meta = sort_meta(metas)

    res_html = ""
    # --------------------BOOK
    res_html += "<div class='dt cont'>"

    # --------------------DIVIDER
    res_html += "<div class='dt-row header'>"
    for _ in range(langs_count):
        res_html += f"<div class='dt-cell divider'><img class='divider-img' src='{DIVIDER_URL}'/></div>"
    res_html += "</div>"

    # --------------------TITLE and AUTHOR
    res_html += "<div class='dt-row header title-cell'>"
    for lang in lang_ordered:
        res_html += "<div class='title-cell dt-cell'>"
        meta = metas["items"][lang]
        title = get_meta(meta, preprocessor.TITLE)
        if title:
            res_html += "<h1 class='lt-title'>" + title + "</h1>"
        res_html += "</div>"
    res_html += "</div>"
    res_html += "<div class='dt-row header'>"
    for lang in lang_ordered:
        res_html += "<div class='author-cell dt-cell'>"
        meta = metas["items"][lang]
        author = get_meta(meta, preprocessor.AUTHOR)
        if author:
            res_html += "<h1 class='lt-author'>" + author + "</h1>"
        res_html += "</div>"
    res_html += "</div>"

    # --------------------PARAGRAPHS
    next_mark, next_meta_par_id = get_next_meta_par_id(metas)
    min_par_len = min([len(paragraphs[x]) for x in paragraphs])
    if par_amount > 0:
        min_par_len = min(min_par_len, par_amount)

    j = 0
    for actual_paragraphs_id in range(min_par_len):
        real_par_id = delimeters[actual_paragraphs_id]

        while next_meta_par_id <= real_par_id:
            res_html = write_next_polyheader(
                res_html, next_mark, metas, lang_ordered, add_string=True
            )
            next_mark, next_meta_par_id = get_next_meta_par_id(metas)

        res_html += "<div class='dt-row'>"
        for lang in lang_ordered:
            res_html += f"<div class='par dt-cell'><div class='book-par-id'>{real_par_id + 1}</div>"

            for k, sent in enumerate(paragraphs[lang][actual_paragraphs_id]):
                sent_cycle_index = (
                    (j + k) % sent_cycle if highlight == "through" else k % sent_cycle
                )
                res_html += (
                    f"<span class='s s{sent_cycle_index} {template}'>{sent}</span>"
                )

            res_html += "</div>"
        j += len(paragraphs[lang][actual_paragraphs_id])
        res_html += "</div>"

    # --------------------END BOOK
    res_html += "</div>"
    return res_html


def get_next_meta_par_id(metas):
    lang_code = metas["main_lang_code"]
    meta = metas["items"][lang_code]
    min_par_id = float("Inf")
    next_mark = ""

    for mark in MARKS:
        if mark in meta and meta[mark]:
            gen = (x for x in meta[mark])
            min_par_mark = min(gen, key=itemgetter(2))
            if min_par_mark[2] < min_par_id:
                next_mark = mark
                min_par_id = min_par_mark[2]
    return next_mark, min_par_id


def write_next_polyheader(
    writer, next_mark, metas_dict, lang_ordered, add_string=False
):
    """Write next header to string or stream"""
    metas = metas_dict["items"]
    main_lang_code = metas_dict["main_lang_code"]
    if add_string:
        writer += "<div class='dt-row header'>"
    else:
        writer.write("<div class='dt-row header'>")
    for lang in lang_ordered:
        # divider is a special case
        if next_mark == preprocessor.DIVIDER:
            el = f"<div class='dt-cell divider'><img class='divider-img' src='{DIVIDER_URL}'/></div>"
            if next_mark in metas[lang] and metas[lang][next_mark]:
                metas[lang][next_mark].pop(0)
            if add_string:
                writer += el
            else:
                writer.write(el)
            continue

        meta = metas[lang][next_mark]
        if meta:
            val = meta.pop(0)
        else:
            val = ("",)
        if next_mark == preprocessor.QUOTE_TEXT:
            el = f"<div class='par dt-cell'><div class='lt-quote lt-quote-text'>{val[0]}</div></div>"
        elif next_mark == preprocessor.QUOTE_NAME:
            el = f"<div class='par dt-cell'><div class='lt-quote lt-quote-name'>{val[0]}</div></div>"
        elif next_mark == preprocessor.IMAGE:
            el = f"<div class='par dt-cell text-center'><div class='img-cont'><div class='lt-image-mask'></div><img class='lt-image' src='{val[0]}'/></div></div>"
        else:
            el = f"<div class='par dt-cell'><{HEADER_HTML_MAPPING[next_mark]}>{val[0]}</{HEADER_HTML_MAPPING[next_mark]}></div>"

        if add_string:
            writer += el
        else:
            writer.write(el)

    if add_string:
        writer += "</div>"
    else:
        writer.write("</div>")

    # pop main lang if not yet popped
    if main_lang_code not in lang_ordered:
        meta = metas[main_lang_code][next_mark]
        if meta:
            meta.pop(0)
    return writer


def prepare_meta(meta, direction):
    """Take all meta for an exact direction"""
    res = dict()
    for key in meta:
        if key.endswith(direction):
            # trim direction, leave mark only
            res[key[: len(key) - len(direction) - 1]] = meta[key]
    return res


def get_meta(meta, mark, occurence=0):
    """Get prepared meta value"""
    if mark in meta and len(meta[mark]) > occurence:
        return meta[mark][occurence][0]
    return ""


def get_meta_from(meta, mark, occurence=0):
    """Get meta value from"""
    key = f"{mark}_from"
    if len(meta[key]) > occurence:
        return meta[key][occurence]
    return ""


def get_meta_to(meta, mark, occurence=0):
    """Get meta value to"""
    key = f"{mark}_to"
    if len(meta[key]) > occurence:
        return meta[key][occurence]
    return ""


def generate_css(styles, cols=2):
    special = ""
    for i, s in enumerate(styles):
        style = json.loads(s)
        special += f".s{i} {{\n"
        for rule in style:
            special += f"{rule}: {style[rule]};"
        special += "\n}"
    col_width = f"{100 // cols}%"

    res = (
        CSS_TEMPLATE.replace("%GENERAL%", CSS_GENERAL)
        .replace("%SPECIAL%", special)
        .replace("%COL_WIDTH%", col_width)
    )
    return res


HEADER_HTML_MAPPING = {
    preprocessor.H1: "h2",
    preprocessor.H2: "h3",
    preprocessor.H3: "h3",
    preprocessor.H4: "h4",
    preprocessor.H5: "h4",
}

# DIVIDER_URL = "https://habrastorage.org/webt/nr/av/qa/nravqa-wy0sg8kgwr3cfli8veym.png"
# DIVIDER_URL = "https://habrastorage.org/webt/q9/1t/cy/q91tcypgjnrrsmrfcviyzzdvfsk.png"
DIVIDER_URL = "https://habrastorage.org/webt/28/2n/1b/282n1b8oxclxp-jacqfc3sytbqm.png"


HTML_FOOTER = """
<div class="lt-footer">🚀 created in lingtrain alignment studio</div>"""

CSS_TEMPLATE = """<style type="text/css">
%GENERAL%
%SPECIAL%
</style>"""

CSS_GENERAL = """
.par {
    font-size: 20px;
    font-family: 'Noto Serif', serif;
    padding: 15px 60px;
    padding: 15px 20px;
    text-indent: 30px;
    font-weight: 400;
}

@media print {
    .par {
        font-size: 12px;
        font-family: 'Noto Serif', serif;
        padding: 15px 10px;
        text-indent: 30px;
        font-weight: 400;
    }
}

h1, h2, h3, h4, h5 {
    font-family: 'Oswald', cursive;
}

h1 {
    font-weight: 400;
}

h2 {
    text-transform: uppercase;
    font-weight: 400;
}

h3 {
    font-weight: 400;
}

.cont {
    margin: 0 150px;
    # background: #fcfcfc;
    # border: 1px solid #efefef;
    # border-radius: 10px;
    # overflow: hidden;
}

@media print {
    .cont {
        margin: 0 0px;
    }
}

.dt {
    display: table;
}
.dt-row {
    display: table-row;
    position: relative;
}

# .dt-row:nth-child(even):not(.header) {
#     background: #fefefe;
# }

.dt-row:nth-child(even):not(.header) > .dt-cell:not(.divider) {
    border-top: 1px solid #f0f0f0;
    border-bottom: 1px solid #f0f0f0;
}

.dt-row:hover > .dt-cell > .book-par-id {
    opacity: 1;
}

.book-par-id {
    position: absolute;
    left: -20px;
    top: 0;
    font-size: 12px;
    opacity: 0;
    font-weight: 600;
}

.dt-cell {
    display: table-cell;
    width: %COL_WIDTH% !important;
    word-break: break-word;
}

.dt-cell-colspan {
    display: table-caption;
}

.divider {
    padding-top: 30px;
    padding-bottom: 30px;
    vertical-align:middle;
    text-align:center;
}

.divider-img {
    # width: 50px;
    # height: 50px;
}

.flag-img {
    height: 50px;
    width: 50px;
    padding: 0 10px 0 0;
}

.inline {
    display: inline-block;
}

.s {
    padding: 0 6px;
}

.text-center {
    text-align: center;
}

.lt-quote-text {
    font-style: italic;
    text-align: right;
    font-size: 0.9em;
}
.lt-quote-name {
    text-align: right;
    font-size: 0.8em;
}
.lt-image {
    max-height: 300px;
    max-width: 800px;
    -webkit-mask-image: url(https://hsto.org/webt/wg/wv/ai/wgwvai84o5fvgpok4mnannr7jca.png);
    mask-image: url(https://hsto.org/webt/wg/wv/ai/wgwvai84o5fvgpok4mnannr7jca.png);
    -webkit-mask-repeat: no-repeat;
    mask-repeat: no-repeat;
    -webkit-mask-size: contain;
    mask-size: contain;
    -webkit-mask-position: center;
    mask-position: center;
    margin: 20px 0;
}
.lt-title {
    font-size: 52px;
    text-align: center;
}
.lt-author {
    margin-top: -10px;
    font-size: 40px;
    text-align: center;
}

.title-cell {
    padding: 20px 20px 0px 20px;
}

.author-cell {
    padding: 0px 0px 30px 0px;
}

@media print {
    .lt-title {
        font-size: 32px;
    }
    .lt-author {
        font-size: 28px;
    }
    .lt-image {
        max-height: 260px;
        max-width: 400px;
    }
    .divider-img {
        height: 30px;
        width: 30px;
    }
}

.lt-footer {
    padding: 20px 0;
    margin-top: 30px;
    text-align:center;
    font-family: 'Josefin Sans', sans-serif;
    text-transform: capitalize;
    color: white;
    background: cornflowerblue;
}

.lt-header {
    padding: 20px 0;
    margin-bottom: 30px;
    text-align:center;
    font-family: 'Josefin Sans', sans-serif;
    text-transform: capitalize;
    color: white;
    background: cornflowerblue;
}
"""


STYLES = {
    "pastel_fill": [
        '{"background": "#cfefd7", "color": "black"}',
        '{"background": "#fadce2", "color": "black"}',
        '{"background": "#cce7eb", "color": "black"}',
        '{"background": "#fefbd6", "color": "black"}',
    ],
    "pastel_start": [
        '{"background": "linear-gradient(90deg, #cfefd7 0px, #ffffff00 150px)", "border-radius": "15px"}',
        '{"background": "linear-gradient(90deg, #fadce2 0px, #ffffff00 150px)", "border-radius": "15px"}',
        '{"background": "linear-gradient(90deg, #cce7eb 0px, #ffffff00 150px)", "border-radius": "15px"}',
        '{"background": "linear-gradient(90deg, #fefbd6 0px, #ffffff00 150px)", "border-radius": "15px"}',
    ],
}
