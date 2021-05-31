from lingtrain_aligner import helper, preprocessor
import json
import pathlib
import copy

H1_MARK = preprocessor.PARAGRAPH_MARK + preprocessor.H1
H2_MARK = preprocessor.PARAGRAPH_MARK + preprocessor.H2
H3_MARK = preprocessor.PARAGRAPH_MARK + preprocessor.H3
H4_MARK = preprocessor.PARAGRAPH_MARK + preprocessor.H4
H5_MARK = preprocessor.PARAGRAPH_MARK + preprocessor.H5
DIVIDER_MARK = preprocessor.PARAGRAPH_MARK + preprocessor.DIVIDER
MARKS = [preprocessor.H1, preprocessor.H2, preprocessor.H3, preprocessor.H4, preprocessor.H5, preprocessor.DIVIDER]

def get_paragraphs(db_path, direction="from"):
    """Read all paragraphs with marks from database"""
    #default direction is 'from'
    if direction != "to": direction = "from"

    index = helper.get_flatten_doc_index(db_path)
    page = list(zip(index, range(len(index))))

    data, _, __ = helper.get_doc_items(page, db_path)

    # extract paragraph info
    from_ids, to_ids = set(), set()
    for item in index:
        from_ids.update(json.loads(item[0][1]))
        to_ids.update(json.loads(item[0][3]))

    splitted_from = helper.get_splitted_from_by_id(db_path, from_ids)
    splitted_to = helper.get_splitted_to_by_id(db_path, to_ids)

    paragraphs_from_dict = helper.get_paragraph_dict(splitted_from)
    paragraphs_to_dict = helper.get_paragraph_dict(splitted_to)

    meta = helper.get_meta_dict(db_path)

    paragraphs_from, paragraphs_to = [], []
    if direction == "from":
        prev_meta = paragraphs_from_dict[json.loads(index[0][0][1])[0]]
    else:
        prev_meta = paragraphs_to_dict[json.loads(index[0][0][3])[0]]

    prev_paragraph = prev_meta[0]
    prev_h1, prev_h2, prev_h3, prev_h4, prev_h5, prev_di = prev_meta[1], prev_meta[2], prev_meta[3], prev_meta[4], prev_meta[5], prev_meta[6]

    curr_from, curr_to = [data[0]["text_from"]], [data[0]["text_to"]]

    for item, texts in zip(index[1:], data[1:]):
        fid = max(json.loads(item[0][1]))
        tid = max(json.loads(item[0][3]))

        if direction == "from":
            curr_paragraph = paragraphs_from_dict[fid][0]
        else:
            curr_paragraph = paragraphs_to_dict[tid][0]

        # print("item", item)
        # print("fid", fid)
        # print("tid", tid)
        # print("prev_paragraph", prev_paragraph)
        # print("curr_paragraph", curr_paragraph)

        curr_h1 = paragraphs_from_dict[fid][1] if direction == "from" else paragraphs_to_dict[tid][1]
        curr_h2 = paragraphs_from_dict[fid][2] if direction == "from"  else paragraphs_to_dict[tid][2]
        curr_h3 = paragraphs_from_dict[fid][3] if direction == "from"  else paragraphs_to_dict[tid][3]
        curr_h4 = paragraphs_from_dict[fid][4] if direction == "from"  else paragraphs_to_dict[tid][4]
        curr_h5 = paragraphs_from_dict[fid][5] if direction == "from"  else paragraphs_to_dict[tid][5]
        curr_di = paragraphs_from_dict[fid][6] if direction == "from"  else paragraphs_to_dict[tid][6]

        if curr_paragraph == prev_paragraph:
            curr_from.append(texts["text_from"])
            curr_to.append(texts["text_to"])
        else:
            paragraphs_from.append(curr_from)
            paragraphs_to.append(curr_to)

            prev_paragraph = curr_paragraph
            curr_from, curr_to = [texts["text_from"]], [texts["text_to"]]

        if curr_h1 != prev_h1:
            paragraphs_from.append(H1_MARK)
            paragraphs_to.append(curr_h1-1)
            prev_h1 = curr_h1

        if curr_h2 != prev_h2:
            paragraphs_from.append(H2_MARK)
            paragraphs_to.append(curr_h2-1)
            prev_h2 = curr_h2

        if curr_h3 != prev_h3:
            paragraphs_from.append(H3_MARK)
            paragraphs_to.append(curr_h3-1)
            prev_h3 = curr_h3

        if curr_h4 != prev_h4:
            paragraphs_from.append(H4_MARK)
            paragraphs_to.append(curr_h4-1)
            prev_h4 = curr_h4

        if curr_h5 != prev_h5:
            paragraphs_from.append(H5_MARK)
            paragraphs_to.append(curr_h5-1)
            prev_h5 = curr_h5

        if curr_di != prev_di:
            paragraphs_from.append(DIVIDER_MARK)
            paragraphs_to.append(curr_di-1)
            prev_di = curr_di

    paragraphs_from.append(curr_from)
    paragraphs_to.append(curr_to)

    return paragraphs_from, paragraphs_to, meta


def get_paragraphs_polybook(db_paths, direction="to"):
    """Read all paragraphs with marks from database"""
    #default direction is 'to'
    if direction != "to": direction = "from"
    direction_main_lang = "to" if direction == "from" else "from"

    main_language_side = 1 if direction == "to" else 0
    target_language_side = 0 if direction == "to" else 1

    indexes, datas, paragraphs_from_dicts, paragraphs_to_dicts, lang_codes = [], [], [], [], []
    splitted_dict, meta_dict = dict(), dict()
    add_main_language = True
    for db_path in db_paths:
        index = helper.get_flatten_doc_index(db_path)
        indexes.append(index)

        #get ordered data
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

        #get metas and splitted dicts
        splitted_target = helper.get_splitted_from(db_path) if direction == "to" else helper.get_splitted_to(db_path)
        meta = helper.get_meta_dict(db_path)
        meta_dict[langs[target_language_side]] = prepare_meta(meta, direction_main_lang)
        splitted_dict[langs[target_language_side]] = splitted_target
        if add_main_language:
            meta_dict[langs[main_language_side]] = prepare_meta(meta, direction)
            splitted_main =  helper.get_splitted_to(db_path) if direction == "to" else helper.get_splitted_from(db_path)
            splitted_dict[langs[main_language_side]] = splitted_main
            main_lang_code = langs[main_language_side]
            add_main_language = False

    paragraphs_dict = dict()  
    sent_mapping_dict = dict()    

    gen_0 = get_next_paragraph(indexes[0], datas[0], paragraphs_from_dicts[0], paragraphs_to_dicts[0], direction)
    par_info = [(par_id, par_sent_ids_from, par_sent_ids_to) for _, _, par_id, par_sent_ids_from, par_sent_ids_to in gen_0]

    #sentences mapping in pragraphs
    sent_mapping_dict[lang_codes[0][target_language_side]] = {}
    sent_mapping_dict[lang_codes[0][target_language_side]][lang_codes[0][target_language_side]] = [(par_id, par_sent_ids_from) for par_id, par_sent_ids_from, _ in par_info]
    sent_mapping_dict[lang_codes[0][target_language_side]][lang_codes[0][main_language_side]] = [(par_id, par_sent_ids_to) for par_id, _, par_sent_ids_to in par_info]

    par_ids = [par_id for par_id, _, _ in par_info]
    for a,b,c,d, lang_code in zip(indexes[1:], datas[1:], paragraphs_from_dicts[1:], paragraphs_to_dicts[1:], lang_codes[1:]):
        gen_curr = get_next_paragraph(a,b,c,d,direction)
        par_info_curr = [(par_id, sent_ids_from, sent_ids_to) for _, _, par_id, sent_ids_from, sent_ids_to in gen_curr]
        sent_mapping_dict[lang_code[target_language_side]] = {}
        sent_mapping_dict[lang_code[target_language_side]][lang_code[target_language_side]] = [(par_id, par_sent_ids_from) for par_id, par_sent_ids_from, _ in par_info_curr]
        sent_mapping_dict[lang_code[target_language_side]][lang_code[main_language_side]] = [(par_id, par_sent_ids_to) for par_id, _, par_sent_ids_to in par_info_curr]
        par_ids = merge_par_ids(par_ids, [par_id for par_id, _, _ in par_info_curr])

    #merge magic
    merge_sentences_mapping(sent_mapping_dict, par_ids)
    merged_mapping = merge_sent_mappings(sent_mapping_dict, lang_codes, target_language_side, main_language_side)
    aux_mappings = get_auxiliary_mapping_dict(sent_mapping_dict, lang_codes, target_language_side, main_language_side)
    aligned_mappings = get_aligned_sentence_mappings(merged_mapping, aux_mappings, lang_codes, target_language_side, main_language_side)

    for lang_code in aligned_mappings:
        paragraphs_dict[lang_code] = [[] for _ in range(len(par_ids))]
        splitted = splitted_dict[lang_code]
        mapping = aligned_mappings[lang_code]
        curr_par = []

        print(len(par_ids), "==",  len(mapping))

        for i,par in enumerate(mapping):
            for sent_ids in par:
                curr_par.append(" ".join(splitted[id] for id in sent_ids))    
            paragraphs_dict[lang_code][i] = curr_par
            curr_par = []

    meta_info = { "items": meta_dict, "main_lang_code": main_lang_code}

    return paragraphs_dict, par_ids, meta_info


def get_aligned_sentence_mappings(mapping_merged, aux_mappings, lang_codes, target_language_side, main_language_side):
    """Get aligned sentence mappings per paragraph"""
    #init
    new_sent_mappings = {}
    for langs in lang_codes:
        new_sent_mappings[langs[target_language_side]] = [[] for _ in range(len(mapping_merged))]
    new_sent_mappings[lang_codes[0][main_language_side]] = [[] for _ in range(len(mapping_merged))]

    add_main_language = True
    for langs in lang_codes:
        target_lang = langs[target_language_side]
        main_lang = langs[main_language_side]
        for i,par in enumerate(mapping_merged):
            for sent_block_ids in par:
                merge_key, new_sents, new_sents_main_lang = [],[],[]
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


def get_auxiliary_mapping_dict(sent_mapping_dict, lang_codes, target_language_side, main_language_side):
    """Calculate auxiliary mappings which we will use for back mapping"""
    sent_aux_mapping_dict = {}
    for langs in lang_codes:
        sent_aux_mapping_dict[langs[target_language_side]] = {}
        sent_aux_mapping_dict[langs[target_language_side]][langs[target_language_side]] = []
        sent_aux_mapping_dict[langs[target_language_side]][langs[main_language_side]] = {}

    for langs in lang_codes:
        counter = 0
        target_lang = langs[target_language_side]
        main_lang = langs[main_language_side]
        for i in range(len(sent_mapping_dict[target_lang][target_lang])):
            for a, b in zip(sent_mapping_dict[target_lang][target_lang][i], sent_mapping_dict[target_lang][main_lang][i]):
                sent_aux_mapping_dict[target_lang][target_lang].append(a)
                sent_aux_mapping_dict[target_lang][main_lang][(tuple(b))] = counter
                counter += 1
    return sent_aux_mapping_dict


def merge_sent_mappings(sent_mapping_dict, lang_codes, target_language_side, main_language_side):
    """Calculate one merge sentence mapping"""
    res, curr, merged_res = [], [], []
    mapping_dict = copy.deepcopy(sent_mapping_dict)
    main_lang_code = lang_codes[0][main_language_side]

    min_len = min([len(mapping_dict[langs[target_language_side]][main_lang_code]) for langs in lang_codes])
    print("min_len", min_len)

    mapping_merged = mapping_dict[lang_codes[0][target_language_side]][main_lang_code]
    for langs in lang_codes[1:]:
        mapping_curr = mapping_dict[langs[target_language_side]][main_lang_code]
        for i in range(min_len):
            left, right = mapping_merged[i].pop(0), mapping_curr[i].pop(0)

            # print("merge_sub_arrays------------ left, right", left, right)
            # print(mapping_merged[i], mapping_curr[i])
            

            merge_sub_arrays(res, curr, mapping_merged[i], mapping_curr[i], left, right, len(left), len(right), left)
            merged_res.append(res)
            res, curr = [], []
        mapping_merged, merged_res = merged_res, []
    return mapping_merged


def merge_sentences_mapping(sent_mapping_dict, par_ids):
    """Merge sentences mapping according to the weakest paragraph mapping (par_ids)"""

    # print("par_ids", par_ids[:100])
    # print("len(par_ids)", len(par_ids))

    # return

    for target_lang_code in sent_mapping_dict:
        for lang in sent_mapping_dict[target_lang_code]:
            res = [[] for _ in range(len(par_ids))]
            
            curr_par_id = 1
            res[0].extend(sent_mapping_dict[target_lang_code][lang][0][1])

            # len(sent_mapping_dict[target_lang_code][lang]) > curr_par_id and 

            for i, par_id in enumerate(par_ids[1:]):
                while len(sent_mapping_dict[target_lang_code][lang]) > curr_par_id and sent_mapping_dict[target_lang_code][lang][curr_par_id][0] <= par_id:

                    if sent_mapping_dict[target_lang_code][lang][curr_par_id][0] < par_id:
                        res[i+1].extend(sent_mapping_dict[target_lang_code][lang][curr_par_id][1])
                    else:
                        res[i+1].extend(sent_mapping_dict[target_lang_code][lang][curr_par_id][1])
                    

                    # if target_lang_code == "uk" and curr_par_id >=47 and curr_par_id < 49:
                    #     print("par_id", par_id)
                    #     print("curr_par_id", sent_mapping_dict[target_lang_code][lang][curr_par_id][0])
                    #     print("res", res[46:49])

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
        if len(a) > 0 and len(b)>0:
            left, right = a.pop(0), b.pop(0)
            merge_sub_arrays(res, [], a, b, left, right, len(left), len(right), left)
    elif len1 > len2:
        curr.extend(right)
        right_next = b.pop(0)
        merge_sub_arrays(res, curr, a, b, [x for x in left if x not in right], right_next, len1, len2 + len(right_next), right_next)
    else:
        curr.extend(left)
        left_next = a.pop(0)
        merge_sub_arrays(res, curr, a, b, left_next, [x for x in right if x not in left], len1 + len(left_next), len2, left_next)


#TODO OPTIMIZE
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


def get_next_paragraph(index, data, paragraphs_from_dict, paragraphs_to_dict, direction="from"):
    """Next paragraph generator"""
    if direction == "from":
        prev_meta_max = paragraphs_from_dict[json.loads(index[0][0][1])[-1]]
    else:
        prev_meta_max = paragraphs_to_dict[json.loads(index[0][0][3])[-1]]

    prev_paragraph_max = prev_meta_max[0]

    # prev_h1, prev_h2, prev_h3, prev_h4, prev_h5, prev_di = prev_meta[1], prev_meta[2], prev_meta[3], prev_meta[4], prev_meta[5], prev_meta[6]

    curr_from, curr_to = [data[0]["text_from"]], [data[0]["text_to"]]
    curr_splitted_ids_from, curr_splitted_ids_to = [json.loads(index[0][0][1])], [json.loads(index[0][0][3])]

    for item, texts in zip(index[1:], data[1:]):
        fid_min, fid_max = min(json.loads(item[0][1])), max(json.loads(item[0][1]))
        tid_min, tid_max = min(json.loads(item[0][3])), max(json.loads(item[0][3]))

        if direction == "from":
            curr_paragraph_min = paragraphs_from_dict[fid_min][0]
            curr_paragraph_max = paragraphs_from_dict[fid_max][0]
        else:
            curr_paragraph_min = paragraphs_to_dict[tid_min][0]
            curr_paragraph_max = paragraphs_to_dict[tid_max][0]

        # if curr_paragraph_min > 63 and curr_paragraph_min < 67:
        #     print("tid_min, tid_max", tid_min, tid_max)
        #     print("curr_paragraph_min", "curr_paragraph_max", curr_paragraph_min, curr_paragraph_max)
        #     print("prev_paragraph_max", prev_paragraph_max)

        if curr_paragraph_min == prev_paragraph_max:
            
            curr_from.append(texts["text_from"])
            curr_to.append(texts["text_to"])
            curr_splitted_ids_from.append(json.loads(item[0][1]))
            curr_splitted_ids_to.append(json.loads(item[0][3]))

            # if curr_paragraph_min == 65 and curr_paragraph_max == 66:
            #     print(curr_from)

            prev_paragraph_max = curr_paragraph_max
                
        else:
            yield curr_from, curr_to, prev_paragraph_max, curr_splitted_ids_from, curr_splitted_ids_to

            prev_paragraph_max = curr_paragraph_max
            curr_from, curr_to = [texts["text_from"]], [texts["text_to"]]
            curr_splitted_ids_from, curr_splitted_ids_to = [json.loads(item[0][1])], [json.loads(item[0][3])]

    yield curr_from, curr_to, prev_paragraph_max, curr_splitted_ids_from, curr_splitted_ids_to


def create_book(paragraphs_from, paragraphs_to, meta, output_path, template, styles=[]):
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

    with open(output_path, "w", encoding="utf8") as res_html:
        # --------------------HEAD
        res_html.write(f"""
        <html><head>
            <link rel="stylesheet" href="main.css">
            <link rel="preconnect" href="https://fonts.gstatic.com">
            <link href="https://fonts.googleapis.com/css2?family=Noto+Serif:wght@400&display=swap" rel="stylesheet">
            <link href="https://fonts.googleapis.com/css2?family=Raleway&display=swap" rel="stylesheet">
            <title>Lingtrain Magic Book</title>
            {css}
        </head>
        <body>""")

        # --------------------BOOK
        res_html.write("<div class='dt cont'>")

        # --------------------IMG
        # res_html.write("<div class='dt-row'><div class='par dt-cell'>")
        # res_html.write("<img class='flag-img' src='img/flags/flag-ru-h.svg'/>")
        # res_html.write("</div>")
        # res_html.write("<div class='par dt-cell'>")
        # res_html.write("<img class='flag-img' src='img/flags/flag-de-h.svg'/>")
        # res_html.write("</div></div>")

        #--------------------TITLE and AUTHOR
        res_html.write("<div class='dt-row header'><div class='par dt-cell'>")

        title_from = get_meta_from(meta, preprocessor.TITLE)
        if title_from:
            res_html.write("<h1>" + title_from[0] + "</h1>")
        author_from = get_meta_from(meta, preprocessor.AUTHOR)
        if author_from:
            res_html.write("<h2>" + author_from[0] + "</h2>")

        res_html.write("</div><div class='par dt-cell'>")

        title_to = get_meta_to(meta, preprocessor.TITLE)
        if title_to:
            res_html.write("<h1>" + title_to[0] + "</h1>")
        author_to = get_meta_to(meta, preprocessor.AUTHOR)
        if author_to:
            res_html.write("<h2>" + author_to[0] + "</h2>")

        res_html.write("</div></div>")

        # --------------------DIVIDER
        res_html.write("<div class='dt-row header'>")
        res_html.write(
            "<div class='dt-cell divider'><img class='divider-img' src='img/divider1.svg'/></div><div class='dt-cell divider'><img class='divider-img' src='img/divider1.svg'/></div>")
        res_html.write("</div>")

        # --------------------FIRST HEADERS IF EXIST
        write_header(res_html, meta, preprocessor.H1, occurence=0)
        write_header(res_html, meta, preprocessor.H2, occurence=0)
        write_header(res_html, meta, preprocessor.H3, occurence=0)
        write_header(res_html, meta, preprocessor.H4, occurence=0)
        write_header(res_html, meta, preprocessor.H5, occurence=0)

        # --------------------PARAGRAPHS
        for p_from, p_to in zip(paragraphs_from, paragraphs_to):

            if p_from == H1_MARK: write_header(res_html, meta, preprocessor.H1, occurence=p_to, add_divider=True)
            elif p_from == H2_MARK: write_header(res_html, meta, preprocessor.H2, occurence=p_to, add_divider=True)
            elif p_from == H3_MARK: write_header(res_html, meta, preprocessor.H3, occurence=p_to, add_divider=True)
            elif p_from == H4_MARK: write_header(res_html, meta, preprocessor.H4, occurence=p_to, add_divider=True)
            elif p_from == H5_MARK: write_header(res_html, meta, preprocessor.H5, occurence=p_to, add_divider=True)

            elif p_from == DIVIDER_MARK:
                res_html.write("<div class='dt-row header'>")
                res_html.write(
                    "<div class='dt-cell divider'><img class='divider-img' src='img/divider1.svg'/></div><div class='dt-cell divider'><img class='divider-img' src='img/divider1.svg'/></div>")
                res_html.write("</div>")
            else:
                res_html.write("<div class='dt-row'><div class='par dt-cell'>")
                for i, sent in enumerate(p_from):
                    res_html.write(f"<span class='sent sent-{i%sent_cycle}'>{sent}</span>")
                res_html.write("</div><div class='par dt-cell'>")
                for i, sent in enumerate(p_to):
                    res_html.write(f"<span class='sent sent-{i%sent_cycle}'>{sent}</span>")
                res_html.write("</div></div>")

        res_html.write("</div>")
        res_html.write("</body></html>")


def create_polybook(lang_ordered, paragraphs, delimeters, metas, output_path, template, styles=[]):
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

    with open(output_path, "w", encoding="utf8") as res_html:
        # --------------------HEAD
        res_html.write(f"""
        <html><head>
            <link rel="stylesheet" href="main.css">
            <link rel="preconnect" href="https://fonts.gstatic.com">
            <link href="https://fonts.googleapis.com/css2?family=Noto+Serif:wght@400&display=swap" rel="stylesheet">
            <link href="https://fonts.googleapis.com/css2?family=Raleway&display=swap" rel="stylesheet">
            <title>Lingtrain Magic Book</title>
            {css}
        </head>
        <body>""")

        # --------------------BOOK
        res_html.write("<div class='dt cont'>")

        #--------------------TITLE and AUTHOR
        res_html.write("<div class='dt-row header'>")
        for lang in lang_ordered:
            res_html.write("<div class='par dt-cell'>")
            meta = metas["items"][lang]
            title = get_meta(meta, preprocessor.TITLE)
            author = get_meta(meta, preprocessor.AUTHOR)
            if title:
                res_html.write("<h1>" + title + "</h1>")
            if author:
                res_html.write("<h1>" + author + "</h1>")
            res_html.write("</div>")
        res_html.write("</div>")

        # --------------------DIVIDER
        res_html.write("<div class='dt-row header'>")
        for _ in range(langs_count):
            res_html.write(
                "<div class='dt-cell divider'><img class='divider-img' src='img/divider1.svg'/></div>")
        res_html.write("</div>")

        # --------------------PARAGRAPHS
        next_mark, next_meta_par_id = get_next_meta_par_id(metas)
        min_par_len = min([len(paragraphs[x]) for x in paragraphs])
            
        for actual_paragraphs_id in range(min_par_len):
            real_par_id = delimeters[actual_paragraphs_id]

            while next_meta_par_id <= real_par_id:
                write_next_polyheader(res_html, next_mark, metas["items"], lang_ordered)
                next_mark, next_meta_par_id = get_next_meta_par_id(metas)
            
            res_html.write("<div class='dt-row'>")
            for lang in lang_ordered:
                
                res_html.write(f"<div class='par dt-cell'>  ")

                for k, sent in enumerate(paragraphs[lang][actual_paragraphs_id]):
                    res_html.write(f"<span class='sent sent-{k%sent_cycle}'>{sent}</span>")
                    
                res_html.write("</div>")

            res_html.write("</div>")

        # --------------------END BOOK
        res_html.write("</div>")
        res_html.write("</body></html>")


def get_next_meta_par_id(metas):
    lang_code = metas["main_lang_code"]
    meta = metas["items"][lang_code]
    min_par_id = float('Inf')
    next_mark = ""
    for mark in MARKS:
        if mark in meta and meta[mark]:
            if meta[mark][0][2] < min_par_id:
                next_mark = mark
                min_par_id = meta[mark][0][2]
    return next_mark, min_par_id


def write_next_polyheader(writer, next_mark, metas, lang_ordered):
    writer.write("<div class='dt-row header'>")
    for lang in lang_ordered:
        meta = metas[lang][next_mark]
        if meta:
            val = meta.pop(0)
        else:
            val = ("",)
        writer.write("<div class='par dt-cell'>")
        writer.write(f'<{HEADER_HTML_MAPPING[next_mark]}>' + val[0] + f'</{HEADER_HTML_MAPPING[next_mark]}>')
        writer.write("</div>")
            
    writer.write("</div>")


def write_header(writer, meta, mark, occurence, add_divider=False):
    meta_from = get_meta_from(meta, mark, occurence)
    meta_to = get_meta_to(meta, mark, occurence)
    if not meta_to: meta_to=meta_from

    if meta_from:
        if add_divider:
            writer.write("<div class='dt-row header'>")
            writer.write(
                "<div class='dt-cell divider'></div><div class='dt-cell divider'></div>")
            writer.write("</div>")
        # left
        writer.write("<div class='dt-row header'><div class='par dt-cell'>")
        writer.write(f'<{HEADER_HTML_MAPPING[mark]}>' + meta_from[0] + f'</{HEADER_HTML_MAPPING[mark]}>')
        writer.write("</div>")
        # right
        writer.write("<div class='par dt-cell'>")
        writer.write(f'<{HEADER_HTML_MAPPING[mark]}>' + get_meta_to(meta,
                                            mark, occurence)[0] + f'</{HEADER_HTML_MAPPING[mark]}>')
        writer.write("</div></div>")


def prepare_meta(meta, direction):
    """Take all meta for an exact direction"""
    res = dict()
    for key in meta:
        if key.endswith(direction):
            #trim direction, leave mark only
            res[key[:len(key)-len(direction)-1]] = meta[key]
    return res


def get_meta(meta, mark, occurence=0):
    """Get prepared meta value"""
    if len(meta[mark]) > occurence:
        return meta[mark][occurence][0]
    return ''


def get_meta_from(meta, mark, occurence=0):
    """Get meta value from"""
    key = f"{mark}_from"
    if len(meta[key]) > occurence:
        return meta[key][occurence]
    return ''


def get_meta_to(meta, mark, occurence=0):
    """Get meta value to"""
    key = f"{mark}_to"
    if len(meta[key]) > occurence:
        return meta[key][occurence]
    return ''


def generate_css(styles, cols=2):
    special = ""
    for i,s in enumerate(styles):
        style = json.loads(s)
        special += f".sent-{i} {{\n"
        for rule in style:
            special += f"{rule}: {style[rule]};"
        special += "\n}"
    col_width = f"{100 // cols}%"

    res = CSS_TEMPLATE.replace("%GENERAL%", CSS_GENERAL).replace("%SPECIAL%", special).replace("%COL_WIDTH%", col_width)
    return res

HEADER_HTML_MAPPING = {
    preprocessor.H1: "h2",
    preprocessor.H2: "h3",
    preprocessor.H3: "h3",
    preprocessor.H4: "h4",
    preprocessor.H5: "h4"
}

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
    font-family: 'Raleway', cursive;
}

.cont {
    margin: 0 150px;
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
}

.dt-row:nth-child(even):not(.header) {
    background: #fafafa;
}

.dt-row:nth-child(even):not(.header) > .dt-cell:not(.divider) {
    border-top: 1px solid #e0e0e0;
    border-bottom: 1px solid #e0e0e0;
}

.dt-cell {
    display: table-cell;
    width: %COL_WIDTH% !important;
}

.dt-cell-colspan {
    display: table-caption;
}

.divider {
    padding-top: 30px;
    vertical-align:middle;
    text-align:center;
}

.divider-img {
    width: 100px;
}

.flag-img {
    height: 50px;
    width: 50px;
    padding: 0 10px 0 0;
}

.inline {
    display: inline-block;
}

.sent {
    padding: 0 6px;
}
"""


STYLES = {
    "pastel_fill": [
        '{"background": "#A2E4B8", "color": "black", "border-bottom": "0px solid red"}',
        '{"background": "#FFC1CC", "color": "black"}',
        '{"background": "#9BD3DD", "color": "black"}',
        '{"background": "#FFFCC9", "color": "black"}'
        ],
    "pastel_start": [
        '{"background": "linear-gradient(90deg, #A2E4B8 0px, #fff 150px)"}',
        '{"background": "linear-gradient(90deg, #FFC1CC 0px, #fff 150px)"}',
        '{"background": "linear-gradient(90deg, #9BD3DD 0px, #fff 150px)"}',
        '{"background": "linear-gradient(90deg, #FFFCC9 0px, #fff 150px)"}'    
        ]

# styles3 = [
#     '{ }',
#     '{"background": "#fafad2", "color": "black"}',
#     ]


# styles4 = [
#     '{ }',
#     '{"background": "crimson", "color": "white", "border-radius": "15px"}',
#     ]

# styles5 = [
#     '{ }',
#     '{"background": "linear-gradient(45deg, #85FFBD 0%, #FFFB7D 100%)"}',
#     ]

# styles6 = [
#     '{"background": "linear-gradient(90deg, #FDEB71 0%, #fff 20%)"}',
#     '{"background": "linear-gradient(90deg, #ABDCFF 0%, #fff 20%)"}',
#     '{"background": "linear-gradient(90deg, #FEB692 0%, #fff 20%)"}',
#     '{"background": "linear-gradient(90deg, #CE9FFC 0%, #fff 20%)"}',
#     '{"background": "linear-gradient(90deg, #81FBB8 0%, #fff 20%)"}'
#     ]

# styles7 = [
#     '{"background": "linear-gradient(90deg, #FDEB71 0px, #fff 150px)"}',
#     '{"background": "linear-gradient(90deg, #ABDCFF 0px, #fff 150px)"}',
#     '{"background": "linear-gradient(90deg, #FEB692 0px, #fff 150px)"}',
#     '{"background": "linear-gradient(90deg, #CE9FFC 0px, #fff 150px)"}',
#     '{"background": "linear-gradient(90deg, #81FBB8 0px, #fff 150px)"}'
#     ]


}