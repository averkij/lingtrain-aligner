from lingtrain_aligner import helper, preprocessor
from shutil import copyfile
import json
import pathlib
import os
import lingtrain_aligner

H1_MARK = preprocessor.PARAGRAPH_MARK + preprocessor.H1
H2_MARK = preprocessor.PARAGRAPH_MARK + preprocessor.H2
H3_MARK = preprocessor.PARAGRAPH_MARK + preprocessor.H3
H4_MARK = preprocessor.PARAGRAPH_MARK + preprocessor.H4
H5_MARK = preprocessor.PARAGRAPH_MARK + preprocessor.H5
DIVIDER_MARK = preprocessor.PARAGRAPH_MARK + preprocessor.DIVIDER


def get_paragraphs(db_path):
    """Read all paragraphs with marks from database"""
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
    prev_meta = paragraphs_from_dict[json.loads(index[0][0][1])[0]]

    prev_paragraph_from = prev_meta[0]
    prev_h1, prev_h2, prev_di = prev_meta[1], prev_meta[2], prev_meta[6]

    curr_from, curr_to = [data[0]["text_from"]], [data[0]["text_to"]]

    for item, texts in zip(index[1:], data[1:]):
        fid = max(json.loads(item[0][1]))

        curr_paragraph_from = paragraphs_from_dict[fid][0]
        curr_h1 = paragraphs_from_dict[fid][1]
        curr_h2 = paragraphs_from_dict[fid][2]
        curr_di = paragraphs_from_dict[fid][6]

        if curr_paragraph_from == prev_paragraph_from:
            curr_from.append(texts["text_from"])
            curr_to.append(texts["text_to"])
        else:
            paragraphs_from.append(curr_from)
            paragraphs_to.append(curr_to)

            prev_paragraph_from = curr_paragraph_from
            curr_from, curr_to = [texts["text_from"]], [texts["text_to"]]

        if curr_h1 != prev_h1:
            paragraphs_from.append(H1_MARK)
            paragraphs_to.append(curr_h1-1)
            prev_h1 = curr_h1

        if curr_h2 != prev_h2:
            paragraphs_from.append(H2_MARK)
            paragraphs_to.append(curr_h2-1)
            prev_h2 = curr_h2

        if curr_di != prev_di:
            paragraphs_from.append(DIVIDER_MARK)
            paragraphs_to.append(curr_di-1)
            prev_di = curr_di

    paragraphs_from.append(curr_from)
    paragraphs_to.append(curr_to)

    return paragraphs_from, paragraphs_to, meta


def create_book(paragraphs_from, paragraphs_to, meta, output_path, template):
    """Generate html"""
    # ensure path is existed and copy styles
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    styles_file = os.path.join(os.path.dirname(
        lingtrain_aligner.__file__), "assets", "main.css")
    copyfile(styles_file, os.path.join(
        pathlib.Path(output_path).parent, "main.css"))

    with open(output_path, "w", encoding="utf8") as res_html:
        # --------------------HEAD
        res_html.write("""
        <html><head>
            <link rel="stylesheet" href="main.css">
            <link rel="preconnect" href="https://fonts.gstatic.com">
            <link href="https://fonts.googleapis.com/css2?family=Noto+Serif:wght@400;700&display=swap" rel="stylesheet">
            <title>Lingtrain Magic Book</title>
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
        res_html.write("<div class='dt-row'><div class='par dt-cell'>")

        author_from = get_meta_from(meta, preprocessor.AUTHOR)
        if author_from:
            res_html.write("<h1>" + author_from + "</h1>")
        title_from = get_meta_from(meta, preprocessor.TITLE)
        if title_from:
            res_html.write("<h2>" + title_from + "</h2>")

        res_html.write("</div>")

        res_html.write("<div class='par dt-cell'>")
        author_to = get_meta_to(meta, preprocessor.AUTHOR)
        if author_to:
            res_html.write("<h1>" + author_to + "</h1>")
        title_to = get_meta_to(meta, preprocessor.TITLE)
        if title_to:
            res_html.write("<h2>" + title_to + "</h2>")

        res_html.write("</div></div>")

        # --------------------FIRST H1 and H2
        res_html.write("<div class='dt-row'><div class='par dt-cell'>")

        h1_from = get_meta_from(meta, preprocessor.H1)
        if h1_from:
            res_html.write("<h2>" + h1_from + "</h2>")
        h2_from = get_meta_from(meta, preprocessor.H2)
        if h2_from:
            res_html.write("<h3>" + h2_from + "</h3>")

        res_html.write("</div>")

        res_html.write("<div class='par dt-cell'>")
        h1_to = get_meta_to(meta, preprocessor.H1)
        if h1_to:
            res_html.write("<h2>" + h1_to + "</h2>")
        h2_to = get_meta_to(meta, preprocessor.H2)
        if h2_to:
            res_html.write("<h3>" + h2_to + "</h3>")

        res_html.write("</div></div>")

        # --------------------PARAGRAPHS
        for p_from, p_to in zip(paragraphs_from, paragraphs_to):
            if p_from == H1_MARK:
                # left
                res_html.write("<div class='dt-row'><div class='par dt-cell'>")
                res_html.write('<h3>' + get_meta_from(meta,
                                                      preprocessor.H1, p_to) + '</h3>')
                res_html.write("</div>")
                # right
                res_html.write("<div class='par dt-cell'>")
                res_html.write('<h3>' + get_meta_from(meta,
                                                      preprocessor.H1, p_to) + '</h3>')
                res_html.write("</div></div>")
            elif p_from == H2_MARK:
                # left
                res_html.write("<div class='dt-row'><div class='par dt-cell'>")
                res_html.write('<h4>' + get_meta_from(meta,
                                                      preprocessor.H2, p_to) + '</h4>')
                res_html.write("</div>")
                # right
                res_html.write("<div class='par dt-cell'>")
                res_html.write('<h4>' + get_meta_from(meta,
                                                      preprocessor.H2, p_to) + '</h4>')
                res_html.write("</div></div>")
            elif p_from == DIVIDER_MARK:
                res_html.write("<div class='dt-row'>")
                res_html.write(
                    "<div class='dt-cell divider'><img class='divider-img' src='img/divider1.svg'/></div><div class='dt-cell divider'><img class='divider-img' src='img/divider1.svg'/></div>")
                res_html.write("</div>")
            else:
                res_html.write("<div class='dt-row'><div class='par dt-cell'>")
                res_html.write(' '.join(p_from))
                res_html.write("</div><div class='par dt-cell'>")
                res_html.write(' '.join(p_to))
                res_html.write("</div></div>")

        res_html.write("</div>")
        res_html.write("</body></html>")


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
