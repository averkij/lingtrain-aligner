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
    prev_h1, prev_h2, prev_h3, prev_h4, prev_h5, prev_di = prev_meta[1], prev_meta[2], prev_meta[3], prev_meta[4], prev_meta[5], prev_meta[6]

    curr_from, curr_to = [data[0]["text_from"]], [data[0]["text_to"]]

    for item, texts in zip(index[1:], data[1:]):
        fid = max(json.loads(item[0][1]))

        curr_paragraph_from = paragraphs_from_dict[fid][0]

        curr_h1 = paragraphs_from_dict[fid][1]
        curr_h2 = paragraphs_from_dict[fid][2]
        curr_h3 = paragraphs_from_dict[fid][3]
        curr_h4 = paragraphs_from_dict[fid][4]
        curr_h5 = paragraphs_from_dict[fid][5]
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


def create_book(paragraphs_from, paragraphs_to, meta, output_path, template, styles=[]):
    """Generate html"""
    # ensure path is existed
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    if template == "custom":
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
            res_html.write("<h1>" + title_from + "</h1>")
        author_from = get_meta_from(meta, preprocessor.AUTHOR)
        if author_from:
            res_html.write("<h2>" + author_from + "</h2>")

        res_html.write("</div><div class='par dt-cell'>")

        title_to = get_meta_to(meta, preprocessor.TITLE)
        if title_to:
            res_html.write("<h1>" + title_to + "</h1>")
        author_to = get_meta_to(meta, preprocessor.AUTHOR)
        if author_to:
            res_html.write("<h2>" + author_to + "</h2>")

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
                res_html.write("<div class='dt-row'>")
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


def write_header(writer, meta, mark, occurence, add_divider=False):
    meta_from = get_meta_from(meta, mark, occurence)
    meta_to = get_meta_to(meta, mark, occurence)
    if not meta_to: meta_to=meta_from

    if meta_from:
        if add_divider:
            writer.write("<div class='dt-row'>")
            writer.write(
                "<div class='dt-cell divider'></div><div class='dt-cell divider'></div>")
            writer.write("</div>")
        # left
        writer.write("<div class='dt-row header'><div class='par dt-cell'>")
        writer.write(f'<{HEADER_HTML_MAPPING[mark]}>' + meta_from + f'</{HEADER_HTML_MAPPING[mark]}>')
        writer.write("</div>")
        # right
        writer.write("<div class='par dt-cell'>")
        writer.write(f'<{HEADER_HTML_MAPPING[mark]}>' + get_meta_to(meta,
                                            mark, occurence) + f'</{HEADER_HTML_MAPPING[mark]}>')
        writer.write("</div></div>")


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


def generate_css(styles):
    special = ""
    for i,s in enumerate(styles):
        style = json.loads(s)
        special += f".sent-{i} {{\n"
        for rule in style:
            special += f"{rule}: {style[rule]};"
        special += "\n}"

    res = CSS_TEMPLATE.replace("%GENERAL%", CSS_GENERAL).replace("%SPECIAL%", special)
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
    width: 50%;
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
