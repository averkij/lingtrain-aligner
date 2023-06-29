import re
import dateutil.parser as date_parser
import dateparser

from collections import defaultdict

# preprocessing cleaning operations
DELETE = "delete"
SPLIT = "split"
PASS = "pass"

PARAGRAPH = "paragraph"
H1, H2, H3, H4, H5 = "h1", "h2", "h3", "h4", "h5"
DIVIDER = "divider"
TITLE = "title"
AUTHOR = "author"
TRANSLATOR = "translator"
QUOTE_TEXT = "qtext"
QUOTE_NAME = "qname"
IMAGE = "image"
SEGMENT = "segment"

MARK_META = [
    H1,
    H2,
    H3,
    H4,
    H5,
    DIVIDER,
    TITLE,
    AUTHOR,
    QUOTE_TEXT,
    QUOTE_NAME,
    IMAGE,
    TRANSLATOR,
]

MARK_COUNTERS = [H1, H2, H3, H4, H5, DIVIDER, QUOTE_TEXT, QUOTE_NAME, IMAGE]
MARKS_FOR_ADDING = [
    H1,
    H2,
    H3,
    H4,
    H5,
    DIVIDER,
    QUOTE_TEXT,
    QUOTE_NAME,
    IMAGE,
    AUTHOR,
    TITLE,
]

PARAGRAPH_MARK = "%%%%%"
LINE_ENDINGS = [".", "!", "?", ";", ":", "。", "？", "！", '"', "'"]  # ”

headings_1 = re.compile(
    r"^часть .*$|^.* band$|^part .*$|^DÍL .*$|^capitolo .*$", re.IGNORECASE
)
headings_2 = re.compile(r"^глава .*$|^.* teil$", re.IGNORECASE)

arabic_nums = re.compile(r"^\d+$")
roman_nums = re.compile("^M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$")

page_number_pat = re.compile(r"^-\d+-$")
headings_inline = re.compile(r"^\d+\s+.*$|^\d+´.*$")

misc_pat = re.compile(r"^x x x*$")

# year range
years_range = re.compile(r"^15\d\d|16\d\d|17\d\d|18\d\d|19\d\d|20\d\d$")


def clean_artifacts(text, config=None, act=True):
    lines = []
    for line in text:
        line = line.strip()

        if re.match(page_number_pat, line):
            print("[page_num]:", line)
            if act and config["page_num"] == DELETE:
                line = ""

        if line and re.match(headings_inline, line) and not re.match(years_range, line):
            num = re.search("\d+", line)
            print(f"[inline], is date: {is_date(line)}: [{num.group(0)}]", line)
            if act:
                if config["date"] == PASS and is_date(line):
                    pass
                elif config["inline"] == DELETE:
                    line = ""
                elif config["inline"] == SPLIT and num:
                    lines.append(num.group(0))
                    line = line.replace(num.group(0), "", 1)

        if line and re.match(arabic_nums, line):
            print("[arabic]:", line)
            if act and config["arabic"] == DELETE:
                line = ""

        if line and re.match(misc_pat, line):
            print("[misc]:", line)
            if act and config["misc"] == DELETE:
                line = ""

        if line:
            lines.append(line.strip())

    return lines


def find_artifacts(path):
    foo = clean_artifacts(path, act=False)


def mark_heading(line, action):
    if action == DELETE:
        return ""
    if action == PASS:
        return line
    marked = f"{line}{PARAGRAPH_MARK}{action}."
    return marked


def mark_headings(text, config=None, act=True):
    lines = []
    line_endings = tuple([x for x in LINE_ENDINGS])
    for line in text:
        line = line.strip()
        end_prev = False

        if re.match(headings_1, line):
            print("[type 1]:", line)
            end_prev = True
            if act:
                line = mark_heading(line, config["type_1"])

        if re.match(headings_2, line):
            print("[type 2]:", line)
            end_prev = True
            if act:
                line = mark_heading(line, config["type_2"])

        if line and re.match(roman_nums, line):
            print("[roman]:", line)
            end_prev = True
            if act:
                line = mark_heading(line, config["roman"])

        if line and re.match(arabic_nums, line):
            print("[arabic]:", line)
            end_prev = True
            if act:
                line = mark_heading(line, config["arabic"])

        if line and re.match(headings_inline, line):
            if is_date(line):
                print("[date]:", line)
                end_prev = True
                if act:
                    line = mark_heading(line, config["date"])

        if line and re.match(misc_pat, line):
            print("[misc]:", line)
            end_prev = True
            if act:
                line = mark_heading(line, config["misc"])

        if line:
            if end_prev and len(lines) > 0:
                prev_line = lines[-1]
                if not prev_line.endswith(line_endings):
                    lines[-1] = prev_line + "."
                    end_prev = False
            lines.append(line)
    return lines


def find_headings(text):
    foo = mark_headings(text, act=False)


def is_date(line):
    res = dateparser.parse(line, languages=["ru", "en", "fr", "it", "de", "zh"])
    return False if not res else True


def mark_paragraphs(lines):
    line_endings = tuple([x for x in LINE_ENDINGS])
    for i, line in enumerate(lines):
        line = line.strip()
        if line.endswith(line_endings):
            lines[i] = line[:-1] + PARAGRAPH_MARK + line[-1]
    return lines


def parse_marked_line(line):
    """Parse marked line for UI view"""
    res = defaultdict(bool)
    p_ending = tuple([PARAGRAPH_MARK + x for x in LINE_ENDINGS])
    if line.endswith(p_ending):
        # remove last occurence of PARAGRAPH_MARK
        line = "".join(line.rsplit(PARAGRAPH_MARK, 1))
        res["pa"] = True
    for mark in MARK_META:
        ending = f"{PARAGRAPH_MARK}{mark}."
        if line.endswith(ending):
            res[mark[:2]] = True
            res["pa"] = False
            line = line[: len(line) - len(ending)]
    res["text"] = line
    return res


def extract_marks(res, line, ix):
    """Extract marks information in exists"""
    p_ending = tuple([PARAGRAPH_MARK + x for x in LINE_ENDINGS])
    if line.endswith(p_ending):
        # remove last occurence of PARAGRAPH_MARK
        line = "".join(line.rsplit(PARAGRAPH_MARK, 1))
    for mark in MARK_META:
        ending = f"{PARAGRAPH_MARK}{mark}."
        if line.endswith(ending):
            res.append((line[: len(line) - len(ending)], ix, mark))
    return res


def get_all_meta_marks():
    return [f"{PARAGRAPH_MARK}{m}" for m in MARK_META]
