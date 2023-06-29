import json
import logging
import sqlite3
from collections import defaultdict


def get_doc_index_original(db_path):
    """Get document index"""
    res = []
    try:
        with sqlite3.connect(db_path) as db:
            cur = db.execute("SELECT contents FROM doc_index")
            res = json.loads(cur.fetchone()[0])
    except:
        logging.warning("can not fetch index db")
    return res


def get_flatten_doc_index(db_path, batch_ids=[]):
    """Get document index"""
    res = []
    try:
        with sqlite3.connect(db_path) as db:
            cur = db.execute("SELECT contents FROM doc_index")
            data = json.loads(cur.fetchone()[0])
        for batch_id, sub_index in enumerate(data):
            if batch_ids and batch_id not in batch_ids:
                continue
            res.extend(list(zip(sub_index, range(len(sub_index)))))
    except:
        logging.warning("can not fetch flatten index")
    return res


def get_flatten_doc_index_with_batch_id(db_path):
    """Get document index"""
    res = []
    try:
        with sqlite3.connect(db_path) as db:
            cur = db.execute("SELECT contents FROM doc_index")
            data = json.loads(cur.fetchone()[0])
        for batch_id, sub_index in enumerate(data):
            res.extend(
                list(zip(sub_index, range(len(sub_index)), [batch_id] * len(sub_index)))
            )
    except:
        logging.warning("can not fetch flatten index")
    return res


def get_clear_flatten_doc_index(db_path):
    """Get document index"""
    res = []
    try:
        with sqlite3.connect(db_path) as db:
            cur = db.execute("SELECT contents FROM doc_index")
            data = json.loads(cur.fetchone()[0])
        for _, sub_index in enumerate(data):
            res.extend(sub_index)
    except:
        logging.warning("can not fetch flatten index")
    return res


def add_empty_processing_line(db, batch_id):
    """Add empty processing line"""
    from_id = db.execute(
        "insert into processing_from(batch_id, text_ids, text) values (:batch_id, :text_ids, :text) ",
        {"batch_id": batch_id, "text_ids": "[]", "text": ""},
    ).lastrowid
    to_id = db.execute(
        "insert into processing_to(batch_id, text_ids, text) values (:batch_id, :text_ids, :text) ",
        {"batch_id": batch_id, "text_ids": "[]", "text": ""},
    ).lastrowid
    return (from_id, to_id)


def add_resolved_processing_line(db, batch_id, text_from, text_to):
    """Add processing line with text"""
    from_id = db.execute(
        "insert into processing_from(batch_id, text_ids, text) values (:batch_id, :text_ids, :text) ",
        {"batch_id": batch_id, "text_ids": "[]", "text": text_from},
    ).lastrowid
    to_id = db.execute(
        "insert into processing_to(batch_id, text_ids, text) values (:batch_id, :text_ids, :text) ",
        {"batch_id": batch_id, "text_ids": "[]", "text": text_to},
    ).lastrowid
    return (from_id, to_id)


def get_processing_from_by_id(db_path, start_id, end_id):
    """Get lines from processing by ids"""
    ids = [x for x in range(start_id, end_id + 1)]
    res = []
    with sqlite3.connect(db_path) as db:
        for id, text_from in db.execute(
            f'select f.id, f.text from processing_from f where f.id in ({",".join([str(x) for x in ids])})'
        ):
            res.append((id, text_from))
    return res


def get_processing_to_by_id(db_path, start_id, end_id):
    """Get lines from processing by ids"""
    ids = [x for x in range(start_id, end_id + 1)]
    res = []
    with sqlite3.connect(db_path) as db:
        for id, text_to, similarity in db.execute(
            f'select t.id, t.text, t.similarity from processing_to t where t.id in ({",".join([str(x) for x in ids])})'
        ):
            res.append((id, text_to, similarity))
    return res


def get_splitted_lenght(db_path):
    """Get splitted_from and splitted_to lenghts"""
    with sqlite3.connect(db_path) as db:
        len_from = db.execute(f"select count(*) from splitted_from").fetchone()[0]
        len_to = db.execute(f"select count(*) from splitted_to").fetchone()[0]
    return len_from, len_to


def get_splitted_from_by_id(db_path, ids):
    """Get lines from splitted_from by ids"""
    res = []
    with sqlite3.connect(db_path) as db:
        for (
            id,
            text_from,
            proxy_from,
            exclude,
            paragraph,
            h1,
            h2,
            h3,
            h4,
            h5,
            divider,
        ) in db.execute(
            f'select f.id, f.text, f.proxy_text, f.exclude, f.paragraph, f.h1, f.h2, f.h3, f.h4, f.h5, f.divider from splitted_from f where f.id in ({",".join([str(x) for x in ids])})'
        ):
            res.append(
                (
                    id,
                    text_from,
                    proxy_from,
                    exclude,
                    paragraph,
                    h1,
                    h2,
                    h3,
                    h4,
                    h5,
                    divider,
                )
            )
    return res


def get_splitted_to_by_id(db_path, ids):
    """Get lines from splitted_to by ids"""
    res = []
    with sqlite3.connect(db_path) as db:
        for (
            id,
            text_to,
            proxy_to,
            exclude,
            paragraph,
            h1,
            h2,
            h3,
            h4,
            h5,
            divider,
        ) in db.execute(
            f'select t.id, t.text, t.proxy_text, t.exclude, t.paragraph, t.h1, t.h2, t.h3, t.h4, t.h5, t.divider from splitted_to t where t.id in ({",".join([str(x) for x in ids])})'
        ):
            res.append(
                (id, text_to, proxy_to, exclude, paragraph, h1, h2, h3, h4, h5, divider)
            )
    return res


def get_splitted_from_by_id_range(db_path, start_id, end_id):
    """Get lines from splitted_from by ids"""
    ids = [x for x in range(start_id, end_id + 1)]
    splitted, proxy = dict(), dict()
    with sqlite3.connect(db_path) as db:
        for id, text_from, proxy_from in db.execute(
            f'select f.id, f.text, f.proxy_text from splitted_from f where f.id in ({",".join([str(x) for x in ids])})'
        ):
            splitted[id] = text_from
            proxy[id] = proxy_from
    return splitted, proxy


def get_splitted_to_by_id_range(db_path, start_id, end_id):
    """Get lines from splitted_to by ids"""
    ids = [x for x in range(start_id, end_id + 1)]
    splitted, proxy = dict(), dict()
    with sqlite3.connect(db_path) as db:
        for id, text_to, proxy_to in db.execute(
            f'select t.id, t.text, t.proxy_text from splitted_to t where t.id in ({",".join([str(x) for x in ids])})'
        ):
            splitted[id] = text_to
            proxy[id] = proxy_to
    return splitted, proxy


def get_splitted_from(db_path, ids=[]):
    """Get lines from splitted_from by ids"""
    res = dict()
    with sqlite3.connect(db_path) as db:
        if not ids:
            for id, text_from in db.execute(
                f"select f.id, f.text from splitted_from f"
            ):
                res[id] = text_from
        else:
            for id, text_from in db.execute(
                f'select f.id, f.text from splitted_from f where f.id in ({",".join([str(x) for x in ids])})'
            ):
                res[id] = text_from
    return res


def get_splitted_to(db_path, ids=[]):
    """Get lines from splitted_to by ids"""
    res = dict()
    with sqlite3.connect(db_path) as db:
        if not ids:
            for id, text_to in db.execute(f"select t.id, t.text from splitted_to t"):
                res[id] = text_to
        else:
            for id, text_to in db.execute(
                f'select t.id, t.text from splitted_to t where t.id in ({",".join([str(x) for x in ids])})'
            ):
                res[id] = text_to
    return res


def get_doc_page(db_path, text_ids):
    """Get processing lines page"""
    res = []
    with sqlite3.connect(db_path) as db:
        db.execute("DROP TABLE If EXISTS temp.text_ids")
        db.execute("CREATE TEMP TABLE text_ids(rank integer primary key, id integer)")
        db.executemany(
            "insert into temp.text_ids(id) values(?)", [(x,) for x in text_ids]
        )
        for batch_id, text_from, text_to in db.execute(
            """SELECT
                f.batch_id, f.text, t.text
            FROM
                processing_from f
                join
                    processing_to t
                        on t.id=f.id
                join
                    temp.text_ids ti
                        on ti.id = f.id
            ORDER BY
                ti.rank
            """
        ):
            res.append((text_from, text_to, batch_id))
    return res


def get_proxy_dict(items):
    """Get proxy sentences as dict"""
    res = dict()
    for item in items:
        res[item[0]] = item[2]
    return res


def get_paragraph_dict(items):
    """Get paragraphs info as dict"""
    res = dict()
    for item in items:
        res[item[0]] = (item[4], item[5], item[6], item[7], item[8], item[9], item[10])
    return res


def get_doc_items(index_items, db_path):
    """Get document items by ids"""
    res = []

    from_ids, to_ids = set(), set()
    for item in index_items:
        from_ids.update(json.loads(item[0][0][1]))
        to_ids.update(json.loads(item[0][0][3]))

    splitted_from = get_splitted_from_by_id(db_path, from_ids)
    splitted_to = get_splitted_to_by_id(db_path, to_ids)

    for i, (data, texts) in enumerate(
        zip(index_items, get_doc_page(db_path, [x[0][0][0] for x in index_items]))
    ):
        res.append(
            {
                "index_id": data[1],  # absolute position in index
                # from
                "batch_id": texts[2],
                "batch_index_id": data[0][1],  # relative position in index batch
                "text_from": texts[0],
                "line_id_from": data[0][0][1],  # array with ids
                # primary key in DB (processing_from)
                "processing_from_id": data[0][0][0],
                # to
                "text_to": texts[1],
                "line_id_to": data[0][0][3],  # array with ids
                # primary key in DB (processing_to)
                "processing_to_id": data[0][0][2],
            }
        )
    return res, get_proxy_dict(splitted_from), get_proxy_dict(splitted_to)


def read_processing(db_path, batch_ids=[]):
    """Read the processsing document"""
    ordered_text_ids = [x[0][0] for x in get_flatten_doc_index(db_path, batch_ids)]
    with sqlite3.connect(db_path) as db:
        db.execute("DROP TABLE If EXISTS temp.dl_ids")
        db.execute("CREATE TEMP TABLE dl_ids(rank integer primary key, id integer)")
        db.executemany(
            "insert into temp.dl_ids(id) values(?)", [(x,) for x in ordered_text_ids]
        )
        res = db.execute(
            """
            SELECT
                f.text, t.text
            FROM
                processing_from f
                join
                    processing_to t
                        on t.id=f.id
                join
                    temp.dl_ids ti
                        on ti.id = f.id
            ORDER BY
                ti.rank
                """
        ).fetchall()
        if not res:
            return [], []
        res = [list(x) for x in zip(*res)]
        return res[0], res[1]


def get_meta_dict(db_path):
    """Get all the meta information as dict"""
    res = defaultdict(list)
    with sqlite3.connect(db_path) as db:
        for key, val, occurence, par_id, id in db.execute(
            f"select m.key, m.val, m.occurence, m.par_id, m.id from meta m where m.deleted = 0"
        ):
            res[key].append((val, occurence, par_id, id))
    return res


def get_meta(db_path, mark, direction, occurence):
    """Get book meta information"""
    direction = "from" if direction == "from" else "to"
    with sqlite3.connect(db_path) as db:
        res = db.execute(
            f'select m.val from meta m where m.key="{mark}_{direction}" and occurence={occurence} and m.deleted = 0'
        ).fetchone()
    return res[0] if res else ""


def add_meta(
    db_path,
    mark,
    val_from,
    val_to,
    par_id_from,
    par_id_to,
    comment_from="",
    comment_to="",
):
    with sqlite3.connect(db_path) as db:
        query = db.execute(
            f"select max(m.occurence) from meta m where m.key=(?) and par_id <= (?)",
            (f"{mark}_from", par_id_from),
        ).fetchone()
        max_from_occurence = query[0] if query[0] is not None else -1
        print(query, max_from_occurence)
        query = db.execute(
            f"select max(m.occurence) from meta m where m.key=(?) and par_id <= (?)",
            (f"{mark}_to", par_id_to),
        ).fetchone()
        max_to_occurence = query[0] if query[0] is not None else -1

        # increment occurence
        db.execute(
            f"update meta set occurence = occurence + 1 where key=(?) and occurence > (?)",
            (f"{mark}_from", max_from_occurence),
        )
        db.execute(
            f"update meta set occurence = occurence + 1 where key=(?) and occurence > (?)",
            (f"{mark}_to", max_to_occurence),
        )

        print(query, max_to_occurence)
        data = [
            (
                f"{mark}_from",
                val_from,
                max_from_occurence + 1,
                par_id_from,
                comment_from,
            ),
            (f"{mark}_to", val_to, max_to_occurence + 1, par_id_to, comment_to),
        ]
        db.executemany(
            "insert into meta(key, val, occurence, par_id, comment) values(?, ?, ?, ?, ?)",
            [
                (key, val, occurence, par_id, comment)
                for key, val, occurence, par_id, comment in data
            ],
        )
    return


def delete_meta(db_path, mark_id):
    """Mark meta as deleted"""
    with sqlite3.connect(db_path) as db:
        db.execute(f"update meta set deleted = 1 where id=(?)", (mark_id,))
    return


def edit_meta(db_path, mark, direction, mark_id, par_id, val):
    """Edit meta"""
    meta_key = f"{mark}_{direction}"
    with sqlite3.connect(db_path) as db:
        curr_par_id = db.execute(
            f"select par_id from meta m where m.id=(?)", (mark_id,)
        ).fetchone()[0]
        if curr_par_id == par_id:
            print("par ids are equal")
            db.execute(f"update meta set val=(?) where id=(?)", (val, mark_id))
        else:
            query = db.execute(
                f"select max(m.occurence) from meta m where m.key=(?) and par_id <= (?)",
                (meta_key, par_id),
            ).fetchone()
            max_occurence = query[0] if query[0] is not None else -1
            # increment occurence
            db.execute(
                f"update meta set occurence = occurence + 1 where key=(?) and occurence > (?)",
                (meta_key, max_occurence),
            )
            db.execute(
                f"update meta set val=(?), par_id=(?) where id=(?)",
                (val, par_id, mark_id),
            )
    return


def get_meta_from(db_path, mark, occurence):
    """Get book meta information 'from'"""
    return get_meta(db_path, mark, "from", occurence)


def get_meta_to(db_path, mark, occurence):
    """Get book meta information 'to'"""
    return get_meta(db_path, mark, "to", occurence)


def get_lang_codes(db_path):
    """Get languages information"""
    with sqlite3.connect(db_path) as db:
        lang_from = db.execute(
            f'select l.val from languages l where l.key="from"'
        ).fetchone()
        lang_to = db.execute(
            f'select l.val from languages l where l.key="to"'
        ).fetchone()
    return lang_from[0], lang_to[0]


def get_files_info(db_path):
    """Get files information"""
    with sqlite3.connect(db_path) as db:
        info_from = db.execute(
            f'select f.name, f.guid from files f where f.direction="from"'
        ).fetchone()
        info_to = db.execute(
            f'select f.name, f.guid from files f where f.direction="to"'
        ).fetchone()
    return info_from[0], info_to[0], info_from[1], info_to[1]


def get_processing_from(db_path):
    """Get lines from processing_from"""
    with sqlite3.connect(db_path) as db:
        res = db.execute(
            f"select f.text from processing_from f order by f.id"
        ).fetchall()
    return [x[0] for x in res]


def get_processing_to(db_path):
    """Get lines from processing_to"""
    with sqlite3.connect(db_path) as db:
        res = db.execute(f"select t.text from processing_to t order by t.id").fetchall()
    return [x[0] for x in res]


def get_batch_info(db_path, batch_id):
    """Get batch alignment parameters"""
    with sqlite3.connect(db_path) as db:
        shift, window = db.execute(
            f"select b.shift, b.window from batches b where b.batch_id=:batch_id",
            {"batch_id": batch_id},
        ).fetchone()
    if shift is not None:
        return shift, window
    return None, None


def get_batches_info(db_path):
    """Get batches info"""
    with sqlite3.connect(db_path) as db:
        res = db.execute(
            "select b.batch_id, b.insert_ts, b.shift, b.window from batches b"
        ).fetchall()
    return res


def get_version(db_path):
    """Get alignment database version"""
    with sqlite3.connect(db_path) as db:
        res = db.execute(f"select v.version from version v").fetchone()
    return res[0]


def set_name(db_path, name):
    """Update alignment name"""
    with sqlite3.connect(db_path) as db:
        db.execute(
            "insert or replace into info (key, val) values ('name',?)",
            (name,),
        )


def get_name(db_path):
    """Get alignment name"""
    with sqlite3.connect(db_path) as db:
        res = db.execute(f"select i.val from info i where i.key='name'").fetchone()
    return res[0]


def get_unique_variants(variants_ids):
    """Get unique variants"""
    res = set()
    for var_ids in variants_ids:
        for ids in var_ids:
            res.add(ids)
    return res


def get_string(dic, ids):
    """Join into string"""
    s = " ".join([dic[x] for x in ids])
    return s


def lazy_property(func):
    """Lazy initialization attribute"""
    attr_name = "_lazy_" + func.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return _lazy_property
