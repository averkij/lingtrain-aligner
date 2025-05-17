import json
import logging
import sqlite3
from collections import defaultdict
import os
from lingtrain_aligner import constants as con
import numpy as np


def create_table_splitted(db, direction):
    """Create tables for splitted lines. Created separately because PK is not needed anymore.
    Tables are recreated while using split confinct feature."""
    if direction == "from":
        db.execute(
            """
            create table splitted_from(
                id integer,
                text text,
                proxy_text text, 
                exclude integer,
                paragraph integer,
                h1 integer,
                h2 integer, 
                h3 integer,
                h4 integer,
                h5 integer,
                divider int,
                embedding text,
                proxy_embedding text
            )
        """
        )
    else:
        db.execute(
            """
            create table splitted_to(
                id integer,
                text text,
                proxy_text text,
                exclude integer,
                paragraph integer, 
                h1 integer,
                h2 integer,
                h3 integer,
                h4 integer,
                h5 integer,
                divider int,
                embedding text,
                proxy_embedding text
            )
        """
        )


def init_document_db(db_path):
    """Init document database (alignment) with tables structure"""
    if os.path.isfile(db_path):
        os.remove(db_path)
    with sqlite3.connect(db_path) as db:
        create_table_splitted(db, "from")
        create_table_splitted(db, "to")
        db.execute(
            "create table processing_from(id integer primary key, batch_id integer, text_ids varchar, initial_id integer, text nvarchar)"
        )
        db.execute(
            "create table processing_to(id integer primary key, batch_id integer, text_ids varchar, initial_id integer, text nvarchar)"
        )
        db.execute("create table doc_index(id integer primary key, contents varchar)")
        db.execute(
            "create table batches(id integer primary key, batch_id integer unique, insert_ts text, shift integer, window integer)"
        )
        db.execute(
            "create table history(id integer primary key, operation text, batch_id integer, insert_ts text, parameters text)"
        )
        db.execute(
            'create table meta(id integer primary key, key text, val text, occurence integer, par_id integer, deleted integer DEFAULT 0, comment text DEFAULT "")'
        )
        db.execute("create table languages(id integer primary key, key text, val text)")
        db.execute(
            "create table files(id integer primary key, direction text, name text, guid text)"
        )
        db.execute("create table info(id integer primary key, key text, val text)")
        db.execute("create table version(id integer primary key, version text)")
        db.execute("insert into version(version) values (?)", (con.DB_VERSION,))


def get_splitted_ids_without_embeddings(
    db_path, direction, line_ids=[], is_proxy=False
):
    """Get splitted ids without embeddings"""
    if direction == "from":
        table_name = "splitted_from"
    else:
        table_name = "splitted_to"
    with sqlite3.connect(db_path) as db:
        if is_proxy:
            if not line_ids:
                res = db.execute(
                    f"select s.id from {table_name} s where s.proxy_embedding is NULL"
                ).fetchall()
            else:
                res = db.execute(
                    f"select s.id from {table_name} s where s.id in ({','.join([str(x) for x in line_ids])}) and s.proxy_embedding is NULL"
                ).fetchall()
        else:
            if not line_ids:
                res = db.execute(
                    f"select s.id from {table_name} s where s.embedding is NULL"
                ).fetchall()
            else:
                res = db.execute(
                    f"select s.id from {table_name} s where s.id in ({','.join([str(x) for x in line_ids])}) and s.embedding is NULL"
                ).fetchall()
    return [x[0] for x in res]


def set_embeddings(db_path, direction, line_ids=[], embeddings=[], is_proxy=False):
    """Fill embeddings in splitted table"""
    if direction == "from":
        table_name = "splitted_from"
    else:
        table_name = "splitted_to"

    # if embeddings are numpy arrays, convert them to lists, if not, leave them as they are
    embeddings = [x.tolist() if isinstance(x, np.ndarray) else x for x in embeddings]
    with sqlite3.connect(db_path) as db:
        if is_proxy:
            db.executemany(
                f"update {table_name} set proxy_embedding=? where id=?",
                [(json.dumps(x), y) for x, y in zip(embeddings, line_ids)],
            )
        else:
            db.executemany(
                f"update {table_name} set embedding=? where id=?",
                [(json.dumps(x), y) for x, y in zip(embeddings, line_ids)],
            )


def get_embeddings(db_path, direction, line_ids=[], is_proxy=False):
    """Get embeddings from splitted table"""
    if direction == "from":
        table_name = "splitted_from"
    else:
        table_name = "splitted_to"
    with sqlite3.connect(db_path) as db:
        if is_proxy:
            res = db.execute(
                f"select s.id, s.proxy_embedding from {table_name} s where s.id in ({','.join([str(x) for x in line_ids])})"
            ).fetchall()
        else:
            res = db.execute(
                f"select s.id, s.embedding from {table_name} s where s.id in ({','.join([str(x) for x in line_ids])})"
            ).fetchall()

    res = [
        (x[0], np.array(json.loads(x[1])) if x[1] is not None else None) for x in res
    ]

    return res


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


def get_flatten_doc_index_with_batch_id(db_path, index=None):
    """Get document index"""
    res = []
    try:
        with sqlite3.connect(db_path) as db:
            cur = db.execute("SELECT contents FROM doc_index")
            if not index:
                index = json.loads(cur.fetchone()[0])
        for batch_id, sub_index in enumerate(index):
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


def check_table_pk(db, table_name):
    cursor = db.execute(f"PRAGMA table_info({table_name});")
    table_info = cursor.fetchall()

    for column in table_info:
        name = column[1]
        pk = column[5]
        if name == "id":
            return True if pk == 1 else False

    return False


def rename_table(db, old_name, new_name):
    """Rename table"""
    db.execute(f"ALTER TABLE `{old_name}` RENAME TO `{new_name}`")


def ensure_splitted_pk_is_not_exists(db_path, direction):
    """Drop PK in splitted table if exists"""
    if direction == "from":
        table_name = "splitted_from"
    else:
        table_name = "splitted_to"

    with sqlite3.connect(db_path) as db:
        pk_exists = check_table_pk(db, table_name)
        if pk_exists:
            print("PK exists, dropiing...", table_name)
            old_table_name = f"old_{table_name}"
            # rename original_table
            rename_table(db, table_name, old_table_name)
            # create table without PK
            create_table_splitted(db, direction)
            # copy data
            db.execute(
                f"insert into {table_name} select * from {old_table_name}",
            )
            # drop old table
            db.execute(f"drop table {old_table_name}")


def update_splitted_text(db_path, direction, line_id, val):
    """Update line value in splitted table"""
    if direction == "from":
        table_name = "splitted_from"
    else:
        table_name = "splitted_to"
    with sqlite3.connect(db_path) as db:
        db.execute(f"update {table_name} set text=? where id=?", (val, line_id))


def update_processing_text(db_path, direction, line_id, val):
    """Update line value in splitted table"""
    if direction == "from":
        table_name = "processing_from"
    else:
        table_name = "processing_to"
    with sqlite3.connect(db_path) as db:
        db.execute(
            f"update {table_name} set text=? where text_ids=?", (val, f"[{line_id}]")
        )


def insert_new_splitted_line(db_path, direction, line_id):
    """Insert line after splitting operation (split conflict feature)."""
    if direction == "from":
        table_name = "splitted_from"
    else:
        table_name = "splitted_to"

    alignment_version = get_version(db_path)

    with sqlite3.connect(db_path) as db:
        db.execute(
            f"update {table_name} set id=id+1 where id>?",
            (line_id,),
        )
        # TODO Recalculate embeddings (leave them empty?)

        if alignment_version >= 7.0:
            db.execute(
                f"""insert into {table_name}(id, text, proxy_text, exclude, paragraph, h1, h2, h3, h4, h5, divider, embedding, proxy_embedding)
                    select {line_id+1}, '', proxy_text, exclude, paragraph, h1, h2, h3, h4, h5, divider, embedding, proxy_embedding from {table_name} where id=?""",
                (line_id,),
            )
        else:
            db.execute(
                f"""insert into {table_name}(id, text, proxy_text, exclude, paragraph, h1, h2, h3, h4, h5, divider)
                    select {line_id+1}, '', proxy_text, exclude, paragraph, h1, h2, h3, h4, h5, divider from {table_name} where id=?""",
                (line_id,),
            )


def update_processing_mapping(db_path, direction, line_id):
    """Update lines mapping in processing table"""
    if direction == "from":
        table_name = "processing_from"
        processing_data = get_processing_from_text_ids_non_empty(db_path)
    else:
        table_name = "processing_to"
        processing_data = get_processing_to_text_ids_non_empty(db_path)

    mapping = {}
    for id, text_ids_json in processing_data:
        text_ids = json.loads(text_ids_json)
        if any(x > line_id for x in text_ids):
            new_values = [x if x <= line_id else x + 1 for x in text_ids]
            mapping[id] = json.dumps(new_values)

    with sqlite3.connect(db_path) as db:
        db.execute(
            "create temporary table temp_mapping (id integer, new_text_ids text)"
        )
        db.executemany(
            "insert into temp_mapping (id, new_text_ids) values (?, ?);",
            mapping.items(),
        )
        db.execute(
            f"""update {table_name}
            set text_ids = (select new_text_ids from temp_mapping where temp_mapping.id = {table_name}.id)
            where id IN (select id from temp_mapping)
        """
        )
        db.execute("drop table temp_mapping")


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


def get_splitted_dict(items):
    """Get splitted sentences as dict"""
    res = dict()
    for item in items:
        res[item[0]] = item[1]
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

    for (data, texts) in zip(index_items, get_doc_page(db_path, [x[0][0][0] for x in index_items])
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
    return (
        res,
        get_proxy_dict(splitted_from),
        get_proxy_dict(splitted_to),
    )


def get_doc_items_with_splitted(index_items, db_path):
    """Get document items by ids"""
    res = []

    from_ids, to_ids = set(), set()
    for item in index_items:
        from_ids.update(json.loads(item[0][0][1]))
        to_ids.update(json.loads(item[0][0][3]))

    splitted_from = get_splitted_from_by_id(db_path, from_ids)
    splitted_to = get_splitted_to_by_id(db_path, to_ids)

    for (data, texts) in zip(index_items, get_doc_page(db_path, [x[0][0][0] for x in index_items])
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
    return (
        res,
        get_proxy_dict(splitted_from),
        get_proxy_dict(splitted_to),
        get_splitted_dict(splitted_from),
        get_splitted_dict(splitted_to),
    )


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


def get_processing_from_text_ids_non_empty(db_path):
    """Get text_ids from processing_from"""
    with sqlite3.connect(db_path) as db:
        res = db.execute(
            f"select f.id, f.text_ids from processing_from f where f.text_ids<>'[]' order by f.id"
        ).fetchall()
    return [x for x in res]


def get_processing_to_text_ids_non_empty(db_path):
    """Get text_ids from processing_to"""
    with sqlite3.connect(db_path) as db:
        res = db.execute(
            f"select t.id, t.text_ids from processing_to t where t.text_ids<>'[]' order by t.id"
        ).fetchall()
    return [x for x in res]


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
    return float(res[0])


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


def get_string_lens(dic, ids):
    """Get lengths of strings"""
    return [len(dic[id]) for id in ids]


def lazy_property(func):
    """Lazy initialization attribute"""
    attr_name = "_lazy_" + func.__name__

    @property
    def _lazy_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)

    return _lazy_property
