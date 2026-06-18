import gc
import json
import logging
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
import os
from lingtrain_aligner import constants as con
import numpy as np


INFO_KEY_NAME = "name"
INFO_KEY_CREATED_AT = "created_at"
INFO_KEY_LAST_EDITED_AT = "last_edited_at"
INFO_KEY_CONTENT_VERSION = "app_content_version"
# Multilingual (.ltm) info keys: ``format`` marks the file as a multibook and
# ``source_lang`` records the structural reference edition (also flagged
# ``is_source`` in the ``languages`` table).
INFO_KEY_FORMAT = "format"
INFO_KEY_SOURCE_LANG = "source_lang"
LTM_FORMAT = "ltm"
INFO_KEY_EMBEDDING_MODEL = "embedding_model"
INFO_KEY_EMBEDDING_MODEL_NAME = "embedding_model_name"
INFO_KEY_EMBEDDING_MODEL_RESOLVED_NAME = "embedding_model_resolved_name"
INFO_KEY_EMBEDDING_INFERENCE = "embedding_model_inference"


def _utc_now_iso():
    return datetime.now(timezone.utc).isoformat()


def _ensure_info_key_index(db):
    """Deduplicate info rows once and enforce key uniqueness for future upserts."""
    index_names = {
        row[1] for row in db.execute("PRAGMA index_list(info)").fetchall()
    }
    if "idx_info_key_unique" in index_names:
        return

    db.execute(
        """
        DELETE FROM info
        WHERE id NOT IN (
            SELECT MAX(id)
            FROM info
            GROUP BY key
        )
        """
    )
    db.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_info_key_unique ON info(key)")


def _ensure_processing_batch_indexes(db):
    """Create indexes used by per-batch processing and compaction paths."""
    tables = {
        row[0]
        for row in db.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    for table_name in ("processing_from", "processing_to"):
        if table_name in tables:
            db.execute(
                f"CREATE INDEX IF NOT EXISTS idx_{table_name}_batch_id_id "
                f"ON {table_name}(batch_id, id)"
            )
    if "history" in tables:
        db.execute(
            "CREATE INDEX IF NOT EXISTS idx_history_batch_id ON history(batch_id)"
        )


def get_info_value_conn(db, key):
    row = db.execute(
        "SELECT val FROM info WHERE key = ? LIMIT 1",
        (key,),
    ).fetchone()
    return row[0] if row else None


def get_info_value(db_path, key):
    with sqlite3.connect(db_path) as db:
        return get_info_value_conn(db, key)


def set_info_value_conn(db, key, val):
    _ensure_info_key_index(db)
    if val is None:
        db.execute("DELETE FROM info WHERE key = ?", (key,))
        return

    db.execute(
        """
        INSERT INTO info(key, val)
        VALUES(?, ?)
        ON CONFLICT(key) DO UPDATE SET val = excluded.val
        """,
        (key, str(val)),
    )


def set_info_value(db_path, key, val):
    with sqlite3.connect(db_path) as db:
        set_info_value_conn(db, key, val)


def get_created_at(db_path):
    return get_info_value(db_path, INFO_KEY_CREATED_AT)


def get_last_edited_at(db_path):
    return get_info_value(db_path, INFO_KEY_LAST_EDITED_AT)


def get_content_version_conn(db):
    value = get_info_value_conn(db, INFO_KEY_CONTENT_VERSION)
    if value is None:
        return None
    try:
        version = int(value)
    except (TypeError, ValueError):
        return None
    return version if version >= 1 else None


def get_content_version(db_path):
    with sqlite3.connect(db_path) as db:
        return get_content_version_conn(db)


def ensure_content_version_conn(db, default=1):
    version = get_content_version_conn(db)
    if version is not None:
        return version
    try:
        version = int(default)
    except (TypeError, ValueError):
        version = 1
    version = max(version, 1)
    set_info_value_conn(db, INFO_KEY_CONTENT_VERSION, version)
    return version


def bump_content_version_conn(db):
    version = ensure_content_version_conn(db) + 1
    set_info_value_conn(db, INFO_KEY_CONTENT_VERSION, version)
    return version


def touch_alignment_change_conn(db, ts=None, bump_version=True):
    version = (
        bump_content_version_conn(db)
        if bump_version
        else ensure_content_version_conn(db)
    )
    set_info_value_conn(db, INFO_KEY_LAST_EDITED_AT, ts or _utc_now_iso())
    return version


def touch_last_edited_conn(db, ts=None):
    set_info_value_conn(db, INFO_KEY_LAST_EDITED_AT, ts or _utc_now_iso())


def touch_last_edited(db_path, ts=None):
    with sqlite3.connect(db_path) as db:
        touch_last_edited_conn(db, ts=ts)


def set_embedding_metadata_conn(
    db,
    *,
    model_key=None,
    model_display_name=None,
    model_name=None,
    inference_type=None,
):
    if model_key:
        set_info_value_conn(db, INFO_KEY_EMBEDDING_MODEL, model_key)
    if model_display_name:
        set_info_value_conn(db, INFO_KEY_EMBEDDING_MODEL_NAME, model_display_name)
    if model_name:
        set_info_value_conn(db, INFO_KEY_EMBEDDING_MODEL_RESOLVED_NAME, model_name)
    if inference_type:
        set_info_value_conn(db, INFO_KEY_EMBEDDING_INFERENCE, inference_type)


def set_embedding_metadata(
    db_path,
    *,
    model_key=None,
    model_display_name=None,
    model_name=None,
    inference_type=None,
):
    with sqlite3.connect(db_path) as db:
        set_embedding_metadata_conn(
            db,
            model_key=model_key,
            model_display_name=model_display_name,
            model_name=model_name,
            inference_type=inference_type,
        )


def clear_embedding_metadata_conn(db):
    for key in (
        INFO_KEY_EMBEDDING_MODEL,
        INFO_KEY_EMBEDDING_MODEL_NAME,
        INFO_KEY_EMBEDDING_MODEL_RESOLVED_NAME,
        INFO_KEY_EMBEDDING_INFERENCE,
    ):
        set_info_value_conn(db, key, None)


def clear_embedding_metadata(db_path):
    with sqlite3.connect(db_path) as db:
        clear_embedding_metadata_conn(db)


def _infer_alignment_timestamp(db, agg):
    candidates = []
    for table_name in ("history", "batches"):
        try:
            row = db.execute(
                f"SELECT {agg}(insert_ts) FROM {table_name} WHERE insert_ts IS NOT NULL"
            ).fetchone()
        except sqlite3.OperationalError:
            row = None
        if row and row[0]:
            candidates.append(row[0])
    if not candidates:
        return None
    return min(candidates) if agg == "MIN" else max(candidates)


def _infer_model_metadata(db):
    for table_name in ("splitted_from", "splitted_to"):
        try:
            row = db.execute(
                f"""
                SELECT model, inference
                FROM {table_name}
                WHERE model IS NOT NULL OR inference IS NOT NULL
                LIMIT 1
                """
            ).fetchone()
        except sqlite3.OperationalError:
            row = None
        if row and (row[0] or row[1]):
            return row[0], row[1]
    return None, None


def _embedding_to_blob(vec):
    """Serialize a numpy array to a float32 binary blob."""
    return np.asarray(vec, dtype=np.float32).tobytes()


def _blob_to_embedding(blob):
    """Deserialize a binary blob back to a numpy float32 array."""
    return np.frombuffer(blob, dtype=np.float32).copy()


def _is_json_embedding(data):
    """Check whether stored embedding data is JSON text (legacy) vs binary blob."""
    return isinstance(data, str)


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
                embedding blob,
                proxy_embedding blob,
                model text,
                inference text,
                verse integer DEFAULT 0
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
                embedding blob,
                proxy_embedding blob,
                model text,
                inference text,
                verse integer DEFAULT 0
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
        _ensure_info_key_index(db)
        _ensure_processing_batch_indexes(db)
        created_at = _utc_now_iso()
        set_info_value_conn(db, INFO_KEY_CREATED_AT, created_at)
        set_info_value_conn(db, INFO_KEY_LAST_EDITED_AT, created_at)
        set_info_value_conn(db, INFO_KEY_CONTENT_VERSION, 1)
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
    """Fill embeddings in splitted table as binary blobs"""
    if direction == "from":
        table_name = "splitted_from"
    else:
        table_name = "splitted_to"

    blobs = [_embedding_to_blob(x) for x in embeddings]
    with sqlite3.connect(db_path) as db:
        if is_proxy:
            db.executemany(
                f"update {table_name} set proxy_embedding=? where id=?",
                [(blob, lid) for blob, lid in zip(blobs, line_ids)],
            )
        else:
            db.executemany(
                f"update {table_name} set embedding=? where id=?",
                [(blob, lid) for blob, lid in zip(blobs, line_ids)],
            )


def get_embeddings(db_path, direction, line_ids=[], is_proxy=False):
    """Get embeddings from splitted table (auto-detects JSON legacy vs binary blob)"""
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

    def _parse_embedding(data):
        if data is None:
            return None
        if _is_json_embedding(data):
            return np.array(json.loads(data))
        return _blob_to_embedding(data)

    return [(x[0], _parse_embedding(x[1])) for x in res]


def get_doc_index_original(db_path):
    """Get document index"""
    res = []
    try:
        with sqlite3.connect(db_path) as db:
            cur = db.execute("SELECT contents FROM doc_index")
            res = json.loads(cur.fetchone()[0])
    except:
        # logging.warning("can not fetch index db")
        pass
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
        # logging.warning("can not fetch flatten index")
        pass
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
        # logging.warning("can not fetch flatten index")
        pass
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
        # logging.warning("can not fetch flatten index")
        pass
    return res


def compact_batches(db_path):
    """Drop empty doc_index batches and renumber survivors to dense batch ids.

    Conflict resolution can absorb a batch and leave holes in batch numbering.
    The web app and several helper flows assume batch ids are dense from 0.
    """
    with sqlite3.connect(db_path) as db:
        try:
            row = db.execute("SELECT contents FROM doc_index").fetchone()
        except sqlite3.OperationalError:
            return {"mapping": {}, "removed": [], "batch_ids": []}

        if not row or not row[0]:
            return {"mapping": {}, "removed": [], "batch_ids": []}

        index = json.loads(row[0])
        compact_index = []
        mapping = {}
        removed = []

        for old_id, batch in enumerate(index):
            if batch:
                mapping[old_id] = len(compact_index)
                compact_index.append(batch)
            else:
                removed.append(old_id)

        moved = {old_id: new_id for old_id, new_id in mapping.items() if old_id != new_id}
        if not removed and not moved:
            return {
                "mapping": mapping,
                "removed": removed,
                "batch_ids": list(range(len(compact_index))),
            }

        db.execute(
            "insert or replace into doc_index (id, contents) values ((select id from doc_index limit 1), ?)",
            (json.dumps(compact_index),),
        )

        tables = ("processing_from", "processing_to", "batches", "history")
        if removed:
            placeholders = ",".join(["?"] * len(removed))
            for table_name in tables:
                try:
                    db.execute(
                        f"DELETE FROM {table_name} WHERE batch_id IN ({placeholders})",
                        tuple(removed),
                    )
                except sqlite3.OperationalError:
                    continue

        if moved:
            db.execute(
                "CREATE TEMP TABLE temp_batch_map(old_id INTEGER PRIMARY KEY, new_id INTEGER NOT NULL)"
            )
            db.executemany(
                "INSERT INTO temp_batch_map(old_id, new_id) VALUES(?, ?)",
                moved.items(),
            )
            temp_offset = 1000000
            for table_name in tables:
                try:
                    db.execute(
                        f"""
                        UPDATE {table_name}
                        SET batch_id = batch_id + {temp_offset}
                        WHERE batch_id IN (SELECT old_id FROM temp_batch_map)
                        """
                    )
                except sqlite3.OperationalError:
                    continue
            for table_name in tables:
                try:
                    db.execute(
                        f"""
                        UPDATE {table_name}
                        SET batch_id = (
                            SELECT new_id
                            FROM temp_batch_map
                            WHERE old_id = {table_name}.batch_id - {temp_offset}
                        )
                        WHERE batch_id >= {temp_offset}
                        """
                    )
                except sqlite3.OperationalError:
                    continue
            db.execute("DROP TABLE temp_batch_map")

        touch_alignment_change_conn(db)
        return {
            "mapping": mapping,
            "removed": removed,
            "batch_ids": list(range(len(compact_index))),
        }


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


def _has_column(db, table, column):
    """True if ``column`` exists on ``table`` (open connection ``db``)."""
    return any(
        c[1] == column for c in db.execute(f"PRAGMA table_info({table})").fetchall()
    )


def get_splitted_from_by_id(db_path, ids):
    """Get lines from splitted_from by ids"""
    res = []
    with sqlite3.connect(db_path) as db:
        # Pre-7.4 alignment DBs lack the `verse` column. `coalesce` only rescues a
        # NULL value, not a MISSING column, so select `verse` only when it exists
        # and fall back to 0 (prose) otherwise. Keeps older artifacts readable
        # without forcing a migration first. See migrate_document_db (schema 7.4).
        verse_expr = "coalesce(f.verse, 0)" if _has_column(db, "splitted_from", "verse") else "0"
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
            verse,
        ) in db.execute(
            f'select f.id, f.text, f.proxy_text, f.exclude, f.paragraph, f.h1, f.h2, f.h3, f.h4, f.h5, f.divider, {verse_expr} from splitted_from f where f.id in ({",".join([str(x) for x in ids])})'
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
                    verse,
                )
            )
    return res


def get_splitted_to_by_id(db_path, ids):
    """Get lines from splitted_to by ids"""
    res = []
    with sqlite3.connect(db_path) as db:
        # See get_splitted_from_by_id: select `verse` only when present so pre-7.4
        # artifacts (no `verse` column) stay readable without a migration.
        verse_expr = "coalesce(t.verse, 0)" if _has_column(db, "splitted_to", "verse") else "0"
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
            verse,
        ) in db.execute(
            f'select t.id, t.text, t.proxy_text, t.exclude, t.paragraph, t.h1, t.h2, t.h3, t.h4, t.h5, t.divider, {verse_expr} from splitted_to t where t.id in ({",".join([str(x) for x in ids])})'
        ):
            res.append(
                (id, text_to, proxy_to, exclude, paragraph, h1, h2, h3, h4, h5, divider, verse)
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
        touch_alignment_change_conn(db)


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
        touch_alignment_change_conn(db)


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
        touch_alignment_change_conn(db)


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
        touch_alignment_change_conn(db)


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
    """Get paragraphs info as dict.

    Value tuple: (paragraph, h1, h2, h3, h4, h5, divider, verse). ``verse`` is the
    poetry stanza index (0 = prose); kept last so existing positional consumers of
    indices 0..6 are unaffected.
    """
    res = dict()
    for item in items:
        verse = item[11] if len(item) > 11 else 0
        res[item[0]] = (item[4], item[5], item[6], item[7], item[8], item[9], item[10], verse)
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
        touch_alignment_change_conn(db)
    return


def delete_meta(db_path, mark_id):
    """Mark meta as deleted"""
    with sqlite3.connect(db_path) as db:
        db.execute(f"update meta set deleted = 1 where id=(?)", (mark_id,))
        touch_alignment_change_conn(db)
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
        touch_alignment_change_conn(db)
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


def migrate_document_db(db_path):
    """Migrate alignment DB to the current schema version.

    Safe to call on any version: checks current version before running.
    Idempotent: will not fail if columns already exist.

    Migration steps applied:
      7.1 -> 7.2: add model TEXT and inference TEXT to splitted_from and splitted_to.
      7.2 -> 7.3: add reliable info-key upserts and backfill DB-level metadata.
      7.3 -> 7.4: add `verse` integer column (poetry stanza index; 0 = prose) to
                  splitted_from and splitted_to.
    """
    with sqlite3.connect(db_path) as db:
        current_version = float(db.execute("SELECT version FROM version").fetchone()[0])

        if current_version < 7.2:
            # Add model + inference columns to splitted_from
            cols_from = [
                col[1]
                for col in db.execute("PRAGMA table_info(splitted_from)").fetchall()
            ]
            if "model" not in cols_from:
                db.execute("ALTER TABLE splitted_from ADD COLUMN model TEXT")
            if "inference" not in cols_from:
                db.execute("ALTER TABLE splitted_from ADD COLUMN inference TEXT")

            # Add model + inference columns to splitted_to
            cols_to = [
                col[1]
                for col in db.execute("PRAGMA table_info(splitted_to)").fetchall()
            ]
            if "model" not in cols_to:
                db.execute("ALTER TABLE splitted_to ADD COLUMN model TEXT")
            if "inference" not in cols_to:
                db.execute("ALTER TABLE splitted_to ADD COLUMN inference TEXT")

            # Bump version
            db.execute("UPDATE version SET version = ?", ("7.2",))
            current_version = 7.2

        if current_version < 7.3 or get_content_version_conn(db) is None:
            _ensure_info_key_index(db)
            created_at = (
                get_info_value_conn(db, INFO_KEY_CREATED_AT)
                or _infer_alignment_timestamp(db, "MIN")
                or _utc_now_iso()
            )
            last_edited_at = (
                get_info_value_conn(db, INFO_KEY_LAST_EDITED_AT)
                or _infer_alignment_timestamp(db, "MAX")
                or created_at
            )
            model_name, inference_type = _infer_model_metadata(db)
            set_info_value_conn(db, INFO_KEY_CREATED_AT, created_at)
            set_info_value_conn(db, INFO_KEY_LAST_EDITED_AT, last_edited_at)
            ensure_content_version_conn(db, default=1)
            if model_name and not get_info_value_conn(db, INFO_KEY_EMBEDDING_MODEL_NAME):
                set_info_value_conn(db, INFO_KEY_EMBEDDING_MODEL_NAME, model_name)
            if model_name and not get_info_value_conn(db, INFO_KEY_EMBEDDING_MODEL_RESOLVED_NAME):
                set_info_value_conn(db, INFO_KEY_EMBEDDING_MODEL_RESOLVED_NAME, model_name)
            if inference_type and not get_info_value_conn(db, INFO_KEY_EMBEDDING_INFERENCE):
                set_info_value_conn(db, INFO_KEY_EMBEDDING_INFERENCE, inference_type)
            db.execute("UPDATE version SET version = ?", (con.DB_VERSION,))

        # 7.3 -> 7.4: add the `verse` column (poetry stanza index). Column-presence
        # gated rather than version-gated so it is idempotent and also repairs a DB
        # whose version was already bumped above. 0 = prose/non-verse (default).
        cols_from = [c[1] for c in db.execute("PRAGMA table_info(splitted_from)").fetchall()]
        if "verse" not in cols_from:
            db.execute("ALTER TABLE splitted_from ADD COLUMN verse integer DEFAULT 0")
        cols_to = [c[1] for c in db.execute("PRAGMA table_info(splitted_to)").fetchall()]
        if "verse" not in cols_to:
            db.execute("ALTER TABLE splitted_to ADD COLUMN verse integer DEFAULT 0")
        db.execute("UPDATE version SET version = ?", (con.DB_VERSION,))

        _ensure_processing_batch_indexes(db)


def set_provenance(db_path, direction, line_ids, model_name, inference_type):
    """Write model name and inference type onto splitted rows (DB >= 7.2 required)."""
    if not line_ids:
        return
    if direction == "from":
        table_name = "splitted_from"
    else:
        table_name = "splitted_to"
    with sqlite3.connect(db_path) as db:
        db.executemany(
            f"UPDATE {table_name} SET model=?, inference=? WHERE id=?",
            [(model_name, inference_type, lid) for lid in line_ids],
        )
        set_info_value_conn(db, INFO_KEY_EMBEDDING_MODEL, None)
        set_embedding_metadata_conn(
            db,
            model_display_name=model_name,
            model_name=model_name,
            inference_type=inference_type,
        )
        touch_alignment_change_conn(db)


def check_model_mismatch(db_path, expected_model):
    """Raise ValueError if DB already contains embeddings from a different model."""
    with sqlite3.connect(db_path) as db:
        row = db.execute(
            "SELECT model FROM splitted_from WHERE model IS NOT NULL LIMIT 1"
        ).fetchone()
    if row is not None and row[0] != expected_model:
        raise ValueError(
            f"Model mismatch: DB contains embeddings from '{row[0]}', "
            f"but resolver returned '{expected_model}'. "
            f"Clear embeddings or use the same model."
        )


def set_name(db_path, name):
    """Update alignment name"""
    with sqlite3.connect(db_path) as db:
        set_info_value_conn(db, INFO_KEY_NAME, name)
        touch_alignment_change_conn(db)


def get_name(db_path):
    """Get alignment name"""
    return get_info_value(db_path, INFO_KEY_NAME) or ""


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


# ---------------------------------------------------------------------------
# Multilingual (.ltm) format — schema authority + accessors
# ---------------------------------------------------------------------------
# A .ltm stores N strictly-1:1:N editions in ONE render-only SQLite file. Unlike
# the bilingual .lt it has NO processing_/doc_index/batches/embeddings: under
# controlled (smart-translator) translation every edition shares the same body
# structure, so the alignment is implicit in the (lang, paragraph, sentence)
# coordinate. The plain .lt schema (init_document_db) is untouched.


def init_multi_db(db_path):
    """Initialise a multilingual (.ltm) book with the multibook table structure.

    Tables:
      * ``version`` — ``LTM_VERSION`` (distinct namespace from ``.lt`` DB_VERSION).
      * ``info`` — ``format='ltm'``, ``source_lang``, created/last_edited/content
        version (key-unique, same upsert discipline as ``.lt``).
      * ``languages(lang, ord, is_source, added_at)`` — one row per edition,
        ``is_source`` flags the structural reference; generalises ``.lt``
        ``languages(key='from'|'to')``.
      * ``structure(paragraph, kind, sentence_count, verse)`` — the canonical body
        skeleton shared by every edition (one row per body line).
      * ``splitted(lang, paragraph, sentence, id, text, …)`` — per-edition sentence
        rows; real key is the ``(lang, paragraph, sentence)`` coordinate (no PK,
        mirroring ``.lt`` ``splitted_*``).
      * ``meta(lang, key, …)`` — per-edition marks with BARE keys (``'title'`` not
        ``'title_from'``).
      * ``files(lang, …)``, ``history`` — provenance / audit.
    """
    if os.path.isfile(db_path):
        # Windows: a lingering sqlite handle from a GC cycle keeps a file lock;
        # collect before removing (mirrors aligner.trivial_alignment).
        gc.collect()
        os.remove(db_path)
    db = sqlite3.connect(db_path)
    try:
        db.execute("create table version(id integer primary key, version text)")
        db.execute("create table info(id integer primary key, key text, val text)")
        db.execute(
            "create table languages(id integer primary key, lang text, ord integer, "
            "is_source integer default 0, added_at text)"
        )
        db.execute("create unique index ux_languages_lang on languages(lang)")
        db.execute(
            "create table structure(paragraph integer primary key, kind text, "
            "sentence_count integer, verse integer default 0)"
        )
        db.execute(
            "create table splitted(lang text, paragraph integer, sentence integer, "
            "id integer, text text, proxy_text text default '', exclude integer default 0, "
            "verse integer default 0)"
        )
        db.execute(
            "create unique index ux_splitted_coord on splitted(lang, paragraph, sentence)"
        )
        db.execute("create index ix_splitted_lang_par on splitted(lang, paragraph)")
        db.execute(
            'create table meta(id integer primary key, lang text, key text, val text, '
            'occurence integer, par_id integer, deleted integer DEFAULT 0, comment text DEFAULT "")'
        )
        db.execute("create index ix_meta_lang_key on meta(lang, key)")
        db.execute(
            "create table files(id integer primary key, lang text, name text, guid text, added_at text)"
        )
        db.execute(
            "create table history(id integer primary key, operation text, lang text, "
            "insert_ts text, parameters text)"
        )
        _ensure_info_key_index(db)
        created_at = _utc_now_iso()
        set_info_value_conn(db, INFO_KEY_FORMAT, LTM_FORMAT)
        set_info_value_conn(db, INFO_KEY_CREATED_AT, created_at)
        set_info_value_conn(db, INFO_KEY_LAST_EDITED_AT, created_at)
        set_info_value_conn(db, INFO_KEY_CONTENT_VERSION, 1)
        db.execute("insert into version(version) values (?)", (con.LTM_VERSION,))
        db.commit()
    finally:
        # sqlite3's `with` only manages the transaction; close explicitly so a
        # re-run / add_language can overwrite the file on Windows.
        db.close()


def is_ltm(db_path):
    """True when ``db_path`` is a multilingual (.ltm) file. Probes
    ``info.format == 'ltm'`` first, then falls back to the presence of the
    ``structure`` table. The routing primitive for ``.lt`` vs ``.ltm``."""
    try:
        with sqlite3.connect(db_path) as db:
            if get_info_value_conn(db, INFO_KEY_FORMAT) == LTM_FORMAT:
                return True
            row = db.execute(
                "select name from sqlite_master where type='table' and name='structure'"
            ).fetchone()
            return row is not None
    except sqlite3.Error:
        return False


def get_ltm_languages(db_path):
    """Editions as ``[{lang, ord, is_source, added_at}]`` ordered by ``ord``.
    Generalises ``get_lang_codes`` to N languages."""
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            "select lang, ord, is_source, added_at from languages order by ord"
        ).fetchall()
    return [
        {"lang": r[0], "ord": r[1], "is_source": bool(r[2]), "added_at": r[3]}
        for r in rows
    ]


def get_ltm_lang_codes(db_path):
    """Ordered list of edition language codes (convenience over get_ltm_languages)."""
    return [d["lang"] for d in get_ltm_languages(db_path)]


def get_ltm_source_lang(db_path):
    """The structural reference edition's language code (``is_source``), falling
    back to the ``info.source_lang`` key."""
    with sqlite3.connect(db_path) as db:
        row = db.execute(
            "select lang from languages where is_source = 1 order by ord limit 1"
        ).fetchone()
        if row:
            return row[0]
        return get_info_value_conn(db, INFO_KEY_SOURCE_LANG)


def get_ltm_structure(db_path):
    """Canonical body skeleton as ``[{paragraph, kind, sentence_count, verse}]``
    ordered by ``paragraph``. ``kind`` is ``'text'``, ``'verse'`` or a structural
    mark (``'h1'..'h5'``/``'divider'``/``'qtext'``/``'qname'``/``'image'``)."""
    with sqlite3.connect(db_path) as db:
        rows = db.execute(
            "select paragraph, kind, sentence_count, verse from structure order by paragraph"
        ).fetchall()
    return [
        {"paragraph": r[0], "kind": r[1], "sentence_count": r[2], "verse": r[3]}
        for r in rows
    ]


def get_ltm_meta_for_lang(db_path, lang):
    """Per-edition meta as ``{mark: [(val, occurence, par_id, id)]}`` with BARE
    keys (no ``_from``/``_to`` suffix to strip). The ``.ltm`` parallel of
    ``reader.prepare_meta`` over ``get_meta_dict``."""
    res = defaultdict(list)
    with sqlite3.connect(db_path) as db:
        for key, val, occurence, par_id, id in db.execute(
            "select key, val, occurence, par_id, id from meta "
            "where lang = ? and deleted = 0 order by par_id, occurence",
            (lang,),
        ):
            res[key].append((val, occurence, par_id, id))
    return res


def touch_ltm_change_conn(db, ts=None):
    """Bump ``app_content_version`` + ``last_edited_at`` on a ``.ltm`` (cache
    invalidation on add-language / replace re-upload). Format-agnostic; reuses the
    ``.lt`` info-table machinery."""
    return touch_alignment_change_conn(db, ts=ts, bump_version=True)
