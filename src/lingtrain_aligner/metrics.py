import json
from lingtrain_aligner import helper


def chain_score(db_path, mode="to"):
    """Get chain score. Mode: ['to', 'both']"""
    index = helper.get_clear_flatten_doc_index(db_path)

    # get flatten chains
    chain_from, chain_to = [], []
    for i, ix in enumerate(index):
        from_ids, to_ids = json.loads(ix[1]), json.loads(ix[3])
        chain_from.extend(from_ids)
        chain_to.extend(to_ids)

    x_len, y_len = len(chain_from), len(chain_to)

    breaks_from = get_breaks_amount(chain_from)
    breaks_to = get_breaks_amount(chain_to)

    if mode == "to":
        return lt_score_first(breaks_to, y_len)

    return lt_score_second(breaks_from, breaks_to, x_len, y_len)


def get_breaks_amount(chain):
    """Get amount of breaks"""
    if not chain:
        return 0
    res, chain_min = 0, min(chain)
    if chain[0] != chain_min:
        res += 1
    for i, c in enumerate(chain[:-1]):
        if chain[i + 1] != c + 1:
            res += 1
    return res


def lt_score_first(breaks_to, y_len):
    """Calculate score only for second chain assuming that first is already straight"""
    res = 1 - (breaks_to / y_len)
    return res


def lt_score_second(breaks_from, breaks_to, x_len, y_len):
    """Calculate score for both chains. Should be used after the conflicts resolving."""
    res = 1 - ((breaks_from + breaks_to) / (x_len + y_len))
    return res