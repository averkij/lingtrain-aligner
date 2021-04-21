"""Visualization helper"""

import json
import os

import numpy as np
from lingtrain_aligner import helper
from matplotlib import pyplot as plt


def visualize_alignment_by_db(db_path, output_path, lang_name_from="ru", lang_name_to="de", batch_size=0, size=(260, 260), batch_ids=[], plt_show=False):
    """Visualize alignment using ids from the database"""
    if batch_size > 0:
        index = helper.get_clear_flatten_doc_index(db_path)
        xs, ys = [], []
        for i, ix in enumerate(index):
            from_ids, to_ids = json.loads(ix[1]), json.loads(ix[3])
            for from_id in from_ids:
                for to_id in to_ids:
                    xs.append(from_id)
                    ys.append(to_id)

        x_max, y_max = max(xs), max(ys)
        is_last = x_max % batch_size > 0
        total_batches = x_max//batch_size + (1 if is_last else 0)
        batches = [[[], []] for _ in range(total_batches)]

        for x, y in zip(xs, ys):
            batch_id = (x-1) // batch_size
            batches[batch_id][0].append(x)
            batches[batch_id][1].append(y)
    else:
        index = helper.get_doc_index_original(db_path)
        batches = [[[], []] for _ in range(len(index))]
        for i, batch in enumerate(index):
            xs, ys = [], []
            for ix in batch:
                from_ids, to_ids = json.loads(ix[1]), json.loads(ix[3])
                for from_id in from_ids:
                    for to_id in to_ids:
                        xs.append(from_id)
                        ys.append(to_id)
            for x, y in zip(xs, ys):
                batches[i][0].append(x)
                batches[i][1].append(y)

    for i, batch in enumerate(batches):
        if i in batch_ids or len(batch_ids) == 0:
            x_min, y_min = min(batch[0]), min(batch[1])
            x_max, y_max = max(batch[0]), max(batch[1])
            align_matrix = np.zeros((x_max-x_min, y_max-y_min))
            for x, y in zip(batch[0], batch[1]):
                align_matrix[x-x_min - 1, y-y_min - 1] = 1
            save_pic(align_matrix, lang_name_to, lang_name_from,
                     output_path, batch_number=i, size=size, plt_show=plt_show)


def save_pic(align_matrix, lang_name_to, lang_name_from, output_path, batch_number, size=(260, 260), plt_show=False):
    """Save the resulted picture"""
    output = "{0}_{1}{2}".format(os.path.splitext(output_path)[
                                 0], batch_number, os.path.splitext(output_path)[1])

    my_dpi = 100
    plt.figure(figsize=(size[0]/my_dpi, size[1]/my_dpi), dpi=my_dpi)

    plt.imshow(align_matrix, cmap='Greens', interpolation='nearest')
    plt.xlabel(lang_name_to, fontsize=12, labelpad=-18)
    plt.ylabel(lang_name_from, fontsize=12, labelpad=-18)
    plt.tick_params(axis='both', which='both', bottom=False, top=False,
                    labelbottom=False, right=False, left=False, labelleft=False)
    plt.savefig(output, dpi=my_dpi)
    # plt.savefig(output, bbox_inches="tight", pad_inches=0, dpi=my_dpi)

    if plt_show:
        plt.show()
