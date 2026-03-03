"""Visualization helper"""

import json
import logging
import os

import numpy as np
from lingtrain_aligner import helper
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

_COLOR_PRIMARY = "#2563eb"
_COLOR_PRIMARY_SUBTLE = "#eff6ff"
_COLOR_TEXT_MUTED = "#6b7280"
_COLOR_BG_SURFACE = "#ffffff"
_COLOR_BORDER = "#e2e8f0"
_COLOR_ERROR = "#dc2626"
_FONT_FAMILY = ["Arial", "sans-serif"]

_CMAP_ALIGNMENT = LinearSegmentedColormap.from_list(
    "lingtrain_blue",
    [_COLOR_BG_SURFACE, _COLOR_PRIMARY_SUBTLE, _COLOR_PRIMARY],
    N=256,
)


def visualize_alignment_by_db(
    db_path,
    output_path,
    lang_name_from="ru",
    lang_name_to="de",
    batch_size=0,
    size=(260, 300),
    batch_ids=[],
    plt_show=False,
    transparent_bg=False,
    show_info=False,
    show_regression=False,
):
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
        total_batches = x_max // batch_size + (1 if is_last else 0)
        batches = [[[], []] for _ in range(total_batches)]

        for x, y in zip(xs, ys):
            batch_id = (x - 1) // batch_size
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

    if len(batch_ids) == 1 and batch_ids[0] == -1:
        batch_ids = []

    for i, batch in enumerate(batches):
        if i in batch_ids or len(batch_ids) == 0:
            if not batch[0] or not batch[1]:
                continue
            y_min, x_min = min(batch[0]), min(batch[1])
            y_max, x_max = max(batch[0]), max(batch[1])
            align_matrix = np.zeros((y_max - y_min, x_max - x_min))
            try:
                for y, x in zip(batch[0], batch[1]):
                    align_matrix[y - y_min - 1, x - x_min - 1] = 1
                shift, window = None, None
                if show_info:
                    shift, window = helper.get_batch_info(db_path, i)
                save_pic(
                    align_matrix,
                    lang_name_to,
                    lang_name_from,
                    output_path,
                    batch_number=i,
                    interval_x=(x_min, x_max),
                    interval_y=(y_min, y_max),
                    size=size,
                    plt_show=plt_show,
                    transparent=transparent_bg,
                    shift=shift,
                    window=window,
                    show_info=show_info,
                    show_regression=show_regression,
                )
            except Exception as e:
                logging.error(e, exc_info=True)


def save_pic(
    align_matrix,
    lang_name_to,
    lang_name_from,
    output_path,
    batch_number,
    interval_x,
    interval_y,
    size=(260, 260),
    plt_show=False,
    transparent=False,
    shift=None,
    window=None,
    show_info=False,
    show_regression=False,
):
    """Save the resulted picture"""
    output = "{0}_{1}{2}".format(
        os.path.splitext(output_path)[0], batch_number, os.path.splitext(output_path)[1]
    )

    dpi = 150
    fig, ax = plt.subplots(
        figsize=(size[0] / dpi, size[1] / dpi),
        dpi=dpi,
    )

    batch_info = restore_batch_info(align_matrix)
    x = np.array(batch_info[1])
    y = np.array(batch_info[0])

    # ── Alignment matrix ─────────────────────────────────────────────────
    ax.imshow(align_matrix, cmap=_CMAP_ALIGNMENT, interpolation="nearest", aspect="auto")

    if show_regression:
        mse = None
        try:
            x_scaler, y_scaler = StandardScaler(), StandardScaler()
            x_train = x_scaler.fit_transform(x[..., None])
            y_train = y_scaler.fit_transform(y[..., None])
            model = HuberRegressor(alpha=0.0, epsilon=1)
            model.fit(x_train, y_train.ravel())
            test_x = np.array([0, len(align_matrix[0])])
            reg_line = y_scaler.inverse_transform(
                model.predict(x_scaler.transform(test_x[..., None]))
            )
            preds = y_scaler.inverse_transform(
                model.predict(x_scaler.transform(x[..., None]))
            )
            ax.plot(test_x, reg_line, color=_COLOR_ERROR, linewidth=0.8, alpha=0.7)
            mse = mean_squared_error(preds, y)
        except Exception as e:
            logging.error(e, exc_info=True)
            coefs, res, rank, s_val, cond = np.polyfit(x, y, 1, full=True)
            m, b = coefs[0], coefs[1]
            preds = m * x + b
            ax.plot(x, preds, color=_COLOR_PRIMARY, linewidth=0.8, alpha=0.7)
            mse = mean_squared_error(preds, y)

    # ── Axis labels ──────────────────────────────────────────────────────
    label_props = dict(fontsize=8, fontfamily=_FONT_FAMILY, color=_COLOR_TEXT_MUTED)
    ax.set_xlabel(lang_name_to, labelpad=2, **label_props)
    ax.set_ylabel(lang_name_from, labelpad=2, **label_props)

    # ── Remove ticks, add subtle border ──────────────────────────────────
    ax.tick_params(
        axis="both", which="both",
        bottom=False, top=False, labelbottom=False,
        right=False, left=False, labelleft=False,
    )
    for spine in ax.spines.values():
        spine.set_color(_COLOR_BORDER)
        spine.set_linewidth(0.6)

    # ── Info text below the chart ────────────────────────────────────────
    if show_info and shift is not None and window is not None:
        info_props = dict(fontsize=6, fontfamily=_FONT_FAMILY, color=_COLOR_TEXT_MUTED)
        if show_regression and mse is not None:
            ax.text(0.0, -0.10, f"s={shift}  w={window}  mse={mse:.2f}",
                    transform=ax.transAxes, **info_props)
        else:
            ax.text(0.0, -0.10, f"s={shift}  w={window}",
                    transform=ax.transAxes, **info_props)
        ax.text(
            0.0, -0.18,
            f"{lang_name_to} {interval_x[0]}\u2013{interval_x[1]}  |  "
            f"{lang_name_from} {interval_y[0]}\u2013{interval_y[1]}",
            transform=ax.transAxes, **info_props,
        )

    # ── Save ─────────────────────────────────────────────────────────────
    fig.tight_layout(pad=0.4)
    fig.savefig(output, dpi=dpi, transparent=transparent)
    if plt_show:
        plt.show()
    plt.close(fig)


def restore_batch_info(m):
    x, y = [], []
    for i, _ in enumerate(m):
        for j, val in enumerate(m[i]):
            if val == 1:
                x.append(j + 1)
                y.append(i + 1)
    return y, x
