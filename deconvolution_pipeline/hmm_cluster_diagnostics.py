"""
hmm_cluster_diagnostics.py

Диагностика качества разделения пептидов на 3 ветки через кластеризацию.
"""

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
from itertools import permutations
from pomegranate import DiscreteDistributionCycle
import logomaker
import seaborn as sns

from peptides_utils import get_frequencies
from hmm_diversity_split import cluster_peptides_to_branches, encode_peptides_positional

AA_LIST = sorted("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
N_AA = len(AA_LIST)


def _get_anchor_state_names(model, anchor_state_names=None):
    if anchor_state_names is not None:
        return anchor_state_names
    emitting = [
        s for s in model.states
        if s.distribution is not None
        and type(s.distribution) is not DiscreteDistributionCycle
    ]
    return [emitting[-2].name, emitting[-1].name] if len(emitting) >= 2 else []


def _build_peptide_to_allele(additional_data) -> dict:
    peptide_to_allele = {}
    if isinstance(additional_data, dict):
        frames = list(additional_data.values())
    elif isinstance(additional_data, list):
        frames = []
        for item in additional_data:
            if isinstance(item, pd.DataFrame):
                frames.append(item)
            elif isinstance(item, dict):
                frames.extend(item.values())
    else:
        print(f"[clustering] WARNING: unexpected additional_data type: {type(additional_data)}")
        return peptide_to_allele

    for df in frames:
        if not isinstance(df, pd.DataFrame):
            continue
        if "peptide" not in df.columns or "source_allele" not in df.columns:
            print(f"[clustering] WARNING: missing columns: {df.columns.tolist()}")
            continue
        for pep, allele in zip(df["peptide"], df["source_allele"]):
            peptide_to_allele[str(pep)] = str(allele)

    print(f"  _build_peptide_to_allele: built {len(peptide_to_allele)} peptide->allele mappings")
    return peptide_to_allele


# ─────────────────────────────────────────────────────────────────────────────
# SEQUENCE LOGO
# ─────────────────────────────────────────────────────────────────────────────

def _build_count_matrix_for_peptides(peptides, model, anchor_state_names):
    emitting = [
        s for s in model.states
        if s.distribution is not None
        and type(s.distribution) is not DiscreteDistributionCycle
    ]
    core_length = len(emitting)
    index_values = np.arange(-core_length + 1, 2 * core_length - 1)
    max_index = max(index_values)
    left_part = int((len(index_values) - core_length) / 2)
    amino_acids_list = list("ACDEFGHIKLMNPQRSTVWY")
    index = pd.MultiIndex.from_product([index_values, amino_acids_list])
    df_counts = pd.Series(1, index=index)
    anchor_name_set = set(anchor_state_names)

    for peptide in peptides:
        try:
            _, path = model.viterbi(peptide)
        except Exception:
            continue
        pep_pos = 0
        core_start = None
        for _idx, state in path:
            if state.distribution is None:
                continue
            if state.name in anchor_name_set:
                core_start = pep_pos
                break
            pep_pos += 1
        if core_start is None:
            continue
        write_start = max(-core_start, -left_part)
        write_end = min(len(peptide) - core_start, max_index + 1)
        row_values = np.arange(write_start, write_end)
        pep_subset_start = max(0, core_start - left_part)
        pep_subset_end = min(len(peptide), core_start + max_index) + 1
        column_values = list(peptide[pep_subset_start:pep_subset_end])
        if len(row_values) != len(column_values):
            continue
        idx = pd.MultiIndex.from_arrays([row_values, column_values])
        df_counts.loc[idx] += 1

    return df_counts.unstack(level=-1)


def _draw_logo(ax, peptides, model, anchor_names, title, max_peptides, rng):
    sample = peptides
    if len(sample) > max_peptides:
        idx = rng.choice(len(sample), max_peptides, replace=False)
        sample = [sample[i] for i in idx]
    try:
        count_df = _build_count_matrix_for_peptides(sample, model, anchor_names)
        info = logomaker.transform_matrix(
            count_df,
            from_type="counts",
            to_type="information",
            background=get_frequencies(count_df.shape[0]).set_index(count_df.index),
        )
        total_ic = info.sum(axis=1).sum()
        logomaker.Logo(info, font_name="Arial", flip_below=True, ax=ax)
        ax.set_title(f"{title}\n(n={len(peptides)}, total IC={total_ic:.2f})")
        emitting = [
            s for s in model.states
            if s.distribution is not None
            and type(s.distribution) is not DiscreteDistributionCycle
        ]
        core_length = len(emitting)
        ax.axvline(x=-0.5, color="red", linestyle="--", alpha=0.5)
        ax.axvline(x=core_length - 0.5, color="red", linestyle="--", alpha=0.5)
    except Exception as e:
        ax.text(0.5, 0.5, f"Logo failed:\n{e}", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_title(title)


# ─────────────────────────────────────────────────────────────────────────────
# МЕТРИКА СООТВЕТСТВИЯ КЛАСТЕРОВ И АЛЛЕЛЕЙ
# ─────────────────────────────────────────────────────────────────────────────

def _compute_cluster_allele_stats(all_peptides, labels, peptide_to_allele):
    """
    Работает для любого числа кластеров и аллелей.
    Accuracy считается через Hungarian algorithm (linear_sum_assignment)
    только если n_clusters == n_alleles.
    """
    allele_labels = [peptide_to_allele.get(p, "unknown") for p in all_peptides]
    known_alleles = sorted(set(a for a in allele_labels if a != "unknown"))
    cluster_ids = sorted(set(labels))
    n_clusters = len(cluster_ids)

    # Таблица: строки = кластеры, столбцы = аллели
    rows = []
    for cluster_id in cluster_ids:
        mask = labels == cluster_id
        cluster_alleles = [a for a, m in zip(allele_labels, mask) if m]
        total = len(cluster_alleles)
        row = {"cluster": cluster_id, "total": total}
        for allele in known_alleles:
            cnt = cluster_alleles.count(allele)
            row[allele] = cnt
            row[f"{allele}_%"] = f"{100*cnt/max(total,1):.1f}%"
        rows.append(row)
    stats_df = pd.DataFrame(rows).set_index("cluster")

    # Accuracy через Hungarian algorithm
    known_mask = [a != "unknown" for a in allele_labels]
    known_labels_arr = np.array([l for l, m in zip(labels, known_mask) if m])
    known_allele_arr = [a for a, m in zip(allele_labels, known_mask) if m]
    n_known = len(known_allele_arr)

    accuracy = None
    if n_known > 0 and len(known_alleles) == n_clusters:
        # Строим матрицу совпадений [кластер x аллель]
        allele_to_col = {a: i for i, a in enumerate(known_alleles)}
        cost_matrix = np.zeros((n_clusters, len(known_alleles)), dtype=np.int64)
        for l, a in zip(known_labels_arr, known_allele_arr):
            row_idx = cluster_ids.index(l)
            col_idx = allele_to_col[a]
            cost_matrix[row_idx, col_idx] += 1

        # Максимизируем совпадения через Hungarian (минимизируем отрицательное)
        row_ind, col_ind = linear_sum_assignment(-cost_matrix)
        best_correct = cost_matrix[row_ind, col_ind].sum()
        accuracy = best_correct / n_known

    return stats_df, accuracy, allele_labels, known_alleles


# ─────────────────────────────────────────────────────────────────────────────
# ГЛАВНАЯ ФУНКЦИЯ
# ─────────────────────────────────────────────────────────────────────────────

def visualize_clustering_split(
    per_name_models,
    train_data,
    experiment_params,
    iteration_name,
    additional_data=None,
    anchor_state_names=None,
    n_clusters=3,
    window=2,
    n_init=30,
    max_peptides_for_logo=5000,
):
    """
    Строит диагностический рисунок для N кластеров:

      [A] PCA — раскраска по кластерам KMeans
      [B] PCA — раскраска по реальным аллелям
      [C] Длины пептидов по кластерам
      [D] Длины пептидов по реальным аллелям
      [E..] Sequence logo для каждого кластера (по два в ряд)

    PCA для графика строится на том же пространстве (StandardScaler → PCA full),
    что и само кластерование — никакой двойной редукции.
    """
    model_training_params = experiment_params.model_training_params
    target_lengths = model_training_params.lengths_to_use

    top_n = min(3, len(per_name_models))
    top_model_names = list(per_name_models.keys())[:top_n]

    anchor_names = anchor_state_names if anchor_state_names is not None \
        else ["s007", "s002", "s004", "s008"]
    print(f"[clustering] Anchor states: {anchor_names}")

    # ── Собираем все пептиды ──────────────────────────────────────────────
    all_peptides = []
    seen = set()
    for allele_data in train_data.values():
        for split_data in allele_data.values():
            for length in target_lengths:
                for pep in split_data.get(length, []):
                    if pep not in seen:
                        all_peptides.append(pep)
                        seen.add(pep)
    print(f"[clustering] Total unique peptides: {len(all_peptides)}")

    # ── Словарь peptide -> аллель ─────────────────────────────────────────
    peptide_to_allele = {}
    if additional_data is not None:
        peptide_to_allele = _build_peptide_to_allele(additional_data)
        n_known = sum(1 for p in all_peptides if p in peptide_to_allele)
        print(f"[clustering] Peptides with known allele: {n_known}/{len(all_peptides)}")

    # ── Кодируем ─────────────────────────────────────────────────────────
    print("[clustering] Encoding peptides via Viterbi...")
    features_list = [
        encode_peptides_positional(
            model=per_name_models[name],
            peptides=all_peptides,
            anchor_state_names=anchor_names,
            window=window,
        )
        for name in top_model_names
    ]
    features = np.mean(features_list, axis=0).astype(np.float32)

    # ── Кластеризуем; получаем labels и PCA-пространство кластеризации ───
    print("[clustering] Running KMeans...")
    labels, clusterer, X_pca_full = cluster_peptides_to_branches(
        features,
        n_clusters=n_clusters,
        n_init=n_init,
    )

    # ── 2D-проекция из того же PCA-пространства (первые 2 компоненты) ────
    # X_pca_full уже StandardScaler → PCA(200), берём первые 2 измерения
    X_pca_2d = X_pca_full[:, :2]

    # Доли дисперсии для подписей осей: повторяем PCA только на 2 компонентах
    # чтобы получить explained_variance_ratio_ (не влияет на X_pca_2d)
    scaler_2d = StandardScaler()
    X_scaled = scaler_2d.fit_transform(features)
    pca_2d = PCA(n_components=2, random_state=42)
    pca_2d.fit(X_scaled)
    evr = pca_2d.explained_variance_ratio_

    # ── Статистика ────────────────────────────────────────────────────────
    stats_df, accuracy, allele_labels, known_alleles = _compute_cluster_allele_stats(
        all_peptides, labels, peptide_to_allele)
    print("[clustering] Cluster vs allele stats:")
    print(stats_df.to_string())
    if accuracy is not None:
        print(f"[clustering] Clustering accuracy vs real alleles: {accuracy:.1%}")

    peptides_per_cluster = {
        c: [p for p, l in zip(all_peptides, labels) if l == c]
        for c in range(n_clusters)
    }

    has_allele_info = len(peptide_to_allele) > 0

    # ── Цвета ─────────────────────────────────────────────────────────────
    base_colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800", "#00BCD4"]
    cluster_color_map = {c: base_colors[c % len(base_colors)] for c in range(n_clusters)}

    allele_palette = ["#1B5E20", "#880E4F", "#E65100", "#01579B", "#4A148C", "#BF360C"]
    allele_color_map = {a: allele_palette[i % len(allele_palette)]
                        for i, a in enumerate(known_alleles)}
    allele_color_map["unknown"] = "#BDBDBD"

    # ── Компоновка сетки ──────────────────────────────────────────────────
    logo_rows = math.ceil(n_clusters / 2)
    total_rows = 2 + logo_rows

    acc_str = f"  |  Accuracy: {accuracy:.1%}" if accuracy is not None else ""
    cluster_counts = " | ".join(
        f"Cluster {c}: {len(peptides_per_cluster[c])}" for c in range(n_clusters))

    fig = plt.figure(figsize=(20, 5 * total_rows))
    fig.suptitle(
        f"Clustering split diagnostics — {iteration_name}\n"
        f"{cluster_counts}{acc_str}",
        fontsize=13,
    )

    gs = gridspec.GridSpec(total_rows, 2, figure=fig, hspace=0.55, wspace=0.35)

    ax_pca_clust  = fig.add_subplot(gs[0, 0])
    ax_pca_allele = fig.add_subplot(gs[0, 1])
    ax_len_clust  = fig.add_subplot(gs[1, 0])
    ax_len_allele = fig.add_subplot(gs[1, 1])

    logo_axes = []
    for row in range(logo_rows):
        for col in range(2):
            cluster_id = row * 2 + col
            if cluster_id < n_clusters:
                logo_axes.append(fig.add_subplot(gs[2 + row, col]))
            else:
                ax_empty = fig.add_subplot(gs[2 + row, col])
                ax_empty.axis("off")

    # ── Joint KDE ─────────────────────────────────────────────────────────
    joint_fig_allele = None
    joint_fig_clust = None

    if has_allele_info and known_alleles:
        print("[clustering] Building joint KDE plots...")
        try:
            pca_df = pd.DataFrame({
                "PC1": X_pca_2d[:, 0],
                "PC2": X_pca_2d[:, 1],
                "cluster": [f"Cluster {l}" for l in labels],
                "allele": [peptide_to_allele.get(p, "unknown") for p in all_peptides],
            })

            g_allele = sns.jointplot(
                data=pca_df[pca_df["allele"] != "unknown"],
                x="PC1", y="PC2", hue="allele", kind="kde", height=7,
                marginal_kws={"fill": True, "alpha": 0.3},
            )
            title = f"KDE — Real alleles | {iteration_name}"
            if accuracy is not None:
                title += f"\naccuracy={accuracy:.1%}"
            g_allele.fig.suptitle(title, y=1.02)
            joint_fig_allele = g_allele.fig

            clust_palette = {f"Cluster {c}": cluster_color_map[c] for c in range(n_clusters)}
            g_clust = sns.jointplot(
                data=pca_df,
                x="PC1", y="PC2", hue="cluster", kind="kde", height=7,
                marginal_kws={"fill": True, "alpha": 0.3},
                palette=clust_palette,
            )
            g_clust.fig.suptitle(f"KDE — KMeans clusters | {iteration_name}", y=1.02)
            joint_fig_clust = g_clust.fig

        except Exception as e:
            print(f"[clustering] Joint KDE failed: {e}")

    # ── [A] PCA по кластерам ──────────────────────────────────────────────
    colors_clust = [cluster_color_map[l] for l in labels]
    ax_pca_clust.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                         c=colors_clust, alpha=0.3, s=8, rasterized=True)
    ax_pca_clust.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax_pca_clust.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax_pca_clust.set_title("PCA. KMeans clusters")
    ax_pca_clust.legend(handles=[
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=cluster_color_map[c], markersize=8,
               label=f"Cluster {c} (n={len(peptides_per_cluster[c])})")
        for c in range(n_clusters)
    ], fontsize=9)

    # ── [B] PCA по реальным аллелям ───────────────────────────────────────
    if has_allele_info and known_alleles:
        colors_allele = [allele_color_map.get(a, "#BDBDBD") for a in allele_labels]
        ax_pca_allele.scatter(X_pca_2d[:, 0], X_pca_2d[:, 1],
                              c=colors_allele, alpha=0.3, s=8, rasterized=True)
        legend_handles = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=allele_color_map[a], markersize=8,
                   label=f"{a} (n={allele_labels.count(a)})")
            for a in known_alleles
        ]
        unknown_cnt = allele_labels.count("unknown")
        if unknown_cnt > 0:
            legend_handles.append(
                Line2D([0], [0], marker='o', color='w', markerfacecolor="#BDBDBD",
                       markersize=8, label=f"unknown (n={unknown_cnt})")
            )
        ax_pca_allele.legend(handles=legend_handles, fontsize=8)
        if accuracy is not None:
            ax_pca_allele.text(
                0.02, 0.98, f"Accuracy: {accuracy:.1%}",
                transform=ax_pca_allele.transAxes, fontsize=10,
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            )
    else:
        ax_pca_allele.text(0.5, 0.5, "No allele info\n(additional_data not provided)",
                           ha="center", va="center",
                           transform=ax_pca_allele.transAxes, fontsize=11, color="gray")

    ax_pca_allele.set_xlabel(f"PC1 ({evr[0]:.1%})")
    ax_pca_allele.set_ylabel(f"PC2 ({evr[1]:.1%})")
    ax_pca_allele.set_title("PCA. Real alleles")

    # ── [C] Длины по кластерам ────────────────────────────────────────────
    all_lengths = sorted(set(len(p) for p in all_peptides))
    x = np.arange(len(all_lengths))
    bar_width = 0.8 / n_clusters
    for c in range(n_clusters):
        counts = [sum(1 for p in peptides_per_cluster[c] if len(p) == l) for l in all_lengths]
        offset = (c - n_clusters / 2 + 0.5) * bar_width
        ax_len_clust.bar(x + offset, counts, bar_width,
                         color=cluster_color_map[c], alpha=0.8,
                         label=f"Cluster {c}")
    ax_len_clust.set_xticks(x)
    ax_len_clust.set_xticklabels(all_lengths)
    ax_len_clust.set_xlabel("Peptide length")
    ax_len_clust.set_ylabel("Count")
    ax_len_clust.set_title("Length distribution. Clusters")
    ax_len_clust.legend(fontsize=9)

    # ── [D] Длины по реальным аллелям ────────────────────────────────────
    if has_allele_info and known_alleles:
        bar_width_a = 0.8 / len(known_alleles)
        for i, allele in enumerate(known_alleles):
            allele_peps = [p for p, a in zip(all_peptides, allele_labels) if a == allele]
            counts = [sum(1 for p in allele_peps if len(p) == l) for l in all_lengths]
            offset = (i - len(known_alleles) / 2 + 0.5) * bar_width_a
            ax_len_allele.bar(x + offset, counts, bar_width_a,
                              color=allele_color_map[allele], alpha=0.8,
                              label=f"{allele} (n={len(allele_peps)})")
        ax_len_allele.set_xticks(x)
        ax_len_allele.set_xticklabels(all_lengths)
        ax_len_allele.set_xlabel("Peptide length")
        ax_len_allele.set_ylabel("Count")
        ax_len_allele.set_title("Length distribution. Real alleles")
        ax_len_allele.legend(fontsize=8)
    else:
        ax_len_allele.text(0.5, 0.5, "No allele info", ha="center", va="center",
                           transform=ax_len_allele.transAxes, fontsize=11, color="gray")
        ax_len_allele.set_title("Length distribution. Real alleles")

    # ── Sequence logos ────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    print("[clustering] Building logos...")
    best_model = per_name_models[top_model_names[0]]
    for c, ax in enumerate(logo_axes):
        _draw_logo(ax, peptides_per_cluster[c], best_model, anchor_names,
                   f"Sequence logo — Cluster {c}", max_peptides_for_logo, rng)

    print("[clustering] Figure ready.")
    return fig, labels, all_peptides, joint_fig_allele, joint_fig_clust