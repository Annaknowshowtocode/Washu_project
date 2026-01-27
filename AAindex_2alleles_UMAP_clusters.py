import os
from pathlib import Path
from protlearn.features import aaindex1
import numpy as np
import pandas as pd
from sklearn.cluster import FeatureAgglomeration
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA, TruncatedSVD
import scanpy as sc
import anndata as ad



PROJECT_ROOT = Path("/Users/annaklimova/Desktop/Washu_project")
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = PROJECT_ROOT / "filtered_df.csv"  # путь к входному CSV с данными!


def open_iedb_data(csv_path: Path) -> pd.DataFrame:
    """Чтение IEDB-таблицы из CSV."""
    return pd.read_csv(csv_path)


def get_data_for_allele(
        df: pd.DataFrame,
        allele: str = "HLA-DPA1*02:02/DPB1*05:01",
        binder_status: int = 1  # (1=Positive, 0=Negative)
) -> pd.DataFrame:
    """Фильтрация данных по аллелю и статусу связывания."""
    return df.loc[(df["Allele"] == allele) & (df["Binder"] == binder_status)
                  & (df["Length"] == 15)]


def get_peptide_splits(df: pd.DataFrame, core_window: int = 9) -> dict:
    """Нарезка пептидов в окна заданной длины."""
    peptide_splits_dict = {}
    for pep in df["Peptide"]:
        if len(pep) >= core_window:
            windows = [pep[p:p + core_window] for p in range(0, len(pep) - core_window + 1)]
            peptide_splits_dict[pep] = [list(w) for w in windows]
    return peptide_splits_dict


def make_amino_pattern_table(peptide_splits_dict, core_window=9) -> pd.DataFrame:
    """Создание таблицы 9-меров (по позициям) из окон пептидов."""
    rows = []
    for peptide, windows in peptide_splits_dict.items():
        for start, w in enumerate(windows):
            if len(w) != core_window:
                continue
            row = {"Peptide": peptide, "Start": start, "Window": "".join(w)}
            for i, aa in enumerate(w, start=1):
                row[i] = aa
            rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df[["Peptide", "Start", "Window"] + list(range(1, core_window + 1))]
    return df



def transform_to_properties(amino_pattern_table: pd.DataFrame, prop_mat: pd.DataFrame,
                            core_window: int = 9) -> pd.DataFrame:
    """Преобразование таблицы аминокислот в таблицу свойств (векторизация)."""
    if amino_pattern_table.empty:
        return pd.DataFrame()

    pos_cols = list(range(1, core_window + 1))
    letters = amino_pattern_table[pos_cols].to_numpy()

    aa_to_row = {aa: i for i, aa in enumerate(prop_mat.index)}
    idx = np.vectorize(aa_to_row.get)(letters)

    prop_values = prop_mat.to_numpy()
    encoded = prop_values[idx]

    X_peptides = encoded.reshape(encoded.shape[0], -1)

    n_props = prop_mat.shape[1]
    col_names = [
        f"{pos}_new_prop_{k}"
        for pos in range(core_window)
        for k in range(n_props)
    ]

    transformed_peptides_table = pd.DataFrame(X_peptides, columns=col_names)
    transformed_peptides_table.insert(0, "Peptide", amino_pattern_table["Peptide"].values)
    return transformed_peptides_table

def filter_correlated_features(df: pd.DataFrame, threshold: float = 0.9) -> pd.DataFrame:
    """Удаление высококоррелирующих признаков."""
    print(f"Начальное число признаков: {df.shape[1]}")

    corr_matrix = df.corr(numeric_only=True).abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = [column for column in upper.columns if (upper[column] > threshold).any()]
    df_filtered = df.drop(columns=to_drop)

    print(f"Удалено признаков: {len(to_drop)}")
    print(f"Конечное число признаков: {df_filtered.shape[1]}")
    return df_filtered

def plot_pca_scatter_by_allele(X_pca: np.ndarray, allele_labels: pd.Series, outpath: Path):
    plt.figure(figsize=(10, 8))

    uniq = pd.unique(allele_labels)
    cmap = plt.get_cmap("tab10")

    for i, a in enumerate(uniq):
        mask = (allele_labels.values == a)
        plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            s=3, alpha=0.5,
            color=cmap(i % 10),
            label=a
        )

    plt.title("PCA: Projected peptides (colored by allele)", fontsize=16)
    plt.xlabel("PC1", fontsize=12)
    plt.ylabel("PC2", fontsize=12)

    plt.legend(title="Allele", markerscale=2, fontsize=10, title_fontsize=11, frameon=True)
    plt.grid(linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"PCA Scatter Plot сохранен в {outpath}")


def plot_umap_scatter_by_cluster(
    X_umap: np.ndarray,
    cluster_labels: pd.Series,
    outpath: Path,
    point_size: float = 3,
    alpha: float = 0.6,
    cmap_name: str = "tab10",
    show_legend: bool = False,
    label_fontsize: int = 14,
    label_color: str = "black",
    label_weight: str = "bold",
):
    """
    Рисует UMAP, раскрашенный по кластерам, и подписывает номер/название кластера на UMAP как на scanpy-картинках.

    X_umap: shape (n_cells, 2)
    cluster_labels: pd.Series длиной n_cells (категории/числа/строки)
    """

    if X_umap.shape[1] != 2:
        raise ValueError("X_umap должен быть формы (n, 2)")

    # чтобы порядок кластеров был стабильным (особенно если это числа)
    cl = pd.Series(cluster_labels).copy()
    try:
        # если кластеры выглядят как числа ("0","1","2"), отсортируем численно
        uniq = sorted(pd.unique(cl), key=lambda x: float(x))
    except Exception:
        uniq = list(pd.unique(cl))

    cmap = plt.get_cmap(cmap_name)

    plt.figure(figsize=(8, 6))

    for i, c in enumerate(uniq):
        mask = (cl.values == c)
        plt.scatter(
            X_umap[mask, 0], X_umap[mask, 1],
            s=point_size,
            alpha=alpha,
            color=cmap(i % cmap.N),
            linewidths=0,
            label=str(c)
        )

        # --- подпись кластера в центре ---
        # медиана обычно устойчивее, чем среднее (меньше влияния выбросов)
        cx = np.median(X_umap[mask, 0])
        cy = np.median(X_umap[mask, 1])

        plt.text(
            cx, cy, str(c),
            fontsize=label_fontsize,
            color=label_color,
            fontweight=label_weight,
            ha="center", va="center",
        )

    plt.title("UMAP projection (colored by cluster)")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")

    if show_legend:
        plt.legend(title="Cluster", markerscale=2, fontsize=9, title_fontsize=10, frameon=True)
    else:
        plt.legend([], [], frameon=False)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"UMAP сохранен в {outpath}")












def main():
    # 1. Загрузка данных
    try:
        train_df = open_iedb_data(CSV_PATH)
    except FileNotFoundError:
        print(f"Ошибка: Файл не найден по пути {CSV_PATH}.")
        return

    # 2. Подготовка AAindex и кластеризация (Шаг 1)
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    prop_values, prop_names = aaindex1(amino_acids, standardize="zscore")
    aa_prop_index = pd.DataFrame(prop_values, columns=prop_names, index=amino_acids).dropna(axis="columns")

    n_clusters = 60
    agg = FeatureAgglomeration(n_clusters=n_clusters, linkage="ward")
    X0 = aa_prop_index.values
    _ = agg.fit_transform(X0)

    labels = agg.labels_
    clusters = {}
    for fid, cid in zip(aa_prop_index.columns, labels):
        clusters.setdefault(cid, []).append(fid)

    aa_prop_reduced_manual = pd.DataFrame(index=aa_prop_index.index)
    for cid, feats in sorted(clusters.items()):
        aa_prop_reduced_manual[f"Cluster_{cid:03d}"] = aa_prop_index[feats].mean(axis=1)

    print(f"Кластеризация AAindex завершена. Получено {n_clusters} агрегированных свойства.")

    # ==========================================================================
    CORE_WINDOW = 9

    ALLELE_A = "HLA-DRB1*15:01"
    ALLELE_B = "HLA-DRB1*03:01"
    print(n_clusters)
    # ==========================================================================

    # 3. Аллель A (binder_status=1)
    allele_a_df = get_data_for_allele(train_df, allele=ALLELE_A, binder_status=1)
    allele_a_df = allele_a_df.sample(n=2000, random_state=42)
    peptide_splits_a_dict = get_peptide_splits(allele_a_df, core_window=CORE_WINDOW)
    amino_pattern_table_a = make_amino_pattern_table(peptide_splits_a_dict, core_window=CORE_WINDOW)

    transformed_a = transform_to_properties(amino_pattern_table_a, aa_prop_reduced_manual, core_window=CORE_WINDOW)
    transformed_a["y"] = 1
    transformed_a["Allele"] = ALLELE_A
    print(f"Аллель {ALLELE_A}: {transformed_a.shape[0]} строк.")

    # 4. Аллель B (binder_status=1)
    allele_b_df = get_data_for_allele(train_df, allele=ALLELE_B, binder_status=1)
    allele_b_df = allele_b_df.sample(n=2000, random_state=42)
    peptide_splits_b_dict = get_peptide_splits(allele_b_df, core_window=CORE_WINDOW)
    amino_pattern_table_b = make_amino_pattern_table(peptide_splits_b_dict, core_window=CORE_WINDOW)

    transformed_b = transform_to_properties(amino_pattern_table_b, aa_prop_reduced_manual, core_window=CORE_WINDOW)
    transformed_b["y"] = 0
    transformed_b["Allele"] = ALLELE_B
    print(f"Аллель {ALLELE_B}: {transformed_b.shape[0]} строк.")

    # 5. Объединение
    full_data = pd.concat([transformed_a, transformed_b], ignore_index=True)
    full_data = full_data.drop_duplicates(keep="first").reset_index(drop=True)

    # Признаки
    X = full_data.drop(columns=["Peptide", "y", "Allele"])
    alleles = full_data["Allele"]

    print(f"Объединение завершено. Общий размер данных: {full_data.shape}")
    print(full_data.iloc[:15,:8])
    # 6. Повторная фильтрация коррелирующих признаков (> 0.9)
    # X_filtered = filter_correlated_features(X, threshold=0.9)


    #7. Снижение размерности

    scaler = StandardScaler(with_mean=True, with_std=True)
    X_scaled = scaler.fit_transform(X)

    # # PCA
    # N_COMPONENTS_MAX = X_scaled.shape[1]
    # pca = PCA(n_components=N_COMPONENTS_MAX)
    # X_pca = pca.fit_transform(X_scaled)
    # plot_pca_scatter_by_allele(X_pca, alleles, OUTPUT_DIR / "pca_scatter_plot.png")

    ## SVD
    # N_COMPONENTS = 10
    # svd = TruncatedSVD(n_components=N_COMPONENTS)
    # X_svd = svd.fit_transform(X_scaled)
    # print(f"SVD: Объясненная дисперсия первыми {N_COMPONENTS} компонентами: {svd.explained_variance_ratio_.sum():.2f}")

    # UMAP
    reducer = umap.UMAP(n_components=2)
    X_umap = reducer.fit_transform(X_scaled)

    adata = ad.AnnData(X_scaled)
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X")  # kNN граф
    sc.tl.leiden(adata, resolution=1.0, random_state=42)  # ↑resolution => больше кластеров

    full_data["Cluster"] = adata.obs["leiden"].astype(int).to_numpy()
    print("N clusters:", full_data["Cluster"].nunique())
    print(full_data[["Allele", "Cluster"]].value_counts().sort_index())

    # # 10) Plot UMAP by leiden clusters + labels
    # plot_umap_scatter_by_cluster(
    #     X_umap=X_umap,
    #     cluster_labels=full_data["Cluster"],
    #     outpath=OUTPUT_DIR / "umap_leiden.png",
    #     point_size=3,
    #     alpha=0.6,
    #     show_legend=False,
    #     label_fontsize=14,
    # )

    out_csv = OUTPUT_DIR / "cores_with_leiden_clusters.csv"
    full_data.to_csv(out_csv, index=False)
    print(f"Файл сохранён: {out_csv}")


if __name__ == "__main__":
    main()
