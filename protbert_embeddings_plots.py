import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import scanpy as sc
import anndata as ad

def plot_pca_scatter_by_allele(
    X_pca: np.ndarray,
    allele_labels: pd.Series,
    outpath: Path,
):
    plt.figure(figsize=(10, 8))

    uniq = pd.unique(allele_labels)
    cmap = plt.get_cmap("tab10")

    for i, a in enumerate(uniq):
        mask = (allele_labels.values == a)
        plt.scatter(
            X_pca[mask, 0],
            X_pca[mask, 1],
            s=3,
            alpha=0.5,
            color=cmap(i % cmap.N),
            label=a,
        )

    plt.title("PCA: Projected peptides (colored by allele)", fontsize=16)
    plt.xlabel("PC1", fontsize=12)
    plt.ylabel("PC2", fontsize=12)

    plt.legend(
        title="Allele",
        markerscale=2,
        fontsize=10,
        title_fontsize=11,
        frameon=True,
    )
    plt.grid(linestyle=":", alpha=0.5)
    plt.tight_layout()

    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()

    print(f"PCA Scatter Plot сохранен в {outpath}")


# ---------- твой плоттер (без изменений) ----------
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
    if X_umap.shape[1] != 2:
        raise ValueError("X_umap должен быть формы (n, 2)")

    cl = pd.Series(cluster_labels).copy()
    try:
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
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"UMAP сохранен в {outpath}")


# ---------- загрузчик npz ----------
def load_embeddings_from_npz(npz_path: Path) -> np.ndarray:
    data = np.load(npz_path, allow_pickle=True)
    if "embeddings" in data.files:
        return np.asarray(data["embeddings"])  # у тебя это точно есть
    # fallback (на всякий)
    preferred_keys = ["embedding", "X", "arr_0"]
    for k in preferred_keys:
        if k in data.files:
            return np.asarray(data[k])
    for k in data.files:
        X = np.asarray(data[k])
        if X.ndim == 2:
            return X
    raise ValueError(f"Не нашла 2D-эмбеддинги в {npz_path}. keys={data.files}")


# ---------- пути ----------
npz_a = Path("/Users/annaklimova/Desktop/Washu_project/protbert_embeddings_HLA_DRB1_15_01.npz")
npz_b = Path("/Users/annaklimova/Desktop/Washu_project/protbert_embeddings_HLA_DRB1_03_01.npz")

out_dir = Path("/Users/annaklimova/Desktop/Washu_project/plots")
out_umap = out_dir / "umap_leiden.png"

Xa = load_embeddings_from_npz(npz_a)
Xb = load_embeddings_from_npz(npz_b)

X = np.vstack([Xa, Xb]).astype(np.float32)

alleles = pd.Series(
    ["HLA_A_02_01"] * Xa.shape[0] + ["HLA_B_27_05"] * Xb.shape[0],
    name="allele"
)

print("Shapes:", Xa.shape, Xb.shape, "->", X.shape)


# ---------- scanpy: neighbors -> umap -> leiden ----------
adata = ad.AnnData(X)
adata.obs["allele"] = alleles.values

# (опционально, но ок) скейл
sc.pp.scale(adata, zero_center=True, max_value=10)

# PCA для ускорения
sc.tl.pca(adata, n_comps=50, svd_solver="arpack", random_state=42)
# PCA
sc.tl.pca(adata, n_comps=50, svd_solver="arpack", random_state=42)

# --- сохраняем PCA plot ---
X_pca = adata.obsm["X_pca"][:, :2]
pca_out = out_dir / "pca_by_allele.png"

plot_pca_scatter_by_allele(
    X_pca,
    alleles,
    pca_out,
)

# граф соседей
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=50, metric="cosine")

# UMAP (с random_state для воспроизводимости)
sc.tl.umap(adata, min_dist=0.1, random_state=42)

# Leiden (воспроизводимость + явный backend)
sc.tl.leiden(
    adata,
    resolution=0.6,
    key_added="leiden",
    flavor="igraph",      # чтобы одинаково работало на разных окружениях
    n_iterations=2,
    random_state=42,
)

# ---------- результаты ----------
X_umap = adata.obsm["X_umap"]
clusters = pd.Series(adata.obs["leiden"].astype(str).values, name="cluster")

# ---------- твой plot ----------
plot_umap_scatter_by_cluster(
    X_umap,
    clusters,
    out_umap,
    point_size=3,
    alpha=0.6,
    show_legend=False,
    label_fontsize=12
)

print("Готово:", out_umap)

# ---------- полезная диагностика: cluster × allele ----------
ct = pd.crosstab(adata.obs["leiden"], adata.obs["allele"])
ct_pct = ct.div(ct.sum(axis=1), axis=0).round(3)

print("\nCounts (cluster x allele):")
print(ct)

print("\nRow % (cluster composition):")
print(ct_pct)

print("\nLeiden cluster sizes:")
print(adata.obs["leiden"].value_counts().sort_index().to_string())
