import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import scanpy as sc
import anndata as ad
from typing import Optional, Tuple
import logomaker
from sklearn.preprocessing import StandardScaler


# =========================
# 1) PCA plot by allele
# =========================
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


# =========================
# 2) UMAP plot by cluster
# =========================
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


# =========================
# 3) Load npz: embeddings + peptides
# =========================
def load_npz_embeddings_and_peptides(npz_path: Path) -> Tuple[np.ndarray, pd.Series]:
    data = np.load(npz_path, allow_pickle=True)

    # --- embeddings ---
    if "embeddings" in data.files:
        X = np.asarray(data["embeddings"])
    else:
        preferred_keys = ["embedding", "X", "arr_0"]
        X = None
        for k in preferred_keys:
            if k in data.files:
                X = np.asarray(data[k])
                break
        if X is None:
            for k in data.files:
                arr = np.asarray(data[k])
                if arr.ndim == 2:
                    X = arr
                    break
        if X is None:
            raise ValueError(f"Не нашла 2D-эмбеддинги в {npz_path}. keys={data.files}")

    # --- peptides ---
    pep_keys = ["peptides", "peptide", "seqs", "seq", "sequences", "sequence", "strings"]
    peptides = None
    for k in pep_keys:
        if k in data.files:
            peptides = np.asarray(data[k])
            break

    if peptides is None:
        raise ValueError(
            f"Не нашла пептиды в {npz_path}. "
            f"Проверь, что npz содержит один из ключей {pep_keys}. keys={data.files}"
        )

    peptides = pd.Series(peptides).astype(str)

    if X.shape[0] != len(peptides):
        raise ValueError(
            f"Несовпадение размеров в {npz_path}: embeddings n={X.shape[0]} vs peptides n={len(peptides)}"
        )

    return X, peptides


# =========================
# 4) Sequence logo helpers
# =========================
AA_ORDER = list("ACDEFGHIKLMNPQRSTVWY")

def seqs_to_pfm(seqs, pseudocount=0.5) -> Optional[pd.DataFrame]:
    if len(seqs) == 0:
        return None

    L = len(seqs[0])
    for s in seqs:
        if len(s) != L:
            raise ValueError("Все пептиды должны быть одинаковой длины")

    pfm = pd.DataFrame(
        pseudocount,
        index=range(L),
        columns=AA_ORDER,
        dtype=float
    )

    for s in seqs:
        for i, aa in enumerate(s):
            if aa in pfm.columns:
                pfm.loc[i, aa] += 1

    pfm = pfm.div(pfm.sum(axis=1), axis=0)  # частоты
    return pfm

def pfm_to_ic(pfm: pd.DataFrame) -> pd.DataFrame:
    """
    Information content (bits) per position:
      IC_pos = log2(20) - H(pos)
      logo heights = p(aa,pos) * IC_pos
    """
    eps = 1e-12
    p = pfm.clip(lower=eps)
    H = -(p * np.log2(p)).sum(axis=1)                 # entropy
    IC = np.log2(p.shape[1]) - H                      # bits
    ic_logo = pfm.mul(IC, axis=0)
    return ic_logo

def plot_logo_for_cluster(
    full_data: pd.DataFrame,
    cluster_id: int,
    allele: Optional[str],
    ax,
    use_information_content: bool = True,
    min_n: int = 50,
    pseudocount: float = 0.5,
):
    df = full_data[full_data["Cluster"] == cluster_id]
    if allele is not None:
        df = df[df["Allele"] == allele]

    seqs = df["Peptide"].dropna().astype(str).tolist()

    ax.set_title(f"Cluster {cluster_id}\n(n={len(seqs)})", fontsize=10)

    if len(seqs) < min_n:
        ax.axis("off")
        ax.text(0.5, 0.5, f"n<{min_n}", ha="center", va="center")
        return

    pfm = seqs_to_pfm(seqs, pseudocount=pseudocount)
    if pfm is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "no seqs", ha="center", va="center")
        return

    mat = pfm_to_ic(pfm) if use_information_content else pfm
    mat = mat.copy()
    mat.index = mat.index + 1  # позиции 1..15 для красоты

    logomaker.Logo(mat, ax=ax)

    ax.set_xlabel("Position")
    ax.set_ylabel("Bits" if use_information_content else "Frequency")
    ax.set_xticks(list(range(1, len(mat.index) + 1)))
    ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)

def plot_cluster_logo_grid(
    full_data: pd.DataFrame,
    clusters,
    allele: Optional[str] = None,
    ncols: int = 4,
    use_information_content: bool = True,
    min_n: int = 50,
    pseudocount: float = 0.5,
    outpath: Optional[Path] = None,
):
    clusters = list(clusters)
    n = len(clusters)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.8 * nrows))
    axes = np.array(axes).reshape(-1)

    for ax, cid in zip(axes, clusters):
        plot_logo_for_cluster(
            full_data=full_data,
            cluster_id=int(cid),
            allele=allele,
            ax=ax,
            use_information_content=use_information_content,
            min_n=min_n,
            pseudocount=pseudocount,
        )

    for ax in axes[len(clusters):]:
        ax.axis("off")

    title = "ALL alleles" if allele is None else allele
    fig.suptitle(f"Sequence logos (9-mers) — {title}", fontsize=14, y=1.02)

    plt.tight_layout()
    if outpath is not None:
        outpath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(outpath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Logo grid сохранен в {outpath}")
    else:
        plt.show()

def consensus_by_cluster(full_data: pd.DataFrame, clusters, allele=None, topk=3, pseudocount=0.5):
    out = {}
    for cid in clusters:
        df = full_data[full_data["Cluster"] == cid]
        if allele is not None:
            df = df[df["Allele"] == allele]

        seqs = df["Peptide"].dropna().astype(str).tolist()
        if len(seqs) == 0:
            out[cid] = None
            continue

        pfm = seqs_to_pfm(seqs, pseudocount=pseudocount)
        if pfm is None:
            out[cid] = None
            continue

        cons = []
        for pos in pfm.index:
            top = pfm.loc[pos].sort_values(ascending=False).head(topk)
            cons.append("/".join([f"{aa}{top[aa]:.2f}" for aa in top.index]))

        out[cid] = cons
    return out


# =========================
# 5) Paths + params
# =========================
npz_a = Path("/Users/annaklimova/Desktop/Washu_project/protbert_embeddings_HLA_B_27_05.npz")
npz_b = Path("/Users/annaklimova/Desktop/Washu_project/protbert_embeddings_HLA_A_02_01.npz")

OUTPUT_DIR = Path("/Users/annaklimova/Desktop/Washu_project/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

out_umap = OUTPUT_DIR / "umap_leiden.png"
out_pca = OUTPUT_DIR / "pca_by_allele.png"

ALLELE_A = "HLA_B_27_05"
ALLELE_B = "HLA_A_02_01"

LOGO_NCOLS = 4
LOGO_USE_INFORMATION_CONTENT = True
LOGO_MIN_N = 50
LOGO_PSEUDOCOUNT = 0.5


# =========================
# 6) Load data
# =========================
Xa, pep_a = load_npz_embeddings_and_peptides(npz_a)
Xb, pep_b = load_npz_embeddings_and_peptides(npz_b)

X = np.vstack([Xa, Xb]).astype(np.float32)

alleles = pd.Series(
    [ALLELE_A] * Xa.shape[0] + [ALLELE_B] * Xb.shape[0],
    name="Allele"
)

peptides = pd.concat([pep_a, pep_b], ignore_index=True).rename("Peptide")

# sanity check: все 15-mers
lens = peptides.str.len()
if lens.nunique() != 1 or int(lens.iloc[0]) != 9:
    raise ValueError(f"Ожидались пептиды длины 15. Найдены длины: {sorted(lens.unique())[:20]} ...")

print("Shapes:", Xa.shape, Xb.shape, "->", X.shape)
print("Peptides:", len(peptides))


# =========================
# 7) Scanpy pipeline
# =========================
adata = ad.AnnData(X)
adata.obs["Allele"] = alleles.values
adata.obs["Peptide"] = peptides.values

scaler = StandardScaler(with_mean=True, with_std=True)
adata.X = scaler.fit_transform(adata.X)   # <-- важно: не пересоздаём AnnData

sc.tl.pca(adata, n_comps=50, svd_solver="arpack", random_state=42)

X_pca = adata.obsm["X_pca"][:, :2]
plot_pca_scatter_by_allele(X_pca, adata.obs["Allele"], out_pca)

sc.pp.neighbors(adata, n_neighbors=15, use_rep="X")
sc.tl.umap(adata, min_dist=0.1, random_state=42)

sc.tl.leiden(
    adata,
    resolution=1.0,
    key_added="leiden",
    flavor="igraph",
    n_iterations=2,
    random_state=42,
)



CLUSTERS_OF_INTEREST = sorted(adata.obs["leiden"].astype(int).unique())

# UMAP plot by cluster
X_umap = adata.obsm["X_umap"]
clusters = pd.Series(adata.obs["leiden"].astype(str).values, name="cluster")

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


# =========================
# 8) Build full_data (Peptide, Allele, Cluster)
# =========================
full_data = pd.DataFrame({
    "Peptide": adata.obs["Peptide"].astype(str).values,
    "Allele": adata.obs["Allele"].astype(str).values,
    "Cluster": adata.obs["leiden"].astype(int).values,
})


print("N clusters:", full_data["Cluster"].nunique())


# =========================
# 9) Diagnostics: cluster × allele
# =========================
ct = pd.crosstab(adata.obs["leiden"], adata.obs["Allele"])
ct_pct = ct.div(ct.sum(axis=1), axis=0).round(3)

print("\nCounts (cluster x allele):")
print(ct)

print("\nRow % (cluster composition):")
print(ct_pct)

print("\nLeiden cluster sizes:")
print(adata.obs["leiden"].value_counts().sort_index().to_string())


# =========================
# 10) Consensus + logo grids
# =========================
cons_all = consensus_by_cluster(
    full_data=full_data,
    clusters=CLUSTERS_OF_INTEREST,
    allele=None,
    topk=3,
    pseudocount=LOGO_PSEUDOCOUNT,
)

print("\nConsensus (ALL alleles combined) for clusters of interest:")
for cid in CLUSTERS_OF_INTEREST:
    print(cid, ":", cons_all.get(cid))

out_logo_all = OUTPUT_DIR / "logos_clusters_interest_ALL.png"
plot_cluster_logo_grid(
    full_data=full_data,
    clusters=CLUSTERS_OF_INTEREST,
    allele=None,
    ncols=LOGO_NCOLS,
    use_information_content=LOGO_USE_INFORMATION_CONTENT,
    min_n=LOGO_MIN_N,
    pseudocount=LOGO_PSEUDOCOUNT,
    outpath=out_logo_all,
)

out_logo_a = OUTPUT_DIR / "logos_clusters_interest_A.png"
plot_cluster_logo_grid(
    full_data=full_data,
    clusters=CLUSTERS_OF_INTEREST,
    allele=ALLELE_A,
    ncols=LOGO_NCOLS,
    use_information_content=LOGO_USE_INFORMATION_CONTENT,
    min_n=LOGO_MIN_N,
    pseudocount=LOGO_PSEUDOCOUNT,
    outpath=out_logo_a,
)

out_logo_b = OUTPUT_DIR / "logos_clusters_interest_B.png"
plot_cluster_logo_grid(
    full_data=full_data,
    clusters=CLUSTERS_OF_INTEREST,
    allele=ALLELE_B,
    ncols=LOGO_NCOLS,
    use_information_content=LOGO_USE_INFORMATION_CONTENT,
    min_n=LOGO_MIN_N,
    pseudocount=LOGO_PSEUDOCOUNT,
    outpath=out_logo_b,
)

print("\nDone.")
print("Logos saved:")
print(" -", out_logo_all)
print(" -", out_logo_a)
print(" -", out_logo_b)
