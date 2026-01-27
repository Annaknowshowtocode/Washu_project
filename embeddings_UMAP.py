from pathlib import Path
import os, faulthandler
faulthandler.enable()

# чтобы токенизатор не создавал лишние потоки
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# жёстко ограничим потоки (часто лечит SIGSEGV от BLAS/OpenMP)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# важное для numba на macOS: другой threading layer
os.environ["NUMBA_THREADING_LAYER"] = "workqueue"

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# pip install umap-learn
import umap


# ----------------------------
# Load ProtBERT locally
# ----------------------------
MODEL_DIR = Path.home() / "hf_local" / "prot_bert_safe"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    do_lower_case=False,
    local_files_only=True,
    use_fast=True,
)

model = AutoModel.from_pretrained(
    MODEL_DIR,
    local_files_only=True,
    use_safetensors=True,
)
model.eval()


@torch.no_grad()
def protbert_embed(
    seqs,
    tokenizer,
    model,
    device="cpu",
    pooling="mean",
    batch_size=32,
    fp16=False,
):
    device = torch.device(device)
    model = model.to(device).eval()
    if fp16 and device.type != "cpu":
        model = model.half()

    # ProtBERT expects spaced AA tokens: "A C D E ..."
    seqs_spaced = [
        " ".join(
            list(
                str(s)
                .strip()
                .upper()
                .replace("U", "X")
                .replace("Z", "X")
                .replace("O", "X")
                .replace("B", "X")
            )
        )
        for s in seqs
    ]

    all_vecs = []
    for i in range(0, len(seqs_spaced), batch_size):
        batch = seqs_spaced[i : i + batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            add_special_tokens=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)
        hidden = out.last_hidden_state  # (B, L, H)
        attn = enc["attention_mask"]    # (B, L)

        # mask out CLS and SEP for pooling
        token_mask = attn.bool()
        token_mask[:, 0] = False  # CLS

        lengths = attn.sum(dim=1)
        sep_idx = (lengths - 1).clamp(min=0)
        token_mask[torch.arange(token_mask.size(0), device=device), sep_idx] = False  # SEP

        if pooling == "mean":
            mask_f = token_mask.unsqueeze(-1).float()
            summed = (hidden * mask_f).sum(dim=1)
            counts = mask_f.sum(dim=1).clamp(min=1.0)
            vecs = summed / counts
        elif pooling == "cls":
            vecs = hidden[:, 0, :]
        elif pooling == "max":
            neg_inf = torch.finfo(hidden.dtype).min
            hidden_masked = hidden.masked_fill(~token_mask.unsqueeze(-1), neg_inf)
            vecs = hidden_masked.max(dim=1).values
        else:
            raise ValueError("pooling must be 'mean', 'cls', or 'max'")

        all_vecs.append(vecs.float().cpu().numpy())

    return np.vstack(all_vecs)


def open_iedb_data(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def filter_two_alleles(df: pd.DataFrame, alleles, binder_status=1, length=9) -> pd.DataFrame:
    return df.loc[
        (df["Allele"].isin(list(alleles)))
        & (df["Binder"] == binder_status)
        & (df["Length"] == length)
    ].copy()


def plot_2class_scatter(X2d: np.ndarray, labels: pd.Series, title: str, outpath: Path):
    """Простой scatter для 2 аллелей."""
    plt.figure(figsize=(8, 6))
    uniq = list(pd.unique(labels))
    cmap = plt.get_cmap("tab10")

    for i, lab in enumerate(uniq):
        m = (labels.values == lab)
        plt.scatter(X2d[m, 0], X2d[m, 1], s=10, alpha=0.7, color=cmap(i), label=str(lab))

    plt.title(title)
    plt.xlabel("Dim 1")
    plt.ylabel("Dim 2")
    plt.grid(linestyle=":", alpha=0.4)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()
    print(f"[OK] Saved: {outpath}")


def main():
    CSV_PATH = Path("/Users/annaklimova/Desktop/Washu_project/filtered_df.csv")
    OUTDIR = Path("/Users/annaklimova/Desktop/Washu_project/emb_plots_2alleles")
    OUTDIR.mkdir(parents=True, exist_ok=True)

    PEPTIDE_COL = "Peptide"
    ALLELES = ["HLA-A*02:01", "HLA-B*27:05"]
    BINDER_STATUS = 1
    LENGTH = 9

    DEVICE = "cpu"   # можно "cuda" / "mps"
    POOLING = "mean"
    BATCH_SIZE = 32

    # UMAP params (хорошие стартовые)
    UMAP_N_NEIGHBORS = 15
    UMAP_MIN_DIST = 0.1
    UMAP_METRIC = "cosine"
    RANDOM_STATE = 42

    df = open_iedb_data(CSV_PATH)
    sub = filter_two_alleles(df, ALLELES, binder_status=BINDER_STATUS, length=LENGTH)

    sub = sub.dropna(subset=[PEPTIDE_COL]).copy()
    sub[PEPTIDE_COL] = sub[PEPTIDE_COL].astype(str).str.strip()
    sub = sub[sub[PEPTIDE_COL] != ""].copy()

    if sub.empty:
        print("После фильтрации не осталось пептидов.")
        return

    sub = sub.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    peptides = sub[PEPTIDE_COL].tolist()
    allele_labels = sub["Allele"].astype(str)

    print("1) embeddings...")
    # Embeddings
    X = protbert_embed(
        peptides,
        tokenizer=tokenizer,
        model=model,
        device=DEVICE,
        pooling=POOLING,
        batch_size=BATCH_SIZE,
        fp16=False,
    )
    print("Embeddings shape:", X.shape)

    print("2) scaling...")
    # Standardize for PCA/UMAP stability
    Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

    # PCA (2D)
    print("3) PCA...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(Xs)
    print("PCA explained variance ratio:", pca.explained_variance_ratio_)

    plot_2class_scatter(
        X2d=X_pca,
        labels=allele_labels,
        title="PCA (ProtBERT mean-pool) — colored by allele",
        outpath=OUTDIR / "pca_A0201_vs_B2705.png",
    )

    # UMAP (2D)
    print("4) UMAP init...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=UMAP_N_NEIGHBORS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
        random_state=RANDOM_STATE,
    )
    print("5) UMAP fit_transform...")
    X_umap = reducer.fit_transform(Xs)

    plot_2class_scatter(
        X2d=X_umap,
        labels=allele_labels,
        title="UMAP (ProtBERT mean-pool) — colored by allele",
        outpath=OUTDIR / "umap_A0201_vs_B2705.png",
    )

    # Save table with coordinates
    out = sub[[PEPTIDE_COL, "Allele", "Binder", "Length"]].copy()
    out["PCA1"] = X_pca[:, 0]
    out["PCA2"] = X_pca[:, 1]
    out["UMAP1"] = X_umap[:, 0]
    out["UMAP2"] = X_umap[:, 1]

    out_csv = OUTDIR / "peptides_pca_umap_A0201_vs_B2705.csv"
    out.to_csv(out_csv, index=False)
    print(f"[OK] Saved coords table: {out_csv}")


if __name__ == "__main__":
    main()
