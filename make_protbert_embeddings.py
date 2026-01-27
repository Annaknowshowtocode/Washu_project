from pathlib import Path
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel

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
    use_safetensors=True,   # <- важно
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

    # ProtBERT ожидает: "A C D E ..."
    seqs_spaced = [
        " ".join(list(str(s).strip().upper()
                      .replace("U", "X").replace("Z", "X").replace("O", "X").replace("B", "X")))
        for s in seqs
    ]

    all_vecs = []

    for i in range(0, len(seqs_spaced), batch_size):
        batch = seqs_spaced[i : i + batch_size]

        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        out = model(**enc)
        hidden = out.last_hidden_state  # (B, L, H)
        attn = enc["attention_mask"]    # (B, L)

        token_mask = attn.bool()
        token_mask[:, 0] = False  # remove CLS

        lengths = attn.sum(dim=1)
        sep_idx = (lengths - 1).clamp(min=0)
        token_mask[torch.arange(token_mask.size(0), device=device), sep_idx] = False  # remove SEP

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


def get_data_for_allele(
    df: pd.DataFrame,
    allele: str = "HLA-DRB1*03:01",
    binder_status: int = 1,
    length: int = 15,
    n: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    filtered = df.loc[
        (df["Allele"] == allele)
        & (df["Binder"] == binder_status)
        & (df["Length"] == length)
    ]

    if len(filtered) > n:
        filtered = filtered.sample(n=n, random_state=random_state)

    return filtered.copy()



def main():
    CSV_PATH = Path("/Users/annaklimova/Desktop/Washu_project/filtered_df.csv")
    ALLELE_A = "HLA-DRB1*03:01"
    PEPTIDE_COL = "Peptide"

    train_df = open_iedb_data(CSV_PATH)
    filtered = get_data_for_allele(
        train_df,
        allele="HLA-DRB1*03:01",
        binder_status=1,
        length=15,
        n=5000,
    )

    peptides = (
        filtered["Peptide"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    peptides = peptides[peptides != ""].tolist()

    print("N peptides:", len(peptides))

    peptides = (
        filtered[PEPTIDE_COL]
        .dropna()
        .astype(str)
        .str.strip()
    )
    peptides = peptides[peptides != ""].tolist()

    if not peptides:
        print("После фильтрации не осталось пептидов.")
        return

    # peptides = peptides[:15]
    X = protbert_embed(peptides, tokenizer=tokenizer, model=model, device="cpu", pooling="mean")
    print("Embeddings shape:", X.shape)
    print(peptides[0])
    print(X[0][:10])  # первые 10 чисел эмбеддинга


    OUT_PATH = Path("/Users/annaklimova/Desktop/Washu_project/protbert_embeddings_HLA_DRB1_03_01.npz")

    np.savez(
        OUT_PATH,
        embeddings=X,
        peptides=np.array(peptides),
    )

    print(f"Saved embeddings to {OUT_PATH}")










































if __name__ == "__main__":
    main()
