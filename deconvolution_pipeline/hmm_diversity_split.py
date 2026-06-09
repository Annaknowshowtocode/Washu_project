from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pomegranate import DiscreteDistributionCycle


AA_LIST = sorted("ACDEFGHIKLMNPQRSTVWY")
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_LIST)}
N_AA = len(AA_LIST)  # 20


def _viterbi_anchor_positions(model, peptide: str) -> list[int]:
    try:
        _, path = model.viterbi(peptide)
    except Exception:
        return []

    emitting = [
        s for s in model.states
        if s.distribution is not None
        and type(s.distribution) is not DiscreteDistributionCycle
    ]
    if len(emitting) < 2:
        return []

    anchor_names = {emitting[-2].name, emitting[-1].name}

    anchor_positions = []
    pep_pos = 0
    for state_idx, state in path:
        if state.distribution is None:          # start / end
            continue
        if state.name in anchor_names:
            anchor_positions.append(pep_pos)
        pep_pos += 1

    return anchor_positions


def encode_peptides_positional(
    model,
    peptides: list[str],
    anchor_state_names: Optional[list[str]] = None,
    window: int = 2,
    use_full_sequence: bool = True,
) -> np.ndarray:
    emitting = [
        s for s in model.states
        if s.distribution is not None
        and type(s.distribution) is not DiscreteDistributionCycle
    ]

    if anchor_state_names is not None:
        target_state_names = anchor_state_names
    else:
        target_state_names = [s.name for s in emitting]

    n_anchors = len(target_state_names)
    win_size = 2 * window + 1

    positional_dim = n_anchors * win_size * N_AA
    bag_dim = N_AA if use_full_sequence else 0
    feature_dim = positional_dim + bag_dim

    features = np.zeros((len(peptides), feature_dim), dtype=np.float32)

    for i, peptide in enumerate(peptides):
        if use_full_sequence:
            bag = np.zeros(N_AA, dtype=np.float32)
            for aa in peptide:
                if aa in AA_TO_IDX:
                    bag[AA_TO_IDX[aa]] += 1
            if len(peptide) > 0:
                bag /= len(peptide)
            features[i, positional_dim:] = bag

        try:
            _, path = model.viterbi(peptide)
        except Exception:
            continue

        state_to_peppos = {}
        pep_pos = 0
        for _state_idx, state in path:
            if state.distribution is None:
                continue
            state_to_peppos[state.name] = pep_pos
            pep_pos += 1

        for anchor_idx, anchor_name in enumerate(target_state_names):
            if anchor_name not in state_to_peppos:
                continue
            anchor_pep_pos = state_to_peppos[anchor_name]
            base_offset = anchor_idx * win_size * N_AA
            for w in range(-window, window + 1):
                pos_in_pep = anchor_pep_pos + w
                if 0 <= pos_in_pep < len(peptide):
                    aa = peptide[pos_in_pep]
                    if aa in AA_TO_IDX:
                        win_offset = (w + window) * N_AA
                        features[i, base_offset + win_offset + AA_TO_IDX[aa]] = 1.0

    return features

def _build_train_data_from_labels(
    all_peptides: list[str],
    labels: np.ndarray,
    reference_train_data: dict,
    target_lengths: list[int],
    branch_id: int,
) -> dict:
    branch_peptides = set(
        pep for pep, lab in zip(all_peptides, labels) if lab == branch_id
    )

    result = {}
    for allele_name, per_split in reference_train_data.items():
        result[allele_name] = {}
        for split_num, per_length in per_split.items():
            result[allele_name][split_num] = {}
            for length in target_lengths:
                seqs = per_length.get(length, np.array([], dtype=object))
                filtered = np.array(
                    [s for s in seqs if s in branch_peptides],
                    dtype=object,
                )
                result[allele_name][split_num][length] = filtered

    return result



def cluster_peptides_to_branches(
    features: np.ndarray,
    n_clusters: int = 4,
    n_init: int = 30,
    random_state: int = 42,
    min_branch_fraction: float = 0.1,
    variance_threshold: float = 0.95,
    max_pca_components: int = 400,
    max_retries: int = 5,
    pca_save_path: Optional[str] = None,
    peptide_names: Optional[list[str]] = None,
) -> tuple[np.ndarray, object, np.ndarray]:

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    n_max = min(max_pca_components, X.shape[0], X.shape[1])
    pca_full = PCA(n_components=n_max, random_state=random_state)
    X_pca_full = pca_full.fit_transform(X)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_threshold) + 1)
    n_components = min(n_components, n_max)
    n_components = max(n_components, 2)

    explained = cumvar[n_components - 1]
    print(f"  PCA: {n_components}/{n_max} components → {explained:.1%} variance explained"
          + (f" (threshold {variance_threshold:.0%} not reached)" if explained < variance_threshold else ""))

    X_pca = X_pca_full[:, :n_components]

    if pca_save_path is not None:
        index = peptide_names if peptide_names is not None else range(len(X_pca))
        pca_df = pd.DataFrame(
            X_pca,
            index=index,
            columns=[f"PC{i+1}" for i in range(n_components)],
        )
        pca_df.index.name = "peptide"

        var_row = pd.DataFrame(
            [pca_full.explained_variance_ratio_[:n_components]],
            index=["explained_variance_ratio"],
            columns=pca_df.columns,
        )
        pd.concat([var_row, pca_df]).to_csv(pca_save_path)
        print(f"  PCA: saved - {pca_save_path}")

    for attempt in range(max_retries):
        current_seed = random_state + attempt

        clusterer = KMeans(
            n_clusters=n_clusters,
            n_init=n_init,
            random_state=current_seed,
            max_iter=500,
        )
        labels = clusterer.fit_predict(X_pca)

        fractions = [(labels == c).mean() for c in range(n_clusters)]
        too_small = [c for c, f in enumerate(fractions) if f < min_branch_fraction]

        for cluster_id, fraction in enumerate(fractions):
            n = (labels == cluster_id).sum()
            print(f"  Attempt {attempt+1}, Cluster {cluster_id}: {n} ({fraction:.1%})"
                  + (" TOO SMALL" if cluster_id in too_small else ""))

        if not too_small:
            print(f"  Clustering OK on attempt {attempt+1} (seed={current_seed})")
            return labels, clusterer, X_pca

        print(f"  Retry {attempt+1}/{max_retries}: clusters {too_small} below {min_branch_fraction:.0%}")

    print(f"  WARNING: could not achieve min_branch_fraction={min_branch_fraction:.0%} "
          f"after {max_retries} attempts. Using last result (seed={current_seed}).")
    return labels, clusterer, X_pca



def _encode_features(
    per_name_models: dict,
    peptides: list[str],
    window: int,
    top_n_models: int,
) -> np.ndarray:
    top_n = min(top_n_models, len(per_name_models))
    top_model_names = list(per_name_models.keys())[:top_n]

    features_list = []
    for model_name in top_model_names:
        model = per_name_models[model_name]

        f = encode_peptides_positional(
            model=model,
            peptides=peptides,
            anchor_state_names=None,
            window=window,
        )
        features_list.append(f)

    return np.concatenate(features_list, axis=1).astype(np.float32)


def get_data_split_by_clustering(
    per_name_models: dict,
    train_data: dict,
    experiment_params,
    anchor_state_names: Optional[list[str]] = None,
    window: int = 2,
    n_init: int = 30,
    n_clusters: int = 2,
    top_n_models: int = 3,
) -> tuple[dict, ...]:
    model_training_params = experiment_params.model_training_params
    target_lengths = model_training_params.lengths_to_use

    all_peptides_ordered = []
    seen = set()
    for allele_data in train_data.values():
        for split_data in allele_data.values():
            for length in target_lengths:
                for pep in split_data.get(length, []):
                    if pep not in seen:
                        all_peptides_ordered.append(pep)
                        seen.add(pep)
    print(f"  Clustering: {len(all_peptides_ordered)} unique peptides")

    print(f"  Clustering: using top {min(top_n_models, len(per_name_models))} models")
    features = _encode_features(
        per_name_models, all_peptides_ordered, window, top_n_models,
    )
    print(f"  Clustering: feature matrix shape = {features.shape}")

    labels, _, _ = cluster_peptides_to_branches(
        features=features,
        n_clusters=n_clusters,
        n_init=n_init,
        peptide_names=all_peptides_ordered,
    )

    result = []
    for branch_id in range(n_clusters):
        td = _build_train_data_from_labels(
            all_peptides_ordered, labels, train_data, target_lengths,
            branch_id=branch_id,
        )
        total = sum(
            len(td[a][s].get(l, []))
            for a in td for s in td[a] for l in target_lengths
        )
        print(f"  Branch {branch_id}: {total} peptides")
        result.append(td)

    return tuple(result)

