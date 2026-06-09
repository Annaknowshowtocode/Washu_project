"""
hmm_metrics.py
==============
Metrics collection for hierarchical HMM motif deconvolution.

Сценарий использования
----------------------
Все пептиды идут под одним техническим именем "dummy_allele", но реально
это смесь из нескольких аллелей (напр. DRB1*01:01, DRB1*03:01, DRB1*04:01).
После каждого шага деконволюции каждый бранч сравнивается со ВСЕМИ эталонами —
так строится матрица branch × allele, которая показывает, куда "расходятся"
мотивы в процессе обучения.

Ключевые функции
----------------
load_reference_matrices(mat_dir, allele_map)
    Загружает .mat файлы (формат NetMHCIIpan/MHCMotifViewer).

MetricsTracker(experiment_params, reference_matrices)
    .record_step(...)  — записывает одну строку метрик на каждом шаге.
                         Автоматически вычисляет sim(branch_i, allele_j)
                         для всех i×j комбинаций.
    .assignment_table() — возвращает текущее назначение branch → allele.
    .to_dataframe()    — все метрики в виде DataFrame.
    .save_csv(path)
    .save_latex_table(path)
    .print_summary()

compare_branches_to_all_references(...)
    Одноразовое сравнение четырёх бранчей со всеми эталонами.
    Возвращает матрицу сходства (branch × allele) и назначение.

Новые метрики (добавлены)
--------------------------
confusion_matrix_metrics(true_labels, pred_labels, allele_names)
    Матрица ошибок + MCC, PPV для мультиклассовой задачи.

shannon_entropy_per_position(matrix)
    Shannon entropy на каждой позиции мотива.

kld_from_background(matrix, background)
    KL-дивергенция между мотивом и фоновым распределением АК.

pearson_correlation_matrices(mat_a, mat_b)
    PCC между двумя матрицами 9×20 (или n×20 в целом).

sequence_logo_data(matrix)
    Данные для построения sequence logo (IC-взвешенные высоты букв).

plot_sequence_logo(matrix, title, save_path)
    Рисует sequence logo через logomaker (если установлен).
"""

from __future__ import annotations

import os
import math
import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr

try:
    import logomaker
    _HAS_LOGOMAKER = True
except ImportError:
    _HAS_LOGOMAKER = False

try:
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:
    _HAS_MPL = False

try:
    from pomegranate import DiscreteDistributionCycle
    _HAS_POMEGRANATE = True
except ImportError:
    _HAS_POMEGRANATE = False

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
_MAT_FILE_AA_ORDER = list("ARNDCQEGHILKMFPSTWYV")

# Фоновые частоты аминокислот (SwissProt / UniRef50, порядок AMINO_ACIDS)
# A  C      D      E      F      G      H      I      K      L
# M  N      P      Q      R      S      T      V      W      Y
_UNIPROT_BACKGROUND = np.array([
    0.0825, 0.0137, 0.0545, 0.0675, 0.0386, 0.0707, 0.0227, 0.0591, 0.0580, 0.0965,
    0.0242, 0.0406, 0.0474, 0.0393, 0.0553, 0.0659, 0.0534, 0.0687, 0.0108, 0.0292,
])
_UNIPROT_BACKGROUND /= _UNIPROT_BACKGROUND.sum()   # нормировка на всякий случай


# ============================================================================
# .mat file loading
# ============================================================================

def load_mat_file(path: str) -> Optional[pd.DataFrame]:
    """Parse a NetMHCIIpan-style PSSM .mat file → probability DataFrame (n_pos × 20).

    Format:
      - Lines starting with '#' or blank → skip
      - Data lines: <pos> <aa_letter> <score_A> <score_R> ... <score_V>
        (20 log-odds scores in _MAT_FILE_AA_ORDER)
      - Softmax per row → probabilities
      - Columns reordered to canonical AMINO_ACIDS order
    """
    rows = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) < 22:
                    continue
                try:
                    scores = np.array(parts[2:22], dtype=float)
                except ValueError:
                    continue
                exp_s = np.exp(scores - scores.max())
                rows.append(exp_s / exp_s.sum())
    except OSError as e:
        warnings.warn(f"Cannot open {path}: {e}")
        return None

    if not rows:
        warnings.warn(f"No data rows in {path}")
        return None

    mat = pd.DataFrame(rows, columns=_MAT_FILE_AA_ORDER)[AMINO_ACIDS]
    mat.index = range(len(mat))
    return mat


def load_reference_matrices(
    mat_dir: str,
    allele_map: Dict[str, str],
) -> Dict[str, pd.DataFrame]:
    """Load .mat files for multiple alleles.

    Parameters
    ----------
    mat_dir   : directory with .mat files
    allele_map: {allele_label: filename_stem}
                e.g. {"DRB1*01:01": "DRB1_0101"}

    Returns
    -------
    {allele_label: probability_DataFrame}
    """
    result: Dict[str, pd.DataFrame] = {}
    for allele_name, stem in allele_map.items():
        path = os.path.join(mat_dir, f"{stem}.mat")
        mat = load_mat_file(path)
        if mat is not None:
            result[allele_name] = mat
            print(f"  [ref] {path}  →  {len(mat)} positions, allele={allele_name}")
        else:
            warnings.warn(f"  [ref] FAILED: {path}")
    return result


# ============================================================================
# Emission matrix extraction
# ============================================================================

def _emission_matrix(per_name_models: dict, fraction: float = 1.0) -> Optional[pd.DataFrame]:
    """Averaged probability matrix (n_states × 20) from a model dict.

    Anchor states (last two non-cycle states) are placed at positions 0 and -1.
    """
    if not per_name_models:
        return None

    model_names = list(per_name_models.keys())
    n = max(int(len(model_names) * fraction), 1)
    accumulator: Optional[pd.DataFrame] = None
    count = 0

    for name in model_names[:n]:
        model = per_name_models[name]
        dists = [
            state.distribution.parameters[0]
            for state in model.states
            if state.distribution is not None
            and not (_HAS_POMEGRANATE
                     and isinstance(state.distribution, DiscreteDistributionCycle))
        ]
        if len(dists) < 2:
            continue
        ordered = [dists[-2]] + dists[:-2] + [dists[-1]]   # anchor-first convention
        mat = pd.DataFrame(ordered, columns=AMINO_ACIDS)
        accumulator = mat if accumulator is None else accumulator + mat
        count += 1

    if accumulator is None or count == 0:
        return None

    avg = accumulator / count
    return avg.div(avg.sum(axis=1), axis=0)


# ============================================================================
# Low-level base metrics
# ============================================================================

def _js(a: np.ndarray, b: np.ndarray) -> float:
    return float(jensenshannon(a, b))


def _sim(model_mat: Optional[pd.DataFrame],
         ref_mat: Optional[pd.DataFrame]) -> float:
    """Mean per-position (1 – JS distance) between model and reference. In [0,1]."""
    if model_mat is None or ref_mat is None:
        return float("nan")
    m = model_mat.reindex(columns=AMINO_ACIDS, fill_value=1e-9).values
    r = ref_mat.reindex(columns=AMINO_ACIDS, fill_value=1e-9).values
    n = min(len(m), len(r))
    if n == 0:
        return float("nan")
    return float(1.0 - np.mean([_js(m[i], r[i]) for i in range(n)]))


def intra_cluster_compactness(matrix: pd.DataFrame) -> float:
    """Mean pairwise JS distance between positions. Lower = more specific motif."""
    vals = matrix.values
    n = len(vals)
    if n < 2:
        return 0.0
    total = sum(_js(vals[i], vals[j])
                for i in range(n) for j in range(i + 1, n))
    pairs = n * (n - 1) // 2
    return total / pairs


def inter_cluster_separation(mats: List[Optional[pd.DataFrame]]) -> float:
    """Mean JS distance between branch-averaged matrices. Higher = better."""
    valid = [m for m in mats if m is not None]
    if len(valid) < 2:
        return 0.0
    uniform = np.ones(len(AMINO_ACIDS)) / len(AMINO_ACIDS)
    n_states = max(len(m) for m in valid)
    padded = []
    for m in valid:
        arr = m.reindex(columns=AMINO_ACIDS, fill_value=1e-9).values.copy()
        if len(arr) < n_states:
            arr = np.vstack([arr, np.tile(uniform, (n_states - len(arr), 1))])
        padded.append(arr)
    total, pairs = 0.0, 0
    for i in range(len(padded)):
        for j in range(i + 1, len(padded)):
            total += sum(_js(padded[i][p], padded[j][p]) for p in range(n_states))
            pairs += n_states
    return total / pairs if pairs else 0.0


def information_content(matrix: pd.DataFrame) -> float:
    """Mean per-position IC (bits). Higher = more specific."""
    if _HAS_LOGOMAKER:
        try:
            from peptides_utils import get_frequencies
            bg = get_frequencies(len(matrix)).set_index(matrix.index)
            info = logomaker.transform_matrix(matrix, from_type="probability",
                                              to_type="information", background=bg)
        except Exception:
            info = logomaker.transform_matrix(matrix, from_type="probability",
                                              to_type="information")
        return float(info.sum(axis=1).mean())
    bg = 1.0 / len(AMINO_ACIDS)
    ic = 0.0
    for _, row in matrix.iterrows():
        p = row.values.clip(1e-9, 1.0)
        ic += float(np.sum(p * np.log2(p / bg)))
    return ic / len(matrix)


def branch_size(train_data: dict) -> int:
    return sum(len(seqs)
               for per_length in train_data.values()
               for seqs in per_length.values())


def branch_entropy(sizes: List[int]) -> float:
    s = np.array([x for x in sizes if x > 0], dtype=float)
    if s.sum() == 0:
        return 0.0
    p = s / s.sum()
    return float(-np.sum(p * np.log2(p)))


def mean_viterbi_logprob(per_name_models: dict, train_data: dict,
                         fraction: float = 0.3, max_peptides: int = 200) -> float:
    if not per_name_models:
        return float("nan")
    peptides: List[str] = []
    for per_length in train_data.values():
        for seqs in per_length.values():
            peptides.extend(list(seqs)[:max(1, max_peptides // max(len(seqs), 1))])
    peptides = list(set(peptides))[:max_peptides]
    if not peptides:
        return float("nan")
    names = list(per_name_models.keys())
    scores = []
    for name in names[:max(int(len(names) * fraction), 1)]:
        model = per_name_models[name]
        for pep in peptides:
            try:
                lp, _ = model.viterbi(pep)
                if np.isfinite(lp):
                    scores.append(lp / len(pep))
            except Exception:
                pass
    return float(np.mean(scores)) if scores else float("nan")


# ============================================================================
# NEW METRIC 1: Shannon Entropy per position
# ============================================================================

def shannon_entropy_per_position(matrix: pd.DataFrame) -> np.ndarray:
    """Shannon entropy H(p) = -Σ p_i * log2(p_i) for each position.

    Parameters
    ----------
    matrix : DataFrame (n_pos × 20), probability matrix

    Returns
    -------
    np.ndarray shape (n_pos,), entropy in bits per position.
    Maximum entropy = log2(20) ≈ 4.32 bits (uniform distribution).
    Lower entropy = more specific / conserved position.
    """
    vals = matrix.reindex(columns=AMINO_ACIDS, fill_value=1e-9).values.clip(1e-9, 1.0)
    return -np.sum(vals * np.log2(vals), axis=1)


def mean_shannon_entropy(matrix: pd.DataFrame) -> float:
    """Mean Shannon entropy across all positions (bits). Lower = more specific motif."""
    if matrix is None or len(matrix) == 0:
        return float("nan")
    return float(np.mean(shannon_entropy_per_position(matrix)))


# ============================================================================
# NEW METRIC 2: KL-divergence from background amino acid distribution
# ============================================================================

def kld_from_background(
    matrix: pd.DataFrame,
    background: Optional[np.ndarray] = None,
) -> np.ndarray:
    """KL-divergence KLD(motif_pos || background) for each position.

    KLD(P||Q) = Σ P_i * log2(P_i / Q_i)

    Parameters
    ----------
    matrix     : DataFrame (n_pos × 20), probability matrix of the motif
    background : array of length 20 with background AA frequencies.
                 If None, uses UniProt/SwissProt background (_UNIPROT_BACKGROUND).

    Returns
    -------
    np.ndarray shape (n_pos,), KLD in bits per position.
    Higher KLD = position differs more from background = more informative.
    """
    if background is None:
        background = _UNIPROT_BACKGROUND
    bg = np.array(background, dtype=float)
    bg = bg / bg.sum()   # ensure normalised
    bg = bg.clip(1e-9, 1.0)

    vals = matrix.reindex(columns=AMINO_ACIDS, fill_value=1e-9).values.clip(1e-9, 1.0)
    # KLD per position: sum over AAs of p * log2(p / q)
    return np.sum(vals * np.log2(vals / bg[None, :]), axis=1)


def mean_kld_from_background(
    matrix: pd.DataFrame,
    background: Optional[np.ndarray] = None,
) -> float:
    """Mean KLD from background across all positions (bits). Higher = more informative."""
    if matrix is None or len(matrix) == 0:
        return float("nan")
    return float(np.mean(kld_from_background(matrix, background)))


# ============================================================================
# NEW METRIC 3: Pearson Correlation Coefficient between two motif matrices
# ============================================================================

def pearson_correlation_matrices(
    mat_a: Optional[pd.DataFrame],
    mat_b: Optional[pd.DataFrame],
) -> Tuple[float, float]:
    """Pearson correlation coefficient between two probability matrices.

    Both matrices are flattened to 1-D vectors (n_pos × 20 elements each).
    If matrices differ in length, the shorter one is zero-padded.

    Parameters
    ----------
    mat_a, mat_b : DataFrames (n_pos × 20)

    Returns
    -------
    (r, p_value) — PCC in [-1, 1] and two-tailed p-value.
    r close to 1 → very similar motifs.
    """
    if mat_a is None or mat_b is None:
        return float("nan"), float("nan")

    a = mat_a.reindex(columns=AMINO_ACIDS, fill_value=0.0).values
    b = mat_b.reindex(columns=AMINO_ACIDS, fill_value=0.0).values

    # Pad to same length
    n = max(len(a), len(b))
    if len(a) < n:
        a = np.vstack([a, np.zeros((n - len(a), len(AMINO_ACIDS)))])
    if len(b) < n:
        b = np.vstack([b, np.zeros((n - len(b), len(AMINO_ACIDS)))])

    a_flat = a.flatten()
    b_flat = b.flatten()

    if np.std(a_flat) < 1e-12 or np.std(b_flat) < 1e-12:
        return float("nan"), float("nan")

    r, p = pearsonr(a_flat, b_flat)
    return float(r), float(p)


def pearson_branch_vs_reference(
    branch_mat: Optional[pd.DataFrame],
    ref_mat: Optional[pd.DataFrame],
) -> float:
    """Convenience wrapper: returns just the PCC (r) value."""
    r, _ = pearson_correlation_matrices(branch_mat, ref_mat)
    return r


# ============================================================================
# NEW METRIC 4: Confusion Matrix + MCC + PPV
# ============================================================================

def confusion_matrix_metrics(
    true_labels: List[str],
    pred_labels: List[str],
    allele_names: Optional[List[str]] = None,
) -> Dict:
    """Compute confusion matrix, MCC, and PPV for multi-class branch assignment.

    Designed for the scenario where each peptide has a known ground-truth allele
    (true_labels) and is assigned to a branch that maps to an allele (pred_labels).

    Parameters
    ----------
    true_labels  : list of ground-truth allele labels per peptide
    pred_labels  : list of predicted allele labels per peptide (same length)
    allele_names : ordered list of class names. If None, inferred from labels.

    Returns
    -------
    dict with keys:
        'confusion_matrix'   — pd.DataFrame (true × pred)
        'MCC'                — Matthews Correlation Coefficient (macro-averaged)
        'PPV_per_class'      — dict {allele: PPV}
        'PPV_macro'          — macro-averaged PPV
        'accuracy'           — overall accuracy
        'MCC_per_class'      — dict {allele: binary MCC (one-vs-rest)}
    """
    if len(true_labels) != len(pred_labels):
        raise ValueError("true_labels and pred_labels must have the same length.")

    if allele_names is None:
        allele_names = sorted(set(true_labels) | set(pred_labels))

    n = len(allele_names)
    idx = {a: i for i, a in enumerate(allele_names)}

    # Build confusion matrix (true rows, pred cols)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        if t in idx and p in idx:
            cm[idx[t], idx[p]] += 1

    cm_df = pd.DataFrame(cm, index=allele_names, columns=allele_names)
    cm_df.index.name = "true \\ pred"

    total = cm.sum()
    accuracy = float(np.trace(cm) / total) if total > 0 else float("nan")

    # PPV (Positive Predictive Value = Precision) per class
    ppv_per_class: Dict[str, float] = {}
    mcc_per_class: Dict[str, float] = {}

    for allele in allele_names:
        i = idx[allele]
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP        # predicted as allele, but wrong
        FN = cm[i, :].sum() - TP        # true allele, but predicted as other
        TN = total - TP - FP - FN

        ppv_per_class[allele] = float(TP / (TP + FP)) if (TP + FP) > 0 else float("nan")

        # Binary MCC (one-vs-rest)
        denom = math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
        mcc_per_class[allele] = float((TP * TN - FP * FN) / denom) if denom > 0 else float("nan")

    ppv_values = [v for v in ppv_per_class.values() if np.isfinite(v)]
    ppv_macro = float(np.mean(ppv_values)) if ppv_values else float("nan")

    mcc_values = [v for v in mcc_per_class.values() if np.isfinite(v)]
    mcc_macro = float(np.mean(mcc_values)) if mcc_values else float("nan")

    return {
        "confusion_matrix": cm_df,
        "MCC": mcc_macro,
        "MCC_per_class": mcc_per_class,
        "PPV_per_class": ppv_per_class,
        "PPV_macro": ppv_macro,
        "accuracy": accuracy,
    }


def print_confusion_metrics(metrics: Dict, title: str = "") -> None:
    """Pretty-print confusion matrix metrics."""
    sep = "=" * 60
    print(f"\n{sep}")
    if title:
        print(f"  Confusion Metrics: {title}")
    print(sep)
    print(metrics["confusion_matrix"].to_string())
    print(f"\n  Accuracy      : {metrics['accuracy']:.4f}")
    print(f"  MCC (macro)   : {metrics['MCC']:.4f}")
    print(f"  PPV (macro)   : {metrics['PPV_macro']:.4f}")
    print("\n  Per-class:")
    for allele in metrics["PPV_per_class"]:
        mcc_v = metrics["MCC_per_class"].get(allele, float("nan"))
        ppv_v = metrics["PPV_per_class"].get(allele, float("nan"))
        print(f"    {allele:20s}  MCC={mcc_v:.4f}  PPV={ppv_v:.4f}")
    print(sep + "\n")


# ============================================================================
# NEW METRIC 5: Sequence logo data + plotting
# ============================================================================

def sequence_logo_data(
    matrix: pd.DataFrame,
    background: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """Compute IC-weighted letter heights for a sequence logo.

    Each cell = p_i * IC_pos  where IC_pos = log2(20) - H_pos.
    This matches the standard Schneider & Stephens (1990) convention.

    Parameters
    ----------
    matrix     : DataFrame (n_pos × 20), probability matrix
    background : optional background frequencies. If None, uses uniform (1/20).

    Returns
    -------
    DataFrame (n_pos × 20) with IC-weighted heights (bits).
    """
    if matrix is None or len(matrix) == 0:
        return pd.DataFrame()

    mat = matrix.reindex(columns=AMINO_ACIDS, fill_value=1e-9)

    if background is None:
        bg = np.ones(len(AMINO_ACIDS)) / len(AMINO_ACIDS)
    else:
        bg = np.array(background, dtype=float)
        bg = bg / bg.sum()

    bg = bg.clip(1e-9, 1.0)
    vals = mat.values.clip(1e-9, 1.0)

    # Shannon entropy per position
    H = -np.sum(vals * np.log2(vals), axis=1, keepdims=True)
    # IC per position (bits) — using KLD-style relative to background
    R = np.sum(vals * np.log2(vals / bg[None, :]), axis=1, keepdims=True)

    logo_heights = vals * R   # negative values (below background) get negative height
    return pd.DataFrame(logo_heights, columns=AMINO_ACIDS, index=mat.index)


def plot_sequence_logo(
    matrix: pd.DataFrame,
    title: str = "",
    save_path: Optional[str] = None,
    background: Optional[np.ndarray] = None,
    figsize: Tuple[float, float] = (12, 3),
) -> None:
    """Plot a sequence logo using logomaker (if available) or a text fallback.

    Parameters
    ----------
    matrix    : DataFrame (n_pos × 20), probability matrix
    title     : plot title
    save_path : if given, saves the figure to this path (PNG/PDF/SVG)
    background: background frequencies for IC computation (default: uniform)
    figsize   : matplotlib figure size

    Notes
    -----
    Requires: logomaker, matplotlib
    """
    if matrix is None or len(matrix) == 0:
        print("  [logo] empty matrix, skipping plot.")
        return

    logo_df = sequence_logo_data(matrix, background)

    if _HAS_LOGOMAKER and _HAS_MPL:
        fig, ax = plt.subplots(figsize=figsize)
        try:
            logo = logomaker.Logo(
                logo_df,
                ax=ax,
                color_scheme="chemistry",
                vpad=0.05,
                width=0.8,
            )
            logo.style_spines(visible=False)
            logo.style_spines(spines=["left", "bottom"], visible=True)
            logo.ax.set_ylabel("bits", fontsize=11)
            logo.ax.set_xlabel("position", fontsize=11)
            if title:
                logo.ax.set_title(title, fontsize=13, fontweight="bold")
        except Exception as e:
            warnings.warn(f"  [logo] logomaker failed: {e}; falling back.")
            _text_logo(matrix, ax, title)

        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  [logo] saved → {save_path}")
        plt.show()
        plt.close(fig)

    elif _HAS_MPL:
        # Simple bar-chart fallback
        fig, ax = plt.subplots(figsize=figsize)
        _text_logo(matrix, ax, title)
        plt.tight_layout()
        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  [logo] saved → {save_path}")
        plt.show()
        plt.close(fig)

    else:
        # Text fallback: print top AA per position
        print(f"\n  [logo] {title or 'Sequence Logo'}")
        print("  pos  top_AA  p     IC(bits)")
        print("  " + "-" * 32)
        ic_vals = kld_from_background(matrix, background)
        for pos in range(len(matrix)):
            row = matrix.iloc[pos]
            top_aa = row.idxmax()
            top_p  = row.max()
            print(f"  {pos+1:3d}  {top_aa:6s}  {top_p:.3f}  {ic_vals[pos]:.3f}")


def _text_logo(matrix: pd.DataFrame, ax, title: str = "") -> None:
    """Fallback: stacked bar chart of top-5 AAs per position."""
    if not _HAS_MPL:
        return
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n_pos = len(matrix)
    top_n = 5
    colors = plt.cm.tab20.colors

    bottoms = np.zeros(n_pos)
    legend_handles = []
    seen_aas: Dict[str, bool] = {}

    # Sort AAs by total frequency descending
    total_freq = matrix.sum(axis=0).sort_values(ascending=False)
    top_aas = total_freq.index[:top_n].tolist()

    for k, aa in enumerate(top_aas):
        heights = matrix[aa].values
        color = colors[k % len(colors)]
        ax.bar(range(n_pos), heights, bottom=bottoms, color=color,
               label=aa, width=0.8, edgecolor="white", linewidth=0.5)
        bottoms += heights
        if aa not in seen_aas:
            legend_handles.append(Patch(facecolor=color, label=aa))
            seen_aas[aa] = True

    ax.set_xticks(range(n_pos))
    ax.set_xticklabels([str(i + 1) for i in range(n_pos)])
    ax.set_ylabel("probability")
    ax.set_xlabel("position")
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              ncol=min(top_n, 5), framealpha=0.7)
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold")


def plot_all_branch_logos(
    branch_matrices: Dict[str, Optional[pd.DataFrame]],
    save_dir: Optional[str] = None,
    background: Optional[np.ndarray] = None,
) -> None:
    """Plot sequence logos for all branches side-by-side.

    Parameters
    ----------
    branch_matrices : {"branch1": mat1, "branch2": mat2, ...}
    save_dir        : directory to save individual PNGs (one per branch)
    background      : background frequencies (default: uniform)
    """
    valid = {k: v for k, v in branch_matrices.items() if v is not None}
    if not valid:
        print("  [logo] no valid branch matrices.")
        return

    for branch_name, mat in valid.items():
        path = (os.path.join(save_dir, f"logo_{branch_name}.png")
                if save_dir else None)
        plot_sequence_logo(mat, title=branch_name, save_path=path,
                           background=background)


# ============================================================================
# Core: branch × allele similarity matrix
# ============================================================================

def branch_allele_similarity_matrix(
    branch_matrices: Dict[str, Optional[pd.DataFrame]],
    reference_matrices: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compute similarity(branch_i, allele_j) for all i×j combinations.

    Parameters
    ----------
    branch_matrices    : {"branch1": mat1, "branch2": mat2, "branch3": mat3, "branch4": mat4}
    reference_matrices : {"DRB1*01:01": ref1, "DRB1*03:01": ref2, ...}

    Returns
    -------
    DataFrame  index=branch labels, columns=allele labels, values=similarity in [0,1]
    """
    branches = list(branch_matrices.keys())
    alleles = list(reference_matrices.keys())
    data = {
        allele: [_sim(branch_matrices[b], reference_matrices[allele])
                 for b in branches]
        for allele in alleles
    }
    return pd.DataFrame(data, index=branches)


def branch_allele_pcc_matrix(
    branch_matrices: Dict[str, Optional[pd.DataFrame]],
    reference_matrices: Dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """PCC(branch_i, allele_j) matrix. Complement to JS-similarity matrix.

    Returns
    -------
    DataFrame  index=branch labels, columns=allele labels, values=PCC in [-1, 1]
    """
    branches = list(branch_matrices.keys())
    alleles = list(reference_matrices.keys())
    data = {
        allele: [pearson_branch_vs_reference(branch_matrices[b],
                                             reference_matrices[allele])
                 for b in branches]
        for allele in alleles
    }
    return pd.DataFrame(data, index=branches)


def assign_branches_to_alleles(sim_df: pd.DataFrame) -> Dict[str, str]:
    """Greedy one-to-one assignment: each branch → best-matching allele.

    Uses a greedy max approach (not full Hungarian, but sufficient for small branch × allele matrices).
    A branch that is best matched to an already-taken allele gets the next best.

    Returns {branch_label: allele_label}
    """
    df = sim_df.copy().astype(float)
    assignment: Dict[str, str] = {}
    used_alleles: set = set()

    # Iterate over branches in order of their best similarity (most certain first)
    branch_best = df.max(axis=1).sort_values(ascending=False)
    for branch in branch_best.index:
        row = df.loc[branch].drop(labels=list(used_alleles), errors="ignore")
        if row.empty:
            assignment[branch] = "unassigned"
        else:
            best_allele = row.idxmax()
            assignment[branch] = best_allele
            used_alleles.add(best_allele)

    return assignment


def print_branch_allele_table(
    sim_df: pd.DataFrame,
    assignment: Dict[str, str],
    iteration_name: str = "",
    step: str = "",
    pcc_df: Optional[pd.DataFrame] = None,
) -> None:
    """Pretty-print the branch × allele similarity matrix with assignment markers.

    If pcc_df is provided, also prints PCC values in parentheses.
    """
    header = f"  Branch→Allele similarity  [{iteration_name} | {step}]"
    print("\n" + "=" * max(len(header), 60))
    print(header)
    print("=" * max(len(header), 60))

    alleles = sim_df.columns.tolist()
    col_w = max(len(a) for a in alleles) + 2
    branch_w = 10

    # header row
    print(f"  {'branch':<{branch_w}}" +
          "".join(f"{a:>{col_w}}" for a in alleles) +
          f"  {'→ assigned allele'}")
    print("  " + "-" * (branch_w + col_w * len(alleles) + 20))

    for branch in sim_df.index:
        sims = sim_df.loc[branch]
        assigned = assignment.get(branch, "?")
        row_str = f"  {branch:<{branch_w}}"
        for allele in alleles:
            val = sims[allele]
            marker = "★" if allele == assigned else " "
            if not np.isnan(val):
                if pcc_df is not None and allele in pcc_df.columns:
                    pcc_val = pcc_df.loc[branch, allele]
                    pcc_str = f"({pcc_val:.2f})" if np.isfinite(pcc_val) else "(nan)"
                    cell = f"{val:.3f}{marker}{pcc_str}"
                else:
                    cell = f"{val:.3f}{marker}"
            else:
                cell = "  nan "
            row_str += f"{cell:>{col_w + (8 if pcc_df is not None else 0)}}"
        row_str += f"  {assigned}"
        print(row_str)

    if pcc_df is not None:
        print("  (values in parentheses = PCC)")
    print("=" * max(len(header), 60) + "\n")


# ============================================================================
# Convenience one-shot function
# ============================================================================

def compare_branches_to_all_references(
    per_name_models1: dict,
    per_name_models2: dict,
    per_name_models3: Optional[dict],
    reference_matrices: Dict[str, pd.DataFrame],
    models_fraction: float = 0.7,
    iteration_name: str = "",
    step: str = "",
    compute_pcc: bool = True,
    per_name_models4: Optional[dict] = None,
) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Compare every branch to every reference allele.

    Supports four branches: branch1, branch2, branch3, branch4.

    Returns
    -------
    sim_df     : DataFrame (branch × allele) with JS-similarity values
    assignment : {branch_label: allele_label} greedy one-to-one assignment
    """
    frac = models_fraction
    branch_mats = {
        "branch1": _emission_matrix(per_name_models1, frac),
        "branch2": _emission_matrix(per_name_models2, frac),
        "branch3": _emission_matrix(per_name_models3 or {}, frac),
        "branch4": _emission_matrix(per_name_models4 or {}, frac),
    }
    # drop branches with no models
    branch_mats = {k: v for k, v in branch_mats.items() if v is not None}

    sim_df = branch_allele_similarity_matrix(branch_mats, reference_matrices)
    pcc_df = branch_allele_pcc_matrix(branch_mats, reference_matrices) if compute_pcc else None
    assignment = assign_branches_to_alleles(sim_df)
    print_branch_allele_table(sim_df, assignment, iteration_name, step, pcc_df=pcc_df)
    return sim_df, assignment


# ============================================================================
# MetricsTracker
# ============================================================================

class MetricsTracker:
    """Collect per-step metrics for a dummy_allele deconvolution run.

    At every recorded step:
    - Standard clustering metrics (IC, compactness, separation, entropy, Viterbi)
    - Shannon entropy per position (mean across positions)
    - KLD from background amino acid distribution (mean across positions)
    - Full branch × allele similarity matrix vs. all reference PSSMs (JS-based)
    - PCC branch × allele matrix
    - Greedy branch→allele assignment
    - Assignment stability across steps (how often assignment changes)
    - PPV and MCC if ground-truth labels provided

    Parameters
    ----------
    experiment_params   : used to read statistics_estimate_model_fraction
    models_fraction     : fraction of sorted models to average
    reference_matrices  : {allele_label: probability_DataFrame}
                          loaded by load_reference_matrices()
    background          : background AA frequencies for KLD/logo (default: UniProt)
    """

    def __init__(
        self,
        experiment_params=None,
        models_fraction: float = 0.7,
        reference_matrices: Optional[Dict[str, pd.DataFrame]] = None,
        background: Optional[np.ndarray] = None,
    ):
        self._rows: List[dict] = []
        self._sim_history: List[dict] = []   # raw sim_df per step
        self._models_fraction = models_fraction
        self._reference_matrices: Dict[str, pd.DataFrame] = reference_matrices or {}
        self._background = background if background is not None else _UNIPROT_BACKGROUND
        self._last_assignment: Optional[Dict[str, str]] = None

        if experiment_params is not None:
            try:
                f = getattr(experiment_params.model_training_params,
                            "statistics_estimate_model_fraction", models_fraction)
                self._models_fraction = f
            except Exception:
                pass

        if self._reference_matrices:
            print(f"  [MetricsTracker] reference alleles: "
                  f"{list(self._reference_matrices.keys())}")

    # ------------------------------------------------------------------

    def record_step(
        self,
        iteration_name: str,
        step: str,
        per_name_models1: dict,
        per_name_models2: dict,
        per_name_models3: Optional[dict] = None,
        train_data1: Optional[dict] = None,
        train_data2: Optional[dict] = None,
        train_data3: Optional[dict] = None,
        peptide_length: int = 9,
        true_labels: Optional[List[str]] = None,
        pred_labels: Optional[List[str]] = None,
        allele_names_for_cm: Optional[List[str]] = None,
        plot_logos: bool = False,
        logo_save_dir: Optional[str] = None,
        per_name_models4: Optional[dict] = None,
        train_data4: Optional[dict] = None,
    ) -> dict:
        """Record one row of metrics.

        Supports four branches: branch1, branch2, branch3, branch4.

        New parameters vs. original:
        ----------------------------
        true_labels, pred_labels : if provided, computes confusion matrix,
                                   MCC, and PPV.
        allele_names_for_cm      : ordered allele list for confusion matrix.
        plot_logos               : if True, plots sequence logos for each branch.
        logo_save_dir            : directory to save logo PNGs.
        """
        frac = self._models_fraction

        branch_models = {
            "branch1": per_name_models1 or {},
            "branch2": per_name_models2 or {},
            "branch3": per_name_models3 or {},
            "branch4": per_name_models4 or {},
        }
        branch_train_data = {
            "branch1": train_data1,
            "branch2": train_data2,
            "branch3": train_data3,
            "branch4": train_data4,
        }
        branch_names = list(branch_models.keys())

        branch_mats_all = {
            branch: _emission_matrix(models, frac)
            for branch, models in branch_models.items()
        }
        branch_mats = {
            branch: mat
            for branch, mat in branch_mats_all.items()
            if mat is not None
        }

        branch_sizes = {
            branch: (
                branch_size(branch_train_data[branch])
                if branch_train_data[branch] else len(branch_models[branch])
            )
            for branch in branch_names
        }
        sizes = [s for s in branch_sizes.values() if s > 0]

        sep = inter_cluster_separation(list(branch_mats_all.values()))
        ent = branch_entropy(sizes)

        ic = {
            branch: information_content(mat) if mat is not None else float("nan")
            for branch, mat in branch_mats_all.items()
        }
        compactness = {
            branch: intra_cluster_compactness(mat) if mat is not None else float("nan")
            for branch, mat in branch_mats_all.items()
        }
        viterbi_lp = {}
        for branch in branch_names:
            models = branch_models[branch]
            td = branch_train_data[branch]
            viterbi_lp[branch] = (
                mean_viterbi_logprob(models, td, min(frac, 0.3))
                if td and models else float("nan")
            )

        # ── NEW: Shannon entropy (mean per branch) ──────────────────────
        shannon_entropy = {
            branch: mean_shannon_entropy(mat) if mat is not None else float("nan")
            for branch, mat in branch_mats_all.items()
        }

        # ── NEW: KLD from background (mean per branch) ──────────────────
        kld_background = {
            branch: mean_kld_from_background(mat, self._background) if mat is not None else float("nan")
            for branch, mat in branch_mats_all.items()
        }

        # ── NEW: PCC between all branch pairs ───────────────────────────
        branch_pair_pcc = {}
        for i, branch_a in enumerate(branch_names):
            for branch_b in branch_names[i + 1:]:
                pcc_value, _ = pearson_correlation_matrices(
                    branch_mats_all[branch_a], branch_mats_all[branch_b])
                branch_pair_pcc[f"pcc_{branch_a}_vs_{branch_b}"] = pcc_value

        # ── branch × allele similarity matrix (JS-based) ────────────────
        sim_df = pd.DataFrame()
        pcc_df = pd.DataFrame()
        assignment: Dict[str, str] = {}
        assignment_changed = False

        if self._reference_matrices and branch_mats:
            sim_df = branch_allele_similarity_matrix(branch_mats, self._reference_matrices)
            pcc_df = branch_allele_pcc_matrix(branch_mats, self._reference_matrices)
            assignment = assign_branches_to_alleles(sim_df)
            print_branch_allele_table(sim_df, assignment, iteration_name, step,
                                      pcc_df=pcc_df)

            if self._last_assignment is not None:
                assignment_changed = assignment != self._last_assignment
            self._last_assignment = assignment.copy()

        self._sim_history.append({
            "iteration": iteration_name,
            "step": step,
            "sim_df":   sim_df.copy() if not sim_df.empty else None,
            "pcc_df":   pcc_df.copy() if not pcc_df.empty else None,
            "assignment": assignment.copy(),
        })

        # ── NEW: Confusion matrix + MCC + PPV ───────────────────────────
        cm_metrics: Dict = {}
        if true_labels and pred_labels:
            cm_metrics = confusion_matrix_metrics(
                true_labels, pred_labels, allele_names_for_cm)
            print_confusion_metrics(
                cm_metrics, title=f"{iteration_name} | {step}")

        # ── NEW: Sequence logos ──────────────────────────────────────────
        if plot_logos and branch_mats:
            plot_all_branch_logos(branch_mats, save_dir=logo_save_dir,
                                  background=self._background)

        # ── Helper lambdas ───────────────────────────────────────────────
        def _r(x):
            return round(float(x), 4) if np.isfinite(x) else float("nan")

        def _nanmean_safe(vals):
            arr = [v for v in vals if np.isfinite(v)]
            return float(np.mean(arr)) if arr else float("nan")

        # Flatten sim_df into row columns: sim_branch1_DRB1*01:01 etc.
        sim_flat: dict = {}
        pcc_flat: dict = {}
        if not sim_df.empty:
            for branch in sim_df.index:
                for allele in sim_df.columns:
                    sim_flat[f"sim_{branch}_{allele}"] = _r(sim_df.loc[branch, allele])
                    if not pcc_df.empty and allele in pcc_df.columns:
                        pcc_flat[f"pcc_{branch}_{allele}"] = _r(pcc_df.loc[branch, allele])

        # Best sim per branch (across alleles) and best allele per branch
        best_sim_per_branch: dict = {}
        if not sim_df.empty:
            for branch in sim_df.index:
                best_sim_per_branch[f"best_sim_{branch}"] = _r(sim_df.loc[branch].max())
                best_sim_per_branch[f"best_allele_{branch}"] = sim_df.loc[branch].idxmax()
                if not pcc_df.empty:
                    best_sim_per_branch[f"best_pcc_{branch}"] = _r(pcc_df.loc[branch].max())

        # Confusion metrics scalars
        cm_scalars: dict = {}
        if cm_metrics:
            cm_scalars["MCC"] = _r(cm_metrics.get("MCC", float("nan")))
            cm_scalars["PPV_macro"] = _r(cm_metrics.get("PPV_macro", float("nan")))
            cm_scalars["accuracy"] = _r(cm_metrics.get("accuracy", float("nan")))
            for allele, ppv in cm_metrics.get("PPV_per_class", {}).items():
                cm_scalars[f"PPV_{allele}"] = _r(ppv)
            for allele, mcc_v in cm_metrics.get("MCC_per_class", {}).items():
                cm_scalars[f"MCC_{allele}"] = _r(mcc_v)

        row = {
            "iteration":         iteration_name,
            "step":              step,
            "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
            **{f"size_{branch}": branch_sizes[branch] for branch in branch_names},
            "total_peptides":    sum(sizes),
            "split_entropy":     _r(ent),
            "inter_cluster_sep": _r(sep),
            # IC
            **{f"IC_{branch}": _r(ic[branch]) for branch in branch_names},
            "IC_mean":           _r(_nanmean_safe(list(ic.values()))),
            # Compactness
            **{f"compactness_{branch}": _r(compactness[branch]) for branch in branch_names},
            "compactness_mean":  _r(_nanmean_safe(list(compactness.values()))),
            # Viterbi
            **{f"viterbi_lp_{branch}": _r(viterbi_lp[branch]) for branch in branch_names},
            "viterbi_lp_mean":   _r(_nanmean_safe(list(viterbi_lp.values()))),
            # NEW: Shannon entropy
            **{f"shannon_entropy_{branch}": _r(shannon_entropy[branch]) for branch in branch_names},
            "shannon_entropy_mean": _r(_nanmean_safe(list(shannon_entropy.values()))),
            # NEW: KLD from background
            **{f"kld_background_{branch}": _r(kld_background[branch]) for branch in branch_names},
            "kld_background_mean": _r(_nanmean_safe(list(kld_background.values()))),
            # NEW: PCC between branch pairs
            **{k: _r(v) for k, v in branch_pair_pcc.items()},
            # Assignment change flag
            "assignment_changed": assignment_changed,
            # Flattened sim/pcc matrices
            **sim_flat,
            **pcc_flat,
            **best_sim_per_branch,
            # NEW: Confusion metrics (if labels provided)
            **cm_scalars,
            # Human-readable assignment string
            "assignment": " | ".join(
                f"{b}→{a}" for b, a in sorted(assignment.items())),
        }

        self._rows.append(row)
        self._print_row(row)
        return row


    # ------------------------------------------------------------------

    def assignment_table(self) -> pd.DataFrame:
        """Return a DataFrame tracking branch→allele assignment over all steps."""
        rows = []
        for entry in self._sim_history:
            base = {"iteration": entry["iteration"], "step": entry["step"]}
            base.update(entry["assignment"])
            rows.append(base)
        return pd.DataFrame(rows)

    def similarity_history(self) -> List[dict]:
        """Raw list of {iteration, step, sim_df, pcc_df, assignment} dicts."""
        return self._sim_history

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._rows)

    def save_csv(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.to_dataframe().to_csv(path, index=False)
        print(f"  [MetricsTracker] {len(self._rows)} rows → {path}")

    def save_assignment_csv(self, path: str) -> None:
        """Save branch→allele assignment table over all steps."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.assignment_table().to_csv(path, index=False)
        print(f"  [MetricsTracker] assignment table → {path}")

    def save_latex_table(self, path: str) -> None:
        df = self.to_dataframe()
        if df.empty:
            return
        # Key columns + all best_sim / best_pcc / CM columns
        fixed = ["iteration", "step", "size_branch1", "size_branch2", "size_branch3", "size_branch4",
                 "split_entropy", "inter_cluster_sep", "IC_mean",
                 "compactness_mean", "viterbi_lp_mean",
                 "shannon_entropy_mean", "kld_background_mean"]
        sim_cols    = sorted(c for c in df.columns if c.startswith("best_sim_"))
        pcc_cols    = sorted(c for c in df.columns if c.startswith("best_pcc_"))
        assign_cols = sorted(c for c in df.columns if c.startswith("best_allele_"))
        cm_cols     = [c for c in ["MCC", "PPV_macro", "accuracy"] if c in df.columns]
        cols = [c for c in fixed + sim_cols + pcc_cols + assign_cols + cm_cols
                if c in df.columns]
        sub = df[cols].copy()
        for c in sub.select_dtypes(float).columns:
            sub[c] = sub[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "–")
        rename = {
            "iteration": "Iter.", "step": "Step",
            "size_branch1": r"$N_1$", "size_branch2": r"$N_2$",
            "size_branch3": r"$N_3$", "size_branch4": r"$N_4$",
            "split_entropy": r"$H$",
            "inter_cluster_sep": r"$\Delta_{JS}$",
            "IC_mean": r"$\overline{IC}$",
            "compactness_mean": r"$\overline{D}_{JS}$",
            "viterbi_lp_mean": r"$\overline{\ell_V}$",
            "shannon_entropy_mean": r"$\overline{H}_{Sh}$",
            "kld_background_mean": r"$\overline{KLD}$",
            "MCC": "MCC",
            "PPV_macro": r"$\overline{PPV}$",
            "accuracy": "Acc.",
        }
        sub = sub.rename(columns=rename)
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(sub.to_latex(index=False, escape=False,
                                 caption="HMM deconvolution metrics.",
                                 label="tab:hmm_metrics"))
        print(f"  [MetricsTracker] LaTeX → {path}")

    def print_summary(self) -> None:
        df = self.to_dataframe()
        if df.empty:
            print("  [MetricsTracker] no metrics yet.")
            return
        fixed = ["iteration", "step", "split_entropy",
                 "inter_cluster_sep", "IC_mean",
                 "shannon_entropy_mean", "kld_background_mean"]
        best_cols  = sorted(c for c in df.columns if c.startswith("best_sim_"))
        pcc_cols   = sorted(c for c in df.columns if c.startswith("best_pcc_"))
        cm_cols    = [c for c in ["MCC", "PPV_macro"] if c in df.columns]
        assign_col = ["assignment"] if "assignment" in df.columns else []
        cols = [c for c in fixed + best_cols + pcc_cols + cm_cols + assign_col
                if c in df.columns]
        print("\n" + "=" * 80)
        print("  HMM Metrics Summary")
        print("=" * 80)
        print(df[cols].to_string(index=False))
        print("=" * 80 + "\n")

    # ------------------------------------------------------------------
    # Logo helpers (call at any time after steps are recorded)
    # ------------------------------------------------------------------

    def plot_logos_for_step(
        self,
        iteration_name: str,
        step: str,
        per_name_models1: dict,
        per_name_models2: dict,
        per_name_models3: Optional[dict] = None,
        save_dir: Optional[str] = None,
        per_name_models4: Optional[dict] = None,
    ) -> None:
        """Plot sequence logos for all branches at a given step."""
        frac = self._models_fraction
        branch_mats = {
            "branch1": _emission_matrix(per_name_models1 or {}, frac),
            "branch2": _emission_matrix(per_name_models2 or {}, frac),
            "branch3": _emission_matrix(per_name_models3 or {}, frac),
            "branch4": _emission_matrix(per_name_models4 or {}, frac),
        }
        branch_mats = {k: v for k, v in branch_mats.items() if v is not None}
        plot_all_branch_logos(branch_mats, save_dir=save_dir,
                              background=self._background)


    # ------------------------------------------------------------------

    @staticmethod
    def _print_row(row: dict) -> None:
        best_sims = {k: v for k, v in row.items() if k.startswith("best_sim_")}
        sim_str = "  ".join(
            f"{k.replace('best_sim_', '')}={v:.3f}"
            for k, v in best_sims.items()
            if isinstance(v, float) and np.isfinite(v)
        )
        mcc_str  = f" | MCC={row['MCC']:.3f}"   if "MCC" in row and np.isfinite(row.get("MCC", float("nan"))) else ""
        kld_str  = f" | KLD={row['kld_background_mean']:.3f}" if np.isfinite(row.get("kld_background_mean", float("nan"))) else ""
        shan_str = f" | H={row['shannon_entropy_mean']:.3f}"  if np.isfinite(row.get("shannon_entropy_mean", float("nan"))) else ""
        print(
            f"  [metrics] {row['iteration']:12s} | {row['step']:15s} | "
            f"IC={row['IC_mean']:.3f} | sep={row['inter_cluster_sep']:.3f}"
            + shan_str + kld_str + mcc_str
            + (f" | {sim_str}" if sim_str else "")
        )