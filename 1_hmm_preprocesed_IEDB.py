from __future__ import annotations

import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import seaborn as sns
import logomaker


# ==========================
# 1. Загрузка и базовая очистка
# ==========================

def load_mhc_data(
    csv_path: str,
    peptide_col: str = "Name",
    allele_col: str = "Name.6"
) -> pd.DataFrame:
    """
    Загружает сырые данные MHC, переименовывает колонки и добавляет длину пептида.
    """
    df = pd.read_csv(csv_path)

    df = (
        df
        .rename(columns={peptide_col: "Peptide", allele_col: "Allele"})
        .assign(Length=lambda d: d["Peptide"].str.len())
    )

    # Оставим только нужные столбцы (по аналогии с твоим кодом)
    # 0 - Peptide, 5 - Allele, 7,8,9 - Class, Qualitative Measurement и т.п.
    df = df.iloc[:, [0, 5, 7, 8, 9]]
    return df


def filter_mhc_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Фильтрует датасет:
    1. Оставляет только нормальные аминокислотные последовательности.
    2. Делает бинарный таргет Binder из Qualitative Measurement.
    3. Убирает дубликаты по ключевым полям.
    """
    filtered_df = df.copy()

    # 1. только стандартные аминокислоты
    filtered_df = filtered_df[
        filtered_df["Peptide"].str.match(r"^[ACDEFGHIKLMNPQRSTVWY]+$", na=False)
    ]

    # 2. бинарный таргет
    binder_list = ["Positive", "Positive-High", "Positive-Intermediate", "Positive-Low"]
    non_binder_list = ["Negative"]

    filtered_df = filtered_df[
        filtered_df["Qualitative Measurement"].isin(binder_list + non_binder_list)
    ]

    filtered_df = filtered_df.copy()
    filtered_df.loc[:, "Binder"] = (
        filtered_df["Qualitative Measurement"].isin(binder_list).astype(int)
    )

    print(f"Before dropping duplicates: {len(filtered_df)}")

    # 3. удаляем дубликаты
    filtered_df = filtered_df.drop_duplicates(
        ["Peptide", "Allele", "Class", "Qualitative Measurement", "Length"]
    ).reset_index(drop=True)

    print(f"After dropping duplicates: {len(filtered_df)}")
    return filtered_df


def select_alleles(df: pd.DataFrame, selected_alleles: List[str]) -> pd.DataFrame:
    """
    Оставляет только заданный список аллелей из датафрейма.
    """
    df_selected = df[df["Allele"].isin(selected_alleles)].reset_index(drop=True)

    binder_percent = df_selected["Binder"].mean() * 100
    binder_count = (df_selected["Binder"] == 1).sum()
    print(f"{binder_percent:.2f}% binders ({binder_count} samples) in selected alleles")

    return df_selected


# ==========================
# 2. Разбиение по аллелям и длинам
# ==========================

def make_per_allele_df(
    df_selected: pd.DataFrame,
    selected_alleles: List[str],
    classI_len_range: Tuple[int, int] = (8, 13),
    classII_len_range: Tuple[int, int] = (12, 18),
) -> Dict[str, pd.DataFrame]:
    """
    Строит словарь:
    { allele_name -> DataFrame для этого аллеля с фильтрацией по длине
      (разные диапазоны для Class I и Class II) }.
    """
    per_allele_df: Dict[str, pd.DataFrame] = {}

    for allele_name in selected_alleles:
        allele_rows = df_selected[df_selected["Allele"] == allele_name]
        if allele_rows.empty:
            print(f"Warning: no rows for allele {allele_name}")
            per_allele_df[allele_name] = allele_rows
            continue

        allele_class = allele_rows["Class"].iloc[0]

        if allele_class == "I":
            L_min, L_max = classI_len_range
        else:
            L_min, L_max = classII_len_range

        allele_df = allele_rows[
            (allele_rows["Length"] >= L_min) & (allele_rows["Length"] <= L_max)
        ].reset_index(drop=True)

        per_allele_df[allele_name] = allele_df

    return per_allele_df


# ==========================
# 3. Вспомогательные функции для логов
# ==========================

def plot_binder_length_distribution(df_selected: pd.DataFrame) -> None:
    """
    Violin-plot распределения длин пептидов по аллелям (только binders).
    """
    df = df_selected[df_selected["Binder"] == 1]

    plt.figure(figsize=(14, 6))
    sns.violinplot(
        data=df,
        x="Allele",
        y="Length",
        inner="box",
        scale="width",
        cut=0,
    )
    sns.stripplot(
        data=df,
        x="Allele",
        y="Length",
        size=1.5,
        alpha=0.1,
        color="white",
    )

    plt.xticks(rotation=90)
    plt.xlabel("Allele")
    plt.ylabel("Peptide length")
    plt.title("Distribution of peptide lengths for binder peptides per allele")
    plt.tight_layout()
    plt.show()


def make_logo(peptides: List[str], name: str, peptide_len: int | None = None, ax=None):
    sns.set_theme(style="white")

    if peptide_len is None and len(peptides) > 0:
        peptide_len = len(peptides[0])

    matrix = logomaker.alignment_to_matrix(peptides)
    info_matrix = logomaker.transform_matrix(
        matrix,
        from_type="counts",
        to_type="information",
    )

    info_matrix.index = np.arange(1, peptide_len + 1)

    if ax is None:
        _, ax = plt.subplots(figsize=(4, 2))

    logomaker.Logo(info_matrix, flip_below=True, ax=ax)

    ax.set_xlim(0.5, peptide_len + 0.5)
    ax.set_xticks(range(1, peptide_len + 1))
    ax.set_title(name)
    ax.set_xlabel("Position")
    ax.set_ylabel("Bits")


def make_logo_restricted(
    df: pd.DataFrame,
    peptide_len: int,
    allele: str | None = None,
    binders_only: bool = True,
    ax=None,
):
    target_df = df.copy()

    if binders_only and "Binder" in target_df.columns:
        target_df = target_df[target_df["Binder"] == 1]

    target_df = target_df[target_df["Length"] == peptide_len]

    if allele is not None:
        target_df = target_df[target_df["Allele"] == allele]

    peptides = target_df["Peptide"].tolist()

    name_parts = [f"len={peptide_len}"]
    if allele is not None:
        name_parts.append(allele)
    name_parts.append(f"n={len(peptides)}")
    name = ", ".join(name_parts)

    if len(peptides) == 0:
        if ax is not None:
            ax.set_visible(False)
        return

    make_logo(peptides, name, peptide_len=peptide_len, ax=ax)


def plot_allele_length_grid(
    per_allele_df: Dict[str, pd.DataFrame],
    allele: str,
    binders_only: bool = True,
    nrows: int = 2,
    ncols: int = 3,
):
    df_allele = per_allele_df[allele].copy()
    unique_lengths = sorted(df_allele["Length"].unique())

    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 7))
    axes = axes.flatten()

    for ax, L in zip(axes, unique_lengths):
        make_logo_restricted(
            df_allele,
            peptide_len=L,
            allele=allele,
            binders_only=binders_only,
            ax=ax,
        )

    for ax in axes[len(unique_lengths):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


# ==========================
# 4. Подготовка данных для HMM
# ==========================

def get_binders_peptides_only(df: pd.DataFrame) -> np.ndarray:
    return df[df["Binder"] == 1]["Peptide"].values


def get_non_binders_peptides_only(df: pd.DataFrame) -> np.ndarray:
    return df[df["Binder"] == 0]["Peptide"].values


def make_per_length_dict(df: pd.DataFrame, binder_flag: int) -> Dict[int, np.ndarray]:
    """
    Возвращает:
    { длина пептида -> np.array(пептидов) с Binder == binder_flag }.
    """
    per_length_dict: Dict[int, np.ndarray] = {}
    subset = df[df["Binder"] == binder_flag]

    for L in sorted(subset["Length"].unique()):
        per_length_dict[L] = subset[subset["Length"] == L]["Peptide"].values

    return per_length_dict


def make_per_kfold_per_length_dict(
    per_length_dict: Dict[int, np.ndarray],
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int | None = 42,
    verbose: bool = True,
) -> Tuple[Dict[int, Dict[int, np.ndarray]], Dict[int, Dict[int, np.ndarray]]]:
    """
    На вход:  per_length_dict[L] -> массив пептидов длины L.

    На выход:
        train[fold][L] -> массив пептидов в train этого фолда и длины L
        test[fold][L]  -> массив пептидов в test этого фолда и длины L

    Т.е. ориентация теперь:
    per k-fold -> per length
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    per_kfold_per_length_train: Dict[int, Dict[int, np.ndarray]] = {}
    per_kfold_per_length_test: Dict[int, Dict[int, np.ndarray]] = {}

    if verbose:
        print("Length \t Train/Test for splits", end="")

    for L, peptides_array in per_length_dict.items():
        if verbose:
            print(f"\n{L}:", end="\t")

        if len(peptides_array) > n_splits:
            # для данной длины L делаем KFold
            for fold_idx, (train_idx, test_idx) in enumerate(kf.split(peptides_array)):
                # инициализируем словари для фолда, если их ещё нет
                if fold_idx not in per_kfold_per_length_train:
                    per_kfold_per_length_train[fold_idx] = {}
                    per_kfold_per_length_test[fold_idx] = {}

                train_peps = peptides_array[train_idx]
                test_peps = peptides_array[test_idx]

                if verbose:
                    print(f"{len(train_peps)}/{len(test_peps)}\t", end="")

                # сохраняем по схеме: train[fold][length] = массив пептидов
                per_kfold_per_length_train[fold_idx][L] = train_peps
                per_kfold_per_length_test[fold_idx][L] = test_peps
        else:
            if verbose:
                print(f"Not enough peptides of length {L}", end="")

    if verbose:
        print()
    return per_kfold_per_length_train, per_kfold_per_length_test


def prepare_mhc_for_hmm(
    csv_path: str,
    selected_alleles: List[str],
    n_splits: int = 5,
    random_state: int | None = 42,
) -> Dict[str, object]:
    """
    Главная функция пайплайна.
    Возвращает словарь со всеми структурами, которые можно сохранить и потом использовать
    в скрипте обучения HMM.
    Теперь структуры кросс-валидации организованы как:
        per k-fold -> per length.
    """
    # 1. загрузка и фильтрация
    mhc_raw = load_mhc_data(csv_path)
    filtered_df = filter_mhc_dataset(mhc_raw)
    mhc_selected = select_alleles(filtered_df, selected_alleles)

    # 2. разбиение по аллелям и длинам
    per_allele_df = make_per_allele_df(mhc_selected, selected_alleles)

    # 3. подготовка структур для HMM
    per_allele_binders_train: Dict[str, np.ndarray] = {}
    per_allele_non_binders_train: Dict[str, np.ndarray] = {}

    # длина -> пептиды (без k-fold, просто удобно иметь под рукой)
    per_allele_per_length_binders_train: Dict[str, Dict[int, np.ndarray]] = {}
    per_allele_per_length_non_binders_train: Dict[str, Dict[int, np.ndarray]] = {}

    # per k-fold -> per length
    # allele -> fold -> length -> np.ndarray
    per_allele_per_kfold_per_length_binders_train: Dict[str, Dict[int, Dict[int, np.ndarray]]] = {}
    per_allele_per_kfold_per_length_binders_test: Dict[str, Dict[int, Dict[int, np.ndarray]]] = {}
    per_allele_per_kfold_per_length_non_binders_train: Dict[str, Dict[int, Dict[int, np.ndarray]]] = {}
    per_allele_per_kfold_per_length_non_binders_test: Dict[str, Dict[int, Dict[int, np.ndarray]]] = {}

    for allele_name in selected_alleles:
        print(f"\n=== Allele {allele_name} ===")
        df_train = per_allele_df[allele_name].sample(
            frac=1.0, random_state=random_state
        ).reset_index(drop=True)

        # массивы всех пептидов-биндеров/не-биндеров
        per_allele_binders_train[allele_name] = get_binders_peptides_only(df_train)
        per_allele_non_binders_train[allele_name] = get_non_binders_peptides_only(df_train)

        # длина -> пептиды
        per_allele_per_length_binders_train[allele_name] = make_per_length_dict(
            df_train, binder_flag=1
        )
        per_allele_per_length_non_binders_train[allele_name] = make_per_length_dict(
            df_train, binder_flag=0
        )

        print("Binders:")
        (
            per_allele_per_kfold_per_length_binders_train[allele_name],
            per_allele_per_kfold_per_length_binders_test[allele_name],
        ) = make_per_kfold_per_length_dict(
            per_allele_per_length_binders_train[allele_name],
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
            verbose=True,
        )

        print("Non-binders:")
        (
            per_allele_per_kfold_per_length_non_binders_train[allele_name],
            per_allele_per_kfold_per_length_non_binders_test[allele_name],
        ) = make_per_kfold_per_length_dict(
            per_allele_per_length_non_binders_train[allele_name],
            n_splits=n_splits,
            shuffle=True,
            random_state=random_state,
            verbose=True,
        )

    # Собираем всё в один словарь
    result = {
        "filtered_df": filtered_df,
        "mhc_selected": mhc_selected,
        "per_allele_df": per_allele_df,

        "per_allele_binders_train": per_allele_binders_train,
        "per_allele_non_binders_train": per_allele_non_binders_train,

        "per_allele_per_length_binders_train": per_allele_per_length_binders_train,
        "per_allele_per_length_non_binders_train": per_allele_per_length_non_binders_train,

        # НОВОЕ: per k-fold -> per length
        "per_allele_per_kfold_per_length_binders_train": per_allele_per_kfold_per_length_binders_train,
        "per_allele_per_kfold_per_length_binders_test": per_allele_per_kfold_per_length_binders_test,
        "per_allele_per_kfold_per_length_non_binders_train": per_allele_per_kfold_per_length_non_binders_train,
        "per_allele_per_kfold_per_length_non_binders_test": per_allele_per_kfold_per_length_non_binders_test,
    }

    return result


# ==========================
# 5. Сохранение / загрузка
# ==========================

def save_prepared_data(obj: dict, output_path: str) -> None:
    with open(output_path, "wb") as f:
        pickle.dump(obj, f)
    print(f"\nSaved prepared data to: {output_path}")


def load_prepared_data(input_path: str) -> dict:
    with open(input_path, "rb") as f:
        return pickle.load(f)


# ==========================
# 6. Удобный блок параметров
# ==========================

if __name__ == "__main__":
    INPUT_CSV = "/Users/annaklimova/Desktop/Washu_project/mhc_ligand_selected.csv"
    OUTPUT_PKL = "/Users/annaklimova/Desktop/Washu_project/1_mhc_prepared_for_hmm.pkl"

    SELECTED_ALLELES = [
        # "H2-Db",                      # mouse I
        # "HLA-A*02:01",               # human I
        # "HLA-B*27:05",               # human I
        # "HLA-C*05:01",               # human I
        # "HLA-DRB1*15:01",            # human II
        # "HLA-DRB1*03:01",            # human II
        # "HLA-DQA1*02:01/DQB1*02:02", # human II
        # "HLA-DPA1*01:03/DPB1*04:01", # human II
        # "H2-IAb",                    # mouse II
        "HLA-DPA1*02:02/DPB1*05:01"
    ]

    N_SPLITS = 5
    RANDOM_STATE = 42

    # --- ЗАПУСК ПАЙПЛАЙНА ---

    prepared = prepare_mhc_for_hmm(
        csv_path=INPUT_CSV,
        selected_alleles=SELECTED_ALLELES,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
    )

    save_prepared_data(prepared, OUTPUT_PKL)
