from pathlib import Path
from training_parameters import (
    PreprocessedIEDBDataParams,
    SimpleModelClassIIParams,
    ExperimentParams,
)
import pandas as pd
from data_reading_methods import (
    remove_unused_lengths,
    calculate_weights_based_on_length_counts,
    transform_data_to_properties_and_join_alleles,
    get_train_test_data,
)
from peptides_utils import  join_dicts
import os
import importlib
import hmm_logic_methods
from hmm_logic_methods import build_model_based_on_params, prepare_multiple_models, train_multiple_models, reorder_models_by_score_and_flatten_to_by_name_list, hierarchically_train_splited_models, get_all_scores_and_sort_by_best, check_model_alignment, save_model_score_tables
import hmm_visualization_methods
from hmm_visualization_methods import plot_distributions_for_states, make_pyviz_graph, save_all_visualization_results, save_figure_to_svg, plot_split_diagnostics, plot_all_scores_as_separate_lines, plot_all_scores_as_lines

from copy import deepcopy
from pathlib import Path

def score_only_and_save(peptide_data, per_name_models, output_dir, tag):
    """
    Считает score уже обученными моделями.
    здесь НЕТ обучения, только scoring.
    """
    print("Модели уже обучены. Сейчас только считаем score, без дообучения.")

    per_allele_data_df = get_all_scores_and_sort_by_best(peptide_data, per_name_models)

    save_model_score_tables(
        per_allele_data_df=per_allele_data_df,
        per_name_models=per_name_models,
        output_dir=str(Path(output_dir) / tag),
    )
    return per_allele_data_df


def combine_scored_tables(per_allele_data_df, source_name):
    """
    Собирает scored-таблицы разных наборов в один DataFrame
    и добавляет метку source_set.
    """
    frames = []
    for allele_name, df in per_allele_data_df.items():
        cur = df.copy()
        cur["source_allele"] = allele_name
        cur["source_set"] = source_name
        frames.append(cur)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def train_all_models_for_one_allele(
    allele_name,
    raw_train,
    raw_test,
    base_hmm_params,
    subfolder_root,
):
    """
    Обучаем ВСЕ модели только на одном аллеле.
    Ничего не отбираем, возвращаем все обученные модели и их истории.
    """
    params = deepcopy(base_hmm_params)
    params.model_training_params.alleles_to_use = [allele_name]

    single_train = {allele_name: raw_train[allele_name]}
    single_test = {allele_name: raw_test[allele_name]}

    train_data, test_data, _, old_train_data, old_test_data, _ = (
        transform_data_to_properties_and_join_alleles(
            single_train,
            single_test,
            params.model_training_params,
        )
    )

    train_data_weights = calculate_weights_based_on_length_counts(
        old_train_data,
        experiment_params=params
    )

    prepared_models = prepare_multiple_models(params)

    result_models, result_histories = train_multiple_models(
        prepared_models=prepared_models,
        train_data=train_data,
        test_data=test_data,
        train_data_weights=train_data_weights,
        experiment_params=params,
        subfolder_to_safe_result=f"{subfolder_root}/{allele_name}",
    )

    per_name_models, per_name_histories, _ = (
        reorder_models_by_score_and_flatten_to_by_name_list(
            histories=result_histories,
            models=result_models,
            experiment_params=params,
            subfolder_to_safe_result=f"{subfolder_root}/{allele_name}",
        )
    )

    print("\n=== VISUALIZATION SAVE PATHS ===")
    print(f"Visualization will be saved under: {params.experiment_result_data_path}/{subfolder_root}/{allele_name}")

    save_all_visualization_results(
        per_name_models,
        per_name_histories,
        experiment_params=params,
        subfolder_to_safe_result=f"{subfolder_root}/{allele_name}",
    )

    # Переименовываем все модели в уникальные публичные имена
    public_models = {}
    public_histories = {}

    for idx, internal_name in enumerate(per_name_models.keys()):
        public_model_name = f"model_{allele_name}_{idx}"
        public_models[public_model_name] = per_name_models[internal_name]
        public_histories[public_model_name] = per_name_histories[internal_name]

    return public_models, public_histories

def score_one_source_allele_with_all_models(
    source_allele,
    raw_train,
    raw_test,
    trained_models,
    base_hmm_params,
    score_root,
):
    """
    Берём пептиды одного source_allele и считаем score всеми уже обученными моделями.
    НИКАКОГО обучения тут нет.
    """
    params = deepcopy(base_hmm_params)
    params.model_training_params.alleles_to_use = [source_allele]

    single_train = {source_allele: raw_train[source_allele]}
    single_test = {source_allele: raw_test[source_allele]}

    train_data, test_data, _, _, _, _ = transform_data_to_properties_and_join_alleles(
        single_train,
        single_test,
        params.model_training_params,
    )

    train_scored = score_only_and_save(
        peptide_data=train_data,
        per_name_models=trained_models,
        output_dir=score_root,
        tag=f"{source_allele}/train_score_only",
    )

    test_scored = score_only_and_save(
        peptide_data=test_data,
        per_name_models=trained_models,
        output_dir=score_root,
        tag=f"{source_allele}/test_score_only",
    )

    train_df = combine_scored_tables(train_scored, "train")
    test_df = combine_scored_tables(test_scored, "test")

    return pd.concat([train_df, test_df], ignore_index=True)


# ============================================================
# ОСНОВНОЙ PIPELINE
# ============================================================

hmm_logic_methods = importlib.reload(hmm_logic_methods)
hmm_visualization_methods = importlib.reload(hmm_visualization_methods)

SUBFOLDER = "IEDB_DRB1_0101_0301_0401"

hmm_params = ExperimentParams(experiment_name="simple_model_enrichment")
hmm_params.data_scenario_params = PreprocessedIEDBDataParams()
hmm_params.data_scenario_params.input_data_path = (
    "/Users/annaklimova/Desktop/Washu_project/"
    "data/simple_model_enrichment/IEDB_data/per_length_per_kfold_split"
)
hmm_params.model_training_params = SimpleModelClassIIParams()
hmm_params.model_training_params.num_runs = 20

(
    per_allele_per_kfold_per_length_binders_train,
    per_allele_per_kfold_per_length_binders_test,
    additional_data,
) = get_train_test_data(hmm_params)

PARSED_ALLELES = list(per_allele_per_kfold_per_length_binders_train.keys())
print("Alleles we use:", PARSED_ALLELES)
per_allele_per_kfold_per_length_binders_train = remove_unused_lengths(
    per_allele_per_kfold_per_length_binders_train,
    experiment_params=hmm_params
)
per_allele_per_kfold_per_length_binders_test = remove_unused_lengths(
    per_allele_per_kfold_per_length_binders_test,
    experiment_params=hmm_params
)

# ============================================================
# 1) УЧИМ 3 МОДЕЛИ ОТДЕЛЬНО — КАЖДУЮ ТОЛЬКО НА СВОЁМ АЛЛЕЛЕ
# ============================================================

all_models = {}
all_histories = {}

for allele_name in PARSED_ALLELES:
    print(f"\n=== TRAIN ALL MODELS FOR {allele_name} ===")

    allele_models, allele_histories = train_all_models_for_one_allele(
        allele_name=allele_name,
        raw_train=per_allele_per_kfold_per_length_binders_train,
        raw_test=per_allele_per_kfold_per_length_binders_test,
        base_hmm_params=hmm_params,
        subfolder_root=SUBFOLDER,
    )

    all_models.update(allele_models)
    all_histories.update(allele_histories)

print("\nAll trained models:")
for k in all_models.keys():
    print(" ", k)
# ============================================================
# 2) БЕРЁМ ВСЕ ПЕПТИДЫ ВСЕХ 3 АЛЛЕЛЕЙ И СЧИТАЕМ SCORE ВСЕМИ 3 МОДЕЛЯМИ
# ============================================================
score_root = f"/Users/annaklimova/Desktop/Washu_project/results/model_score_tables_{SUBFOLDER}"

all_scored_frames = []
for source_allele in PARSED_ALLELES:

    print(f"\n=== SCORE PEPTIDES FROM {source_allele} WITH ALL MODELS ===")

    cur_df = score_one_source_allele_with_all_models(
        source_allele=source_allele,
        raw_train=per_allele_per_kfold_per_length_binders_train,
        raw_test=per_allele_per_kfold_per_length_binders_test,
        trained_models=all_models,
        base_hmm_params=hmm_params,
        score_root=score_root,
    )

    all_scored_frames.append(cur_df)

all_scores_df = pd.concat(all_scored_frames, ignore_index=True)

print("all_scores_df columns BEFORE rename:")
print(all_scores_df.columns.tolist())

rename_map = {
    f"model_{i}": model_name
    for i, model_name in enumerate(all_models.keys())
}
print("rename_map:", rename_map)

all_scores_df = all_scores_df.rename(columns=rename_map)

score_cols = list(all_models.keys())

preferred_meta_cols = [
    "peptide",
    "source_allele",
    "source_set",
    "split",
    "length",
]

meta_cols = [c for c in preferred_meta_cols if c in all_scores_df.columns]
other_cols = [c for c in all_scores_df.columns if c not in meta_cols + score_cols]

all_scores_df = all_scores_df[meta_cols + score_cols + other_cols]

out_csv = f"{score_root}/all_peptides_x_{len(PARSED_ALLELES)}models_scores.csv"
all_scores_df.to_csv(out_csv, index=False)

print("Saved:", out_csv)
print(all_scores_df.head())