import argparse
import os

from pomegranate import *

from training_parameters import (
    SimulatedDataParams,
    SimpleModelClassIIParams,
    ExperimentParams,
)
from data_reading_methods import (
    remove_unused_lengths,
    calculate_weights_based_on_length_counts,
    transform_data_to_properties_and_join_alleles,
    get_train_test_data,
)
from hmm_logic_methods import (
    build_model_based_on_params,
    prepare_multiple_models,
    train_multiple_models,
    reorder_models_by_score_and_flatten_to_by_name_list,
    hierarchically_train_splited_models,
)
from hmm_visualization_methods import save_all_visualization_results


def describe_lengths_structure(per_allele_per_kfold_per_length):
    """Собираем статистику: какие длины есть и сколько пептидов."""
    lengths = set()
    total_peptides = 0
    for allele, per_split in per_allele_per_kfold_per_length.items():
        for split_idx, per_length in per_split.items():
            for length, arr in per_length.items():
                lengths.add(length)
                total_peptides += len(arr)
    return sorted(lengths), total_peptides


# создаём объект параметров эксперимента и переключаемся на симулированные данные

# === РУЧКИ ДЛЯ ЭКСПЕРИМЕНТА ===
NUM_RUNS = 10               # сколько запусков моделей на сплит
TOTAL_LAYERS_TO_DO = 5      # сколько слоёв в иерархическом обучении
SUBFOLDER = "root"          # подпапка внутри папки эксперимента


 # создаём объект параметров эксперимента и переключаемся на симулированные данные
hmm_params = ExperimentParams(experiment_name="simple_model_enrichment")
hmm_params.data_scenario_params = SimulatedDataParams()
hmm_params.model_training_params = SimpleModelClassIIParams()


def parse_cli_and_override_input(hmm_params: ExperimentParams):
    """
    Позволяет выбрать входной файл через аргументы командной строки.

    Пример:
        python main.py --input-file /path/to/file.csv
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input-file",
        dest="input_file",
        help="Путь к входному CSV-файлу с симулированными данными",
    )
    args = parser.parse_args()

    if args.input_file:
        data_params = hmm_params.data_scenario_params
        input_path = os.path.abspath(args.input_file)
        data_dir = os.path.dirname(input_path)
        filename = os.path.basename(input_path)

        # Для всех сценариев задаём директорию как input_data_path
        data_params.input_data_path = data_dir

        # Для симулированных данных отдельно указываем имя файла
        if isinstance(data_params, SimulatedDataParams):
            data_params.simulated_exact_file_name = filename

        print("\n=== OVERRIDE INPUT FILE FROM CLI ===")
        print("Selected input file:", input_path)

    return args


# разбираем аргументы командной строки и, при необходимости, переопределяем входной файл
_cli_args = parse_cli_and_override_input(hmm_params)

# настраиваем число запусков моделей
hmm_params.model_training_params.num_runs = NUM_RUNS

print("=== PATHS ===")
print("DATA_PATH:", hmm_params.data_scenario_params.input_data_path)
print("RESULTS_ROOT:", hmm_params.experiment_result_data_path)
print("NUM_RUNS:", hmm_params.model_training_params.num_runs)
print("TOTAL_LAYERS_TO_DO:", TOTAL_LAYERS_TO_DO)

# читаем train/test из CSV c simulated данными
(
    per_allele_per_kfold_per_length_binders_train,
    per_allele_per_kfold_per_length_binders_test,
    additional_data,
) = get_train_test_data(hmm_params)

PARSED_ALLELES = list(per_allele_per_kfold_per_length_binders_train.keys())
model_training_params = hmm_params.model_training_params
allele_name = hmm_params.data_scenario_params.dummy_allele_name

print("\n=== ALLELES INFO ===")
print("Alleles found in data:", sorted(PARSED_ALLELES))
print("Dummy allele name from params:", allele_name)

# выбираем только dummy_allele из распарсенных аллелей
model_training_params.alleles_to_use = [
    allele for allele in PARSED_ALLELES if allele == allele_name
]
removed_alleles = sorted(set(PARSED_ALLELES) - set(model_training_params.alleles_to_use))

print("Alleles selected for training:", model_training_params.alleles_to_use)
print("Alleles removed:", removed_alleles)

if not model_training_params.alleles_to_use:
    raise ValueError(f"{allele_name} не найден в данных, PARSED_ALLELES={PARSED_ALLELES}")

print("\n=== LENGTHS BEFORE FILTERING ===")
train_lengths_before, train_peptides_before = describe_lengths_structure(
    per_allele_per_kfold_per_length_binders_train
)
print("Lengths present in TRAIN:", train_lengths_before)
print("Total peptides in TRAIN:", train_peptides_before)

# убираем длины пептидов, которые не указаны в model_training_params.lengths_to_use
per_allele_per_kfold_per_length_binders_train = remove_unused_lengths(
    per_allele_per_kfold_per_length_binders_train,
    experiment_params=hmm_params,
)
per_allele_per_kfold_per_length_binders_test = remove_unused_lengths(
    per_allele_per_kfold_per_length_binders_test,
    experiment_params=hmm_params,
)

print("\n=== LENGTHS AFTER FILTERING ===")
train_lengths_after, train_peptides_after = describe_lengths_structure(
    per_allele_per_kfold_per_length_binders_train
)
removed_lengths = sorted(set(train_lengths_before) - set(train_lengths_after))

print("Lengths present in TRAIN after filter:", train_lengths_after)
print("Removed lengths:", removed_lengths)
print("Total peptides in TRAIN after filter:", train_peptides_after)

# преобразуем данные (при необходимости — к свойствам) и объединяем аллели
(
    train_data,
    test_data,
    NEW_ALLELES,
    old_train_data,
    old_test_data,
    OLD_ALLELES,
) = transform_data_to_properties_and_join_alleles(
    per_allele_per_kfold_per_length_binders_train,
    per_allele_per_kfold_per_length_binders_test,
    model_training_params,
)

print("\n=== AFTER MERGING / TRANSFORM ===")
print("OLD_ALLELES:", OLD_ALLELES)
print("NEW_ALLELES:", NEW_ALLELES)
print("train_data keys (alleles/combos):", list(train_data.keys()))

# считаем веса для train-данных по длинам пептидов (на неслитых данных)
train_data_weights = calculate_weights_based_on_length_counts(
    old_train_data,
    experiment_params=hmm_params,
)

print("\n=== SAMPLE SLICE FOR DEBUG ===")
target_allele = model_training_params.alleles_to_use[0]
target_length = model_training_params.lengths_to_use[2]
peptides = train_data[target_allele][0][target_length]
print(f"Target allele: {target_allele}")
print(f"Target length: {target_length}")
print(f"Num peptides for allele={target_allele}, split=0, length={target_length}: {len(peptides)}")

# make_logo_for_data(peptides, f"{target_allele} Binders length {target_length}")


print("\n=== MODEL BUILD / TRAIN ===")
prepared_model = build_model_based_on_params(
    model_training_params,
    verbose=True,
)

per_allele_per_run_per_split_prepared_models = prepare_multiple_models(hmm_params)

print(f"Training results will be saved under: {hmm_params.experiment_result_data_path}/{SUBFOLDER}")
result_models, result_histories = train_multiple_models(
    prepared_models=per_allele_per_run_per_split_prepared_models,
    train_data=train_data,
    test_data=test_data,
    train_data_weights=train_data_weights,
    experiment_params=hmm_params,
    subfolder_to_safe_result=SUBFOLDER,
)

# сортируем модели по качеству + приводим в удобный формат: "имя модели" -> модель/история
per_name_models, per_name_histories, original_alelle_sortings = (
    reorder_models_by_score_and_flatten_to_by_name_list(
        histories=result_histories,
        models=result_models,
        experiment_params=hmm_params,
        subfolder_to_safe_result=SUBFOLDER,
    )
)

print("\n=== VISUALIZATION SAVE PATHS ===")
print(f"Visualization will be saved under: {hmm_params.experiment_result_data_path}/{SUBFOLDER}")
save_all_visualization_results(
    per_name_models,
    per_name_histories,
    experiment_params=hmm_params,
    subfolder_to_safe_result=SUBFOLDER,
)

# Сохранить начальные модели и скоры (до иерархического обучения)
print("\n=== SAVING INITIAL MODELS AND SCORES ===")
from hmm_logic_methods import get_all_scores_and_sort_by_best
import json

initial_save_dir = f"{hmm_params.experiment_result_data_path}/{SUBFOLDER}/initial_models/"
os.makedirs(initial_save_dir, exist_ok=True)

# Сохранить информацию о начальных моделях
initial_models_info = {
    f'model_{i}': {
        'model_name': model_name,
        'model_index': i
    }
    for i, model_name in enumerate(per_name_models.keys())
}

with open(f"{initial_save_dir}/models_info.json", 'w') as f:
    json.dump(initial_models_info, f, indent=2)

# Рассчитать и сохранить скоры для начальных моделей
initial_scores_data = get_all_scores_and_sort_by_best(train_data, per_name_models)

for allele_name, df in initial_scores_data.items():
    df.to_csv(f"{initial_save_dir}/{allele_name}_scores.csv", index=False)
    df.to_pickle(f"{initial_save_dir}/{allele_name}_scores.pkl")

print(f"Initial models info and scores saved to: {initial_save_dir}")

print("\n=== HIERARCHICAL TRAINING ===")
print(f"Hierarchical training results under: {hmm_params.experiment_result_data_path}/{SUBFOLDER}")
hierarchically_train_splited_models(
    per_name_models=per_name_models,
    train_data=train_data,
    test_data=test_data,
    additional_data=additional_data,
    experiment_params=hmm_params,
    total_layers_to_do=TOTAL_LAYERS_TO_DO,
)


# В конце: вывести все файлы, которые реально создались
print("\n=== FILES CREATED IN RESULTS_ROOT ===")
results_root = hmm_params.experiment_result_data_path
for root, dirs, files in os.walk(results_root):
    for name in files:
        full_path = os.path.join(root, name)
        print(full_path)

print("\n=== DONE ===")