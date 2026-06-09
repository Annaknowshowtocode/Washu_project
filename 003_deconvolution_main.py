import sys
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent / "deconvolution_pipeline"
sys.path.insert(0, str(PIPELINE_DIR))

from pomegranate import *
from training_parameters import (
    PreprocessedIEDBDataParams,
    SimpleModelClassIIParams,
    ExperimentParams,
)
from data_reading_methods import (
    remove_unused_lengths,
    calculate_weights_based_on_length_counts,
    transform_data_to_properties_and_join_alleles,
    get_train_test_data,
)
from peptides_utils import  join_dicts
import importlib
import hmm_logic_cluster_split_trash
from hmm_logic_cluster_split_trash import prepare_multiple_models, train_multiple_models, reorder_models_by_score_and_flatten_to_by_name_list, hierarchically_train_splited_models, get_all_scores_and_sort_by_best, check_model_alignment
import hmm_visualization_methods
from hmm_visualization_methods import save_all_visualization_results, save_figure_to_svg



hmm_logic_methods= importlib.reload(hmm_logic_cluster_split_trash)
hmm_visualization_methods= importlib.reload(hmm_visualization_methods)
import os
SUBFOLDER = "IEDB_root_trash"

hmm_params = ExperimentParams(experiment_name="simple_model_enrichment")
hmm_params.data_scenario_params = PreprocessedIEDBDataParams()

hmm_params.model_training_params = SimpleModelClassIIParams()
model_training_params = hmm_params.model_training_params
hmm_params.model_training_params.num_runs = 5 # n runs

(
    per_allele_per_kfold_per_length_binders_train,
    per_allele_per_kfold_per_length_binders_test,
    additional_data,
) = get_train_test_data(hmm_params, clean = False, collapse_to_dummy = True)

PARSED_ALLELES = list(per_allele_per_kfold_per_length_binders_train.keys())
model_training_params.alleles_to_use = PARSED_ALLELES  # один dummy-аллель
print("Alleles we use:", model_training_params.alleles_to_use)

per_allele_per_kfold_per_length_binders_train = remove_unused_lengths(per_allele_per_kfold_per_length_binders_train, experiment_params=hmm_params)
per_allele_per_kfold_per_length_binders_test = remove_unused_lengths(per_allele_per_kfold_per_length_binders_test,experiment_params=hmm_params)


# train-peptides txt for GibbsCluster
peptides_for_gibbs = sorted(set(
    peptide
    for allele_dict in per_allele_per_kfold_per_length_binders_train.values()
    for kfold_dict in allele_dict.values()
    for length_peptides in kfold_dict.values()
    for peptide in length_peptides
))

gibbs_path = os.path.join(
    hmm_params.data_scenario_params.input_data_path,
    "peptides_for_gibbscluster.txt"
)

with open(gibbs_path, "w") as f:
    f.write("\n".join(peptides_for_gibbs))

print(f"Saved {len(peptides_for_gibbs)} peptides to {gibbs_path}")


train_data, test_data, NEW_ALLELES, \
old_train_data, old_test_data, OLD_ALLELES = transform_data_to_properties_and_join_alleles(
    per_allele_per_kfold_per_length_binders_train,
    per_allele_per_kfold_per_length_binders_test,
    hmm_params.model_training_params
)

train_data_weights = calculate_weights_based_on_length_counts(old_train_data, experiment_params=hmm_params)


# {
#   allele_name: {
#     run_index: {
#       split_num: prepared_model
#     }
#   }
# }

per_allele_per_run_per_split_prepared_models = prepare_multiple_models(hmm_params)
print("\n=== PREPARED MULTIPLE MODELS ===")

result_models, result_histories = train_multiple_models(
    prepared_models=per_allele_per_run_per_split_prepared_models,
    train_data=train_data,
    test_data=test_data,
    train_data_weights=train_data_weights,
    experiment_params=hmm_params,
    subfolder_to_safe_result=SUBFOLDER,
)



(per_name_models, per_name_histories, original_alelle_sortings) = (
    reorder_models_by_score_and_flatten_to_by_name_list(
        histories=result_histories,
        models=result_models,
        experiment_params=hmm_params,
        subfolder_to_safe_result=SUBFOLDER,
    )
)

save_all_visualization_results(
    per_name_models,
    per_name_histories,
    experiment_params=hmm_params,
    subfolder_to_safe_result=SUBFOLDER,
)

allele_data_df = join_dicts(train_data)[PARSED_ALLELES[0]]  # ~52000 peptides x 3 columns (peptides, split, length)
allele_name = PARSED_ALLELES[0].replace('-', '_').replace('*', '_').replace(':', '_') #dummy

info_matrix, _, fig = check_model_alignment(per_name_models[list(per_name_models.keys())[1]], allele_data_df, allele_name)
filename=f"{allele_name}-clean_split-root-example_of_alignment.svg"
path_to_save_pictures = f"{hmm_params.experiment_result_data_path}"
save_figure_to_svg(fig, dir=path_to_save_pictures, filename=f"{filename}")


print("\n=== HIERARCHICAL TRAINING ===")
print(f"Hierarchical training results under: {hmm_params.experiment_result_data_path}/{SUBFOLDER}")
hierarchically_train_splited_models(
    per_name_models=per_name_models,
    train_data=train_data,
    test_data=test_data,
    additional_data=additional_data,
    experiment_params=hmm_params,
    total_layers_to_do=3,
    n_branches = 2,
)
print("\n=== DONE ===")
