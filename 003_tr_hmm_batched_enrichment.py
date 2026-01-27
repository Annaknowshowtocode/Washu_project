import pomegranate.io
from pomegranate import *

import sys
PATH_TO_PREDICTOR_HOME = "../.."
sys.path.append(PATH_TO_PREDICTOR_HOME)
METHOD = "hmm_pomegranate"


import training_parameters
from peptides_utils import defineClass, defineOrganism, make_logo_for_data
from data_reading_methods import remove_unused_lengths, transform_data_to_properties_and_join_alleles, calculate_weights_based_on_length_counts

hmm_params = training_parameters.ExperimentParams(experiment_name="simple_model_enrichment")
hmm_params.data_scenario_params = training_parameters.SimulatedDataParams()

model_training_params = hmm_params.model_training_params

hmm_params.data_scenario_params.simulated_exact_file_name
# ________________________________________________________
from data_reading_methods import get_available_alleles, get_train_test_data, join_dicts

per_allele_per_kfold_per_length_binders_train, \
per_allele_per_kfold_per_length_binders_test, additional_data = get_train_test_data(hmm_params)

PARSED_ALLELES = list(per_allele_per_kfold_per_length_binders_train.keys())

print(sorted(PARSED_ALLELES))


data_params = hmm_params.data_scenario_params
DATA_PATH = data_params.input_data_path
simulated_exact_file = data_params.simulated_exact_file_name
dummy_allele_name = data_params.dummy_allele_name
simulated_scenario = data_params.simulated_scenario
SIMULATED_DATA_PATH = f"{DATA_PATH}/{simulated_scenario}/{simulated_exact_file}"
ALLELES = [dummy_allele_name]

allele_df = pd.read_csv(SIMULATED_DATA_PATH, sep=";")



for i in range(len(allele_df.allele.unique())):
    name = f"Dummy_allele_{i}"
    make_logo_for_data(allele_df[(allele_df.allele == name) & (allele_df.noize == 0)].core.values, name=name)



# _____________________________________________________
model_training_params: training_parameters.ModelTrainingParams = hmm_params.model_training_params
#model_training_params.alleles_to_use = [ item for item in PARSED_ALLELES if item in ['HLA-DRB1*04:01']]
model_training_params.alleles_to_use = [ item for item in PARSED_ALLELES if item in ['dummy_allele']]
#model_training_params.alleles_to_use = [ item for item in PARSED_ALLELES if item in ['decoy-allele-78']]
# remove unused lengths (used are specified in ModelTrainingParams.model_training_params
per_allele_per_kfold_per_length_binders_train = remove_unused_lengths(per_allele_per_kfold_per_length_binders_train, experiment_params=hmm_params)
per_allele_per_kfold_per_length_binders_test = remove_unused_lengths(per_allele_per_kfold_per_length_binders_test,experiment_params=hmm_params)

# transform to properties if needed and join multiple alleles
train_data, test_data, NEW_ALLELES, \
old_train_data, old_test_data, OLD_ALLELES = transform_data_to_properties_and_join_alleles(
    per_allele_per_kfold_per_length_binders_train,
    per_allele_per_kfold_per_length_binders_test,
    hmm_params.model_training_params
)
# Calculate weights for the training data based on peptide couns/lengths for unmerged data
train_data_weigths = calculate_weights_based_on_length_counts(old_train_data, experiment_params=hmm_params)



from hmm_logic_methods import train_model_prepared,  \
    train_model_batched, save_model, \
    add_more_states_and_reset_transitions, \
    build_model_based_on_params, \
    prepare_multiple_models, train_multiple_models, \
reorder_models_by_score_and_flatten_to_by_name_list, hierarchically_train_splited_models


# ----------------------------------------------------------------------------------------------------------------------

from hmm_visualization_methods import *

prepared_model = build_model_based_on_params(hmm_params.model_training_params, verbose=True)

with open('model.png', 'w+') as f:
    prepared_model.plot(file=f, crop_zero=True, rotation=90, rotate=90)

plot_distributions_for_states(prepared_model,"test", horizontal=False,
                              discrete = hmm_params.model_training_params.aa_labels_training,
                              initial_params = hmm_params.model_training_params.initial_params)
make_pyviz_graph(prepared_model, "test", precision=3)






#prepare models
per_allele_per_run_per_split_prepared_models = prepare_multiple_models(hmm_params)
#do training
result_models, \
result_histories =  train_multiple_models(
    prepared_models=per_allele_per_run_per_split_prepared_models,
    train_data=train_data,
    test_data=test_data,
    train_data_weigths=train_data_weigths,
    experiment_params=hmm_params,
    subfolder_to_safe_result="root")


per_name_models, per_name_histories, original_alelle_sortings = reorder_models_by_score_and_flatten_to_by_name_list(histories=result_histories,
                                                                models=result_models, experiment_params=hmm_params,subfolder_to_safe_result="root" )
save_all_visualization_results(per_name_models, per_name_histories, experiment_params=hmm_params, subfolder_to_safe_result="root")

hierarchically_train_splited_models(per_name_models=per_name_models,
                                    train_data=train_data,
                                    test_data=test_data,
                                    additional_data=additional_data,
                                    experiment_params=hmm_params,
                                    total_layers_to_do=6)

import importlib
import hmm_logic_methods
import hmm_visualization_methods
hmm_logic_methods= importlib.reload(hmm_logic_methods)
hmm_visualization_methods= importlib.reload(hmm_visualization_methods)
from hmm_logic_methods import get_all_scores_and_sort_by_best
from hmm_visualization_methods import plot_all_scores_as_lines, plot_all_scores_as_separate_lines, plot_split_diagnostics

per_allele_data_df = get_all_scores_and_sort_by_best(train_data, per_name_models)


target_df = plot_all_scores_as_lines(per_allele_data_df, per_name_models)

plot_all_scores_as_separate_lines(per_allele_data_df, per_name_models)


t = plot_split_diagnostics(per_allele_data_df, additional)

import math

# def split_model(per_name_models, train_data, iteration, split_type = 0.5, original_model_index=0):
#     ####!!!! TODO: change to per allele model
#    # selected_model = get_average_model_for_runs(per_name_models)
#     selected_model = per_name_models[list(per_name_models.keys())[original_model_index]]
#     ####!!!
#     per_allele_list_of_dfs = defaultdict(list)
#     for allele_name in ALLELES:
#         for current_split in range(TOTAL_SPLITS):
#             for current_length in TARGET_LENGTHS:
#                 peptides = train_data[allele_name][current_split][current_length]
#                 df = pd.DataFrame({'peptide': peptides,'split': current_split, 'length':current_length})
#                 per_allele_list_of_dfs[allele_name].append(df)
#
#     per_allele_data_df = dict()
#     per_allele_first_df = dict()
#     per_allele_second_df =dict()
#     for allele_name in ALLELES:
#         df = pd.concat(per_allele_list_of_dfs[allele_name])
#         df['score'] = [selected_model.log_probability(peptide)/len(peptide) for peptide in df.peptide]
#         df = df.sort_values(by='score', ascending=False)
#         #df1 = df.loc[df.score > -3].copy()
#         #df2 = df[~df.apply(tuple,axis=1).isin(df1.apply(tuple,axis=1))].copy()
#         #df1 = df[df['score'] >= df['score'].max() - np.log(2)].copy()
#         #df2 = df[df['score'] < df['score'].max() - np.log(2)].copy()
#         df1 = df[df['score'] >= df['score'].mean()].copy()
#         df2 = df[df['score'] < df['score'].mean()].copy()
#         per_allele_first_df[allele_name] = df1
#         per_allele_second_df[allele_name] = df2
#         per_allele_data_df[allele_name] = df
#     new_train_data1 = split_to_dicts(per_allele_first_df, ALLELES, TARGET_LENGTHS, TOTAL_SPLITS)
#     new_train_data2 = split_to_dicts(per_allele_second_df, ALLELES, TARGET_LENGTHS, TOTAL_SPLITS)
#     train_data_weights1 = get_weights(new_train_data1)
#     train_data_weights2 = get_weights(new_train_data2)
#     per_allele_per_run_per_split_prepared_models1 = prepare_all_models()
#     per_allele_per_run_per_split_prepared_models2 = prepare_all_models()
#
#     #1
#     per_allele_per_run_per_split_prepared_models1, \
#     per_allele_per_run_per_split_new_histories1 = train_all_models(train_data=new_train_data1,
#                                                                   test_data=test_data,
#                                                                   train_data_weights=train_data_weights1,
#                                                                   save_dir=TARGET_PATH_TO_RESULTS + f'/new_model{iteration}1/',
#                                                                   prepared_models=per_allele_per_run_per_split_prepared_models1
#                                                                   )
#     #2
#     per_allele_per_run_per_split_prepared_models2, \
#     per_allele_per_run_per_split_new_histories2 = train_all_models(train_data=new_train_data2,
#                                                                   test_data=test_data,
#                                                                   train_data_weights=train_data_weights2,
#                                                                   save_dir=TARGET_PATH_TO_RESULTS +f'/new_model{iteration}2/',
#                                                                   prepared_models=per_allele_per_run_per_split_prepared_models2
#                                                                   )
#     per_name_models1, per_name_histories1 = reorder_models_by_score(histories=per_allele_per_run_per_split_new_histories1,
#                                                                 models=per_allele_per_run_per_split_prepared_models1)
#     per_name_models2, per_name_histories2 = reorder_models_by_score(histories=per_allele_per_run_per_split_new_histories2,
#                                                                     models=per_allele_per_run_per_split_prepared_models2)
#     save_all_visualization_results(per_name_models1, per_name_histories1, save_dir=TARGET_PATH_TO_RESULTS + f'/new_model{iteration}1/')
#     save_all_visualization_results(per_name_models2, per_name_histories2, save_dir=TARGET_PATH_TO_RESULTS + f'/new_model{iteration}2/')
#
#     return per_name_models1, per_name_models2, new_train_data1, new_train_data2





NUM_RUNS = 8
NUM_RUNS

from data_reading_methods import split_to_dicts

def split_model_and_enrich(per_name_models, train_data, iteration,
                           split_type = 0.5,
                           original_model_index=0,
                           enrichment_steps = 3,
                           NUM_RUNS =20,
                           DECREASE_ANCHOR_ACIDS_STEPS = 1,
                           model_index_start=0):
    ####!!!! TODO: change to per allele model
   # selected_model = get_average_model_for_runs(per_name_models)
    selected_model = per_name_models[list(per_name_models.keys())[original_model_index]]
    ####!!!
    per_allele_list_of_dfs = defaultdict(list)
    for allele_name in ALLELES:
        for current_split in range(TOTAL_SPLITS):
            for current_length in TARGET_LENGTHS:
                peptides = train_data[allele_name][current_split][current_length]
                df = pd.DataFrame({'peptide': peptides,'split': current_split, 'length':current_length})
                per_allele_list_of_dfs[allele_name].append(df)

    per_allele_data_df = dict()
    per_allele_first_df = dict()
    per_allele_second_df =dict()
    for allele_name in ALLELES:
        df = pd.concat(per_allele_list_of_dfs[allele_name])
        df['score'] = [selected_model.log_probability(peptide)/len(peptide) for peptide in df.peptide]
        df = df.sort_values(by='score', ascending=False)
        #df1 = df.loc[df.score > -3].copy()
        #df2 = df[~df.apply(tuple,axis=1).isin(df1.apply(tuple,axis=1))].copy()
        #df1 = df[df['score'] >= df['score'].max() - np.log(2)].copy()
        #df2 = df[df['score'] < df['score'].max() - np.log(2)].copy()
        df1 = df[df['score'] >= df['score'].mean()].copy()
        df2 = df[df['score'] < df['score'].mean()].copy()
        per_allele_first_df[allele_name] = df1
        per_allele_second_df[allele_name] = df2
        per_allele_data_df[allele_name] = df
    new_train_data1 = split_to_dicts(per_allele_first_df, ALLELES, TARGET_LENGTHS, TOTAL_SPLITS)
    new_train_data2 = split_to_dicts(per_allele_second_df, ALLELES, TARGET_LENGTHS, TOTAL_SPLITS)
    train_data_weights1 = get_weights(new_train_data1)
    train_data_weights2 = get_weights(new_train_data2)
    per_allele_per_run_per_split_prepared_models1 = prepare_all_models(RUN_START=model_index_start,
                                                                       NUM_RUNS=NUM_RUNS,
                                                                       DECREASE_ANCHOR_ACIDS=DECREASE_ANCHOR_ACIDS_STEPS)
    per_allele_per_run_per_split_prepared_models2 = prepare_all_models(RUN_START=model_index_start + NUM_RUNS*DECREASE_ANCHOR_ACIDS_STEPS,
                                                                       NUM_RUNS=NUM_RUNS,
                                                                       DECREASE_ANCHOR_ACIDS=DECREASE_ANCHOR_ACIDS_STEPS)

    #1
    per_allele_per_run_per_split_prepared_models1, \
    per_allele_per_run_per_split_new_histories1 = train_all_models(train_data=new_train_data1,
                                                                  test_data=test_data,
                                                                  train_data_weights=train_data_weights1,
                                                                  save_dir=TARGET_PATH_TO_RESULTS + f'/new_model{iteration}1/',
                                                                  prepared_models=per_allele_per_run_per_split_prepared_models1
                                                                  )
    #2
    per_allele_per_run_per_split_prepared_models2, \
    per_allele_per_run_per_split_new_histories2 = train_all_models(train_data=new_train_data2,
                                                                  test_data=test_data,
                                                                  train_data_weights=train_data_weights2,
                                                                  save_dir=TARGET_PATH_TO_RESULTS +f'/new_model{iteration}2/',
                                                                  prepared_models=per_allele_per_run_per_split_prepared_models2
                                                                  )
    per_name_models1, per_name_histories1,_ = reorder_models_by_score(histories=per_allele_per_run_per_split_new_histories1,
                                                                models=per_allele_per_run_per_split_prepared_models1)
    per_name_models2, per_name_histories2,_ = reorder_models_by_score(histories=per_allele_per_run_per_split_new_histories2,
                                                                    models=per_allele_per_run_per_split_prepared_models2)

    # ## Here we want to plot UMAP for freshly splited models
    # total_models = dict()
    # total_models.update(per_name_models1)
    # total_models.update(per_name_models2)
    # print(f"length of models is {len(total_models)}" )
    # per_allele_scores_data = get_all_scores_and_sort_by_best(train_data, total_models)
    # plot_split_diagnostics(per_allele_scores_data, additional)



    ## enrichment step
    print("Enrichment")
    for i in range(enrichment_steps):
        print("enrichment iteration", i)
        for allele_name in ALLELES:
            df = per_allele_data_df[allele_name]
            selected_model1 = per_name_models1[list(per_name_models1)[0]]
            selected_model2 = per_name_models2[list(per_name_models2)[0]]
            df[f'score1_step{i}'] = [selected_model1.log_probability(peptide)/len(peptide) for peptide in df.peptide]
            df[f'score2_step{i}'] = [selected_model2.log_probability(peptide)/len(peptide) for peptide in df.peptide]
            df1 = df[df[f'score1_step{i}'] >= df[f'score2_step{i}']].copy()
            df2 = df[df[f'score2_step{i}'] >= df[f'score1_step{i}']].copy()
            per_allele_first_df[allele_name] = df1
            per_allele_second_df[allele_name] = df2
            per_allele_data_df[allele_name] = df
        new_train_data1 = split_to_dicts(per_allele_first_df, ALLELES, TARGET_LENGTHS, TOTAL_SPLITS)
        new_train_data2 = split_to_dicts(per_allele_second_df, ALLELES, TARGET_LENGTHS, TOTAL_SPLITS)
        train_data_weights1 = get_weights(new_train_data1)
        train_data_weights2 = get_weights(new_train_data2)
        per_allele_per_run_per_split_prepared_models1 = prepare_all_models(RUN_START=model_index_start,
                                                                           NUM_RUNS=NUM_RUNS,
                                                                           DECREASE_ANCHOR_ACIDS=DECREASE_ANCHOR_ACIDS_STEPS)
        per_allele_per_run_per_split_prepared_models2 = prepare_all_models(RUN_START=model_index_start + NUM_RUNS*DECREASE_ANCHOR_ACIDS_STEPS,
                                                                           NUM_RUNS=NUM_RUNS,
                                                                           DECREASE_ANCHOR_ACIDS=DECREASE_ANCHOR_ACIDS_STEPS)
         #1
        print(f"Model {iteration}1")
        per_allele_per_run_per_split_prepared_models1, \
        per_allele_per_run_per_split_new_histories1 = train_all_models(train_data=new_train_data1,
                                                                      test_data=test_data,
                                                                      train_data_weights=train_data_weights1,
                                                                      save_dir=TARGET_PATH_TO_RESULTS + f'/new_model{iteration}1/',
                                                                      prepared_models=per_allele_per_run_per_split_prepared_models1
                                                                      )
        #2
        print(f"Model {iteration}2")
        per_allele_per_run_per_split_prepared_models2, \
        per_allele_per_run_per_split_new_histories2 = train_all_models(train_data=new_train_data2,
                                                                      test_data=test_data,
                                                                      train_data_weights=train_data_weights2,
                                                                      save_dir=TARGET_PATH_TO_RESULTS +f'/new_model{iteration}2/',
                                                                      prepared_models=per_allele_per_run_per_split_prepared_models2
                                                                      )
        per_name_models1, per_name_histories1,_ = reorder_models_by_score(histories=per_allele_per_run_per_split_new_histories1,
                                                                    models=per_allele_per_run_per_split_prepared_models1)
        per_name_models2, per_name_histories2,_ = reorder_models_by_score(histories=per_allele_per_run_per_split_new_histories2,
                                                                        models=per_allele_per_run_per_split_prepared_models2)

    save_all_visualization_results(per_name_models1, per_name_histories1, save_dir=TARGET_PATH_TO_RESULTS + f'/new_model{iteration}1/')
    save_all_visualization_results(per_name_models2, per_name_histories2, save_dir=TARGET_PATH_TO_RESULTS + f'/new_model{iteration}2/')

    return per_name_models1, per_name_models2, new_train_data1, new_train_data2, per_allele_data_df


%%time
from collections import deque
import sys
orig_stdout = sys.stdout
with open('out.txt', 'w') as f:
  #  sys.stdout = f
    # try single split and multiple enrichments
    stats_storage = dict()
    iteration = "1"
    models_deque = deque()
    data_deque = deque()
    iteration_deque = deque()

    models_deque.append(per_name_models)
    data_deque.append(train_data)
    iteration_deque.append("1")

    total_models = dict()

    for i in range(7):
        per_name_current_models = models_deque.popleft()
        this_train_data = data_deque.popleft()
        iteration = iteration_deque.popleft()
        print("iteration", iteration)
        # run current data
        per_name_models1, per_name_models2, new_train_data1, new_train_data2, enrichmenet_stats = split_model_and_enrich(per_name_current_models, this_train_data, iteration, enrichment_steps=3, model_index_start=len(total_models),NUM_RUNS=NUM_RUNS, DECREASE_ANCHOR_ACIDS_STEPS=2 )
        # Add current models to the total representation and visualize UMAP of this representation
        total_models.update(per_name_models1)
        total_models.update(per_name_models2)
        print(f"length of models is {len(total_models)}" )
        per_allele_scores_data = get_all_scores_and_sort_by_best(train_data, total_models)
        plot_split_diagnostics(per_allele_scores_data, additional)

        # add left to queue
        models_deque.append(per_name_models1)
        stats_storage[iteration] = enrichmenet_stats

        #data_deque.append(new_train_data1)
        # add whole train data
        data_deque.append(new_train_data1)
        iteration_deque.append(f'{iteration}1')
        models_deque.append(per_name_models2)
        data_deque.append(new_train_data2)
        # add whole train data
       # data_deque.append(train_data)
        iteration_deque.append(f'{iteration}2')
        if i in {0, 2, 6}:
            params['emission_pseudocount']=  params['emission_pseudocount']/2
        if i in {0, 2, 6}:
            #params['anchor_default_pseudocount']=  params['anchor_default_pseudocount']/2
            #params['anchor_top_aas'] = params['anchor_top_aas'] - 1
            pass
        if i in {0, 2, 6}:
            # make total data df (should be per allele)
            per_allele_list_of_dfs = defaultdict(list)
            for allele_name in ALLELES:
                for current_split in range(TOTAL_SPLITS):
                    for current_length in TARGET_LENGTHS:
                        peptides = train_data[allele_name][current_split][current_length]
                        df = pd.DataFrame({'peptide': peptides,'split': current_split, 'length':current_length})
                        per_allele_list_of_dfs[allele_name].append(df)
            #
            per_allele_X_matrix = dict()
            for allele_name in ALLELES:
                total_data = pd.concat(per_allele_list_of_dfs[allele_name])
                for model_num in range(len(models_deque)):
                    current_models = models_deque[model_num]
                    selected_model = current_models[list(current_models.keys())[0]]
                    total_data[f'score_general_{model_num}'] = [selected_model.log_probability(peptide)/len(peptide) for peptide in total_data.peptide]
                X_matrix = total_data[[f'score_general_{i}' for i in range(len(models_deque))]].copy()
                X_matrix['max_column'] = X_matrix.idxmax(axis=1)
            #
                per_allele_X_matrix[allele_name] = X_matrix
            #
            new_train_data = list()
            for model_num in range(len(models_deque)):
                per_allele_new_df = dict()
                for allele_name in ALLELES:
                    X_matrix = per_allele_X_matrix[allele_name]
                    new_df = total_data[X_matrix['max_column'] == f'score_general_{model_num}'].copy()
                    per_allele_new_df[allele_name] =new_df
                new_train_data = split_to_dicts(per_allele_new_df, ALLELES, TARGET_LENGTHS, TOTAL_SPLITS)
                data_deque[model_num] = new_train_data
                # nor we want to reorder all peptides according to scores of 2-4-6-8
   # sys.stdout = orig_stdout





per_allele_scores_data = get_all_scores_and_sort_by_best(train_data, total_models)
df = per_allele_scores_data[allele_name]
target_df = df[[col for col in df.columns if col.startswith("model")]]
sns.clustermap(target_df)

sns.clustermap(target_df, metric="correlation" )

df = per_allele_scores_data[ALLELES[0]]

pd

t = dict()
t['model_1x'] = list(df.model_1.values)
t['model_12']= list(df.model_2.values)
t['model_3x'] = list(df.model_3.values)



pd.concat([df, pd.DataFrame(t, index=df.index )], axis=1)

df

df = per_allele_scores_data[allele_name]
target_df = df[[col for col in df.columns if col.startswith("model")]]
sns.clustermap(target_df)
pca = PCA(n_components=30)
pca_result = pca.fit_transform(target_df.values)
ann_df = additional[0][allele_name]
ann_df = ann_df[ann_df.split == 0]
print("this method supports 1 split for now")
ann_df = ann_df[['peptide', 'allele']]
annotated_df = df.merge(ann_df, on='peptide', how='left', validate='1:1')
to_plot = pd.DataFrame({"x": pca_result[:, 1], "y": pca_result[:, 2], "allele":annotated_df.allele.values})
sns.jointplot(to_plot, x="x", y="y", hue="allele", kind='kde')
fit = umap.UMAP()
u = fit.fit_transform(pca_result)
to_plot = pd.DataFrame({"x": u[:, 0], "y": u[:, 1], "allele": annotated_df.allele.values, "peptide": annotated_df.peptide.values})
sns.jointplot(to_plot, x="x", y="y", hue='allele', kind='kde')

stats_storage.keys()

enrichmenet_stats = stats_storage["11"]

df = enrichmenet_stats[ALLELES[0]]

df1 = df[df['score'] >= df['score'].mean()].copy()
df2 = df[df['score'] < df['score'].mean()].copy()
len(df1), len(df2)

lengths = list()
for step in range(5):
    #print("Step", step)
    df1 = df[df[f'score1_step{step}'] >= df[f'score2_step{step}']].copy()
    df2 = df[df[f'score2_step{step}'] >= df[f'score1_step{step}']].copy()
    lengths.append(f"{len(df1)}/{len(df2)}")

print(" -> ".join(lengths))

df = enrichmenet_stats[ALLELES[0]]

current_step = 0
columns = [f"score1_step{current_step}",f"score2_step{current_step}"]
t_df = df.copy()
t_df = t_df.sort_values(by=f"score1_step{current_step}", ascending=False)
t_df = t_df.reset_index(drop=True)
target_df =t_df.melt(id_vars=['peptide'],value_vars=columns, value_name = 'score', var_name='model', ignore_index=False)
ax = sns.lineplot(target_df, x=target_df.index, y=f'score', hue='model', palette=sns.color_palette("husl", 2))
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


df = enrichmenet_stats[ALLELES[0]]

current_step = 1
columns = [f"score1_step{current_step}",f"score2_step{current_step}"]
t_df = df.copy()
t_df = t_df.sort_values(by=f"score1_step{current_step}", ascending=False)
t_df = t_df.reset_index(drop=True)
target_df =t_df.melt(id_vars=['peptide'],value_vars=columns, value_name = 'score', var_name='model', ignore_index=False)
ax = sns.lineplot(target_df, x=target_df.index, y=f'score', hue='model', palette=sns.color_palette("husl", 2))
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

df = enrichmenet_stats[ALLELES[0]]

current_step = 2
columns = [f"score1_step{current_step}",f"score2_step{current_step}"]
t_df = df.copy()
t_df = t_df.sort_values(by=f"score1_step{current_step}", ascending=False)
t_df = t_df.reset_index(drop=True)
target_df =t_df.melt(id_vars=['peptide'],value_vars=columns, value_name = 'score', var_name='model', ignore_index=False)
ax = sns.lineplot(target_df, x=target_df.index, y=f'score', hue='model', palette=sns.color_palette("husl", 2))
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


df = enrichmenet_stats[ALLELES[0]]

current_step = 3
columns = [f"score1_step{current_step}",f"score2_step{current_step}"]
t_df = df.copy()
t_df = t_df.sort_values(by=f"score1_step{current_step}", ascending=False)
t_df = t_df.reset_index(drop=True)
target_df =t_df.melt(id_vars=['peptide'],value_vars=columns, value_name = 'score', var_name='model', ignore_index=False)
ax = sns.lineplot(target_df, x=target_df.index, y=f'score', hue='model', palette=sns.color_palette("husl", 2))
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


df = enrichmenet_stats[ALLELES[0]]

current_step = 4
columns = [f"score1_step{current_step}",f"score2_step{current_step}"]
t_df = df.copy()
t_df = t_df.sort_values(by=f"score1_step{current_step}", ascending=False)
t_df = t_df.reset_index(drop=True)
target_df =t_df.melt(id_vars=['peptide'],value_vars=columns, value_name = 'score', var_name='model', ignore_index=False)
ax = sns.lineplot(target_df, x=target_df.index, y=f'score', hue='model', palette=sns.color_palette("husl", 2))
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

a = [iteration_deque.popleft() for i in range(6)]
m = [models_deque.popleft() for i in range(6)]
d = [data_deque.popleft() for i in range(6)]

iteration_deque

per_name_current_models = models_deque.popleft()
this_train_data = data_deque.popleft()
iteration = iteration_deque.popleft()
print("iteration", iteration)
# run current data
per_name_models1, per_name_models2, new_train_data1, new_train_data2, enrichmenet_stats = split_model_and_enrich(per_name_current_models, this_train_data, iteration, enrichment_steps=5)
# add left to queue
models_deque.append(per_name_models1)
stats_storage[iteration] = enrichmenet_stats

#data_deque.append(new_train_data1)
# add whole train data
data_deque.append(new_train_data1)
iteration_deque.append(f'{iteration}1')
models_deque.append(per_name_models2)
data_deque.append(new_train_data2)
# add whole train data
# data_deque.append(train_data)
iteration_deque.append(f'{iteration}2')
if i in {0, 2, 6}:
    #params['anchor_default_pseudocount']=  params['anchor_default_pseudocount']/2
    params['emission_pseudocount']=  params['emission_pseudocount']/2
if i in {0, 2, 6}:
    # make total data df (should be per allele)
    per_allele_list_of_dfs = defaultdict(list)
    for allele_name in ALLELES:
        for current_split in range(TOTAL_SPLITS):
            for current_length in TARGET_LENGTHS:
                peptides = train_data[allele_name][current_split][current_length]
                df = pd.DataFrame({'peptide': peptides,'split': current_split, 'length':current_length})
                per_allele_list_of_dfs[allele_name].append(df)
    #
    per_allele_X_matrix = dict()
    for allele_name in ALLELES:
        total_data = pd.concat(per_allele_list_of_dfs[allele_name])
        for model_num in range(len(models_deque)):
            current_models = models_deque[model_num]
            selected_model = current_models[list(current_models.keys())[0]]
            total_data[f'score_general_{model_num}'] = [selected_model.log_probability(peptide)/len(peptide) for peptide in total_data.peptide]
        X_matrix = total_data[[f'score_general_{i}' for i in range(len(models_deque))]].copy()
        X_matrix['max_column'] = X_matrix.idxmax(axis=1)
    #
        per_allele_X_matrix[allele_name] = X_matrix
    #
    new_train_data = list()
    for model_num in range(len(models_deque)):
        per_allele_new_df = dict()
        for allele_name in ALLELES:
            X_matrix = per_allele_X_matrix[allele_name]
            new_df = total_data[X_matrix['max_column'] == f'score_general_{model_num}'].copy()
            per_allele_new_df[allele_name] =new_df
        new_train_data = split_to_dicts(per_allele_new_df, ALLELES, TARGET_LENGTHS, TOTAL_SPLITS)
        data_deque[model_num] = new_train_data
        # nor we want to reorder all peptides according to scores of 2-4-6-8

NUM_RUNS = 3

from collections import deque

# try two splits  and multiple enrichments
stats_storage = dict()
iteration = "1"
models_deque = deque()
data_deque = deque()
iteration_deque = deque()

models_deque.append(per_name_models)
data_deque.append(train_data)
iteration_deque.append("1")

for i in range(3):
    per_name_current_models = models_deque.popleft()
    this_train_data = data_deque.popleft()
    iteration = iteration_deque.popleft()
    # run current data
    per_name_models1, per_name_models2, new_train_data1, new_train_data2, enrichmenet_stats = split_model_and_enrich(per_name_current_models, this_train_data, iteration, enrichment_steps=5)
    # add left to queue
    models_deque.append(per_name_models1)
    stats_storage[iteration] = enrichmenet_stats
    #data_deque.append(new_train_data1)
    # add whole train data
    data_deque.append(new_train_data1)
    iteration_deque.append(f'{iteration}1')
    models_deque.append(per_name_models2)
    data_deque.append(new_train_data2)
    # add whole train data
   # data_deque.append(train_data)
    iteration_deque.append(f'{iteration}2')
    if i in {0, 2, 6}:
        #params['anchor_default_pseudocount']=  params['anchor_default_pseudocount']/2
        params['emission_pseudocount']=  params['emission_pseudocount']/2



current_step = 3
columns = [f"score1_step{current_step}",f"score2_step{current_step}"]
t_df = df.copy()
t_df = t_df.sort_values(by=f"score1_step{current_step}", ascending=False)
target_df =t_df.melt(id_vars=['peptide'],value_vars=columns, value_name = 'score', var_name='model', ignore_index=False)
ax = sns.lineplot(target_df, x=target_df.index, y=f'score', hue='model', palette=sns.color_palette("husl", 2))
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

enrichmenet_stats[ALLELES[0]]

for i in range(7):
    per_name_current_models = models_deque.popleft()
    this_train_data = data_deque.popleft()
    iteration = iteration_deque.popleft()
    # run current data
    per_name_models1, per_name_models2, new_train_data1, new_train_data2 = split_model(per_name_current_models, this_train_data, iteration)
    # add left to queue
    models_deque.append(per_name_models1)

    #data_deque.append(new_train_data1)
    # add whole train data
    data_deque.append(new_train_data1)
    iteration_deque.append(f'{iteration}1')
    models_deque.append(per_name_models2)
    data_deque.append(new_train_data2)
    # add whole train data
   # data_deque.append(train_data)
    iteration_deque.append(f'{iteration}2')
    if i in {0, 2, 6}:
        #params['anchor_default_pseudocount']=  params['anchor_default_pseudocount']/2
        params['emission_pseudocount']=  params['emission_pseudocount']/2

from collections import deque

result_models1 = list()
result_models2 = list()
result_train_data1 = list()
result_train_data2 = list()
iteration = "1"
split_types = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

for split_type in split_types:
    split_type = round(split_type, 2)
    rest_part = np.round(split_type, 2)
    iteration = f"-({split_type},{rest_part})-"
    #pet_name models is the same base model
    #train_data is the same train data
    per_name_models1, per_name_models2, new_train_data1, new_train_data2 = split_model(per_name_models, train_data, iteration, split_type=split_type)
    result_models1.append(per_name_models1)
    result_models2.append(per_name_models2)
    result_train_data1.append(new_train_data1)
    result_train_data2.append(new_train_data2)


len(result_models1)

def get_average_model_for_runs(per_name_models):
    example_model = per_name_models[list(per_name_models.keys())[0]]
    transition_matrix_cube = np.zeros((NUM_RUNS, len(example_model.states), len(example_model.states)))
    emission_matrix_cube = np.zeros((NUM_RUNS, len(example_model.states)-2, 20))

    for i, model in enumerate(per_name_models.values()):
        transition_matrix_cube[i, :, :] = model.dense_transition_matrix()
        for j,state in enumerate(model.states):
            if state.distribution is not None:
                params = list(state.distribution.parameters[0].values())
                emission_matrix_cube[i, j, :] = np.array(params)

    mean_emission_table = np.median(emission_matrix_cube, axis=0)
    mean_transition_table = np.median(transition_matrix_cube, axis=0)

    distributions = []
    amino_acids_list = list('ACDEFGHIKLMNPQRSTVWY')
    for j in range(len(example_model.states)-2):
        d_dict = {key: prob for key, prob in zip(amino_acids_list, mean_emission_table[j, :])}
        curr_distribution = DiscreteDistribution(d_dict)
        distributions.append(curr_distribution)

    model = HiddenMarkovModel.from_matrix(
        name="average model",
        transition_probabilities=mean_transition_table[:-2,:-2],
        starts=mean_transition_table[-2, :],
        distributions=distributions,
        ends=mean_transition_table[:, -1],
        state_names = ["s{:03d}".format(i) for i in range(len(example_model.states)-2)],
        verbose=True)
    model.bake()
    return model

per_allele_list_of_dfs = defaultdict(list)

for allele_name in ALLELES:
    for current_split in range(TOTAL_SPLITS):
        for current_length in TARGET_LENGTHS:
            peptides = train_data[allele_name][current_split][current_length]
            df = pd.DataFrame({'peptide': peptides,'split': current_split, 'length':current_length})
            per_allele_list_of_dfs[allele_name].append(df)

per_allele_data_df = dict()
for allele_name in ALLELES:
    per_allele_data_df[allele_name] = pd.concat(per_allele_list_of_dfs[allele_name])
for allele_name in ALLELES:
    df = per_allele_data_df[allele_name]
    for i, model in enumerate(per_name_models.values()):
        df[f'model_{i}'] = [model.log_probability(peptide)/len(peptide) for peptide in df.peptide]
        df = df.sort_values(by='model_0', ascending=False)
        per_allele_data_df[allele_name] = df

for allele_name in ALLELES:
    df = per_allele_data_df[allele_name]
    for current_split, models_dict in zip(split_types,result_models1):
        selected_model = get_average_model_for_runs(models_dict)
        df[f'model_{current_split}'] = [selected_model.log_probability(peptide)/len(peptide) for peptide in df.peptide]
        per_allele_data_df[allele_name] = df

per_allele_data_df[allele_name]

df = df.reset_index(drop=True)
target_df =df.melt(id_vars=['peptide'],value_vars=['model_0'] + [f'model_{i}' for i in split_types], value_name = 'score', var_name='model', ignore_index=False)

for allele_name in ALLELES:
    df = per_allele_data_df[allele_name]
    for current_split, models_dict in zip(split_types,result_models2):
        selected_model = get_average_model_for_runs(models_dict)
        df[f'model_{current_split}_alt'] = [selected_model.log_probability(peptide)/len(peptide) for peptide in df.peptide]
        per_allele_data_df[allele_name] = df

per_allele_data_df[allele_name]
df = df.reset_index(drop=True)
target_df = df.melt(id_vars=['peptide'], value_vars=['model_0'] + [f'model_{i}_alt' for i in split_types],
                    value_name='score', var_name='model', ignore_index=False)

target_df

df

ax = sns.lineplot(target_df, x=target_df.index, y=f'score', hue='model')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))


target_model = 0.7
new_df = df.sort_values(by=f'model_0', ascending=False)
new_df = new_df.reset_index(drop=True)
target_df = new_df.melt(id_vars=['peptide'], value_vars=['model_0'] + [f'model_{i}_alt' for i in [ target_model]] + [f'model_{i}' for i in [target_model]],
                    value_name='score', var_name='model', ignore_index=False)
ax = sns.lineplot(target_df, x=target_df.index, y=f'score', hue='model')
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))

def get_average_model_for_runs(per_name_models):
    example_model = per_name_models[list(per_name_models.keys())[0]]
    transition_matrix_cube = np.zeros((NUM_RUNS, len(example_model.states), len(example_model.states)))
    emission_matrix_cube = np.zeros((NUM_RUNS, len(example_model.states)-2, 20))

    for i, model in enumerate(per_name_models.values()):
        transition_matrix_cube[i, :, :] = model.dense_transition_matrix()
        for j,state in enumerate(model.states):
            if state.distribution is not None:
                params = list(state.distribution.parameters[0].values())
                emission_matrix_cube[i, j, :] = np.array(params)

    mean_emission_table = np.median(emission_matrix_cube, axis=0)
    mean_transition_table = np.median(transition_matrix_cube, axis=0)

    distributions = []
    amino_acids_list = list('ACDEFGHIKLMNPQRSTVWY')
    for j in range(len(example_model.states)-2):
        d_dict = {key: prob for key, prob in zip(amino_acids_list, mean_emission_table[j, :])}
        curr_distribution = DiscreteDistribution(d_dict)
        distributions.append(curr_distribution)

    model = HiddenMarkovModel.from_matrix(
        name="average model",
        transition_probabilities=mean_transition_table[:-2,:-2],
        starts=mean_transition_table[-2, :],
        distributions=distributions,
        ends=mean_transition_table[:, -1],
        state_names = ["s{:03d}".format(i) for i in range(len(example_model.states)-2)],
        verbose=True)
    model.bake()
    return model

selected_model = get_average_model_for_runs(per_name_models)

per_allele_list_of_dfs = defaultdict(list)

for allele_name in ALLELES:
    for current_split in range(TOTAL_SPLITS):
        for current_length in TARGET_LENGTHS:
            peptides = train_data[allele_name][current_split][current_length]
            df = pd.DataFrame({'peptide': peptides,'split': current_split, 'length':current_length})
            per_allele_list_of_dfs[allele_name].append(df)

per_allele_data_df = dict()
for allele_name in ALLELES:
    per_allele_data_df[allele_name] = pd.concat(per_allele_list_of_dfs[allele_name])

per_allele_data_df[ALLELES[0]]

for allele_name in ALLELES:
    df = per_allele_data_df[allele_name]
    df['score'] = [selected_model.log_probability(peptide)/len(peptide) for peptide in df.peptide]
    df = df.sort_values(by='score', ascending=False)
    per_allele_data_df[allele_name] = df

df

df[df.score > -3].length.value_counts()

df1 = df.loc[df.score > -3].copy()
df2 = df[~df.apply(tuple,axis=1).isin(df1.apply(tuple,axis=1))].copy()

df1

df[df.index.isin(df1.index)]

per_allele_data_df[allele_name].to_csv('original_predictions.csv', sep=';')

import seaborn as sns

per_allele_first_df = dict()
per_allele_second_df = dict()
for allele_name in ALLELES:
    df = per_allele_data_df[allele_name]
    df['score'] = [selected_model.log_probability(peptide)/len(peptide) for peptide in df.peptide]
    df = df.sort_values(by='score', ascending=False)
    df1 = df.iloc[:int(len(df)/2)].copy()
    df2 = df.iloc[int(len(df)/2):].copy()
    per_allele_first_df[allele_name] = df1
    per_allele_second_df[allele_name] = df2
    per_allele_data_df[allele_name] = df

sns.displot(df, x='score')

len(train_data[ALLELES[0]][0][15])

from collections import deque

iteration = "1"
split_types = np.linspace(0.05, 0.95, num=20)


for split_type in split_types:
    split_type = round(split_type, 2)
    rest_part =round(1-split_type, 2)
    iteration = f"-({split_type},{rest_part})-"
    #pet_name models is the same base model
    #train_data is the same train data
    per_name_models1, per_name_models2, new_train_data1, new_train_data2 = split_model(per_name_models, train_data, iteration, split_type=split_type)


split_type = round(split_types[2], 2)
iteration = f"-({split_type},{1-split_type})-"

split_type

####!!!! TODO: change to per allele model
# selected_model = get_average_model_for_runs(per_name_models)
selected_model = per_name_models[list(per_name_models.keys())[0]]
####!!!
per_allele_list_of_dfs = defaultdict(list)
for allele_name in ALLELES:
    for current_split in range(TOTAL_SPLITS):
        for current_length in TARGET_LENGTHS:
            peptides = train_data[allele_name][current_split][current_length]
            df = pd.DataFrame({'peptide': peptides,'split': current_split, 'length':current_length})
            per_allele_list_of_dfs[allele_name].append(df)

per_allele_data_df = dict()
per_allele_first_df = dict()
per_allele_second_df =dict()
for allele_name in ALLELES:
    df = pd.concat(per_allele_list_of_dfs[allele_name])
    df['score'] = [selected_model.log_probability(peptide)/len(peptide) for peptide in df.peptide]
    df = df.sort_values(by='score', ascending=False)
    #df1 = df.loc[df.score > -3].copy()
    #df2 = df[~df.apply(tuple,axis=1).isin(df1.apply(tuple,axis=1))].copy()
    df1 = df.iloc[:int(len(df)*split_type)].copy()
    df2 = df.iloc[int(len(df)*split_type):].copy()
    per_allele_first_df[allele_name] = df1
    per_allele_second_df[allele_name] = df2
    per_allele_data_df[allele_name] = df
new_train_data1 = split_to_dicts(per_allele_first_df, ALLELES, TARGET_LENGTHS, TOTAL_SPLITS)
new_train_data2 = split_to_dicts(per_allele_second_df, ALLELES, TARGET_LENGTHS, TOTAL_SPLITS)
train_data_weights1 = get_weights(new_train_data1, old_test_data)
train_data_weights2 = get_weights(new_train_data2, old_test_data)
per_allele_per_run_per_split_prepared_models1 = prepare_all_models()
per_allele_per_run_per_split_prepared_models2 = prepare_all_models()

#1
per_allele_per_run_per_split_prepared_models1, \
per_allele_per_run_per_split_new_histories1 = train_all_models(train_data=new_train_data1,
                                                              test_data=test_data,
                                                              train_data_weights=train_data_weights1,
                                                              save_dir=TARGET_PATH_TO_RESULTS + f'/new_model{iteration}1/',
                                                              prepared_models=per_allele_per_run_per_split_prepared_models1
                                                              )
#2
per_allele_per_run_per_split_prepared_models2, \
per_allele_per_run_per_split_new_histories2 = train_all_models(train_data=new_train_data2,
                                                              test_data=test_data,
                                                              train_data_weights=train_data_weights2,
                                                              save_dir=TARGET_PATH_TO_RESULTS +f'/new_model{iteration}2/',
                                                              prepared_models=per_allele_per_run_per_split_prepared_models2
                                                              )
per_name_models1, per_name_histories1 = reorder_models_by_score(histories=per_allele_per_run_per_split_new_histories1,
                                                            models=per_allele_per_run_per_split_prepared_models1)
per_name_models2, per_name_histories2 = reorder_models_by_score(histories=per_allele_per_run_per_split_new_histories2,
                                                                models=per_allele_per_run_per_split_prepared_models2)
save_all_visualization_results(per_name_models1, per_name_histories1, save_dir=TARGET_PATH_TO_RESULTS + f'/new_model{iteration}1/')
save_all_visualization_results(per_name_models2, per_name_histories2, save_dir=TARGET_PATH_TO_RESULTS + f'/new_model{iteration}2/')


from collections import deque

iteration = "1"
models_deque = deque()
data_deque = deque()
iteration_deque = deque()

models_deque.append(per_name_models)
data_deque.append(train_data)
iteration_deque.append("1")

for i in range(3):
    per_name_models = models_deque.popleft()
    train_data = data_deque.popleft()
    iteration = iteration_deque.popleft()
    # run current data
    per_name_models1, per_name_models2, new_train_data1, new_train_data2 = split_model(per_name_models, train_data, iteration)
    # add left to queue
    models_deque.append(per_name_models1)
    data_deque.append(new_train_data1)
    iteration_deque.append(f'{iteration}1')
    models_deque.append(per_name_models2)
    data_deque.append(new_train_data2)
    iteration_deque.append(f'{iteration}2')



df = join_dicts(train_data)

df

iteration

iteration_deque



