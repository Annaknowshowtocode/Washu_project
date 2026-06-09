import pandas as pd
import json
from typing import Optional
from pomegranate import DiscreteDistribution, DiscreteDistributionAnchor, HiddenMarkovModel, DiscreteDistributionCycle
from pomegranate.io import BatchedDataGenerator, SequenceGenerator
import numpy as np
from hmm_visualization_methods import convert_graph_to_good_format, save_all_visualization_results
from networkx import all_simple_paths
from tqdm import tqdm
from collections import defaultdict
from training_parameters import *
from copy import deepcopy
from peptides_utils import split_to_dicts, join_dicts, get_frequencies
from data_reading_methods import calculate_weights_based_on_length_counts
from collections import deque
from hmm_visualization_methods import plot_split_diagnostics, plot_info_matrices_to_file
from scipy.spatial.distance import jensenshannon
import os
import matplotlib.pyplot as plt
import pickle
import logomaker
from hmm_diversity_split import get_data_split_by_clustering
from hmm_cluster_diagnostics import visualize_clustering_split
from hmm_visualization_methods import save_figure_to_svg
from hmm_metrics import (
    MetricsTracker,
    load_reference_matrices,
)

REFERENCE_MATRICES = load_reference_matrices(
    mat_dir=str(MAT_DIR),
    allele_map={
        "DRB1*01:01": "DRB1_0101",
        "DRB1*03:01": "DRB1_0301",
        "DRB1*04:01": "DRB1_0401",
        "DRB1*07:01": "DRB1_0701",
    },
)

def train_model_prepared(
        per_kfold_per_length_data,
        model_training_params: ModelTrainingParams,
        per_kfold_per_length_test_data=None,
        per_kfold_per_length_weights_for_data=None,
        prepared_models=None
):
    target_lengths = model_training_params.lengths_to_use
    multiple_check_input = model_training_params.multiple_check_input
    algorithm = model_training_params.algorithm
    lr_decay = model_training_params.lr_decay
    minibatch_training = model_training_params.minibatch_training
    batches_per_epoch = model_training_params.batches_per_epoch
    batch_size = model_training_params.batch_size
    min_iterations = model_training_params.min_iters
    max_iterations = model_training_params.maxiters
    emission_pseudocount = model_training_params.emission_pseudocount
    transition_pseudocount = model_training_params.transition_pseudocount
    use_pseudocount = model_training_params.use_pseudocounts
    edge_inertia = model_training_params.edge_inertia
    distribution_inertia = model_training_params.distribution_inertia
    stop_threshold = model_training_params.stop_threshold
    verbose = model_training_params.verbose

    use_test = False
    n_jobs = 1

    per_split_model = dict()
    per_split_history = dict()

    assert type(prepared_models) in [dict, type(None)], \
        "Prepared models should be either None (for new models) or per split models dict"

    for split_num, per_length_data in per_kfold_per_length_data.items():
        per_length_current_weights = per_kfold_per_length_weights_for_data[split_num]
        per_length_test_data = per_kfold_per_length_test_data[split_num]

        binders_array = np.array(
            [per_length_data[tl][i] for tl in target_lengths for i in range(len(per_length_data[tl]))],
            dtype=object)
        weights_array = np.array(
            [per_length_current_weights[tl][i] for tl in target_lengths for i in range(len(per_length_current_weights[tl]))],
            dtype=object)
        binders_test_array = np.array(
            [per_length_test_data[tl][i] for tl in target_lengths for i in range(len(per_length_test_data[tl]))],
            dtype=object)

        rng = np.random.default_rng()
        new_indexes = rng.permutation(len(binders_array))
        binders_array = binders_array[new_indexes]
        weights_array = weights_array[new_indexes]
        rng.shuffle(binders_test_array)

        sample_X = np.array([[char for char in item] for item in binders_array], dtype=object)
        sample_X_test = np.array([[char for char in item] for item in binders_test_array], dtype=object)

        sequence_test_generator = SequenceGenerator(sample_X_test) if use_test else None

        if minibatch_training:
            sequence_generator = BatchedDataGenerator(sample_X,
                                                      batches_per_epoch=batches_per_epoch,
                                                      batch_size=batch_size)
            sequence_generator.reset()
        else:
            sequence_generator = sample_X

        if split_num in prepared_models:
            model = prepared_models[split_num]
            model, history = model.fit(sequence_generator,
                                       sequences_test=sequence_test_generator,
                                       stop_threshold=stop_threshold,
                                       return_history=True,
                                       verbose=verbose,
                                       multiple_check_input=multiple_check_input,
                                       n_jobs=n_jobs,
                                       algorithm=algorithm,
                                       lr_decay=lr_decay,
                                       distribution_inertia=distribution_inertia,
                                       edge_inertia=edge_inertia,
                                       batches_per_epoch=batches_per_epoch,
                                       min_iterations=min_iterations,
                                       max_iterations=max_iterations,
                                       emission_pseudocount=emission_pseudocount,
                                       transition_pseudocount=transition_pseudocount,
                                       use_pseudocount=use_pseudocount,
                                       weights=weights_array)
            per_split_model[split_num] = model
            per_split_history[split_num] = history
        else:
            print("skip training for this split since model was not provided")

        break

    return per_split_model, per_split_history


# Эта функция создаёт эмиссионное распределение (emission distribution) для состояния HMM
# каждому AA даём случайный вес
def init_distribution(amino_acids=None,
                      initial_means=None,
                      initial_stds=None,
                      discrete_distribution=False, # TRUE!
                      equal_probs=False, # начальные вероятности аминокислот не равные  #Случайная инициализация помогает EM/baum-welch “разойтись” по разным локальным решениям, а равномерная — более стабильная, но иногда хуже стартует.
                      anchor=False,
                      anchor_max_components=5,
                      anchor_default_pseudocount=50, #20
                      cycle_max_components=20, #5
                      cycle_default_pseudocount=50,
                      cycle=False,
                      position_component=False,
                      prepared_probs=None): # вероятности инициализируются случайно


    amino_acids_list = list(sorted(set(amino_acids))) #['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L']

    position_variants = [str(i) for i in range(19)] #['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']

    if not discrete_distribution:

        components_d  = list()

        for i in range(len(initial_means)):

            mean_value = initial_means[i]
            std_value = initial_stds[i]
            d = NormalDistribution(mean_value, std_value)
            components_d.append(d)
        total_d = IndependentComponentsDistribution(components_d)

    else:

        if prepared_probs is not None:
            d_dict = {key: prepared_probs[key] for key in amino_acids_list}

        elif not equal_probs: #True (random init probs)"
            d_dict = {key: prob for key, prob in zip(
                amino_acids_list,
                np.random.default_rng().uniform(0, 1, size=len(amino_acids_list))
            )}
        else:
            d_dict = {key: 1 for key in amino_acids_list}

    # Нормализация вероятностей чтоб в сумме были 1
        d_sum = sum(d_dict.values())
        d_dict = {key: value/d_sum for key, value in d_dict.items()}

        if anchor:
            aa_dist = DiscreteDistributionAnchor(d_dict)
            aa_dist.max_components = anchor_max_components
            aa_dist.default_pseudocount = anchor_default_pseudocount

        elif cycle:
            aa_dist = DiscreteDistributionCycle(d_dict)
            aa_dist.max_components = cycle_max_components
            aa_dist.default_pseudocount = cycle_default_pseudocount

        else: #True
            aa_dist = DiscreteDistribution(d_dict)
        total_d = aa_dist

    # Эмиссия становится двухкомпонентной (вероятность, позиция)
        if position_component: #False
            pos_d_dict = {key: prob for key, prob in
                          zip(position_variants, np.random.default_rng().uniform(0, 1, size=len(position_variants)))}
            position_distribution = DiscreteDistribution(pos_d_dict)

            total_d = IndependentComponentsDistribution([aa_dist, position_distribution], weights=[0.9, 0.1])
    return total_d



# Эта функция собирает Hidden Markov Model  из ModelTrainingParams:
# она создаёт все состояния, задаёт им эмиссионные распределения (через init_distribution),
# потом строит матрицу переходов, задаёт start/end вероятности, добавляет переходы (циклы с псевдосчётами),
# делает bake() и возвращает готовую модель

def build_model_based_on_params( model_training_params:ModelTrainingParams, amino_acids='ACDEFGHIKLMNPQRSTVWY',verbose=False,
                                 prepared_emission_matrix=None
                                 ):

    states_per_group = model_training_params.states_per_group
    number_of_groups = model_training_params.groups
    model_complexity = model_training_params.model_complexity

    # architecture
    start_cycle = model_training_params.start_cycle
    end_cycle = model_training_params.end_cycle
    self_cycle = model_training_params.self_cycle
    intermediate_cycle = model_training_params.intermediate_cycle
    cycle_chain = model_training_params.cycle_chain
    chain_length = model_training_params.cycle_chain_length
    anchor_states = model_training_params.anchor_states


    # Distribution restrictions
    freeze_start_end_distr = model_training_params.freeze_start_end_cycle
    freeze_int_cycle_distr = model_training_params.freeze_intermediate_cycle
    tie_start_end_cycle_states = model_training_params.tie_start_end_cycle_states
    tie_anchor_states = model_training_params.tie_anchor_states

    # Type of distribtion
    discrete_distribution = model_training_params.aa_labels_training
    position_component = model_training_params.position_component
    equal_probs_for_cycle = model_training_params.equal_probs_for_cycle
    equal_probs_for_intermediate_cycle = model_training_params.equal_probs_for_intermediate_cycle
    initial_params = model_training_params.initial_params

    # Regularization for distributions
    in_cycle_tr_pseudocount = model_training_params.in_cycle_tr_pseudocount
    cycle_max_components = model_training_params.cycle_top_aas
    anchor_max_components = model_training_params.anchor_top_aas
    anchor_default_pseudocount = model_training_params.anchor_default_pseudocount
    cycle_default_pseudocount = model_training_params.cycle_default_pseudocount

    INTRAGROUP_WEIGHT = 1
    INTERGROUP_WEIGHT = 1
    ##  0.16  and 0.83 situation both ends
    START_WEIGHT = 1
    TO_FIRST_CYCLE_WEIGHT = 5
    TO_LAST_CYCLE_WEIGHT = 5

    if start_cycle or end_cycle:
        TO_FIRST_JOINER_WEIGHT = 5  # just to make transitions even inside groups
        TO_LAST_JOINER_WEIGHT = 1
    else:
        TO_FIRST_JOINER_WEIGHT = 1
        TO_LAST_JOINER_WEIGHT = 2.5

    # IN_FIRST_CYCLE_WEIGHTS = np.linspace(0.1, 0.9, num=model_complexity)
    IN_FIRST_CYCLE_WEIGHTS = np.repeat(0.5, model_complexity)
    FROM_FIRST_CYCLE_WEIGHTS = 1 - IN_FIRST_CYCLE_WEIGHTS
    # IN_LAST_CYCLE_WEIGHTS = np.linspace(0.1, 0.9, num=model_complexity)[::-1]
    IN_LAST_CYCLE_WEIGHTS = np.repeat(0.5, model_complexity)
    FROM_LAST_CYCLE_WEIGHTS = 1 - IN_LAST_CYCLE_WEIGHTS
    END_WEIGHT = 1

    joiner_positions = {0: list(), 1: list()}
    if anchor_states:
        joiner_positions[0] = [i for i in range(model_complexity)]
        joiner_positions[1] = [i for i in range(model_complexity, model_complexity * 2)]

    start_cycle_labels = [i for i in range(model_complexity)] if not cycle_chain else [i for i in range(chain_length)]
    end_cycle_labels = [i for i in range(model_complexity, 2 * model_complexity)] if not cycle_chain else [i for i in
                                                                                                           range(
                                                                                                               chain_length,
                                                                                                               2 * chain_length)]
    number_of_joiners = sum(len(positions) for positions in joiner_positions.values())
    number_of_groups = number_of_groups

    if type(states_per_group) == list:
        assert number_of_groups == len(
            states_per_group), f"Number of groups is expected to be {number_of_groups} but got {len(states_per_group)} in a list of states per group"
    else:
        states_per_group = [states_per_group for i in range(number_of_groups)]
    distributions = list()

    mean_values = list()
    stds = list()
    if initial_params:
        for initial_distr_params in initial_params:
            mean_values.append(initial_distr_params['mean'])
            stds.append(initial_distr_params['std'])
        if verbose:
            print(f"mean values : {mean_values}")
            print(f"std values : {stds}")
    if prepared_emission_matrix is not None:
        print("will try to set probabilities from the matrix")
    # initialize group states
    for group_number in range(number_of_groups):
        states_in_group = states_per_group[group_number]
        for state_num in range(states_in_group):
            total_d = init_distribution(amino_acids=amino_acids,
                                        initial_means=mean_values,
                                        initial_stds=stds,
                                        discrete_distribution=discrete_distribution,
                                        position_component=position_component,
                                        prepared_probs=dict(prepared_emission_matrix.loc[len(distributions), :]) if prepared_emission_matrix is not None else None)
            distributions.append(total_d)
    # initialize joiner states (random here)
    joiner_label_to_index_in_model = dict()
    # Start joiners  (tie if necessary)
    for joiner_name in range(int(number_of_joiners / 2)):
        total_d = init_distribution(amino_acids=amino_acids,
                                    initial_means=mean_values,
                                    initial_stds=stds,
                                    discrete_distribution=discrete_distribution,
                                    anchor=anchor_states,
                                    anchor_max_components=anchor_max_components,
                                    anchor_default_pseudocount=anchor_default_pseudocount,
                                    position_component=position_component,
                                    prepared_probs=dict(prepared_emission_matrix.loc[len(distributions), :]) if prepared_emission_matrix is not None else None)
        joiner_label_to_index_in_model[joiner_name] = len(distributions)
        distributions.append(total_d)

    # End joiners (tie with paired start joiner if necessary)
    for joiner_name in range(int(number_of_joiners / 2), number_of_joiners):
        if tie_anchor_states:
            pair_joiner_name = number_of_joiners - joiner_name - 1
            total_d = distributions[joiner_label_to_index_in_model[pair_joiner_name]]
        else:
            total_d = init_distribution(amino_acids=amino_acids,
                                        initial_means=mean_values,
                                        initial_stds=stds,
                                        discrete_distribution=discrete_distribution,
                                        anchor=anchor_states,
                                        anchor_max_components=anchor_max_components,
                                        anchor_default_pseudocount=anchor_default_pseudocount,
                                        position_component=position_component,
                                        prepared_probs=dict(prepared_emission_matrix.loc[len(distributions), :]) if prepared_emission_matrix is not None else None)
        joiner_label_to_index_in_model[joiner_name] = len(distributions)
        distributions.append(total_d)

    # initialize cycle states
    cycle_label_to_index_in_model = dict()
    start_cycle_label_to_index_in_model = dict()
    number_of_start_cycles = 0
    if start_cycle:
        start_cycle_d = init_distribution(amino_acids=amino_acids,
                                          initial_means=mean_values,
                                          initial_stds=stds,
                                          discrete_distribution=discrete_distribution,
                                          equal_probs=equal_probs_for_cycle,
                                          cycle=True,
                                          cycle_max_components=cycle_max_components,
                                          cycle_default_pseudocount=cycle_default_pseudocount,
                                          position_component=position_component)

        for start_cycle_name in start_cycle_labels:
            if tie_start_end_cycle_states:
                start_d = start_cycle_d
            else:
                start_d = init_distribution(amino_acids=amino_acids,
                                            initial_means=mean_values,
                                            initial_stds=stds,
                                            cycle=True,
                                            cycle_max_components=cycle_max_components,
                                            cycle_default_pseudocount=cycle_default_pseudocount,
                                            discrete_distribution=discrete_distribution,
                                            equal_probs=equal_probs_for_cycle,
                                            position_component=position_component)

            if freeze_start_end_distr:
                start_d.freeze()

            cycle_label_to_index_in_model[start_cycle_name] = len(distributions)
            distributions.append(start_d)
            number_of_start_cycles += 1
    if verbose:
        print("start cycles added", number_of_start_cycles)
    number_of_end_cycles = 0
    if end_cycle:
        start_cycle_d = init_distribution(amino_acids=amino_acids,
                                          initial_means=mean_values,
                                          initial_stds=stds,
                                          cycle=True,
                                          cycle_max_components=cycle_max_components,
                                          cycle_default_pseudocount=cycle_default_pseudocount,
                                          discrete_distribution=discrete_distribution,
                                          equal_probs=equal_probs_for_cycle,
                                          position_component=position_component)

        for end_cycle_name in end_cycle_labels:
            if tie_start_end_cycle_states:
                end_d = start_cycle_d
            else:
                end_d = init_distribution(amino_acids=amino_acids,
                                          initial_means=mean_values,
                                          initial_stds=stds,
                                          cycle=True,
                                          cycle_max_components=cycle_max_components,
                                          cycle_default_pseudocount=cycle_default_pseudocount,
                                          discrete_distribution=discrete_distribution,
                                          equal_probs=equal_probs_for_cycle,
                                          position_component=position_component)
            if freeze_start_end_distr:
                end_d.freeze()

            cycle_label_to_index_in_model[end_cycle_name] = len(distributions)
            distributions.append(end_d)
            number_of_end_cycles += 1
    if verbose:
        print("end cycles added", number_of_end_cycles)
    number_of_int_cycles = 0
    if intermediate_cycle:
        number_of_int_cycles = 1
        # initialize cycle states
        # here only one cycle
        int_cycle_name = 0
        int_cycle_label_to_index_in_model = dict()
        # Same distribution for all cycles
        int_cycle_d = init_distribution(amino_acids=amino_acids,
                                        initial_means=mean_values,
                                        initial_stds=stds,
                                        cycle=True,
                                        cycle_max_components=cycle_max_components,
                                        cycle_default_pseudocount=cycle_default_pseudocount,
                                        discrete_distribution=discrete_distribution,
                                        equal_probs=equal_probs_for_intermediate_cycle,
                                        position_component=position_component)
        int_cycle_label_to_index_in_model[int_cycle_name] = len(distributions)
        if freeze_int_cycle_distr:
            print("will freeze")
            int_cycle_d.freeze()
        distributions.append(int_cycle_d)

    num_states = sum(states_per_group) + \
                 number_of_joiners + \
                 number_of_start_cycles + \
                 number_of_end_cycles + \
                 number_of_int_cycles
    if verbose:
        print(
        f"states will be groups:{sum(states_per_group)} + joiners:{number_of_joiners} + start/end cycles:{number_of_start_cycles + number_of_end_cycles} + int_cycles:{number_of_int_cycles}")

    assert num_states == len(distributions)

    state_names = ["s{:03d}".format(i) for i in range(num_states)]
    transition_matrix = np.zeros((num_states, num_states))

    # start probabilities
    first_joiners = set(joiner_label_to_index_in_model[label] for label in joiner_positions[0])
    last_joiners = set(
        joiner_label_to_index_in_model[label] for label in joiner_positions[max(joiner_positions.keys())])
    # connect start cycle to the first joiners
    start_cycles = list(cycle_label_to_index_in_model[label] for label in start_cycle_labels if
                        start_cycle)  # because each joiner has it's own cycle

    end_cycles = list(cycle_label_to_index_in_model[label] for label in end_cycle_labels if
                      end_cycle)  # because each joiner has it's own cycle
    if verbose:
        print("One more time number of 'cycle' states", len(start_cycles), len(end_cycles))
        print("For example start cycle states", start_cycles)

    #### Initial probabilities
    if anchor_states:
        # start only for first cycle and first joiner
        start_probabilities = [START_WEIGHT
                               if i in start_cycles or i in first_joiners
                               else 0
                               for i in range(num_states)]
    else:
        # start only for first cycle and first group
        start_probabilities = [START_WEIGHT
                               if i in start_cycles or i < states_per_group[0]
                               else 0
                               for i in range(num_states)]

    #### End probabilities
    if anchor_states:
        # end only for last cycle and last joiner
        end_probabilities = [END_WEIGHT
                             if i in end_cycles or i in last_joiners
                             else 0
                             for i in range(num_states)]
    else:
        # end only for last cycle and last group
        last_group_start_state = sum(states_per_group[:-1])
        last_group_end_state = sum(states_per_group)

        end_probabilities = [END_WEIGHT
                             if i in end_cycles or last_group_start_state <= i < last_group_end_state
                             else 0
                             for i in range(num_states)]

    #  start cycle transitions
    if start_cycle:
        # from start cycles to all joiners
        if cycle_chain:
            start_cycle_idx = start_cycles[-1]
            for joiner_idx in first_joiners:
                transition_matrix[start_cycle_idx][joiner_idx] = TO_FIRST_JOINER_WEIGHT  # from cycle to joiners
        else:  # usual case when снсду is alone
            for start_cycle_idx, joiner_idx in zip(start_cycles, first_joiners):
                transition_matrix[start_cycle_idx][joiner_idx] = TO_FIRST_JOINER_WEIGHT  # from cycle to joiners

    # connect last joiners to last cycle
    if end_cycle:
        if cycle_chain:
            end_cycle_idx = end_cycles[0]
            for joiner_idx in last_joiners:
                transition_matrix[joiner_idx][end_cycle_idx] = TO_LAST_CYCLE_WEIGHT  # from cycle to joiners
        else:
            for end_cycle_idx, joiner_idx in zip(end_cycles, last_joiners):
                transition_matrix[joiner_idx][end_cycle_idx] = TO_LAST_CYCLE_WEIGHT
                # for joiner_idx in last_joiners:
                #     transition_matrix[joiner_idx][end_cycle_idx] = TO_LAST_CYCLE_WEIGHT
    if intermediate_cycle:
        for int_cycle_index in int_cycle_label_to_index_in_model.values():
            transition_matrix[int_cycle_index][int_cycle_index] = 1

    # Transitions inside groups
    for group_number in range(number_of_groups):
        states_in_group = states_per_group[group_number]
        for i in range(states_in_group):
            state_num_from = sum(states_per_group[:group_number]) + i
            # fill transitions inside each group with INTRAGROUP_WEIGHT
            if intermediate_cycle:
                transition_matrix[state_num_from][int_cycle_label_to_index_in_model[0]] = 1
                transition_matrix[int_cycle_label_to_index_in_model[0]][state_num_from] = 1
            for j in range(states_in_group):

                state_num_to = sum(states_per_group[:group_number]) + j
                if state_num_from != state_num_to:
                    transition_matrix[state_num_from][state_num_to] = INTRAGROUP_WEIGHT

            if self_cycle:
                transition_matrix[state_num_from][state_num_from] = INTRAGROUP_WEIGHT

            # transition to next group
            if group_number != number_of_groups - 1:
                for j in range(states_in_group):
                    state_num_to = sum(states_per_group[:group_number + 1]) + j
                    transition_matrix[state_num_from][state_num_to] = INTERGROUP_WEIGHT
                    # through the cycle

    # connect first joiner to first group
    for joiner_index in first_joiners:
        group_number = 0
        state_num_from = joiner_index
        states_in_group = states_per_group[group_number]
        for j in range(states_in_group):
            state_num_to = sum(states_per_group[:group_number]) + j
            transition_matrix[state_num_from][state_num_to] = INTERGROUP_WEIGHT
        # and to intermediate cycle
        if intermediate_cycle:
            transition_matrix[state_num_from][int_cycle_label_to_index_in_model[0]] = INTERGROUP_WEIGHT

    # connect last group to the last joiner
    for joiner_index in last_joiners:
        group_number = number_of_groups - 1
        state_num_to = joiner_index
        states_in_group = states_per_group[group_number]
        for j in range(states_in_group):
            state_num_from = sum(states_per_group[:group_number]) + j
            transition_matrix[state_num_from][state_num_to] = TO_LAST_JOINER_WEIGHT
        # and to intermediate cycle
        if intermediate_cycle:
            transition_matrix[int_cycle_label_to_index_in_model[0]][state_num_to] = TO_LAST_JOINER_WEIGHT
    model_name = 'grouped_model'

    model = HiddenMarkovModel.from_matrix(
        name=model_name,
        transition_probabilities=transition_matrix,
        starts=start_probabilities,
        distributions=distributions,
        ends=end_probabilities,
        state_names=state_names,
        verbose=False)

    # Update created model with specific pseudocounts
    if start_cycle:
        if cycle_chain:
            for idx_from, idx_to in zip(start_cycles, start_cycles[1:]):
                model.add_transition(model.states[idx_from], model.states[idx_to], TO_FIRST_JOINER_WEIGHT)
        else:
            CURR_CYCLE_INDEX = 0
            for start_cycle_idx, joiner_idx in zip(start_cycles, first_joiners):
                model.add_transition(model.states[start_cycle_idx], model.states[start_cycle_idx],
                                     IN_FIRST_CYCLE_WEIGHTS[CURR_CYCLE_INDEX], pseudocount=in_cycle_tr_pseudocount)
                model.add_transition(model.start, model.states[start_cycle_idx], TO_FIRST_CYCLE_WEIGHT,
                                     pseudocount=in_cycle_tr_pseudocount)
                model.add_transition(model.states[start_cycle_idx], model.states[joiner_idx],
                                     FROM_FIRST_CYCLE_WEIGHTS[CURR_CYCLE_INDEX], pseudocount=in_cycle_tr_pseudocount)
                CURR_CYCLE_INDEX += 1
    if end_cycle:
        if cycle_chain:
            for idx_from, idx_to in zip(end_cycles, end_cycles[1:]):
                model.add_transition(model.states[idx_from], model.states[idx_to], TO_LAST_JOINER_WEIGHT)
        else:
            CURR_CYCLE_INDEX = 0
            for end_cycle_idx, joiner_idx in zip(end_cycles, last_joiners):
                model.add_transition(model.states[end_cycle_idx], model.states[end_cycle_idx],
                                     IN_LAST_CYCLE_WEIGHTS[CURR_CYCLE_INDEX], pseudocount=in_cycle_tr_pseudocount)
                model.add_transition(model.states[end_cycle_idx], model.end, FROM_LAST_CYCLE_WEIGHTS[CURR_CYCLE_INDEX],
                                     pseudocount=in_cycle_tr_pseudocount)
                model.add_transition(model.states[joiner_idx], model.states[end_cycle_idx], TO_LAST_CYCLE_WEIGHT,
                                     pseudocount=in_cycle_tr_pseudocount)
                CURR_CYCLE_INDEX += 1
    model.bake(verbose=False)
    return model



def save_model(TARGET_PATH_TO_RESULTS, model, history=None, p_model_folder=None):
    json_object = model.to_json()
    if p_model_folder:
        MODEL_FOLDER = f"{TARGET_PATH_TO_RESULTS}{p_model_folder}/"
    else:
        MODEL_FOLDER = f"{TARGET_PATH_TO_RESULTS}{model.name}/"
    if not os.path.exists(MODEL_FOLDER):
        os.makedirs(MODEL_FOLDER)
    with open(f"{MODEL_FOLDER}/model.json", 'w+') as f:
        json.dump(json_object, f)
    if history:
        history_df = pd.DataFrame()
        history_df['total_improvement'] = history.total_improvement
        history_df['improvements'] = history.improvements
        history_df['log_probabilities'] = history.log_probabilities
        history_df['epoch_start_times'] = history.epoch_start_times
        history_df['epoch_end_times'] = history.epoch_end_times
        history_df['epoch_durations'] = history.epoch_durations
        history_df['epochs'] = history.epochs
        history_df['learning_rates'] = history.learning_rates
        history_df['n_seen_batches'] = history.n_seen_batches
        history_df['test_log_probability'] = history.test_log_probability
        history_df['batch_log_probability'] = history.batch_log_probability
        history_df.to_csv(f'{MODEL_FOLDER}/history.csv', index=False)
    return

def load_model(TARGET_PATH_TO_RESULTS, model_name):
    MODEL_FOLDER = f"{TARGET_PATH_TO_RESULTS}{model_name}/"
    with open(f"{MODEL_FOLDER}/model.json") as f:
        json_object = json.load(f)
        model = HiddenMarkovModel.from_json(json_object)
    return model

## Paths finding
def get_all_paths_with_probs(model, edge_value="log_probability", cutoff=None, precision=4, min_path_length=9,
                             max_path_length=9):

    new_graph, name_to_level = convert_graph_to_good_format(model, "", False, precision=precision,
                                                            edge_value=edge_value)
    # +start +end
    min_path_length = min_path_length + 2
    max_path_length = max_path_length + 2
    cutoff = cutoff + 2
    print("Path finding (slow operation)...")
    path_generator = all_simple_paths(new_graph, model.start.name, model.end.name, cutoff=cutoff)
    print(f"\tCondition: Took only paths with transitions above {10 ** -precision}...")
    print(f"\tCondition: Cutoff for path finding was {cutoff - 2}...")
    print(
        f"\tCondition: Took only paths of acceptable lengths: min {min_path_length - 2} , max {max_path_length - 2}...")

    paths = [path for path in tqdm(path_generator) if
             max_path_length >= len(path) >= min_path_length]  # Might be very slow
    print(f"Paths with correct length: {len(paths)}")
    result_paths = list()
    result_probs = list()
    result_normalized_probs = list()
    result_lens = list()
    edge_to_prob = dict()
    i = 0
    print(f"Calculating probs for selected paths...")
    for path in tqdm(paths):
        prob_for_path = 0 if edge_value == "log_probability" else 1
        path_edges = path.copy()
        node_from = path_edges.pop(0)
        i += 1
        while path_edges:

            node_to = path_edges.pop(0)
            edge = new_graph.get_edge_data(node_from, node_to)
            edge_to_prob[(node_from, node_to)] = edge['label']
            if edge_value == "log_probability":
                prob_for_path = prob_for_path + edge['label']

            else:
                prob_for_path = prob_for_path * edge['label']
            node_from = node_to

        result_paths.append(">".join(path))
        result_lens.append(len(path) - 2)
        result_probs.append(prob_for_path)
        # normalize
        result_normalized_probs.append(prob_for_path / len(path))
    df = pd.DataFrame(
        {"path": result_paths, "prob": result_probs, "normalized_prob": result_normalized_probs, "len": result_lens})
    df = df.sort_values('normalized_prob', ascending=False)
    df = df.reset_index(drop=True)

    return df, edge_to_prob

def get_state_to_name_dict(model):
    d = {state.name: state for state in model.states}
    return d

def get_prob_for_path_and_sequence(model,edge_to_prob, state_path, sequence, edge_value="log_probability"):
    prob_for_path = 0 if edge_value == "log_probability" else 1
    path_nodes = state_path.copy()
    node_from = path_nodes.pop(0)
    emissions = list(sequence)
    while path_nodes:
        node_to = path_nodes.pop(0)
        tr_prob = edge_to_prob[(node_from.name, node_to.name)]
        if node_from.name != model.start.name:
            emission = emissions.pop(0)
            e_prob = node_from.distribution.parameters[0][emission]
            e_prob = np.log(e_prob) if edge_value == "log_probability" else e_prob
            if edge_value == "log_probability":
                prob_for_path = prob_for_path + e_prob + tr_prob
            else:
                prob_for_path = prob_for_path * e_prob * tr_prob

        else:
            if edge_value == "log_probability":
                prob_for_path =  prob_for_path + tr_prob
            else:
                prob_for_path =  prob_for_path * tr_prob
        node_from = node_to
    return prob_for_path

def get_emission_prob_for_path_and_sequence(path, sequence, name_to_state, start_state_name,
                                            edge_value="log_probability"):
    prob_for_path = 0 if edge_value == "log_probability" else 1
    path_nodes = path.copy()
    node_from = path_nodes.pop(0)
    emissions = list(sequence)
    while path_nodes:
        node_to = path_nodes.pop(0)
        # tr_prob = edge_to_prob[(node_from, node_to)]
        if node_from != start_state_name:
            emission = emissions.pop(0)
            e_prob = name_to_state[node_from].distribution.parameters[0][emission]
            e_prob = np.log(e_prob) if edge_value == "log_probability" else e_prob
            if edge_value == "log_probability":
                prob_for_path = prob_for_path + e_prob
            else:
                prob_for_path = prob_for_path * e_prob

        else:
            if edge_value == "log_probability":
                prob_for_path = prob_for_path
            else:
                prob_for_path = prob_for_path
        node_from = node_to
    return prob_for_path

def get_peptide_to_paths_df(model, peptides, edge_value="log_probability", top_paths=None, precision=4):
    print(f"Model {model.name}")
    name_to_state = get_state_to_name_dict(model)
    print(f"\tFind all paths")
    cutoff = max(map(len, peptides))
    min_length = min(map(len, peptides))
    max_length = max(map(len, peptides))
    paths_df, edge_to_prob = get_all_paths_with_probs(model, edge_value=edge_value, cutoff=cutoff, precision=precision,
                                                      min_path_length=min_length, max_path_length=max_length)

    paths_df = paths_df.sort_values('normalized_prob', ascending=False)
    paths_df = paths_df.reset_index(drop=True)
    if top_paths:
        paths_df = paths_df[:top_paths]
        print(f"Took only {top_paths} top paths")

    path_to_index = {str(item.path): item.Index for item in paths_df.itertuples()}
    result_repr_matrix = np.zeros(shape=(len(peptides), len(path_to_index)), dtype=np.float)

    # make emission matrix
    name_to_index = {name: i for i, name in enumerate(name_to_state.keys())}
    acid_to_index = {acid: i for i, acid in enumerate(sorted(model.keymap[0].keys()))}
    emission_matrix = np.zeros(shape=(len(name_to_index), len(acid_to_index)))

    for name, state in name_to_state.items():
        if name != model.start.name and name != model.end.name:
            for acid, prob in state.distribution.parameters[0].items():
                emission_matrix[name_to_index[name], acid_to_index[acid]] = np.log(prob)

    print(f"\tCalculate probs for peptides")
    for _, path, tr_prob, tr_normalized_prob, path_length in tqdm(paths_df.itertuples(), total=paths_df.shape[0]):
        curr_path = path[1:]
        curr_path_indexes = [name_to_index[name] for name in curr_path]
        path_index = path_to_index[str(path)]
        for i, peptide in enumerate(peptides):
            if edge_value == "log_probability":
                emission_prob = sum([emission_matrix[state_index, acid_to_index[emission]] for state_index, emission in
                                     zip(curr_path_indexes, peptide)])
                prob_for_path = tr_prob + emission_prob
            else:
                emission_prob = get_emission_prob_for_path_and_sequence(path, peptide, name_to_state, model.start.name,
                                                                        edge_value=edge_value)
                prob_for_path = tr_prob * emission_prob
            result_repr_matrix[i, path_index] = prob_for_path
    df = pd.DataFrame(result_repr_matrix, index=peptides)
    if edge_value == "log_probability":
        df.replace([-np.inf], -55, inplace=True)
    return df, paths_df, emission_matrix, name_to_index

def get_peptide_to_paths_df_external_paths(model, peptides, paths_df, edge_value="log_probability", top_paths=None):
    print(f"Model {model.name}")
    name_to_state = get_state_to_name_dict(model)

    paths_df = paths_df.sort_values('norm_viterbi_prob', ascending=False)
    paths_df = paths_df.reset_index(drop=True)
    if top_paths:
        paths_df = paths_df[:top_paths]  # assumed to be sorted
        print(f"Took only {top_paths} top paths")

    path_to_index = {str(item.path): item.Index for item in paths_df.itertuples()}
    result_repr_matrix = np.zeros(shape=(len(peptides), len(path_to_index)), dtype=np.float)

    # make emission matrix
    name_to_index = {name: i for i, name in enumerate(name_to_state.keys())}
    acid_to_index = {acid: i for i, acid in enumerate(sorted(model.keymap[0].keys()))}
    emission_matrix = np.zeros(shape=(len(name_to_index), len(acid_to_index)))

    for name, state in name_to_state.items():
        if name != model.start.name and name != model.end.name:
            for acid, prob in state.distribution.parameters[0].items():
                emission_matrix[name_to_index[name], acid_to_index[acid]] = np.log(prob)

    print(f"\tCalculate probs for peptides")

    for _, path, tr_prob, path_length, mean_viterbi, mean_bw_probs, path_pep_cnt in tqdm(paths_df.itertuples(),
                                                                                         total=paths_df.shape[0]):
        curr_path = path.split(">")
        curr_path = curr_path[1:]
        curr_path_indexes = [name_to_index[name] for name in curr_path]
        path_index = path_to_index[str(path)]
        for i, peptide in enumerate(peptides):
            if edge_value == "log_probability":

                if len(peptide) == len(curr_path_indexes) - 1:  # -1 because we have end state at the end
                    emission_prob = sum(
                        [emission_matrix[state_index, acid_to_index[emission]] for state_index, emission in
                         zip(curr_path_indexes, peptide)])
                    prob_for_path = tr_prob + emission_prob
                else:
                    prob_for_path = -np.nan
            else:
                if len(peptide) == len(curr_path_indexes):
                    emission_prob = get_emission_prob_for_path_and_sequence(path, peptide, name_to_state,
                                                                            model.start.name, edge_value=edge_value)
                    prob_for_path = tr_prob * emission_prob

                else:
                    prob_for_path = 0

            result_repr_matrix[i, path_index] = prob_for_path

    df = pd.DataFrame(result_repr_matrix, index=peptides)
    return df, paths_df, emission_matrix, name_to_index

def get_all_viterbi_paths_for_model_and_sequences(model, peptides, edge_value="log_probability", precision=4):
    print(f"Model {model.name}")
    model_graph = model.graph

    print("Get viterbi paths for each peptide...")

    result_probs = list()
    result_normalized_probs = list()
    result_lens = list()
    result_paths = list()

    path_code_to_path = dict()
    path_code_to_cnt = defaultdict(int)
    path_code_to_mean_viterbi_prob = defaultdict(float)

    for i, peptide in tqdm(enumerate(peptides), total=len(peptides)):
        prob_for_path, path = model.viterbi(peptide)
        path_state_indexes = "|".join([str(path_tuple[0]) for path_tuple in path])  # path as string
        path_code_to_path[path_state_indexes] = [path_tuple[1] for path_tuple in path]
        n = path_code_to_cnt[path_state_indexes]
        mean_viterbi = path_code_to_mean_viterbi_prob[path_state_indexes]
        mean_viterbi = (mean_viterbi * n + prob_for_path) / (n + 1)
        path_code_to_mean_viterbi_prob[path_state_indexes] = mean_viterbi
        path_code_to_cnt[path_state_indexes] += 1

    print("Select unique paths...")
    unique_paths = list(path_code_to_path.values())
    mean_viterbies = list(path_code_to_mean_viterbi_prob.values())
    cnts = list(path_code_to_cnt.values())
    for path in tqdm(unique_paths):
        prob_for_path = 0 if edge_value == "log_probability" else 1
        path_edges = path.copy()
        node_from = path_edges.pop(0)
        while path_edges:
            node_to = path_edges.pop(0)
            data = model_graph.get_edge_data(node_from, node_to, default=None)
            log_probability = data['probability']
            # print(node_from, node_to,  np.exp(edge['label']), "total_prob", prob_for_path)
            if edge_value == "log_probability":
                prob_for_path = prob_for_path + log_probability
            else:
                prob_for_path = prob_for_path * np.exp(log_probability)
            # print(f"edge {edge}")
            node_from = node_to
        result_paths.append(">".join([node.name for node in path]))
        result_lens.append(len(path) - 2)
        result_probs.append(prob_for_path)
        # normalize
        result_normalized_probs.append(prob_for_path / len(path))

    paths_df = pd.DataFrame(
        {"path": result_paths, "prob": result_probs, "normalized_prob": result_normalized_probs, "len": result_lens,
         "mean_viterbi": mean_viterbies, "peptide_cnts": cnts})
    paths_df = paths_df.sort_values('normalized_prob', ascending=False)
    paths_df = paths_df.reset_index(drop=True)
    return paths_df

def generate_radom_paths_for_model_mark_core(model, n, per_model_important_states, edge_value="log_probability"):
    name_to_state = get_state_to_name_dict(model)
    name_to_index = {name: i for i, name in enumerate(name_to_state.keys())}
    acid_to_index = {acid: i for i, acid in enumerate(sorted(model.keymap[0].keys()))}
    emission_matrix = np.zeros(shape=(len(name_to_index), len(acid_to_index)))
    for name, state in name_to_state.items():
        if name != model.start.name and name != model.end.name:
            for acid, prob in state.distribution.parameters[0].items():
                emission_matrix[name_to_index[name], acid_to_index[acid]] = np.log(prob)
    print(f"Model {model.name}")
    model_graph = model.graph
    result_tr_probs = list()
    result_normalized_probs = list()
    result_lens = list()
    result_paths = list()
    result_simplified_paths = list()
    result_viterbi_probs = list()
    result_total_probs = list()
    result_bw_probs = list()
    result_start_core = list()
    result_end_core = list()
    peptides = list()
    print(f"Generate {n} peptides and paths")
    for i in range(n):
        peptide_arr, path = model.sample(path=True)
        peptide = "".join(peptide_arr)
        peptides.append(peptide)
        viterbi_prob, viterbi_path = model.viterbi(peptide)
        bw_prob = model.log_probability(peptide)
        tr_prob = 0 if edge_value == "log_probability" else 1
        path_nodes = [node for node in path]
        path_length = len(path_nodes)
        result_paths.append(">".join([node.name for node in path_nodes]))
        count_al = [i for i, node in enumerate(path_nodes[1:]) if node.name in per_model_important_states[model.name]]
        start_core = count_al[0] if len(count_al) > 0 else np.nan
        end_core = count_al[-1] if len(count_al) > 0 else np.nan
        curr_path = path_nodes[1:]
        curr_path_indexes = [name_to_index[node.name] for node in curr_path]

        probs_dif = 1 - np.round((viterbi_prob - bw_prob) / path_length, 2)

        if np.isnan(start_core) or np.isnan(end_core):
            simplified_path = np.array([probs_dif if node.name in per_model_important_states[model.name] else 0 for node
                                        in path_nodes[1:-1]]).astype(np.float64)
        else:
            simplified_path = np.array([probs_dif if node.name in per_model_important_states[model.name] else 0 for node
                                        in path_nodes[1:-1]]).astype(np.float64)
            assert start_core is not np.nan
            assert end_core is not np.nan
            # simplified_path[start_core:end_core+1] = probs_dif
        result_simplified_paths.append(simplified_path)
        node_from = path_nodes.pop(0)
        while path_nodes:
            node_to = path_nodes.pop(0)
            data = model_graph.get_edge_data(node_from, node_to, default=None)
            log_probability = data['probability']
            # print(node_from, node_to,  np.exp(edge['label']), "total_prob", prob_for_path)
            if edge_value == "log_probability":
                tr_prob = tr_prob + log_probability
            else:
                tr_prob = tr_prob * np.exp(log_probability)
            # print(f"edge {edge}")
            node_from = node_to

        if edge_value == "log_probability":
            emission_prob = sum(
                [emission_matrix[state_index, acid_to_index[emission]] for state_index, emission in
                 zip(curr_path_indexes, peptide)])
            prob_for_path = tr_prob + emission_prob
        else:
            emission_prob = get_emission_prob_for_path_and_sequence(path, peptide, name_to_state,
                                                                    model.start.name, edge_value=edge_value)
            prob_for_path = tr_prob * emission_prob

        result_lens.append(path_length - 2)
        result_tr_probs.append(tr_prob)
        result_normalized_probs.append(tr_prob / path_length)
        result_viterbi_probs.append(viterbi_prob)
        result_bw_probs.append(bw_prob)
        result_total_probs.append(prob_for_path)
        result_start_core.append(start_core)
        result_end_core.append(end_core)
    paths_df = pd.DataFrame(
        {"path": result_paths,
         "simplified_path": result_simplified_paths,
         "core_start": result_start_core, "core_end": result_end_core,
         "tr_prob": result_tr_probs, "normalized_prob": result_normalized_probs, "len": result_lens,
         "viterbi_probs": result_viterbi_probs,
         "bw_probs": result_bw_probs,
         "total_probs": result_total_probs}, index=peptides)
    return paths_df


def get_all_viterbi_paths_for_model_and_sequences_mark_core(model, peptides, per_model_important_states,
                                                            per_model_skip_states=None,
                                                            edge_value="log_probability", precision=4,
                                                            unique_paths=True, generate_peptides=False,
                                                            generate_count=50000, fill_whole_core=False,
                                                            cumulative_metric=False,
                                                            ):

    print(f"Model {model.name}")
    model_graph = model.graph
    if per_model_important_states == None:
        mark_core = False
        print("Warning: no important states are not defined, simplified path column will be absent")
    else:
        mark_core = True
    if mark_core:
        assert unique_paths is False, "If you want core estimates then you want each individual peptide to be counted, not unique paths"

    if generate_peptides:
        assert peptides is None, "If you want to generate peptides -- you don't need peptides as argument"
        peptides = list()

    print("Get viterbi paths for each peptide...")

    result_probs = list()
    result_lens = list()
    result_paths = list()
    result_simplified_paths = list()
    result_peptides = list()
    result_viterbi_probs = list()
    result_bw_probs = list()
    result_start_core = list()
    result_end_core = list()
    result_cnts = list()

    path_code_to_path = dict()
    path_code_to_cnt = defaultdict(int)
    path_code_to_total_viterbi_prob = defaultdict(float)
    path_code_to_total_bw_prob = defaultdict(float)
    path_code_to_peptide = defaultdict(str)

    peptide_count = generate_count if generate_peptides else len(peptides)

    for i in tqdm(range(peptide_count)):
        if not generate_peptides:  # peptides are given, calculate all from peptide
            peptide = peptides[i]
            viterbi_prob, path = model.viterbi(peptide)
            path = [path_tuple[1] for path_tuple in path]
            bw_prob = model.log_probability(peptide)
        else:
            peptide_arr, path = model.sample(path=True)
            peptide = "".join(peptide_arr)
            peptides.append(peptide)
            viterbi_prob, viterbi_path = model.viterbi(peptide)
            bw_prob = model.log_probability(peptide)

        if unique_paths:
            path_index = "|".join([state.name for state in path])  # path as string
        else:
            path_index = peptide
            path_code_to_peptide[path_index] = peptide
        path_code_to_path[path_index] = path
        path_code_to_total_viterbi_prob[path_index] += viterbi_prob
        path_code_to_total_bw_prob[path_index] += bw_prob
        path_code_to_cnt[path_index] += 1
    print("Subselect unique paths (if unique path argument passed)..")
    print(f"\tUnique paths: {len(path_code_to_path)} ")
    # unique_paths = list(path_code_to_path.values())
    # mean_viterbies = list(viterbi_prob/ cnt for viterbi_prob, cnt in zip(path_code_to_total_viterbi_prob.values(), path_code_to_cnt.values()))
    # mean_bws = list(bw_prob/ cnt for bw_prob, cnt in zip(path_code_to_total_bw_prob.values(), path_code_to_cnt.values()))

    for path_code, path_nodes in tqdm(path_code_to_path.items()):
        if mark_core:
            count_al = [i for i, node in enumerate(path_nodes[1:]) if
                        node.name in per_model_important_states[model.name]]
            start_core = count_al[0] if len(count_al) > 0 else np.nan
            end_core = count_al[-1] if len(count_al) > 0 else np.nan
        peptide_cnt = path_code_to_cnt[path_code]
        viterbi_prob = path_code_to_total_viterbi_prob[path_code] / peptide_cnt
        bw_prob = path_code_to_total_bw_prob[path_code] / peptide_cnt
        path_length = len(path_nodes)

        tr_prob_for_path = 0 if edge_value == "log_probability" else 1
        result_paths.append(">".join([node.name for node in path_nodes]))
        path_nodes_copy = path_nodes.copy()

        node_from = path_nodes_copy.pop(0)
        i = 0
        cumulative_probs = np.zeros(len(path_nodes) - 2)
        cummulative_prob = 0

        while path_nodes_copy:
            node_to = path_nodes_copy.pop(0)
            data = model_graph.get_edge_data(node_from, node_to, default=None)
            log_probability = data['probability']
            # print(node_from, node_to,  np.exp(edge['label']), "total_prob", prob_for_path)
            if edge_value == "log_probability":
                tr_prob_for_path = tr_prob_for_path + log_probability
                if mark_core:
                    peptide = path_code_to_peptide[path_code]
                    if node_to.distribution:
                        cummulative_prob = cummulative_prob + log_probability + np.log(
                            node_to.distribution.parameters[0][peptide[i]])
                        cumulative_probs[i] = cummulative_prob
                        i += 1
            else:
                tr_prob_for_path = tr_prob_for_path * np.exp(log_probability)
            # print(f"edge {edge}")
            node_from = node_to

        # normalize by peptide length
        peptide_len = path_length - 2
        viterbi_prob = viterbi_prob / peptide_len
        bw_prob = bw_prob / peptide_len
        tr_prob_for_path = tr_prob_for_path / peptide_len

        # probs_dif = viterbi_prob

        if mark_core:
            probs_dif = 1
            # probs_dif = 1 - np.round(viterbi_prob-bw_prob,2)
            cur_path = path_nodes[1:-1]
            if np.isnan(start_core) or np.isnan(end_core):
                simplified_path = np.array([probs_dif if node.name in per_model_important_states[model.name] else 0 for
                                            node in cur_path]).astype(np.float64)

            else:
                simplified_path = np.array([probs_dif if node.name in per_model_important_states[model.name] else 0 for
                                            node in cur_path]).astype(np.float64)
                assert start_core is not np.nan
                assert end_core is not np.nan
                if fill_whole_core:

                    simplified_path[start_core:end_core] = [
                        probs_dif if cur_path[i].name not in per_model_skip_states[model.name] else 0 for i in
                        range(start_core, end_core)]

                    if cumulative_metric:
                        cumulative_probs = cumulative_probs / peptide_len
                        # simplified_path[start_core:end_core] = cumulative_probs[start_core:end_core]
                        simplified_path = cumulative_probs

            result_simplified_paths.append(simplified_path)
            result_start_core.append(start_core)
            result_end_core.append(end_core)
        result_lens.append(peptide_len)
        result_probs.append(tr_prob_for_path)
        result_viterbi_probs.append(viterbi_prob)
        result_bw_probs.append(bw_prob)

        result_cnts.append(path_code_to_cnt[path_code])
        if not unique_paths:
            result_peptides.append(path_code_to_peptide[path_code])
    if unique_paths:
        paths_df = pd.DataFrame(
            {"path": result_paths,
             "norm_tr_prob": result_probs, "len": result_lens,
             "norm_viterbi_prob": result_viterbi_probs,
             "norm_bw_prob": result_bw_probs,
             "peptide_cnt": result_cnts})
    else:
        paths_df = pd.DataFrame(
            {"path": result_paths,
             "norm_tr_prob": result_probs, "len": result_lens,
             "norm_viterbi_prob": result_viterbi_probs,
             "norm_bw_prob": result_bw_probs,
             "peptide_cnt": result_cnts}, index=result_peptides)
    if mark_core:
        paths_df['simplified_path'] = result_simplified_paths
        paths_df['core_start'] = result_start_core
        paths_df['core_end'] = result_end_core

    return paths_df



def get_all_scores_and_sort_by_best(per_allele_per_split_per_length_data, per_name_models):
    per_allele_list_of_dfs = defaultdict(list)
    ALLELES = per_allele_per_split_per_length_data.keys()

    for allele_name in ALLELES:
        # TOTAL_SPLITS = len(per_allele_per_split_per_length_data[allele_name].keys())
        TOTAL_SPLITS = 1
        print("currently take only 1 split")
        for current_split in range(TOTAL_SPLITS):
            TARGET_LENGTHS = list(per_allele_per_split_per_length_data[allele_name][current_split].keys())
            for current_length in TARGET_LENGTHS:
                peptides = per_allele_per_split_per_length_data[allele_name][current_split][current_length]
                df = pd.DataFrame({'peptide': peptides, 'split': current_split, 'length': current_length})
                per_allele_list_of_dfs[allele_name].append(df)

    per_allele_data_df = dict()
    for allele_name in ALLELES:
        per_allele_data_df[allele_name] = pd.concat(per_allele_list_of_dfs[allele_name])

    for allele_name in ALLELES:
        df = per_allele_data_df[allele_name]
        dict_of_result_predictions = dict()
        for i, model in enumerate(per_name_models.values()):
            dict_of_result_predictions[f'model_{i}'] = [model.log_probability(peptide) / len(peptide) for peptide in
                                                        df.peptide]
        predictions_df = pd.DataFrame(dict_of_result_predictions, index=df.index)
        df = pd.concat([df, predictions_df], axis=1)
        df = df.sort_values(by='model_0', ascending=False)
        per_allele_data_df[allele_name] = df
    return per_allele_data_df

def prepare_multiple_models(experiment_params: ExperimentParams, custom_models_id: str = "root"):
    model_training_params: ModelTrainingParams = experiment_params.model_training_params
    data_scenario_params: DataScenarioParams = experiment_params.data_scenario_params
    original_num_runs = model_training_params.num_runs #20
    alleles_to_use = model_training_params.alleles_to_use
    total_runs = original_num_runs * model_training_params.decrease_anchor_aas_steps
    per_allele_per_run_per_split_prepared_models = dict()
    for allele_name in alleles_to_use:
        target_allele_name = allele_name.replace('-', '_').replace('*', '_').replace(':', '_').replace('/', '_')
        per_allele_per_run_per_split_prepared_models[allele_name] = dict()
        for run_num in range(total_runs):
            run_index = f"{custom_models_id}[{run_num:04d}]"
            per_allele_per_run_per_split_prepared_models[allele_name][run_index] = dict()
            for split_num in range(data_scenario_params.splits_to_read):
                acids_to_substract = run_num // original_num_runs
                current_params = deepcopy(model_training_params)
                current_params.anchor_top_aas = current_params.anchor_top_aas - acids_to_substract
                prepared_model = build_model_based_on_params(current_params)                                              #f()
                prepared_model.name = f'{run_index}_run-{current_params.get_model_common_names()}_model-{target_allele_name}-{split_num}'
                per_allele_per_run_per_split_prepared_models[allele_name][run_index][split_num] = prepared_model
    return per_allele_per_run_per_split_prepared_models

def check_model_alignment(current_model, allele_data_df, allele_name):
    model_states = [state for state in current_model.states if
                    (state.distribution is not None) and (
                            type(state.distribution) is not DiscreteDistributionCycle)]
    current_core_length = len(model_states)
    first_anchor = model_states[-2].name
    last_anchor = model_states[-1].name
    important_states = {current_model.name: [first_anchor, last_anchor]}
    skip_states = {current_model.name: []}
    viterbi_paths = get_all_viterbi_paths_for_model_and_sequences_mark_core(current_model,
                                                                            allele_data_df.peptide.values,
                                                                            important_states,
                                                                            unique_paths=False,
                                                                            fill_whole_core=True,
                                                                            cumulative_metric=False,
                                                                            per_model_skip_states=skip_states)
    # define  2K - 2  set of distributions
    index_values = np.arange(-current_core_length + 1, current_core_length + current_core_length - 1)
    max_index = max(index_values)
    amino_acids_list = list('ACDEFGHIKLMNPQRSTVWY')
    counts_matrix = np.zeros((len(index_values), len(amino_acids_list)), dtype=int)
    count_df = pd.DataFrame(counts_matrix, index=index_values, columns=amino_acids_list)
    frame_size = len(index_values)
    left_part = int((frame_size - current_core_length) / 2)
    right_part = left_part
    row_names = count_df.index.values
    col_names = count_df.columns
    # create multiindex series
    index = pd.MultiIndex.from_product([row_names, col_names])
    df = pd.Series(1, index=index)

    # Convert to 2d dataframe
    for peptide, row in viterbi_paths.iterrows():
        core_start = row.core_start
        core_end = row.core_end
        write_start = max(-core_start, -left_part)
        write_end = min(len(peptide) - core_start, max_index + 1)
        row_values = np.arange(write_start, write_end)

        peptide_subset_start = max(0, core_start - left_part)
        peptide_subset_end = min(len(peptide), core_start + max_index) + 1
        column_values = list(peptide[peptide_subset_start:peptide_subset_end])
        idx = pd.MultiIndex.from_arrays([row_values, column_values])
        df.loc[idx] += 1

    count_df = df.unstack(level=-1)
    info_matrix = logomaker.transform_matrix(count_df,
                                             from_type='counts', to_type='information',
                                             background=get_frequencies(count_df.shape[0]).set_index(count_df.index))
    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(5, 6))
    ww_logo = logomaker.Logo(info_matrix, font_name='Arial', flip_below=True,
                             ax=axes)
    ww_logo.ax.set_title(f"{allele_name}, peptides num used: {len(allele_data_df)}")
    return info_matrix, viterbi_paths, fig

def check_models_integrity_and_shift_params(per_split_models, allele_data_df, experiment_params):
    per_split_models_to_update = dict()

    save_dir = str(Path(experiment_params.experiment_result_data_path) / "models_debug")

    for split_num in per_split_models.keys():
        current_model = per_split_models[split_num]

        save_path = os.path.join(save_dir, f"current_model_split_{split_num}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(current_model, f)

        print(f"saved: {save_path}")

        model_states = [state for state in current_model.states if
                        (state.distribution is not None) and (
                                type(state.distribution) is not DiscreteDistributionCycle)]
        current_core_length = len(model_states)

        model_distributions = [state.distribution.parameters[0] for state in current_model.states if
                               (state.distribution is not None) and (
                                       type(state.distribution) is not DiscreteDistributionCycle)]

        first_anchor = model_states[-2].name
        last_anchor = model_states[-1].name
        important_states = {current_model.name: [first_anchor, last_anchor]}
        skip_states = {current_model.name: []}
        viterbi_paths = get_all_viterbi_paths_for_model_and_sequences_mark_core(
            current_model,
            allele_data_df.peptide.values,
            important_states,
            unique_paths=False,
            fill_whole_core=True,
            cumulative_metric=False,
            per_model_skip_states=skip_states
        )
        # define  2K - 2  set of distributions
        index_values = np.arange(-current_core_length + 1, current_core_length + current_core_length - 1)
        max_index = max(index_values)
        amino_acids_list = list('ACDEFGHIKLMNPQRSTVWY')
        counts_matrix = np.zeros((len(index_values), len(amino_acids_list)), dtype=int)
        count_df = pd.DataFrame(counts_matrix, index=index_values, columns=amino_acids_list)
        frame_size = len(index_values)
        left_part = int((frame_size - current_core_length) / 2)
        right_part = left_part
        row_names = count_df.index.values
        col_names = count_df.columns
        # create multiindex series
        index = pd.MultiIndex.from_product([row_names, col_names])
        df = pd.Series(1, index=index)

        # Convert to 2d dataframe
        for peptide, core_start, core_end in zip(viterbi_paths.index.values, viterbi_paths.core_start.values, viterbi_paths.core_end.values):
            write_start = max(-core_start, -left_part)
            write_end = min(len(peptide) - core_start, max_index + 1)
            row_values = np.arange(write_start, write_end)

            peptide_subset_start = max(0, core_start - left_part)
            peptide_subset_end = min(len(peptide), core_start + max_index) + 1
            column_values = list(peptide[peptide_subset_start:peptide_subset_end])

            idx = pd.MultiIndex.from_arrays([row_values, column_values])
            df.loc[idx] += 1

        count_df = df.unstack(level=-1)
        print("transform to info matrix")
        info_matrix = logomaker.transform_matrix(count_df,
                                                 from_type='counts', to_type='information',
                                                 background=get_frequencies(
                                                     count_df.shape[0]).set_index(count_df.index))
        # now info matrix contains estimation of the information for all possible shifts inside model
        info_vector = info_matrix.sum(axis=1).values
        dummy_threshold = 0.5
        new_core_count_df = None
        per_pos_sum = np.zeros(len(info_vector) - current_core_length + 1)  # 1....16xxxx25
        print(f"Will calculate sliding window sum for f{len(per_pos_sum)} positions with original vector of size {len(info_vector)}")
        for pos in range(len(per_pos_sum)):
            if pos == 0:
                print(f"calculate shifts for fragments of length {len(info_vector[pos:pos + current_core_length])}")
            per_pos_sum[pos] = sum(info_vector[pos:pos + current_core_length])
        max_core_start_pos = np.argmax(per_pos_sum)

        if max_core_start_pos < left_part:
            print("Seems to be shift to the left, since there is an alignment hit outside the model frame")
            print("Shifting model frame to the best pick leftwise")
            # better_pos_numerical_index = info_vector[: left_part].argmax()
            better_pos_numerical_index = max_core_start_pos
            # new_core_count_df = count_df.iloc[left_part - better_pos_numerical_index - 1:left_part + current_core_length + better_pos_numerical_index - 1, :].copy()
            new_core_count_df = count_df.iloc[
                                better_pos_numerical_index:current_core_length + better_pos_numerical_index, :].copy()
            print(f"shape of new matrix f{new_core_count_df.shape}")
        if max_core_start_pos > left_part:
            # if sum(info_vector[-right_part:] > dummy_threshold) > 0:
            print("Seems to be shift to the right, since there is an alignment hit outside the model frame")
            print("Shifting model frame to the best pick rightwise")
            better_pos_numerical_index = max_core_start_pos
            new_core_count_df = count_df.iloc[
                                better_pos_numerical_index:current_core_length + better_pos_numerical_index, :].copy()
            print(f"shape of new matrix f{new_core_count_df.shape}")

            # new_core_count_df = count_df.iloc[left_part + better_pos_numerical_index + 1:left_part + current_core_length + better_pos_numerical_index + 1, :].copy()

        # reinitialize model with new distributions
        if new_core_count_df is not None:
            new_core_count_df = new_core_count_df.reset_index(drop=True)
            new_prob_matrix = logomaker.transform_matrix(new_core_count_df,
                                                         from_type='counts', to_type='probability',
                                                         pseudocount=1,
                                                         background=get_frequencies(new_core_count_df.shape[
                                                                                        0]).set_index(new_core_count_df.index), )
            new_prob_matrix = new_prob_matrix.reindex(
                [i for i in range(1, current_core_length - 1)] + [0, current_core_length - 1]).reset_index(drop=True)

            # prepare new model with this matrix
            model_training_params: ModelTrainingParams = experiment_params.model_training_params
            data_scenario_params: DataScenarioParams = experiment_params.data_scenario_params
            current_params = deepcopy(model_training_params)

            prepared_model = build_model_based_on_params(current_params, prepared_emission_matrix=new_prob_matrix)
            prepared_model.name = current_model.name
            per_split_models_to_update[split_num] = prepared_model
            print("finished model construction")

    return per_split_models_to_update

def train_multiple_models(prepared_models,
                          train_data,
                          test_data,
                          train_data_weights,
                          experiment_params: ExperimentParams,
                          subfolder_to_safe_result: str):
    model_training_params: ModelTrainingParams = experiment_params.model_training_params
    path_to_save_models = f"{experiment_params.experiment_result_data_path}/{subfolder_to_safe_result}/"
    per_name_models = dict()
    result_histories = dict()
    result_models = dict()

    ALLELES = list(train_data.keys())
    per_allele_data_df = join_dicts(train_data)

    for allele_name in ALLELES:
        print(f"Allele {allele_name}:", end=" ")
        result_histories[allele_name] = dict()
        result_models[allele_name] = dict()
        for run_index in prepared_models[allele_name].keys():
            print(f"{run_index}", end=" ")
            per_kfold_per_length_data = train_data[allele_name]
            per_kfold_per_length_weights = train_data_weights[allele_name]

            models, histories = train_model_prepared(
                per_kfold_per_length_data=per_kfold_per_length_data,
                model_training_params=model_training_params,
                per_kfold_per_length_weights_for_data=per_kfold_per_length_weights,
                prepared_models=prepared_models[allele_name][run_index],
                per_kfold_per_length_test_data=test_data[allele_name],

            )
            # check models integrity
            if model_training_params.check_model_integrity:
                models_to_be_updated = check_models_integrity_and_shift_params(models, per_allele_data_df[
                    allele_name], experiment_params=experiment_params)
                if (models_to_be_updated):
                    print("some models had shifts, we will try one more iteration of training")
                    extra_models, extra_histories = train_model_prepared(
                        per_kfold_per_length_data=per_kfold_per_length_data,
                        model_training_params=model_training_params,
                        per_kfold_per_length_weights_for_data=per_kfold_per_length_weights,
                        prepared_models=models_to_be_updated,
                        per_kfold_per_length_test_data=test_data[allele_name],
                    )
                    models.update(extra_models)
                    histories.update(extra_histories)

            result_models[allele_name][run_index] = models
            result_histories[allele_name][run_index] = histories
            # per_allele_per_run_per_split_models[allele_name][run_num] = models
            for split_num, model in models.items():
                per_name_models[model.name] = model
            # save models
            for split_num, model in models.items():
                history = histories[split_num]
                save_model(path_to_save_models, model, history)
        print(" ")
    return result_models, result_histories

# Функция для каждого allele и split ранжирует модели по финальному log_probability,
# переименовывает их согласно новому рангу, сохраняет на диск и возвращает плоские словари моделей/историй
# плюс таблицы исходной сортировки.

def reorder_models_by_score_and_flatten_to_by_name_list(histories, models, experiment_params: ExperimentParams,
                                                        subfolder_to_safe_result: str):
    path_to_save_models = f"{experiment_params.experiment_result_data_path}/{subfolder_to_safe_result}/"
    ALLELES = list(models.keys())
    RUNS = list(models[ALLELES[0]].keys())
    SPLITS = list(models[ALLELES[0]][RUNS[0]])
    # Sort by score
    per_name_new_models = dict()
    per_name_histories = dict()
    per_split_allele_original_sortings = dict()
    for split_index in SPLITS:
        per_split_allele_original_sortings[split_index] = dict()
        for allele_name in ALLELES:
            original_runs = models[allele_name].keys()
            df = pd.DataFrame({"model": original_runs,
                               "log_probability": [histories[allele_name][i][split_index].log_probabilities[-1]
                                                   for i in original_runs],
                               })
            df = df.sort_values(by='log_probability', ascending=False).reset_index(drop=True)
            per_split_allele_original_sortings[split_index][allele_name] = df
            original_indexes = original_runs
            for new_index, model_run_index in zip(sorted(original_indexes), df.model):
                original_model = models[allele_name][model_run_index][split_index]
                original_history = histories[allele_name][model_run_index][split_index]
                original_model_name = original_model.name
                rest_part = original_model_name.split("_", 1)[1]
                # additional_model_id = model_run_index.split("#")[0]
                new_name = f"{new_index}_{rest_part}"
                original_model.name = new_name
                per_name_new_models[new_name] = original_model
                per_name_histories[new_name] = original_history
    per_name_models = per_name_new_models
    for name, model in per_name_models.items():
        history = per_name_histories[name]
        save_model(path_to_save_models, model, history)
    return per_name_models, per_name_histories, per_split_allele_original_sortings


## Hierarchical split enrich stop models training
def get_average_model_for_runs(per_name_models):
    example_model = per_name_models[list(per_name_models.keys())[0]]
    NUM_RUNS = len(per_name_models)
    transition_matrix_cube = np.zeros((NUM_RUNS, len(example_model.states), len(example_model.states)))
    emission_matrix_cube = np.zeros((NUM_RUNS, len(example_model.states) - 2, 20))

    for i, model in enumerate(per_name_models.values()):
        transition_matrix_cube[i, :, :] = model.dense_transition_matrix()
        for j, state in enumerate(model.states):
            if state.distribution is not None:
                params = list(state.distribution.parameters[0].values())
                emission_matrix_cube[i, j, :] = np.array(params)

    mean_emission_table = np.median(emission_matrix_cube, axis=0)
    mean_transition_table = np.median(transition_matrix_cube, axis=0)

    distributions = []
    amino_acids_list = list('ACDEFGHIKLMNPQRSTVWY')
    for j in range(len(example_model.states) - 2):
        d_dict = {key: prob for key, prob in zip(amino_acids_list, mean_emission_table[j, :])}
        curr_distribution = DiscreteDistribution(d_dict)
        distributions.append(curr_distribution)

    model = HiddenMarkovModel.from_matrix(
        name="average model",
        transition_probabilities=mean_transition_table[:-2, :-2],
        starts=mean_transition_table[-2, :],
        distributions=distributions,
        ends=mean_transition_table[:, -1],
        state_names=["s{:03d}".format(i) for i in range(len(example_model.states) - 2)],
        verbose=True)
    model.bake()
    return model

def get_median_scores_for_top_models(per_name_models, allele_data_df, models_fraction=0.7):
    current_runs = list(per_name_models.keys())
    number_of_models_to_estimate = max([int(len(current_runs) * models_fraction), 1])
    scores = np.zeros(shape=(len(allele_data_df), number_of_models_to_estimate))
    print(f"Number of models used for this split is {number_of_models_to_estimate}")
    for model_index in range(number_of_models_to_estimate):
        selected_model = per_name_models[current_runs[model_index]]
        scores[:, model_index] = [selected_model.log_probability(peptide) / len(peptide) for peptide in
                                  allele_data_df.peptide]
    median_scores = np.median(scores, axis=1)
    return median_scores



def reassign_data_to_splitted_models_according_to_strategy(per_name_models1, per_name_models2, per_allele_data_df,
                                                           experiment_params: ExperimentParams):
    model_training_params: ModelTrainingParams = experiment_params.model_training_params
    data_split_strategy = model_training_params.data_split_strategy
    ALLELES = list(per_allele_data_df.keys())
    TOTAL_SPLITS = per_allele_data_df[ALLELES[0]].split.unique().astype(list)
    TARGET_LENGTHS = per_allele_data_df[ALLELES[0]].length.unique().astype(list)

    if data_split_strategy == "best_model":
        best_model_index = 0
        selected_model1 = per_name_models1[list(per_name_models1.keys())[best_model_index]]
        selected_model2 = per_name_models2[list(per_name_models2.keys())[best_model_index]]
        for allele_name in ALLELES:
            df = per_allele_data_df[allele_name]
            df['score1'] = [selected_model1.log_probability(peptide) / len(peptide) for peptide in df.peptide]
            df['score2'] = [selected_model2.log_probability(peptide) / len(peptide) for peptide in df.peptide]
            per_allele_data_df[allele_name] = df
    elif data_split_strategy == "median_score_of_best_models":
        for allele_name in ALLELES:
            df = per_allele_data_df[allele_name]
            df[
                'score1'] = get_median_scores_for_top_models(per_name_models=per_name_models1, allele_data_df=df, models_fraction=model_training_params.enrichment_split_decision_models_fraction)
            df[
                'score2'] = get_median_scores_for_top_models(per_name_models=per_name_models2, allele_data_df=df, models_fraction=model_training_params.enrichment_split_decision_models_fraction)
            per_allele_data_df[allele_name] = df
    elif data_split_strategy == "single_average_model":
        for allele_name in ALLELES:
            df = per_allele_data_df[allele_name]
            selected_model1 = get_average_model_for_runs(per_name_models1)
            selected_model2 = get_average_model_for_runs(per_name_models2)
            df['score1'] = [selected_model1.log_probability(peptide) / len(peptide) for peptide in df.peptide]
            df['score2'] = [selected_model2.log_probability(peptide) / len(peptide) for peptide in df.peptide]
            per_allele_data_df[allele_name] = df
    else:
        raise AttributeError("Unknown type of strategy")

    # now split the data
    per_allele_first_df = dict()
    per_allele_second_df = dict()
    for allele_name in ALLELES:
        df = per_allele_data_df[allele_name]
        df1 = df[df[f'score1'] >= df[f'score2']].copy()
        df2 = df[df[f'score2'] > df[f'score1']].copy()
        print(f"split is {len(df1)}/{len(df2)}")
        per_allele_first_df[allele_name] = df1
        per_allele_second_df[allele_name] = df2

    new_train_data1 = split_to_dicts(per_allele_first_df, ALLELES, TARGET_LENGTHS, TOTAL_SPLITS)
    new_train_data2 = split_to_dicts(per_allele_second_df, ALLELES, TARGET_LENGTHS, TOTAL_SPLITS)
    new_train_data_weights1 = calculate_weights_based_on_length_counts(new_train_data1,
                                                                       experiment_params=experiment_params)
    new_train_data_weights2 = calculate_weights_based_on_length_counts(new_train_data2,
                                                                       experiment_params=experiment_params)
    return new_train_data1, new_train_data2, new_train_data_weights1, new_train_data_weights2

def get_information_content_estimation_for_models(per_name_models, models_fraction):
    # TODO fix this to be per allele for models as well
    # Try to get average distribution matrix
    model_names = list(per_name_models.keys())
    observed_alleles = list()
    model_models_counters = defaultdict(int)  # count models for each allele
    model_allele_to_matrix = dict()
    model_name_to_allele = {
        model_name: f"{model_name.split('-')[-2]}" for model_name in model_names
    }
    number_of_models_to_estimate = max(int(len(model_names) * models_fraction), 1)
    print(f"For information content {number_of_models_to_estimate} will be summarized")
    for model_name in model_names[:number_of_models_to_estimate]:
        current_model = per_name_models[model_name]
        current_allele = model_name_to_allele[model_name]
        model_name_to_allele[model_name] = current_allele
        model_distributions = [state.distribution.parameters[0] for state in current_model.states if
                               (state.distribution is not None) and (
                                       type(state.distribution) is not DiscreteDistributionCycle)]
        first_anchor = model_distributions[-2]
        last_anchor = model_distributions[-1]
        model_distributions = [first_anchor] + model_distributions[:-2] + [last_anchor]
        current_matrix = pd.DataFrame(model_distributions)
        current_matrix.index.name = 'pos'
        model_models_counters[current_allele] += 1
        if current_allele not in model_allele_to_matrix:
            observed_alleles.append(current_allele)
            model_allele_to_matrix[current_allele] = current_matrix
        else:
            model_allele_to_matrix[current_allele] = model_allele_to_matrix[current_allele] + current_matrix
    print(f"check how many models we added at the end: {model_models_counters}")
    for allele_name in model_models_counters:
        model_allele_to_matrix[allele_name] = model_allele_to_matrix[allele_name].div(
            model_allele_to_matrix[allele_name].sum(axis=1), axis=0)
    allele_to_info_matrix = {allele_name: logomaker.transform_matrix(prob_matrix,
                                                                     from_type='probability',
                                                                     to_type='information',
                                                                     background=get_frequencies(prob_matrix.shape[0]))
                             for allele_name, prob_matrix in model_allele_to_matrix.items()}
    model_trained_allele_to_info_vector = {allele_name: logomaker.transform_matrix(prob_matrix,
                                                                                   from_type='probability',
                                                                                   to_type='information',
                                                                                   background=get_frequencies(
                                                                                       prob_matrix.shape[0])).sum(
        axis=1).values for allele_name, prob_matrix in model_allele_to_matrix.items()}

    allele_to_info_value = {allele_name: sum(item) / len(item) for allele_name, item in
                            model_trained_allele_to_info_vector.items()}
    print("info calculated ", allele_to_info_value)
    model_name_to_info_value = {model_name: allele_to_info_value[model_name_to_allele[model_name]] for model_name in
                                model_names}
    return model_name_to_info_value, allele_to_info_matrix, allele_to_info_value


def get_average_prob_matrix_for_models(per_name_models, fraction_of_models=0.7):
    print("geting the average model matrix")
    model_names = list(per_name_models.keys())
    observed_alleles = list()
    model_models_counters = defaultdict(int)  # count models for each allele
    model_allele_to_matrix = dict()
    model_name_to_allele = {
        model_name: f"{model_name.split('-')[-2]}" for model_name in model_names
    }
    number_of_models_to_estimate = max(int(len(model_names) * fraction_of_models), 1)
    for model_name in model_names[:number_of_models_to_estimate]:
        current_model = per_name_models[model_name]
        current_allele = model_name_to_allele[model_name]
        model_name_to_allele[model_name] = current_allele
        model_distributions = [state.distribution.parameters[0] for state in current_model.states if
                               (state.distribution is not None) and (
                                       type(state.distribution) is not DiscreteDistributionCycle)]
        first_anchor = model_distributions[-2]
        last_anchor = model_distributions[-1]
        model_distributions = [first_anchor] + model_distributions[:-2] + [last_anchor]
        current_matrix = pd.DataFrame(model_distributions)
        current_matrix.index.name = 'pos'
        model_models_counters[current_allele] += 1
        if current_allele not in model_allele_to_matrix:
            observed_alleles.append(current_allele)
            model_allele_to_matrix[current_allele] = current_matrix
        else:
            model_allele_to_matrix[current_allele] = model_allele_to_matrix[current_allele] + current_matrix
    print(f"check how many models we added at the end: {model_models_counters}")
    for allele_name in model_models_counters:
        model_allele_to_matrix[allele_name] = model_allele_to_matrix[allele_name].div(
            model_allele_to_matrix[allele_name].sum(axis=1), axis=0)
    return model_allele_to_matrix, model_name_to_allele


def get_model_distances(per_name_models1, per_name_models2, models_fraction):
    # Get models distribution matrices
    ## Models 1
    model_allele_to_matrix1, model_name_to_allele1 = get_average_prob_matrix_for_models(per_name_models1, fraction_of_models=models_fraction)
    model_allele_to_matrix2, model_name_to_allele2 = get_average_prob_matrix_for_models(per_name_models2, fraction_of_models=models_fraction)
    allele_to_model_distance = dict()
    for allele_name in model_allele_to_matrix1.keys():
        matrix1 = model_allele_to_matrix1[allele_name]
        matrix2 = model_allele_to_matrix2[allele_name]
        average_distance = sum(jensenshannon(matrix1.values, matrix2.values, axis=1)) / len(matrix1)
        allele_to_model_distance[allele_name] = average_distance

    model_names1 = per_name_models1.keys()
    model_names2 = per_name_models2.keys()
    model_name_to_distance1 = {model_name: allele_to_model_distance[model_name_to_allele1[model_name]] for model_name in
                               model_names1}
    model_name_to_distance2 = {model_name: allele_to_model_distance[model_name_to_allele2[model_name]] for model_name in
                               model_names2}

    return allele_to_model_distance, model_name_to_distance1, model_name_to_distance2


def build_trash_train_data(
    per_kfold_per_length_data: dict,
    multiplier: int = 3,
    rng=None,
) -> dict:
    if rng is None:
        rng = np.random.default_rng()

    nat_freqs = {
        "A": 0.066470, "C": 0.021098, "D": 0.047650, "E": 0.070905,
        "F": 0.035178, "G": 0.062337, "H": 0.025244, "I": 0.043243,
        "K": 0.057427, "L": 0.096161, "M": 0.048262, "N": 0.035963,
        "P": 0.059660, "Q": 0.046936, "R": 0.054387, "S": 0.080709,
        "T": 0.052072, "V": 0.058573, "W": 0.011643, "Y": 0.026081,
    }
    amino_acids = list(nat_freqs.keys())
    probs = np.array(list(nat_freqs.values()))
    probs /= probs.sum()

    trash_data = {}
    for split_num, per_length_data in per_kfold_per_length_data.items():
        trash_data[split_num] = {}
        for length, sequences in per_length_data.items():
            n = max(len(sequences) * multiplier, 50)
            seqs = [
                "".join(rng.choice(amino_acids, size=length, p=probs))
                for _ in range(n)
            ]
            trash_data[split_num][length] = np.array(seqs, dtype=object)
    return trash_data

def build_uniform_weights(per_kfold_per_length_data: dict) -> dict:
    return {
        split_num: {
            length: np.ones(len(sequences), dtype=float)
            for length, sequences in per_length_data.items()
        }
        for split_num, per_length_data in per_kfold_per_length_data.items()
    }


def train_trash_models(
    train_data: dict,
    test_data: dict,
    experiment_params: ExperimentParams,
    random_multiplier: int = 3,
    return_histories: bool = False,
) -> dict:
    rng = np.random.default_rng(seed=42)
    per_allele_trash_models = {}
    per_allele_trash_histories = {}

    for allele_name, per_kfold_per_length_data in train_data.items():
        print(f"Training trash model for allele {allele_name}")

        single_allele_params = deepcopy(experiment_params)
        single_allele_params.model_training_params.alleles_to_use = [allele_name]

        trash_kfold_data = build_trash_train_data(
            per_kfold_per_length_data, multiplier=random_multiplier, rng=rng)
        trash_weights = build_uniform_weights(trash_kfold_data)
        trash_prepared = prepare_multiple_models(
            single_allele_params, custom_models_id="trash")

        trash_result_models, trash_result_histories = train_multiple_models(
            prepared_models=trash_prepared,
            train_data={allele_name: trash_kfold_data},
            test_data=test_data,
            train_data_weights={allele_name: trash_weights},
            experiment_params=single_allele_params,
            subfolder_to_safe_result="trash_models",
        )
        per_allele_trash_models[allele_name] = trash_result_models.get(allele_name, {})
        per_allele_trash_histories[allele_name] = trash_result_histories.get(allele_name, {})

    if return_histories:
        return per_allele_trash_models, per_allele_trash_histories
    return per_allele_trash_models

def _flatten_trash_models_and_histories(
    per_allele_trash_models: dict,
    per_allele_trash_histories: Optional[dict] = None,
) -> tuple[dict, dict]:

    per_name_trash_models = {}
    per_name_trash_histories = {}

    for allele_name, per_run_models in per_allele_trash_models.items():
        for run_idx, per_split_models in per_run_models.items():
            for split_num, model in per_split_models.items():
                unique_name = f"{model.name}__trash__{allele_name}__{run_idx}__{split_num}"
                per_name_trash_models[unique_name] = model

                if per_allele_trash_histories is not None:
                    history = (
                        per_allele_trash_histories
                        .get(allele_name, {})
                        .get(run_idx, {})
                        .get(split_num)
                    )
                    if history is not None:
                        per_name_trash_histories[unique_name] = history

    return per_name_trash_models, per_name_trash_histories


def _filter_trash_from_data_df(
    per_allele_data_df: dict,
    per_name_signal_models: dict,
    per_allele_trash_models: dict,
    model_training_params: ModelTrainingParams,
) -> tuple[dict, dict]:
    models_fraction = model_training_params.enrichment_split_decision_models_fraction

    per_allele_clean_df = {}
    per_allele_trash_df = {}

    for allele_name, df in per_allele_data_df.items():
        df = df.copy()

        df['score_signal'] = get_median_scores_for_top_models(
            per_name_models=per_name_signal_models,
            allele_data_df=df,
            models_fraction=models_fraction,
        )

        trash_models_flat = {
            f"trash__{allele_name}__{run_idx}__{split_num}": model
            for run_idx, per_split in per_allele_trash_models.get(allele_name, {}).items()
            for split_num, model in per_split.items()
        }

        if not trash_models_flat:
            print(f"  Warning: no trash models for allele {allele_name}, skipping filter")
            per_allele_clean_df[allele_name] = df
            per_allele_trash_df[allele_name] = df.iloc[0:0].copy()
            continue

        df['score_trash'] = get_median_scores_for_top_models(
            per_name_models=trash_models_flat,
            allele_data_df=df,
            models_fraction=models_fraction,
        )

        is_trash = df['score_trash'] > df['score_signal']

        n_trash = is_trash.sum()
        n_total = len(df)
        print(f"  Allele {allele_name}: {n_trash}/{n_total} peptides moved to trash "
              f"({100 * n_trash / max(n_total, 1):.1f}%)")

        per_allele_clean_df[allele_name] = df[~is_trash].copy()
        per_allele_trash_df[allele_name] = df[is_trash].copy()

    return per_allele_clean_df, per_allele_trash_df


def _save_trash_peptides(
    per_allele_trash_df: dict,
    experiment_params: ExperimentParams,
    iteration_name: str,
) -> None:
    base = f"{experiment_params.experiment_result_data_path}/trash/{iteration_name}"
    os.makedirs(base, exist_ok=True)

    for allele_name, df in per_allele_trash_df.items():
        if len(df) == 0:
            print(f"  No trash peptides for allele {allele_name}")
            continue
        path = f"{base}/{allele_name}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved {len(df)} trash peptides for allele {allele_name} → {path}")



def make_single_split_and_enrich(
    per_name_models,
    train_data,
    test_data,
    iteration_name,
    experiment_params: ExperimentParams,
    additional_data,
    per_allele_trash_models: Optional[dict] = None,
    per_allele_trash_histories: Optional[dict] = None,
    metrics_tracker=None,
    n_branches: int = 4,
):
    print(f"Starting split {iteration_name}")

    if metrics_tracker is None:
        metrics_tracker = MetricsTracker(
            experiment_params,
            reference_matrices=REFERENCE_MATRICES,
        )

    _peptide_length = experiment_params.model_training_params.lengths_to_use
    model_training_params: ModelTrainingParams = experiment_params.model_training_params
    data_scenario_params: DataScenarioParams = experiment_params.data_scenario_params

    per_allele_data_df = join_dicts(train_data)
    ALLELES = list(per_allele_data_df.keys())
    TARGET_LENGTHS = model_training_params.lengths_to_use
    TOTAL_SPLITS = np.arange(data_scenario_params.splits_to_read)
    ANCHOR_STATES_FOR_CLUSTERING = ["s007", "s002", "s004", "s008"]

    # --- Trash models ---
    if per_allele_trash_models is None:
        print("Training trash (background) models...")
        per_allele_trash_models, per_allele_trash_histories = train_trash_models(
            train_data=train_data,
            test_data=test_data,
            experiment_params=experiment_params,
            return_histories=True,
        )
    else:
        print("Using pre-trained trash models.")

    per_name_trash_models, per_name_trash_histories = _flatten_trash_models_and_histories(
        per_allele_trash_models=per_allele_trash_models,
        per_allele_trash_histories=per_allele_trash_histories,
    )
    print(f"  Trash models available: {len(per_name_trash_models)}")

    print("Filtering trash peptides with trained root models...")
    per_allele_clean_df, per_allele_trash_df = _filter_trash_from_data_df(
        per_allele_data_df=per_allele_data_df,
        per_name_signal_models=per_name_models,
        per_allele_trash_models=per_allele_trash_models,
        model_training_params=model_training_params,
    )

    _save_trash_peptides(per_allele_trash_df, experiment_params, iteration_name)

    if per_name_trash_models and per_name_trash_histories:
        save_all_visualization_results(
            per_name_models=per_name_trash_models,
            per_name_histories=per_name_trash_histories,
            experiment_params=experiment_params,
            subfolder_to_safe_result=f"{iteration_name}_trash",
            subset=5,
        )

    train_data = split_to_dicts(per_allele_data_df, ALLELES, TARGET_LENGTHS, TOTAL_SPLITS)

    n_before = sum(len(per_allele_data_df[a]) for a in ALLELES)
    n_after  = sum(len(per_allele_clean_df[a]) for a in ALLELES)
    print(f"  Peptides after trash filtering: {n_after}/{n_before} "
          f"({100 * n_after / max(n_before, 1):.1f}% kept)")

    per_allele_data_df = per_allele_clean_df

    print("Retraining root models...")
    train_data_clean_weights = calculate_weights_based_on_length_counts(
        train_data, experiment_params=experiment_params)

    per_allele_per_run_per_split_clean_models = prepare_multiple_models(
        experiment_params, custom_models_id=f"{iteration_name}_root")

    clean_result_models, clean_result_histories = train_multiple_models(
        prepared_models=per_allele_per_run_per_split_clean_models,
        train_data=train_data,
        test_data=test_data,
        train_data_weights=train_data_clean_weights,
        experiment_params=experiment_params,
        subfolder_to_safe_result=f"{iteration_name}_root",
    )

    per_name_models, _, _ = reorder_models_by_score_and_flatten_to_by_name_list(
        histories=clean_result_histories,
        models=clean_result_models,
        experiment_params=experiment_params,
        subfolder_to_safe_result=f"{iteration_name}_root",
    )
    print(f"  Clean root models ready: {len(per_name_models)}")

    print(f"  Initial split into {n_branches} branches via clustering...")
    train_data_branches_list = get_data_split_by_clustering(
        per_name_models=per_name_models,
        train_data=train_data,
        experiment_params=experiment_params,
        anchor_state_names=None,
        window=2,
        n_init=30,
        n_clusters=n_branches,
        top_n_models=10,
    )

    if not isinstance(train_data_branches_list, (list, tuple)):
        train_data_branches_list = [train_data_branches_list]
    train_data_branches = list(train_data_branches_list)

    fig_clustering, cluster_labels, cluster_peptides, joint_fig_allele, joint_fig_clust = \
        visualize_clustering_split(
            per_name_models=per_name_models,
            train_data=train_data,
            experiment_params=experiment_params,
            iteration_name=iteration_name,
            additional_data=additional_data,
            anchor_state_names=ANCHOR_STATES_FOR_CLUSTERING,
            n_clusters=n_branches,
        )

    clustering_plots_dir = str(Path(experiment_params.experiment_result_data_path) / "plots")
    save_figure_to_svg(fig_clustering,
                       dir=clustering_plots_dir,
                       filename=f"clustering_split_{iteration_name}.svg")
    if joint_fig_allele is not None:
        save_figure_to_svg(joint_fig_allele,
                           dir=clustering_plots_dir,
                           filename=f"clustering_kde_alleles_{iteration_name}.svg")
    if joint_fig_clust is not None:
        save_figure_to_svg(joint_fig_clust,
                           dir=clustering_plots_dir,
                           filename=f"clustering_kde_clusters_{iteration_name}.svg")
    plt.close("all")

    def _train_branch(branch_id, td, tdw):
        prepared = prepare_multiple_models(
            experiment_params, custom_models_id=f"{iteration_name}{branch_id}")
        trained, histories = train_multiple_models(
            prepared_models=prepared,
            train_data=td, test_data=test_data,
            train_data_weights=tdw,
            experiment_params=experiment_params,
            subfolder_to_safe_result=f"{iteration_name}{branch_id}",
        )
        models, hists, _ = reorder_models_by_score_and_flatten_to_by_name_list(
            histories=histories,
            models=trained,
            experiment_params=experiment_params,
            subfolder_to_safe_result=f"{iteration_name}{branch_id}",
        )
        return models, hists

    train_data_weights_branches = [
        calculate_weights_based_on_length_counts(td, experiment_params=experiment_params)
        for td in train_data_branches
    ]

    per_name_models_branches = []
    per_name_histories_branches = []
    for i, (td, tdw) in enumerate(zip(train_data_branches, train_data_weights_branches)):
        models, hists = _train_branch(i + 1, td, tdw)
        per_name_models_branches.append(models)
        per_name_histories_branches.append(hists)

    metrics_tracker.record_step(
        iteration_name=iteration_name,
        step="initial_split",
        **{f"per_name_models{i+1}": m for i, m in enumerate(per_name_models_branches)},
        **{f"train_data{i+1}": td for i, td in enumerate(train_data_branches)},
        peptide_length=_peptide_length,
    )

    # --- Enrichment ---
    print("Enrichment")
    enrichment_steps_count = model_training_params.enrichment_steps

    for step_i in range(enrichment_steps_count):
        print(f"enrichment iteration {step_i}")

        reassigned = reassign_data_to_splitted_models(
            per_name_models_branches=per_name_models_branches,
            per_allele_data_df=per_allele_data_df,
            experiment_params=experiment_params,
        )
        train_data_branches, train_data_weights_branches = reassigned

        per_name_models_branches = []
        per_name_histories_branches = []
        for i, (td, tdw) in enumerate(zip(train_data_branches, train_data_weights_branches)):
            models, hists = _train_branch(i + 1, td, tdw)
            per_name_models_branches.append(models)
            per_name_histories_branches.append(hists)

        metrics_tracker.record_step(
            iteration_name=iteration_name,
            step=f"enrich_{step_i}",
            **{f"per_name_models{i+1}": m for i, m in enumerate(per_name_models_branches)},
            **{f"train_data{i+1}": td for i, td in enumerate(train_data_branches)},
            peptide_length=_peptide_length,
        )

    for i, (models, hists) in enumerate(
            zip(per_name_models_branches, per_name_histories_branches)):
        save_all_visualization_results(
            per_name_models=models,
            per_name_histories=hists,
            experiment_params=experiment_params,
            subfolder_to_safe_result=f"{iteration_name}{i+1}",
            subset=10,
        )

    print("Now estimating this split result...")
    statistics_estimate_models_fraction = model_training_params.statistics_estimate_model_fraction

    info_values_list = []   # list of (name_to_info, allele_to_matrix, allele_to_value)
    for models in per_name_models_branches:
        nv, am, av = get_information_content_estimation_for_models(
            models, models_fraction=statistics_estimate_models_fraction)
        info_values_list.append((nv, am, av))

    # Попарные расстояния между всеми ветками
    from itertools import combinations
    pair_distances = {}  # (i, j) -> allele_to_distance_norm
    for i, j in combinations(range(n_branches), 2):
        dist, _, _ = get_model_distances(
            per_name_models_branches[i],
            per_name_models_branches[j],
            models_fraction=statistics_estimate_models_fraction,
        )
        info_i = info_values_list[i][2]
        info_j = info_values_list[j][2]
        pair_distances[(i, j)] = {
            allele: d / max(2 * (info_i.get(allele, 1e-6) + info_j.get(allele, 1e-6)), 1e-8)
            for allele, d in dist.items()
        }

    all_alleles = set().union(*[d.keys() for d in pair_distances.values()])

    allele_to_distance_norm = {
        allele: np.mean([
            pair_distances[(i, j)].get(allele, 0)
            for (i, j) in pair_distances
        ])
        for allele in all_alleles
    }

    # --- Info filter ---
    info_threshold     = model_training_params.trash_model_information_threshold
    distance_threshold = model_training_params.same_model_distance_threshold

    passed_filters = []
    for b_i, (name_to_info, _, _) in enumerate(info_values_list):
        passed = {k for k, v in name_to_info.items() if v > info_threshold}
        print(f"Branch {b_i+1}: {len(passed)}/{len(per_name_models_branches[b_i])} passed info filter")
        passed_filters.append(passed)

    print(f"Distance values (normalized): {allele_to_distance_norm}")

    finalized_alleles = {
        allele for allele, dist in allele_to_distance_norm.items()
        if dist <= distance_threshold
    }
    model_name_to_allele = {
        model_name: model_name.split('-')[-2]
        for model_name in per_name_models.keys()
    }
    finalized_models = {
        k for k in per_name_models
        if model_name_to_allele.get(k) in finalized_alleles
    }

    per_name_models_branches = [
        {k: v for k, v in models.items() if k in passed}
        for models, passed in zip(per_name_models_branches, passed_filters)
    ]

    path_to_save_pictures = str(Path(experiment_params.experiment_result_data_path) / "plots")
    allele_names = info_values_list[0][1].keys()

    for allele_name in allele_names:
        info_vals = [av.get(allele_name, 0) for _, _, av in info_values_list]
        data_sizes = [
            sum(len(peps) for per_kfold in td[allele_name].values()
                for peps in per_kfold.values())
            for td in train_data_branches
        ]
        ic_str    = " / ".join(str(np.round(v, 2)) for v in info_vals)
        split_str = " / ".join(str(s) for s in data_sizes)
        dist_str  = "  ".join(
            f"dist{i+1}{j+1}={pair_distances[(i,j)].get(allele_name, 0):.2f}"
            for (i, j) in pair_distances
        )

        info_matrices = [am.get(allele_name) for _, am, _ in info_values_list]
        plot_info_matrices_to_file(
            *info_matrices,
            folder_to_save=path_to_save_pictures,
            allele_name=allele_name,
            main_title=(
                f"Allele {allele_name} | {iteration_name}\n"
                f"IC ({ic_str})\n"
                f"{dist_str}\n"
                f"split: {split_str}"
            ),
            filename=f"{allele_name}-split-{iteration_name}.svg",
        )

    metrics_tracker.record_step(
        iteration_name=iteration_name,
        step="final",
        **{f"per_name_models{i+1}": m for i, m in enumerate(per_name_models_branches)},
        **{f"train_data{i+1}": td for i, td in enumerate(train_data_branches)},
        peptide_length=_peptide_length,
    )

    _mdir = f"{experiment_params.experiment_result_data_path}/metrics"
    metrics_tracker.save_csv(f"{_mdir}/{iteration_name}_metrics.csv")
    metrics_tracker.save_assignment_csv(f"{_mdir}/{iteration_name}_assignment.csv")

    return (
        *per_name_models_branches,
        *train_data_branches,
        per_allele_data_df,
        finalized_models,
        metrics_tracker,
    )


def reassign_data_to_splitted_models(
    per_name_models_branches: list,  # list of n_branches dicts
    per_allele_data_df: dict,
    experiment_params: ExperimentParams,
):
    model_training_params: ModelTrainingParams = experiment_params.model_training_params
    ALLELES = list(per_allele_data_df.keys())
    TOTAL_SPLITS = per_allele_data_df[ALLELES[0]].split.unique().astype(list)
    TARGET_LENGTHS = per_allele_data_df[ALLELES[0]].length.unique().astype(list)
    models_fraction = model_training_params.enrichment_split_decision_models_fraction
    n_branches = len(per_name_models_branches)

    per_allele_dfs = [{} for _ in range(n_branches)]

    for allele_name in ALLELES:
        df = per_allele_data_df[allele_name].copy()

        score_cols = []
        for i, models in enumerate(per_name_models_branches):
            col = f'score{i+1}'
            df[col] = get_median_scores_for_top_models(
                models, df, models_fraction=models_fraction)
            score_cols.append(col)

        best = df[score_cols].idxmax(axis=1)

        sizes = []
        for i, col in enumerate(score_cols):
            per_allele_dfs[i][allele_name] = df[best == col].copy()
            sizes.append(len(per_allele_dfs[i][allele_name]))

        print(f"  {allele_name}: {' / '.join(map(str, sizes))}")

    train_data_branches = [
        split_to_dicts(pad, ALLELES, TARGET_LENGTHS, TOTAL_SPLITS)
        for pad in per_allele_dfs
    ]
    weights_branches = [
        calculate_weights_based_on_length_counts(td, experiment_params=experiment_params)
        for td in train_data_branches
    ]

    return train_data_branches, weights_branches


def hierarchically_train_splited_models(
    per_name_models,
    train_data,
    test_data,
    additional_data,
    experiment_params: ExperimentParams,
    total_layers_to_do: int = 3,
    n_branches: int = 3,
):
    data_scenario_params: DataScenarioParams = experiment_params.data_scenario_params
    model_training_params: ModelTrainingParams = experiment_params.model_training_params
    per_layer_split_diagnostic = dict()

    with open('out.txt', 'w') as f:
        global_metrics_tracker = MetricsTracker(
            experiment_params,
            reference_matrices=REFERENCE_MATRICES,
        )

        iteration = "br1"
        models_deque = deque()
        data_deque = deque()
        iteration_deque = deque()

        models_deque.append(per_name_models)
        data_deque.append(train_data)
        iteration_deque.append(iteration)

        result_models = dict()
        layer_counter = 0
        current_total_models = dict()

        while iteration_deque:
            per_name_current_models = models_deque.popleft()
            this_train_data = data_deque.popleft()
            iteration = iteration_deque.popleft()
            print("iteration", iteration)

            split_result = make_single_split_and_enrich(
                per_name_models=per_name_current_models,
                train_data=this_train_data,
                test_data=test_data,
                iteration_name=iteration,
                experiment_params=experiment_params,
                additional_data=additional_data,
                metrics_tracker=global_metrics_tracker,
                n_branches=n_branches,
            )

            per_name_models_branches = list(split_result[:n_branches])
            new_train_data_branches  = list(split_result[n_branches:2*n_branches])
            per_allele_data_df       = split_result[2*n_branches]
            finalized_models         = split_result[2*n_branches + 1]

            for i, (branch_models, branch_data) in enumerate(
                    zip(per_name_models_branches, new_train_data_branches)):
                branch_label = i + 1
                if len(branch_models) != 0:
                    current_total_models.update(branch_models)
                    models_deque.append(branch_models)
                    data_deque.append(branch_data)
                    iteration_deque.append(f'{iteration}{branch_label}')
                else:
                    print(f"Branch {branch_label} of iteration {iteration}: "
                          f"no models remain, splitting finished for some alleles.")

            if len(finalized_models) > 0:
                print("Some models were finalized, adding to result collection.")
                result_models[iteration] = {
                    name: model
                    for name, model in per_name_current_models.items()
                    if name in finalized_models
                }
            elif layer_counter == (total_layers_to_do - 1):
                print("Out of iterations - dumping remaining branch models.")
                for i, branch_models in enumerate(per_name_models_branches):
                    result_models[f'{iteration}{i+1}'] = dict(branch_models)

            if (len(iteration_deque) == 0) or (len(iteration) < len(iteration_deque[0])):
                print("Finished layer.")
                print(f"New models for analysis: {len(current_total_models)}")

                if len(current_total_models) == 0:
                    print("No new models produced. Finishing.")
                    return result_models, per_layer_split_diagnostic

                for cur_result_models_dict in result_models.values():
                    current_total_models.update(cur_result_models_dict)

                per_allele_scores_data = get_all_scores_and_sort_by_best(
                    train_data, current_total_models)
                rest, fig = plot_split_diagnostics(
                    per_allele_data_df=per_allele_scores_data,
                    ann_df=additional_data,
                    layer_counter=layer_counter,
                    save_dir=str(Path(experiment_params.experiment_result_data_path) / "plots" / "umaps"),
                )
                per_layer_split_diagnostic[layer_counter] = fig

                layer_counter += 1
                if layer_counter >= total_layers_to_do:
                    return result_models, per_layer_split_diagnostic

                # перераспределение данных для нового слоя
                print(f"Branches in queue: {list(iteration_deque)}")
                print(f"Finalized iterations: {list(result_models.keys())}")

                per_allele_total_data = join_dicts(train_data)
                ALLELES = list(per_allele_total_data.keys())
                TOTAL_SPLITS = np.arange(data_scenario_params.splits_to_read)
                TARGET_LENGTHS = model_training_params.lengths_to_use

                per_allele_X_matrix = {}
                for allele_name in per_allele_total_data.keys():
                    total_data = per_allele_total_data[allele_name]
                    n_queue = len(models_deque)

                    # Скоры для веток, которые ещё в очереди
                    for model_num, current_models in enumerate(models_deque):
                        total_data[f'score_general_{model_num}'] = \
                            get_median_scores_for_top_models(
                                current_models, total_data,
                                models_fraction=model_training_params.reassignment_decision_models_fraction,
                            )

                    # Скоры для уже финализированных веток
                    for offset, cur_result_dict in enumerate(result_models.values()):
                        total_data[f'score_general_{n_queue + offset}'] = \
                            get_median_scores_for_top_models(
                                cur_result_dict, total_data,
                                models_fraction=model_training_params.reassignment_decision_models_fraction,
                            )

                    n_total_score_cols = n_queue + len(result_models)
                    X_matrix = total_data[
                        [f'score_general_{i}' for i in range(n_total_score_cols)]
                    ].copy()
                    X_matrix['max_column'] = X_matrix.idxmax(axis=1)
                    per_allele_X_matrix[allele_name] = X_matrix

                # Переназначаем данные только для веток в очереди
                for model_num in range(len(models_deque)):
                    per_allele_new_df = {
                        allele_name: per_allele_total_data[allele_name][
                            per_allele_X_matrix[allele_name]['max_column']
                            == f'score_general_{model_num}'
                        ].copy()
                        for allele_name in per_allele_X_matrix
                    }
                    data_deque[model_num] = split_to_dicts(
                        per_allele_new_df, ALLELES, TARGET_LENGTHS, TOTAL_SPLITS)

                current_total_models = {}

    _mdir = f"{experiment_params.experiment_result_data_path}/metrics"
    global_metrics_tracker.save_csv(f"{_mdir}/all_iterations_metrics.csv")
    global_metrics_tracker.save_assignment_csv(f"{_mdir}/all_iterations_assignment.csv")
    global_metrics_tracker.save_latex_table(f"{_mdir}/all_iterations_metrics.tex")
    global_metrics_tracker.print_summary()

    return result_models, per_layer_split_diagnostic, global_metrics_tracker