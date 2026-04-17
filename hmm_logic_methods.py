import json
import pomegranate
from pomegranate import DiscreteDistribution, DiscreteDistributionAnchor, HiddenMarkovModel, DiscreteDistributionCycle
from pomegranate.io import BatchedDataGenerator, SequenceGenerator
import numpy as np
from hmm_visualization_methods import convert_graph_to_good_format, save_all_visualization_results, save_figure_to_svg
from networkx import all_simple_paths
from tqdm import tqdm
from collections import defaultdict
from training_parameters import *
from copy import deepcopy
from peptides_utils import split_to_dicts, join_dicts, get_frequencies
from data_reading_methods import calculate_weights_based_on_length_counts
from collections import deque
import sys
from hmm_visualization_methods import plot_split_diagnostics, plot_info_matrices_to_file
from scipy.spatial.distance import jensenshannon
import os
import matplotlib.pyplot as plt
import pickle
import logomaker

MINUS_INF_CONSTANT = -20
MINUS_INF_CONSTANT_NORMALIZED = -0.5


def train_model_prepared(
        per_kfold_per_length_data,
        model_training_params:ModelTrainingParams,
        per_kfold_per_length_test_data=None,
        per_kfold_per_length_weights_for_data=None,
        prepared_models=None
):
    """
    Обучает (дообучает) заранее подготовленные модели по каждому split (KFold),
    используя данные, сгруппированные по длинам последовательностей.


    per_kfold_per_length_data : dict[int, dict[int, np.ndarray[str] | list[str]]]
        Данные для обучения по сплитам и длинам.
            {
              split_num (0): {
                  length (14): np.ndarray(['AAHCFTVDDKEHSIKV', ...]
              }
            }
    """

    target_lengths = model_training_params.lengths_to_use #[12, 13, 14, 15, 16, 17, 18]
    multiple_check_input = model_training_params.multiple_check_input #True
    algorithm = model_training_params.algorithm #viterbi
    lr_decay = model_training_params.lr_decay # затухание learning rate = 0.0
    minibatch_training = model_training_params.minibatch_training # обучение батчами = True
    batches_per_epoch = model_training_params.batches_per_epoch # 100 батчей
    batch_size = model_training_params.batch_size # размер одного батча = 1
    min_iterations = model_training_params.min_iters #40
    max_iterations = model_training_params.maxiters #80
    emission_pseudocount = model_training_params.emission_pseudocount # псевдосчётчики эмиссии = 0.1
    transition_pseudocount = model_training_params.transition_pseudocount # псевдосчётчики переходов = 0.1
    use_pseudocount = model_training_params.use_pseudocounts #True
    edge_inertia = model_training_params.edge_inertia # “инерция” обновления переходов = 0.4
    distribution_inertia = model_training_params.distribution_inertia # “инерция” обновления распределений emission = 0.4
    stop_threshold = model_training_params.stop_threshold #критерий остановки (если улучшение меньше порога - остановиться) = 3.0
    verbose = model_training_params.verbose #уровень подробности логов внутри model.fit False

    use_test = False     # Здесь  принудительно выключен, значит тестовые данные НЕ используются в fit (sequence_test_generator будет None)
    init_batch_size = None
    n_jobs=1 #количество потоков обучения

    #на каждый split кладём обученную модель и историю обучения
    per_split_model = dict()
    per_split_history = dict()

    # ожидается: dict вида {split_num: model}
    assert type(prepared_models) in [dict, type(
        None)], "Prepared models should be either None (for new models) or per split models dict"

    for split_num, per_length_data in per_kfold_per_length_data.items():
        #веса для тренировочных
        per_length_current_weights = per_kfold_per_length_weights_for_data[split_num] #type=<class 'dict'> keys=[12, 13, 14, 15, 16, 17, 18]
        #тестовые данные этого сплита
        per_length_test_data = per_kfold_per_length_test_data[split_num] #type=<class 'dict'> keys=[16, 17, 18, 12, 13, 14, 15]


        # Собираем ВСЕ тренировочные последовательности по всем target_lengths в один массив
        # per_length_data[target_length] - список последовательностей конкретной длины
        binders_array = np.array([per_length_data[target_length][i] for target_length in target_lengths
                                  for i in range(len(per_length_data[target_length]))], dtype=object)
        weights_array = np.array([per_length_current_weights[target_length][i] for target_length in target_lengths
                                  for i in range(len(per_length_current_weights[target_length]))], dtype=object)
        binders_test_array = np.array([per_length_test_data[target_length][i] for target_length in target_lengths
                                       for i in range(len(per_length_test_data[target_length]))], dtype=object)

        # Перемешиваем тренировочные данные и веса ОДИНАКОВО (одними и теми же индексами)
        rng = np.random.default_rng()
        new_indexes = rng.permutation(len(binders_array))
        binders_array = binders_array[new_indexes]
        weights_array = weights_array[new_indexes]
        rng.shuffle(binders_test_array)

        #ABCD" -> ["A","B","C","D"]
        sample_X = np.array([[char for char in item] for item in binders_array], dtype=object)
        sample_X_test = np.array([[char for char in item] for item in binders_test_array], dtype=object)

        #use_test = False
        sequence_test_generator = SequenceGenerator(sample_X_test) if use_test else None

       # print("Example of data", sample_X[0])

        # делаем генератор батчей для обучения if TRUE
        # Иначе: используем весь sample_X целиком (как набор последовательностей)
        if minibatch_training:

            sequence_generator = BatchedDataGenerator(sample_X,
                                                      batches_per_epoch=batches_per_epoch,
                                                      batch_size=batch_size)
            sequence_generator.reset()
        else:
            sequence_generator = sample_X

        # Если для этого split уже передали готовую модель - обучаем её дальше/заново на данных этого split
        if split_num in prepared_models:
            model = prepared_models[split_num] # model=root[0000]_run-7-st-v-alg-40-min_iters-12-18_model-dummy_allele-0:{

            # Запускаем обучение модели.
            # Внутрь передаём:
            # - данные/генератор sequence_generator
            # - параметры остановки и алгоритма
            # - псевдосчётчики/сглаживание (если включено)
            # - weights_array, чтобы обучение учитывало веса примеров
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
                                       weights=weights_array
                                       )

            per_split_model[split_num] = model
            per_split_history[split_num] = history
        else:
            print("skip training for this split since model was not provided")

        break

    # Возвращаем словари: {split_num: model} и {split_num: history}
    return per_split_model, per_split_history


#
# # НЕ ИСПОЛЬЗУЕМ
def train_model(allele_name,
                per_kfold_per_length_data,
                target_lengths, name_prefix="",
                per_kfold_per_length_test_data=None,
                per_kfold_per_length_weights_for_data=None,
                prepared_models=None,
                num_states=12,
                verbose=False,
                multiple_check_input=True,
                n_jobs=1,
                algorithm='baum-welch',
                lr_decay=0.0,
                minibatch_training=False,
                batches_per_epoch=None,
                batch_size=1, min_iterations=30, max_iterations=1e8,
                init_batch_size=None,
                emission_pseudocount=0.0,
                transition_pseudocount=0.0,
                use_pseudocount=False,
                edge_inertia=0.0,
                distribution_inertia=0.0,
                stop_threshold=5e-5,
                use_test=False
                ):
    per_split_model = dict()
    per_split_history = dict()

    assert type(prepared_models) in [dict, type(
        None)], "Prepared models should be either None (for new models) or per split models dict"

    for split_num, per_length_data in per_kfold_per_length_data.items():
        per_length_current_weights = per_kfold_per_length_weights_for_data[split_num]
        per_length_test_data = per_kfold_per_length_test_data[split_num]
        per_length_counts = {target_length: len(per_length_data[target_length]) for target_length in target_lengths}

        binders_array = np.array([per_length_data[target_length][i] for target_length in target_lengths
                                  for i in range(len(per_length_data[target_length]))], dtype=object)
        weights_array = np.array([per_length_current_weights[target_length][i] for target_length in target_lengths
                                  for i in range(len(per_length_current_weights[target_length]))], dtype=object)
        binders_test_array = np.array([per_length_test_data[target_length][i] for target_length in target_lengths
                                       for i in range(len(per_length_test_data[target_length]))], dtype=object)


        print("split {}".format(split_num))
        print("train size {}".format(len(binders_array)))
        print("Shuffling for fairness ")

        rng = np.random.default_rng()
        new_indexes = rng.permutation(len(binders_array))
        binders_array = binders_array[new_indexes]
        weights_array = weights_array[new_indexes]
        rng.shuffle(binders_test_array)
        sample_X = np.array([[char for char in item] for item in binders_array], dtype=object)
        print("data shuffled")
        sample_X_test = np.array([[char for char in item] for item in binders_test_array], dtype=object)

        if minibatch_training:
            sample_X = np.array([[char for char in item] for item in binders_array], dtype=object)


            sequence_generator = BatchedDataGenerator(sample_X, batches_per_epoch=batches_per_epoch,
                                                      batch_size=batch_size)
            total_batches = int(len(sequence_generator) / sequence_generator.batch_size)
            if init_batch_size is None:
                if batches_per_epoch is None:
                    init_batch_size = total_batches
                else:
                    assert batches_per_epoch < total_batches, f"Batches per epoch should be {total_batches} or lower for given batch size of {batch_size}"
                    init_batch_size = batches_per_epoch
            print(f"Mini_batch: Train Length {len(sequence_generator)}, "
                  f"batch_size {sequence_generator.batch_size}, batches_per_epoch {sequence_generator.batches_per_epoch}, "
                  f"init_batch_size {init_batch_size}")
            sequence_test_generator =  SequenceGenerator(sample_X_test) if use_test else None

        else:
            sequence_generator = sample_X
            sequence_test_generator = SequenceGenerator(sample_X_test) if use_test else None
            print(f"Simple training: Train Length {len(sequence_generator)}, "
                  f"init_batch_size {init_batch_size}")
            if init_batch_size is not None:
                assert init_batch_size < len(
                    sequence_generator), f"init batch size should be {len(sequence_generator)} or lower " \
 \
                    # init_batch_size = max(int(batches_per_epoch / 10), 1) if batches_per_epoch is not None else int(len(sequence_generator)/10)
        # init_batch_size = 1 if batch_size is None and batches_per_epoch is None else init_batch_size

        if prepared_models is None:
            lengths_string = "-".join(map(str, target_lengths))
            name_postfix = name_prefix + "-" + lengths_string
            state_names = ["s{:03d}".format(i) for i in range(num_states)]
            target_allele_name = allele_name.replace('-', '_').replace('*', '_').replace(':', '_')
            # fill model name
            params = {
                "st": num_states,
                "alg": algorithm,
                "min_iters": min_iterations
            }
            if max_iterations != 1e8:
                params["max_iters"] = max_iterations
            params_string = "-".join([f"{value}-{key}" for key, value in params.items()])
            model_name = f"{name_postfix}_model"
            model, history = HiddenMarkovModel.from_samples(distribution=DiscreteDistribution,
                                                            n_components=num_states,
                                                            X=sequence_generator,
                                                            X_test=sequence_test_generator,
                                                            end_state=True,
                                                            stop_threshold=stop_threshold,
                                                            name=model_name,
                                                            state_names=state_names,
                                                            return_history=True,
                                                            verbose=verbose,
                                                            multiple_check_input=multiple_check_input,
                                                            lr_decay=lr_decay,
                                                            edge_inertia=edge_inertia,
                                                            distribution_inertia=distribution_inertia,
                                                            min_iterations=min_iterations,
                                                            max_iterations=max_iterations,
                                                            initialization_batch_size=init_batch_size,
                                                            emission_pseudocount=emission_pseudocount,
                                                            transition_pseudocount=transition_pseudocount,
                                                            use_pseudocount=use_pseudocount,
                                                            n_jobs=n_jobs,
                                                            algorithm=algorithm,
                                                            weights=weights_array
                                                            )
            model_name = f"{name_postfix}-{params_string}_model-{target_allele_name}-{split_num}"
            model.name = model_name
        else:
            bb = next(sequence_generator.batches())

            print("Example of data", sample_X[0])
            print("Example of batch", bb)
            repr(bb)
            sequence_generator.reset()
            print("Use pseudocount is", use_pseudocount)

            print(sequence_generator)
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
                                       weights=weights_array
                                       )
        per_split_model[split_num] = model
        per_split_history[split_num] = history
        break
    return per_split_model, per_split_history




# Add states to existing model
#НЕ ИСПОЛЬЗУЕМ
def add_more_states_and_reset_transitions(source_model, ADD_STATES_NUM, new_sates_names=[], new_name="new_model",
                                          new_probs="fair"):
    old_start_index = source_model.start_index
    old_end_index = source_model.end_index
    old_states = source_model.states.copy()
    old_state_name_to_index = {state.name: i for i, state in enumerate(old_states)}
    old_states_names = {state.name for state in old_states}
    old_transitions = source_model.dense_transition_matrix()
    state_distributions = []
    model = source_model.copy()

    print(f"Initial length {len(model.states)}")
    keymap = model.keymap[0]
    for i in range(ADD_STATES_NUM):
        d = DiscreteDistribution({key: prob for key, prob in zip(keymap, np.random.uniform(0, 1, size=len(keymap)))})
        state_distributions.append(d)

    state_names = new_sates_names
    states_to_add = [State(distribution, name=name) for name, distribution in zip(state_names, state_distributions)]
    print(f"num states to add {len(states_to_add)}")
    model.add_states(states_to_add)
    model.name = new_name
    model.start.name = f"{new_name}-start"
    model.end.name = f"{new_name}-end"
    model.bake(verbose=True, merge=None)  # merge None because we don't want to delete the states without nodes
    print(f"After length {len(model.states)}")

    # total states after adding
    states = model.states
    states.remove(model.start)
    states.remove(model.end)
    n = len(states)
    print(f" N is  {n}")

    # Connect the start of the model to the appropriate state
    sum_from_start = 0

    from_start_probs = list()
    for to_state in states:
        if to_state.name in old_states_names:
            from_start_probs.append(old_transitions[old_start_index, old_state_name_to_index[to_state.name]])

    median_from_start_prob = np.mean(from_start_probs)

    # start probabilities
    for state in states:
        if state.name in old_states_names:
            prob = old_transitions[old_start_index, old_state_name_to_index[state.name]]
        else:
            if new_probs == "fair":
                prob = median_from_start_prob  # we want to have fair prob(median)
                # prob = 1 / ADD_STATES_NUM  # we want to have 0.5 probability in final normalized version for new states
            elif new_probs == "tiny":
                prob = 0.04
            else:
                raise RuntimeError("new_probs argument should be 'fair' or  'tiny'")
        if prob != 0:
            # print(f"ADD transition from {model.start.name} {state.name} with prob {prob}")
            model.add_transition(model.start, state, prob)
            sum_from_start += prob
    print(f"sum from start {sum_from_start}")

    # Connect all states to each other if they have a non-zero probability
    for i, from_state in enumerate(states):

        # calculate from probs
        from_probs = list()
        median_from_prob = 0
        for to_state in states:
            if from_state.name in old_states_names and to_state.name in old_states_names:
                from_probs.append(
                    old_transitions[old_state_name_to_index[from_state.name], old_state_name_to_index[to_state.name]])
        if from_state.name in old_states_names:
            median_from_prob = np.mean(from_probs)

        for j, to_state in enumerate(states):
            if from_state.name in old_states_names and to_state.name in old_states_names:
                prob = old_transitions[old_state_name_to_index[from_state.name], old_state_name_to_index[to_state.name]]
            elif from_state.name in old_states_names:
                if new_probs == "fair":
                    prob = median_from_prob  # probabilities from old to new will be median of all from probabilities
                    # prob = 1 / ADD_STATES_NUM  # probabilities from old to new will be 1/(num_new)
                elif new_probs == "tiny":
                    prob = 0.04
                else:
                    raise RuntimeError("new_probs argument should be 'fair' or  'tiny'")
            else:  # from new state
                if to_state.name in old_states_names:
                    prob = 1.0  # to old state prob is 1
                else:
                    prob = 0  # from new to old prob is 0
            if prob != 0:
                # print(f"ADD transition from {from_state.name} {to_state.name} with prob {prob}")
                model.add_transition(from_state, to_state, prob)

    # Connect states to the end of the model if a non-zero probability
    for state in states:
        if state.name in old_states_names:
            prob = old_transitions[old_state_name_to_index[state.name], old_end_index]
        else:
            prob = 1.0  # if from new state  equal probs from all the states + end state
        if prob != 0:
            # print(f"ADD transition from {state.name} {model.end.name} with prob {prob}")
            model.add_transition(state, model.end, prob)
    model.bake(verbose=False)
    return model


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
            #[('A', 0.785376846695213), ('C', 0.9490376811452166), ('D', 0.9003402231729263), ('E', 0.017993510902062693), ('F', 0.25732488532134146)]

        else:
            d_dict = {key: 1 for key in amino_acids_list}

    # Нормализация вероятностей чтоб в сумме были 1
        d_sum = sum(d_dict.values()) #d_sum=9.646785102992697
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
    tie_intermediate_cycle_states = model_training_params.tie_intermediate_cycle_states
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
        # for each joiner its own cycle
        # Same distribution for all cycles

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
        # Same distribution for all cycles

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
                 number_of_int_cycles  # groups, joiners, all cycles
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

    # assert sum(start_probabilities) == len(joiner_positions[0])* START_WEIGHT + TO_CYCLE_WEIGHT, start_probabilities
    # assert sum(end_probabilities) ==  (len(joiner_positions[max(joiner_positions.keys())]) + 1)  * END_WEIGHT, end_probabilities + list(last_joiners)

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
        #
        # for joiner_idx in first_joiners:
        #         model.add_transition(model.states[start_cycle_idx], model.states[joiner_idx], TO_FIRST_JOINER_WEIGHT, pseudocount=CYCLE_PSEUDOCOUNT)
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
            # for joiner_idx in last_joiners:
            #     model.add_transition(model.states[joiner_idx], model.states[end_cycle_idx], TO_LAST_CYCLE_WEIGHT, pseudocount=CYCLE_PSEUDOCOUNT)
        # model.add_transition(model.states[end_cycle_idx], model.states[end_cycle_idx], IN_CYCLE_WEIGHT, pseudocount=CYCLE_PSEUDOCOUNT)

    model.bake(verbose=False)
    return model



# Add states to existing model
def add_more_states_to_the_end(source_model, ADD_STATES_NUM, new_sates_names=[], new_name="new_model", new_probs="fair",
                               edge_pseudocount=None):
    old_start_index = source_model.start_index
    old_end_index = source_model.end_index
    old_states = source_model.states.copy()
    old_state_name_to_index = {state.name: i for i, state in enumerate(old_states)}
    old_states_names = {state.name for state in old_states}
    old_transitions = source_model.dense_transition_matrix()
    state_distributions = []
    model = source_model.copy()

    print(f"Initial length {len(model.states)}")
    keymap = model.keymap[0]
    for i in range(ADD_STATES_NUM):
        d = DiscreteDistribution({key: prob for key, prob in zip(keymap, np.random.uniform(0, 1, size=len(keymap)))})
        state_distributions.append(d)

    state_names = new_sates_names
    states_to_add = [State(distribution, name=name) for name, distribution in zip(state_names, state_distributions)]
    print(f"num states to add {len(states_to_add)}")
    model.add_states(states_to_add)
    model.name = new_name
    model.start.name = f"{new_name}-start"
    model.end.name = f"{new_name}-end"
    model.bake(verbose=True, merge=None)  # merge None because we don't want to delete the states without nodes
    print(f"After length {len(model.states)}")

    # total states after adding
    states = model.states
    states.remove(model.start)
    states.remove(model.end)
    n = len(states)
    print(f" N is  {n}")

    # Connect the start of the model to the appropriate state
    sum_from_start = 0
    for state in states:
        if state.name in old_states_names:
            prob = old_transitions[old_start_index, old_state_name_to_index[state.name]]
        else:
            if new_probs == "fair":
                prob = 1 / ADD_STATES_NUM  # we want to have 0.5 probability in final normalized version for new states
            elif new_probs == "tiny":
                prob = 0.05
            else:
                raise RuntimeError("new_probs argument should be 'fair' or  'tiny'")
        if prob != 0:
            # print(f"ADD transition from {model.start.name} {state.name} with prob {prob}")
            model.add_transition(model.start, state, prob, pseudocount=edge_pseudocount)
            sum_from_start += prob
    print(f"sum from start {sum_from_start}")

    # Connect all states to each other if they have a non-zero probability
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            if from_state.name in old_states_names and to_state.name in old_states_names:
                if to_state.name != model.end.name:
                    prob = old_transitions[
                        old_state_name_to_index[from_state.name], old_state_name_to_index[to_state.name]]
                # else:
                #
                #     prob = 0
            elif from_state.name in old_states_names:
                if new_probs == "fair":
                    prob = 1 / ADD_STATES_NUM  # probabilities from old to new will be 1/(num_new)
                elif new_probs == "tiny":
                    prob = 0.05
                else:
                    raise RuntimeError("new_probs argument shoul be 'fair' or  'tiny'")
            else:  # from new state
                prob = 0
            if prob != 0:
                # print(f"ADD transition from {from_state.name} {to_state.name} with prob {prob}")
                model.add_transition(from_state, to_state, prob, pseudocount=edge_pseudocount)

    # Connect states to the end of the model if a non-zero probability
    for state in states:
        if state.name in old_states_names:
            prob = 0
        else:
            prob = 1.0  # if from new state  equal probs from all the states + end state
        if prob != 0:
            # print(f"ADD transition from {state.name} {model.end.name} with prob {prob}")
            model.add_transition(state, model.end, prob, pseudocount=edge_pseudocount)
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
    # df = pd.DataFrame(columns=["path", "prob"])
    # remove empty nodes (for speed up and only possible paths)

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
    # print(f"Total paths found: {len(paths)}")
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
            # print(node_from, node_to,  np.exp(edge['label']), "total_prob", prob_for_path)
            edge_to_prob[(node_from, node_to)] = edge['label']
            if edge_value == "log_probability":
                prob_for_path = prob_for_path + edge['label']

            else:
                prob_for_path = prob_for_path * edge['label']
            # print(f"edge {edge}")
            node_from = node_to

        # print(f"Total prob for path is {prob_for_path}")

        result_paths.append(">".join(path))
        result_lens.append(len(path) - 2)
        result_probs.append(prob_for_path)
        # normalize
        result_normalized_probs.append(prob_for_path / len(path))
        # row = {"path": path, "prob": prob_for_path}
        # df = df.append(row, ignore_index=True)
    df = pd.DataFrame(
        {"path": result_paths, "prob": result_probs, "normalized_prob": result_normalized_probs, "len": result_lens})
    df = df.sort_values('normalized_prob', ascending=False)
    df = df.reset_index(drop=True)

    # print("Take top 50 most probable paths")
    # df = df.head(50)
    return df, edge_to_prob


def get_state_for_name(model, state_name):
    state = [state for state in model.states if state.name == state_name][0]
    return state


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
        # print(f"edge {edge}")
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
        # print(f"edge {edge}")
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

    # paths_df = paths_df.loc[paths_df['len'].between(min_length, max_length)].copy()
    paths_df = paths_df.sort_values('normalized_prob', ascending=False)
    paths_df = paths_df.reset_index(drop=True)
    if top_paths:
        paths_df = paths_df[:top_paths]  # assumed to be sorted
        print(f"Took only {top_paths} top paths")

    # df = pd.DataFrame(columns=["peptide", "path", "prob"])
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
            # result_paths.append(path)
            # result_probs.append(prob_for_path)

    df = pd.DataFrame(result_repr_matrix, index=peptides)
    if edge_value == "log_probability":
        df.replace([-np.inf], -55, inplace=True)
    return df, paths_df, emission_matrix, name_to_index


def get_peptide_to_paths_df_external_paths(model, peptides, paths_df, edge_value="log_probability", top_paths=None):
    print(f"Model {model.name}")
    name_to_state = get_state_to_name_dict(model)

    # paths_df = paths_df.loc[paths_df['len'].between(min_length, max_length)].copy()
    paths_df = paths_df.sort_values('norm_viterbi_prob', ascending=False)
    paths_df = paths_df.reset_index(drop=True)
    if top_paths:
        paths_df = paths_df[:top_paths]  # assumed to be sorted
        print(f"Took only {top_paths} top paths")

    # df = pd.DataFrame(columns=["peptide", "path", "prob"])
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
            # result_paths.append(path)
            # result_probs.append(prob_for_path)

    df = pd.DataFrame(result_repr_matrix, index=peptides)
    # if edge_value == "log_probability":
    #     df.replace([-np.inf], MINUS_INF_CONSTANT_NORMALIZED, inplace=True)
    return df, paths_df, emission_matrix, name_to_index


# def get_peptide_to_paths_df(model, peptides, edge_value="log_probability", top_paths=None):
#     print(f"Model {model.name}")
#     name_to_state = get_state_to_name_dict(model)
#     print(f"\tFind all paths")
#     paths_df, edge_to_prob = get_all_paths_with_probs(model, edge_value=edge_value)
#
#     if top_paths:
#         paths_df = paths_df[:top_paths] # assumed to be sorted
#         print(f"Took only {top_paths} top paths")
#     # df = pd.DataFrame(columns=["peptide", "path", "prob"])
#     result_peptides = []
#     result_paths = []
#     result_probs = []
#     print(f"\tCalculate probs for peptides")
#     for peptide in peptides:
#         for path in paths_df['path']:
#             state_path = [name_to_state[state_name] for state_name in path]
#             prob_for_path = get_prob_for_path_and_sequence(convert_graph_to_good_formatmodel, edge_to_prob, state_path, peptide, edge_value=edge_value)
#             result_peptides.append(peptide)
#             result_paths.append(path)
#             result_probs.append(prob_for_path)
#     df = pd.DataFrame({"peptide": result_peptides, "path":result_paths, "prob":result_probs})
#     if edge_value == "log_probability":
#         df.replace([-np.inf], -55, inplace=True)
#     return df


def get_all_viterbi_paths_for_model_and_sequences(model, peptides, edge_value="log_probability", precision=4):
    print(f"Model {model.name}")
    model_graph = model.graph

    # new_graph, name_to_level = convert_graph_to_good_format(model,"", False, precision=precision, edge_value=edge_value)
    # transition_matrix = model.dense_transition_matrix()
    print("Get viterbi paths for each peptide...")
    repeated_paths = list()
    repeated_path_state_indexes = list()

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
    print(f"\tUnique paths: {len(path_code_to_path)}")
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
    """
    Wil leave only last peptide in the peptide df if path was duplicated by multiple peptides
    :param model:
    :param peptides:
    :param per_model_important_states:
    :param edge_value:
    :param precision:
    :return:
    """
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


def calculate_model_core_guesses(allele_peptides, models_dict, per_model_important_states, per_model_skip_states=None,
                                 selected_peptides=None):
    total_df = pd.DataFrame(index=allele_peptides)
    selected_peptide_to_paths = defaultdict(list)

    for i, (model_key, model) in enumerate(models_dict.items()):
        viterbi_paths = get_all_viterbi_paths_for_model_and_sequences_mark_core(model, allele_peptides, per_model_important_states, per_model_skip_states=per_model_skip_states, unique_paths=False, fill_whole_core=True, cumulative_metric=False)
        if 'simplified_path' not in total_df.columns:
            total_df['simplified_path'] = viterbi_paths['simplified_path']

        else:
            total_df['simplified_path'] = total_df['simplified_path'].values + viterbi_paths['simplified_path'].values

        total_df[f'core_start_{i}'] = viterbi_paths['simplified_path'].apply(lambda x: np.where(np.array(x) > 0)[
            0]).apply(lambda x: x[0] if len(x) > 0 else 0)
        total_df[f'core_end_{i}'] = viterbi_paths['simplified_path'].apply(lambda x: np.where(np.array(x) > 0)[
            0]).apply(lambda x: x[-1] if len(x) > 0 else 1)
        total_df[f'core_{i}'] = viterbi_paths['simplified_path']
        total_df[f'prob_{i}'] = viterbi_paths['norm_viterbi_prob']

        if selected_peptides:
            for selected_peptide in selected_peptides:
                selected_peptide_to_paths[selected_peptide].append(
                    viterbi_paths.loc[selected_peptide]['simplified_path'])
    total_df['peptide'] = total_df.index.values

    return total_df, selected_peptide_to_paths


def get_all_viterbi_paths_for_model_and_sequences_preserve_peptides_check_importance_masks(model, peptides,
                                                                                           per_model_important_states,
                                                                                           edge_value="log_probability",
                                                                                           precision=4):
    print(f"Model {model.name}")
    model_graph = model.graph

    # new_graph, name_to_level = convert_graph_to_good_format(model,"", False, precision=precision, edge_value=edge_value)
    # transition_matrix = model.dense_transition_matrix()
    print("Get viterbi paths for each peptide...")
    repeated_paths = list()
    repeated_path_state_indexes = list()

    result_probs = list()
    result_normalized_probs = list()
    result_lens = list()
    result_paths = list()
    result_peptides = list()
    result_simplified_paths = list()

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
    print(f"\tUnique paths: {len(path_code_to_path)}")
    unique_paths = list(path_code_to_path.values())
    mean_viterbies = list(path_code_to_mean_viterbi_prob.values())
    cnts = list(path_code_to_cnt.values())
    for path in tqdm(unique_paths):
        prob_for_path = 0 if edge_value == "log_probability" else 1
        path_edges = path.copy()
        saved_path_nodes = path_edges.copy()
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
        result_simplified_paths.append(np.array([1 if node.name in per_model_important_states[model.name] else 0 for
                                                 node in saved_path_nodes[1:]]))
        # normalize
        result_normalized_probs.append(prob_for_path / len(path))

    paths_df = pd.DataFrame(
        {"path": result_paths, "simplified_path": result_simplified_paths, "prob": result_probs,
         "normalized_prob": result_normalized_probs, "len": result_lens,
         "mean_viterbi": mean_viterbies, "peptide_cnts": cnts})
    paths_df = paths_df.sort_values('normalized_prob', ascending=False)
    paths_df = paths_df.reset_index(drop=True)
    return paths_df


def get_all_viterbi_paths_for_model_and_sequences_preserve_peptides(model, peptides, edge_value="log_probability",
                                                                    precision=4):
    print(f"Model {model.name}")
    model_graph = model.graph

    # new_graph, name_to_level = convert_graph_to_good_format(model,"", False, precision=precision, edge_value=edge_value)
    # transition_matrix = model.dense_transition_matrix()
    print("Get viterbi paths for each peptide...")
    repeated_paths = list()
    repeated_path_state_indexes = list()

    result_probs = list()
    result_normalized_probs = list()
    result_lens = list()
    result_paths = list()
    result_peptides = list()

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
    print(f"\tUnique paths: {len(path_code_to_path)}")
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


def get_model_representation2(model, peptides, edge_value="log_probability", top_paths=None):
    assert isinstance(peptides, set)
    assert edge_value in ['log_probability', 'probability']
    peptide_to_path_df = get_peptide_to_paths_df(model, peptides, edge_value=edge_value, top_paths=top_paths)
    print("Got result dataset")
    unique_paths = peptide_to_path_df['path'].sort_values().apply(str).unique()
    path_to_index = {path: i for i, path in enumerate(unique_paths)}
    peptide_to_path_df['path_index'] = peptide_to_path_df["path"].map(str).map(path_to_index)
    # pivot
    pivot_df = peptide_to_path_df.pivot(index='peptide', columns='path_index', values='prob')
    print("Finished")
    return pivot_df, path_to_index


def get_model_representation(model, peptides, paths_df=None, edge_value="log_probability", top_paths=None, precision=4):
    assert len(peptides) == len(set(peptides))
    assert edge_value in ['log_probability', 'probability']
    pivot_df, paths_df, emission_matrix, name_to_index = \
        get_peptide_to_paths_df_external_paths(model, peptides, paths_df=paths_df, edge_value=edge_value,
                                               top_paths=top_paths)
    # old version with finding paths
    #     print("Precision will not be counted since we have external paths")
    #     pivot_df, paths_df, emission_matrix, name_to_index = get_peptide_to_paths_df(model, peptides,, edge_value=edge_value, top_paths=top_paths)

    # print("Got result dataset")
    # unique_paths = peptide_to_path_df['path'].sort_values().apply(str).unique()
    # path_to_index = {path : i for  i , path in enumerate(unique_paths)}
    # peptide_to_path_df['path_index'] = peptide_to_path_df["path"].map(str).map(path_to_index)
    # #pivot
    # pivot_df = peptide_to_path_df.pivot(index='peptide', columns='path_index', values='prob')
    print("Finished")
    return pivot_df, paths_df, emission_matrix, name_to_index


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
    # the amount of models could be bigger than num_runs due to hp tuning parameters
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

        #  print("Positions", write_start, write_end, peptide_subset_start, peptide_subset_end)
        # print(len(column_values), len(row_values), core_start, core_end, peptide, row_values, column_values)
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

    save_dir = "/Users/annaklimova/Desktop/Washu_project/data/saved_models_debug"
    os.makedirs(save_dir, exist_ok=True)

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

            #  print("Positions", write_start, write_end, peptide_subset_start, peptide_subset_end)
            # print(len(column_values), len(row_values), core_start, core_end, peptide, row_values, column_values)
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
#
# def check_models_integrity_and_shift_params(per_split_models, allele_data_df, experiment_params):
#     per_split_models_to_update = dict()
#     for split_num in per_split_models.keys():
#         current_model = per_split_models[split_num]
#         model_states = [state for state in current_model.states if
#                         (state.distribution is not None) and (
#                                 type(state.distribution) is not DiscreteDistributionCycle)]
#         current_core_length = len(model_states)
#
#         model_distributions = [state.distribution.parameters[0] for state in current_model.states if
#                                (state.distribution is not None) and (
#                                        type(state.distribution) is not DiscreteDistributionCycle)]
#
#         first_anchor = model_states[-2].name
#         last_anchor = model_states[-1].name
#         important_states = {current_model.name: [first_anchor, last_anchor]}
#         skip_states = {current_model.name: []}
#         viterbi_paths = get_all_viterbi_paths_for_model_and_sequences_mark_core(current_model,
#                                                                                 allele_data_df.peptide.values,
#                                                                                 important_states,
#                                                                                 unique_paths=False,
#                                                                                 fill_whole_core=True,
#                                                                                 cumulative_metric=False, per_model_skip_states=skip_states)
#         # define  2K - 2  set of distributions
#         index_values = np.arange(-current_core_length + 1, current_core_length + current_core_length - 1)
#         max_index = max(index_values)
#         amino_acids_list = list('ACDEFGHIKLMNPQRSTVWY')
#         counts_matrix = np.zeros((len(index_values), len(amino_acids_list)), dtype=int)
#         count_df = pd.DataFrame(counts_matrix, index=index_values, columns=amino_acids_list)
#         frame_size = len(index_values)
#         left_part = int((frame_size - current_core_length) / 2)
#         right_part = left_part
#         row_names = count_df.index.values
#         col_names = count_df.columns
#         # create multiindex series
#         index = pd.MultiIndex.from_product([row_names, col_names])
#         df = pd.Series(1, index=index)
#
#         # Convert to 2d dataframe
#         for peptide, core_start, core_end in zip(viterbi_paths.index.values, viterbi_paths.core_start.values, viterbi_paths.core_end.values):
#             write_start = max(-core_start, -left_part)
#             write_end = min(len(peptide) - core_start, max_index + 1)
#             row_values = np.arange(write_start, write_end)
#
#             peptide_subset_start = max(0, core_start - left_part)
#             peptide_subset_end = min(len(peptide), core_start + max_index) + 1
#             column_values = list(peptide[peptide_subset_start:peptide_subset_end])
#
#             #  print("Positions", write_start, write_end, peptide_subset_start, peptide_subset_end)
#             # print(len(column_values), len(row_values), core_start, core_end, peptide, row_values, column_values)
#             idx = pd.MultiIndex.from_arrays([row_values, column_values])
#             df.loc[idx] += 1
#
#         count_df = df.unstack(level=-1)
#         print("transform to info matrix")
#         info_matrix = logomaker.transform_matrix(count_df,
#                                                  from_type='counts', to_type='information',
#                                                  background=get_frequencies(
#                                                      count_df.shape[0]).set_index(count_df.index))
#         # now info matrix contains estimation of the information for all possible shifts inside model
#         info_vector = info_matrix.sum(axis=1).values
#         dummy_threshold = 0.5
#         new_core_count_df = None
#         per_pos_sum = np.zeros(len(info_vector) - current_core_length + 1)  # 1....16xxxx25
#         print(f"Will calculate sliding window sum for f{len(per_pos_sum)} positions with original vector of size {len(info_vector)}")
#         for pos in range(len(per_pos_sum)):
#             if pos == 0:
#                 print(f"calculate shifts for fragments of length {len(info_vector[pos:pos + current_core_length])}")
#             per_pos_sum[pos] = sum(info_vector[pos:pos + current_core_length])
#         max_core_start_pos = np.argmax(per_pos_sum)
#
#         if max_core_start_pos < left_part:
#             print("Seems to be shift to the left, since there is an alignment hit outside the model frame")
#             print("Shifting model frame to the best pick leftwise")
#             # better_pos_numerical_index = info_vector[: left_part].argmax()
#             better_pos_numerical_index = max_core_start_pos
#             # new_core_count_df = count_df.iloc[left_part - better_pos_numerical_index - 1:left_part + current_core_length + better_pos_numerical_index - 1, :].copy()
#             new_core_count_df = count_df.iloc[
#                                 better_pos_numerical_index:current_core_length + better_pos_numerical_index, :].copy()
#             print(f"shape of new matrix f{new_core_count_df.shape}")
#         if max_core_start_pos > left_part:
#             # if sum(info_vector[-right_part:] > dummy_threshold) > 0:
#             print("Seems to be shift to the right, since there is an alignment hit outside the model frame")
#             print("Shifting model frame to the best pick rightwise")
#             better_pos_numerical_index = max_core_start_pos
#             new_core_count_df = count_df.iloc[
#                                 better_pos_numerical_index:current_core_length + better_pos_numerical_index, :].copy()
#             print(f"shape of new matrix f{new_core_count_df.shape}")
#
#             # new_core_count_df = count_df.iloc[left_part + better_pos_numerical_index + 1:left_part + current_core_length + better_pos_numerical_index + 1, :].copy()
#
#         # reinitialize model with new distributions
#         if new_core_count_df is not None:
#             new_core_count_df = new_core_count_df.reset_index(drop=True)
#             new_prob_matrix = logomaker.transform_matrix(new_core_count_df,
#                                                          from_type='counts', to_type='probability',
#                                                          pseudocount=1,
#                                                          background=get_frequencies(new_core_count_df.shape[
#                                                                                         0]).set_index(new_core_count_df.index), )
#             new_prob_matrix = new_prob_matrix.reindex(
#                 [i for i in range(1, current_core_length - 1)] + [0, current_core_length - 1]).reset_index(drop=True)
#
#             # prepare new model with this matrix
#             model_training_params: ModelTrainingParams = experiment_params.model_training_params
#             data_scenario_params: DataScenarioParams = experiment_params.data_scenario_params
#             current_params = deepcopy(model_training_params)
#
#             prepared_model = build_model_based_on_params(current_params, prepared_emission_matrix=new_prob_matrix)
#             prepared_model.name = current_model.name
#             per_split_models_to_update[split_num] = prepared_model
#             print("finished model construction")
#
#     return per_split_models_to_update


def train_multiple_models(prepared_models,
                          train_data,
                          test_data,
                          train_data_weights,
                          experiment_params: ExperimentParams,
                          subfolder_to_safe_result: str):
    model_training_params: ModelTrainingParams = experiment_params.model_training_params
    path_to_save_models = f"{experiment_params.experiment_result_data_path}/{subfolder_to_safe_result}/"
    print(f"Сюда сохраняются {path_to_save_models}")
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
                               # "test_log_probability":  [per_allele_per_run_per_split_new_histories[allele_name][i][split_num].test_log_probability[-1]
                               #                   for i in range(NUM_RUNS)]
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


def get_data_split_according_to_strategy(
        per_name_models,
        per_allele_data_df,
        experiment_params: ExperimentParams,
        split_point="mean"
):
    """
    TODO: make it able to work with multiple alleles (now we take all models)
    :param per_name_models:
    :param per_allele_data_df:
    :param experiment_params:
    :param split_point:
    :return:
    """
    model_training_params: ModelTrainingParams = experiment_params.model_training_params
    data_split_strategy = model_training_params.data_split_strategy

    ALLELES = list(per_allele_data_df.keys())
    TOTAL_SPLITS = per_allele_data_df[ALLELES[0]].split.unique().astype(list)
    TARGET_LENGTHS = model_training_params.lengths_to_use

    if data_split_strategy == "best_model":
        best_model_index = 0
        selected_model = per_name_models[list(per_name_models.keys())[best_model_index]]
        for allele_name in ALLELES:
            df = per_allele_data_df[allele_name]
            df['score'] = [selected_model.log_probability(peptide) / len(peptide) for peptide in df.peptide]
            per_allele_data_df[allele_name] = df
    elif data_split_strategy == "median_score_of_best_models":
        for allele_name in ALLELES:
            df = per_allele_data_df[allele_name]
            df[
                'score'] = get_median_scores_for_top_models(per_name_models=per_name_models, allele_data_df=df, models_fraction=model_training_params.split_decision_models_fraction)
            per_allele_data_df[allele_name] = df
    elif data_split_strategy == "single_average_model":
        for allele_name in ALLELES:
            df = per_allele_data_df[allele_name]
            selected_model = get_average_model_for_runs(per_name_models)
            df['score'] = [selected_model.log_probability(peptide) / len(peptide) for peptide in df.peptide]
            per_allele_data_df[allele_name] = df
    # now split data
    per_allele_first_df = dict()
    per_allele_second_df = dict()
    for allele_name in ALLELES:
        df = per_allele_data_df[allele_name]
        df = df.sort_values(by='score', ascending=False)
        if split_point == "mean":
            df1 = df[df['score'] >= df['score'].mean()].copy()
            df2 = df[df['score'] < df['score'].mean()].copy()
            print(f"split is {len(df1)}/{len(df2)}")
        else:
            # split point is a number of percents
            assert isinstance(split_point, float)
            total_length = len(df)
            df1 = df.iloc[:int(total_length * split_point)].copy()
            df2 = df.iloc[int(total_length * split_point):].copy()
        per_allele_first_df[allele_name] = df1
        per_allele_second_df[allele_name] = df2

    new_train_data1 = split_to_dicts(per_allele_first_df, ALLELES, TARGET_LENGTHS, TOTAL_SPLITS)
    new_train_data2 = split_to_dicts(per_allele_second_df, ALLELES, TARGET_LENGTHS, TOTAL_SPLITS)
    new_train_data_weights1 = calculate_weights_based_on_length_counts(new_train_data1, experiment_params=experiment_params)
    new_train_data_weights2 = calculate_weights_based_on_length_counts(new_train_data2, experiment_params=experiment_params)
    return new_train_data1, new_train_data2, new_train_data_weights1, new_train_data_weights2


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

from pathlib import Path
import pandas as pd

def save_model_score_tables(per_allele_data_df, per_name_models, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    real_model_names = list(per_name_models.keys())
    short_model_names = [f"model_{i}" for i in range(len(real_model_names))]
    model_name_map = dict(zip(short_model_names, real_model_names))

    # сохраняем соответствие коротких и реальных имён моделей
    model_name_mapping = pd.DataFrame({
        "model": short_model_names,
        "real_model_name": real_model_names
    })
    model_name_mapping.to_csv(output_dir / "model_name_mapping.csv", index=False)

    for allele_name, df in per_allele_data_df.items():
        df = df.copy().reset_index(drop=True)

        if "peptide" not in df.columns:
            df["peptide"] = df.index.astype(str)
        else:
            df["peptide"] = df["peptide"].astype(str)

        # wide -> long
        target_df = df.melt(
            id_vars=["peptide"],
            value_vars=[c for c in short_model_names if c in df.columns],
            value_name="score",
            var_name="model",
            ignore_index=False
        ).reset_index(names="peptide_order")

        target_df["real_model_name"] = target_df["model"].map(model_name_map)

        # удобный порядок колонок
        target_df = target_df[
            ["peptide_order", "peptide", "model", "real_model_name", "score"]
        ]

        target_df.to_csv(
            output_dir / f"{allele_name}_model_scores_long.csv",
            index=False
        )

        # заодно можно сохранить и wide-матрицу как есть
        df.to_csv(
            output_dir / f"{allele_name}_model_scores_wide.csv",
            index=False
        )

    return model_name_mapping

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


def make_single_split_and_enrich(per_name_models,
                                 train_data,
                                 test_data,
                                 iteration_name,
                                 experiment_params: ExperimentParams):
    print(f"Starting split {iteration_name}")
    model_training_params: ModelTrainingParams = experiment_params.model_training_params
    data_scenario_params: DataScenarioParams = experiment_params.data_scenario_params

    per_allele_data_df = join_dicts(train_data)
    # print("length of the data before split", len(per_allele_data_df[list(per_allele_data_df.keys())[0]]))

    train_data1, train_data2, \
        train_data_weights1, train_data_weights2 = get_data_split_according_to_strategy(per_name_models=per_name_models,
                                                                                        per_allele_data_df=per_allele_data_df,
                                                                                        experiment_params=experiment_params)

    # prepare new models
    per_allele_per_run_per_split_prepared_models1 = prepare_multiple_models(experiment_params, custom_models_id=f"{iteration_name}1")
    per_allele_per_run_per_split_prepared_models2 = prepare_multiple_models(experiment_params, custom_models_id=f"{iteration_name}2")

    # train new models
    # 1
    per_allele_per_run_per_split_prepared_models1, \
        per_allele_per_run_per_split_new_histories1 = train_multiple_models(
        prepared_models=per_allele_per_run_per_split_prepared_models1,
        train_data=train_data1,
        test_data=test_data,
        train_data_weights=train_data_weights1,
        experiment_params=experiment_params,
        subfolder_to_safe_result=f"{iteration_name}1")

    # 2
    per_allele_per_run_per_split_prepared_models2, \
        per_allele_per_run_per_split_new_histories2 = train_multiple_models(
        prepared_models=per_allele_per_run_per_split_prepared_models2,
        train_data=train_data2,
        test_data=test_data,
        train_data_weights=train_data_weights2,
        experiment_params=experiment_params,
        subfolder_to_safe_result=f"{iteration_name}2")

    per_name_models1, per_name_histories1, _ = reorder_models_by_score_and_flatten_to_by_name_list(
        histories=per_allele_per_run_per_split_new_histories1,
        models=per_allele_per_run_per_split_prepared_models1,
        experiment_params=experiment_params, subfolder_to_safe_result=f"{iteration_name}1")
    per_name_models2, per_name_histories2, _ = reorder_models_by_score_and_flatten_to_by_name_list(
        histories=per_allele_per_run_per_split_new_histories2,
        models=per_allele_per_run_per_split_prepared_models2,
        experiment_params=experiment_params, subfolder_to_safe_result=f"{iteration_name}2")

    # Enrichment procedure
    print("Enrichment")
    enrichment_steps_count = model_training_params.enrichment_steps
    ALLELES = list(per_allele_data_df.keys())
    for i in range(enrichment_steps_count):
        print("enrichment iteration", i)
        train_data1, train_data2, \
            train_data_weights1, train_data_weights2 = reassign_data_to_splitted_models_according_to_strategy(per_name_models1=per_name_models1,
                                                                                                              per_name_models2=per_name_models2,
                                                                                                              per_allele_data_df=per_allele_data_df,
                                                                                                              experiment_params=experiment_params)
        # prepare new models
        per_allele_per_run_per_split_prepared_models1 = prepare_multiple_models(experiment_params,
                                                                                custom_models_id=f"{iteration_name}1")
        per_allele_per_run_per_split_prepared_models2 = prepare_multiple_models(experiment_params,
                                                                                custom_models_id=f"{iteration_name}2")
        # 1
        per_allele_per_run_per_split_prepared_models1, \
            per_allele_per_run_per_split_new_histories1 = train_multiple_models(
            prepared_models=per_allele_per_run_per_split_prepared_models1,
            train_data=train_data1,
            test_data=test_data,
            train_data_weights=train_data_weights1,
            experiment_params=experiment_params,
            subfolder_to_safe_result=f"{iteration_name}1")

        # 2
        per_allele_per_run_per_split_prepared_models2, \
            per_allele_per_run_per_split_new_histories2 = train_multiple_models(
            prepared_models=per_allele_per_run_per_split_prepared_models2,
            train_data=train_data2,
            test_data=test_data,
            train_data_weights=train_data_weights2,
            experiment_params=experiment_params,
            subfolder_to_safe_result=f"{iteration_name}2")

        # reorder these models by score in folders
        per_name_models1, per_name_histories1, _ = reorder_models_by_score_and_flatten_to_by_name_list(
            histories=per_allele_per_run_per_split_new_histories1,
            models=per_allele_per_run_per_split_prepared_models1,
            experiment_params=experiment_params, subfolder_to_safe_result=f"{iteration_name}1")
        per_name_models2, per_name_histories2, _ = reorder_models_by_score_and_flatten_to_by_name_list(
            histories=per_allele_per_run_per_split_new_histories2,
            models=per_allele_per_run_per_split_prepared_models2,
            experiment_params=experiment_params, subfolder_to_safe_result=f"{iteration_name}2")

    save_all_visualization_results(per_name_models=per_name_models1, per_name_histories=per_name_histories1, experiment_params=experiment_params, subfolder_to_safe_result=f"{iteration_name}1", subset=10)
    save_all_visualization_results(per_name_models=per_name_models2, per_name_histories=per_name_histories2, experiment_params=experiment_params, subfolder_to_safe_result=f"{iteration_name}2", subset=10)

    # Now we performed enrichment and want to estimate if result models are trash
    print("Now estimating this split result...")
    statistics_estimate_models_fraction = model_training_params.statistics_estimate_model_fraction
    name_to_info_value1, allele_to_info_matrix1, allele_to_info_value1 = get_information_content_estimation_for_models(per_name_models1, models_fraction=statistics_estimate_models_fraction)
    name_to_info_value2, allele_to_info_matrix2, allele_to_info_value2 = get_information_content_estimation_for_models(per_name_models2, models_fraction=statistics_estimate_models_fraction)
    allele_to_distance, name_to_distance1, name_to_distance2 = get_model_distances(per_name_models1, per_name_models2, models_fraction=statistics_estimate_models_fraction)
    allele_to_distance = {
        allele_name: distance / (2 * (allele_to_info_value1[allele_name] + allele_to_info_value2[allele_name])) for
        allele_name, distance in allele_to_distance.items()}
    model_name_to_allele1 = {
        model_name: f"{model_name.split('-')[-2]}" for model_name in per_name_models1.keys()
    }
    model_name_to_allele2 = {
        model_name: f"{model_name.split('-')[-2]}" for model_name in per_name_models2.keys()
    }

    name_to_distance1 = {model_name: distance / (2 * (
                allele_to_info_value1[model_name_to_allele1[model_name]] + allele_to_info_value2[
            model_name_to_allele1[model_name]])) for model_name, distance in name_to_distance1.items()}
    name_to_distance2 = {model_name: distance / (2 * (
                allele_to_info_value1[model_name_to_allele2[model_name]] + allele_to_info_value2[
            model_name_to_allele2[model_name]])) for model_name, distance in name_to_distance2.items()}

    path_to_save_pictures = f"{experiment_params.experiment_result_data_path}"

    info_threshold = model_training_params.trash_model_information_threshold
    distance_threshold = model_training_params.same_model_distance_threshold

    passed_filter1 = {key for key, info_value in name_to_info_value1.items() if info_value > info_threshold}
    passed_filter2 = {key for key, info_value in name_to_info_value2.items() if info_value > info_threshold}

    print(f"For branch 1 {len(passed_filter1)} models passed the info filter out of {len(per_name_models1)}")
    print(f"For branch 2 {len(passed_filter2)} models passed the info filter out of {len(per_name_models2)}")

    passed_distance_cutoff1 = {key for key, distance_value in name_to_distance1.items() if
                               distance_value > distance_threshold}
    passed_distance_cutoff2 = {key for key, distance_value in name_to_distance2.items() if
                               distance_value > distance_threshold}
    alleles_passed_distance_cutoff = {key for key, similarity_value in allele_to_distance.items() if
                                      similarity_value > distance_threshold}
    # Prepare list of not passed the cutoff models
    finalized_alleles = {key for key, similarity_value in allele_to_distance.items() if
                         similarity_value <= distance_threshold}
    model_name_to_allele = {
        model_name: f"{model_name.split('-')[-2]}" for model_name in per_name_models.keys()
    }
    finalized_models = {key for key in per_name_models.keys() if model_name_to_allele[key] in finalized_alleles}

    print("distance values calculated for this split:", allele_to_distance)
    print(f"For branch 1 {len(passed_distance_cutoff1)} models are new models out of {len(per_name_models1)} ({len(alleles_passed_distance_cutoff)} alleles out of {len(allele_to_distance)})")
    print(f"For branch 2 {len(passed_distance_cutoff2)}  models are new models out of {len(per_name_models2)} (should match with previous")

    print("Exclude bad and finalized models from splitting")
    per_name_models1 = {name: model for name, model in per_name_models1.items() if name in passed_filter1}
    per_name_models2 = {name: model for name, model in per_name_models2.items() if name in passed_filter2}

    per_name_models1 = {name: model for name, model in per_name_models1.items() if name in passed_distance_cutoff1}
    per_name_models2 = {name: model for name, model in per_name_models2.items() if name in passed_distance_cutoff2}

    for allele_name in allele_to_info_matrix1.keys():
        info_value1 = allele_to_info_value1[allele_name]
        info_value2 = allele_to_info_value2[allele_name]

        passed1 = info_value1 > info_threshold
        passed2 = info_value2 > info_threshold
        print(f"train_data1[allele_name].items() {train_data1[allele_name].items()}")
        data_size1 = sum(
            len(peptides) for kfold, per_length_data in train_data1[allele_name].items() for cur_length, peptides in
            per_length_data.items())
        data_size2 = sum(
            len(peptides) for kfold, per_length_data in train_data2[allele_name].items() for cur_length, peptides in
            per_length_data.items())

        plot_info_matrices_to_file(
            allele_to_info_matrix1[allele_name],
            allele_to_info_matrix2[allele_name],
            folder_to_save=path_to_save_pictures,
            allele_name=allele_name,
            main_title=(
                f"Allele {allele_name} for model {iteration_name}\n"
                f"info ({np.round(info_value1, 2)}/{np.round(info_value2, 2)}) "
                f"passed ({passed1}/{passed2}),\n"
                f"distance {allele_to_distance[allele_name]:.2f}, "
                f"final split was ({data_size1} / {data_size2})"
            ),
            filename=f"{allele_name}-split-{iteration_name}.svg"
        )

    return per_name_models1, per_name_models2, train_data1, train_data2, per_allele_data_df, finalized_models


## This is a code for splited training based on original models, train and test data

def hierarchically_train_splited_models(per_name_models, train_data, test_data, additional_data,
                                        experiment_params: ExperimentParams, total_layers_to_do=3):
    orig_stdout = sys.stdout
    data_scenario_params: DataScenarioParams = experiment_params.data_scenario_params
    model_training_params: ModelTrainingParams = experiment_params.model_training_params
    per_layer_split_diagnostic = dict()
    with open('out.txt', 'w') as f:
        #  sys.stdout = f
        # try single split and multiple enrichments
        stats_storage = dict()
        iteration = "br1"
        models_deque = deque()
        data_deque = deque()
        iteration_deque = deque()

        models_deque.append(per_name_models)
        data_deque.append(train_data)
        iteration_deque.append(iteration)

        # current_total_models = dict()
        result_models = dict()
        layer_counter = 0
        current_total_models = dict()
        while iteration_deque:
            per_name_current_models = models_deque.popleft()
            this_train_data = data_deque.popleft()
            iteration = iteration_deque.popleft()
            print("iteration", iteration)
            # run current data
            per_name_models1, per_name_models2, \
                new_train_data1, new_train_data2, per_allele_data_df, finalized_models = make_single_split_and_enrich(per_name_models=per_name_current_models,
                                                                                                                      train_data=this_train_data,
                                                                                                                      test_data=test_data,
                                                                                                                      iteration_name=iteration,
                                                                                                                      experiment_params=experiment_params)
            # Add current models to the total representation and visualize UMAP of this representation
            if len(per_name_models1) != 0:
                current_total_models.update(per_name_models1)
                # add left to queue
                models_deque.append(per_name_models1)
                data_deque.append(new_train_data1)
                iteration_deque.append(f'{iteration}1')
            else:
                print(f"in branch 1 of iteration {iteration} some models were removed. Splitting is finished for some alleles")

            if len(per_name_models2) != 0:
                current_total_models.update(per_name_models2)
                models_deque.append(per_name_models2)
                data_deque.append(new_train_data2)
                iteration_deque.append(f'{iteration}2')
            else:
                print(f"in branch 2 of iteration {iteration} some models were removed. Splitting is finished for some alleles")

            if len(finalized_models) > 0:
                print(f"Some models were finalized, add previous step models to result collection")
                result_models[iteration] = {name: model for name, model in per_name_current_models.items() if
                                            name in finalized_models}
            elif layer_counter == (total_layers_to_do - 1):
                print(f"We are out of iterations and will finish at the end of the level just dump what we had remaining for this branch")
                result_models[f'{iteration}1'] = {name: model for name, model in per_name_models1.items()}
                result_models[f'{iteration}2'] = {name: model for name, model in per_name_models2.items()}

            if (len(iteration_deque) == 0) or (len(iteration) < len(iteration_deque[0])):  # we finished the whole layer
                print("finished layer")
                print("Now estimate total models + result models")
                print(f"length of new models to be used for analysis {len(current_total_models)}")
                if len(current_total_models) == 0:
                    print("Could not produce any new models for these parameters. finish here")
                    return result_models, per_layer_split_diagnostic
                for iter, cur_result_models_dict in result_models.items():
                    current_total_models.update(cur_result_models_dict)
                # get result pictures for the layer
                per_allele_scores_data = get_all_scores_and_sort_by_best(train_data, current_total_models)
                rest, fig = plot_split_diagnostics(per_allele_data_df=per_allele_scores_data, ann_df=additional_data, layer_counter=layer_counter)
                print("diagnostic method finished")
                per_layer_split_diagnostic[layer_counter] = fig

                layer_counter = layer_counter + 1
                if layer_counter >= total_layers_to_do:
                    # Finalize results
                    return result_models, per_layer_split_diagnostic

                # Prepare for the new  layer

                # redistribute data after each layer
                # make total data df (should be per allele)
                # then calculate scores for current models to be splited as well as result models to redistribute the data
                print(f"Models participating in analysis {[iteration_deque]}")
                print(f"Current finalized models {list(result_models.keys())}")

                per_allele_total_data = join_dicts(train_data)
                print(f"list(per_allele_total_data.keys()) {list(per_allele_total_data.keys())}")
                ALLELES = list(per_allele_total_data.keys())
                TOTAL_SPLITS = np.arange(data_scenario_params.splits_to_read)
                TARGET_LENGTHS = model_training_params.lengths_to_use

                per_allele_X_matrix = dict()
                for allele_name in per_allele_total_data.keys():
                    total_data = per_allele_total_data[allele_name]
                    for model_num in range(len(models_deque)):
                        current_models = models_deque[model_num]
                        median_score_for_current_models = get_median_scores_for_top_models(current_models, total_data, models_fraction=model_training_params.reassignment_decision_models_fraction)
                        total_data[f'score_general_{model_num}'] = median_score_for_current_models
                    analysis_models_observed = len(models_deque)
                    # also add result models scores
                    resul_models_num = len(result_models)
                    for model_num, (iteration_num, current_result_models_dict) in enumerate(result_models.items()):
                        median_score_for_current_models = get_median_scores_for_top_models(current_result_models_dict, total_data, models_fraction=model_training_params.reassignment_decision_models_fraction)
                        total_data[
                            f'score_general_{analysis_models_observed + model_num}'] = median_score_for_current_models

                    X_matrix = total_data[
                        [f'score_general_{i}' for i in range(analysis_models_observed + resul_models_num)]].copy()
                    X_matrix['max_column'] = X_matrix.idxmax(axis=1)
                    #
                    per_allele_X_matrix[allele_name] = X_matrix
                #
                # now select only analysis max columns for the data
                for model_num in range(len(models_deque)):
                    per_allele_new_df = dict()
                    for allele_name in per_allele_X_matrix.keys():
                        X_matrix = per_allele_X_matrix[allele_name]
                        new_df = total_data[X_matrix['max_column'] == f'score_general_{model_num}'].copy()
                        per_allele_new_df[allele_name] = new_df
                    new_train_data = split_to_dicts(per_allele_new_df, ALLELES, TARGET_LENGTHS, TOTAL_SPLITS)
                    data_deque[model_num] = new_train_data
                current_total_models = dict()
                # nor we want to reorder all peptides according to scores of 2-4-6-8
    return result_models, per_layer_split_diagnostic

