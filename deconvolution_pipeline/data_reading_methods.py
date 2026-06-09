from os import listdir
from os.path import isfile, join
import mhcgnomes
import pandas as pd
import numpy as np
import itertools
from training_parameters import *
from peptides_utils import split_to_dicts, join_dicts
from collections import Counter

def balance_peptides_preserve_length_distribution(per_allele_dict, alleles, seed=42):
    rng = np.random.default_rng(seed)
    out = {a: {} for a in alleles}
    split_sets = [set(per_allele_dict[a].keys()) for a in alleles]
    splits = sorted(set.intersection(*split_sets))

    for s in splits:
        for a in alleles:
            out[a][s] = {}
        length_counter = Counter()
        for a in alleles:
            for L, peps in per_allele_dict[a][s].items():
                length_counter[L] += len(peps)

        lengths = sorted(length_counter.keys())
        total_all = sum(length_counter.values())
        p = {L: (length_counter[L] / total_all) for L in lengths}  # доли

        totals_per_allele = [sum(len(peps) for peps in per_allele_dict[a][s].values()) for a in alleles]
        target_total = min(totals_per_allele)

        target_counts = {L: int(round(p[L] * target_total)) for L in lengths}
        diff = target_total - sum(target_counts.values())
        lengths_by_weight = sorted(lengths, key=lambda L: p[L], reverse=True)
        i = 0
        while diff != 0 and lengths_by_weight:
            L = lengths_by_weight[i % len(lengths_by_weight)]
            if diff > 0:
                target_counts[L] += 1
                diff -= 1
            else:
                if target_counts[L] > 0:
                    target_counts[L] -= 1
                    diff += 1
            i += 1

        for a in alleles:
            for L in lengths:
                arr = np.array(per_allele_dict[a][s].get(L, []))
                k = min(target_counts[L], len(arr))  # не превышаем наличие
                if k == 0:
                    out[a][s][L] = arr[:0]
                elif k == len(arr):
                    out[a][s][L] = arr
                else:
                    idx = rng.choice(len(arr), size=k, replace=False)
                    out[a][s][L] = arr[idx]

    return out

def collapse_alleles_to_one(per_allele_dict, dummy_allele_name="dummy"):
    """
    per_allele_dict: dict[allele][split][length] -> np.array(peptides)
    return: dict[dummy_allele_name][split][length] -> np.array(peptides)
    """
    # соберём все splits и lengths по всем аллелям
    all_splits = set()
    all_lengths = set()
    for a in per_allele_dict:
        for s in per_allele_dict[a]:
            all_splits.add(s)
            for L in per_allele_dict[a][s]:
                all_lengths.add(L)

    all_splits = sorted(all_splits)
    all_lengths = sorted(all_lengths)

    collapsed = {dummy_allele_name: {}}
    for s in all_splits:
        collapsed[dummy_allele_name][s] = {}
        for L in all_lengths:
            peptides = []
            for a in per_allele_dict:
                if s in per_allele_dict[a] and L in per_allele_dict[a][s]:
                    peptides.extend(list(per_allele_dict[a][s][L]))
            collapsed[dummy_allele_name][s][L] = np.unique(peptides)

    return collapsed

def get_available_alleles(data_path, do_not_parse_alleles=False):
    result_alleles = []
    only_files = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    per_allele_train_df = dict()
    for file_name in only_files:
        allele_name = file_name.split("#")[0]
        if do_not_parse_alleles:
            allele_name = allele_name
        else:
            allele_name = mhcgnomes.parse(allele_name, raise_on_error=False).to_string()
        # per_allele_train_df[allele_name] =  pd.read_csv(DATA_PATH + file_name,sep=";")
        if allele_name is not None:
            result_alleles.append(allele_name)
    result_alleles = list(sorted(set(result_alleles)))
    return result_alleles


def read_data(data_path, alleles_to_read, dataset_type, data_type, do_not_parse_alleles=True):
    assert data_type in ["binders", "nonbinders", 'immunogenic', 'nonimmunogenic']
    assert dataset_type in ["train", "test"]
    per_allele_per_kfold_per_length_peptides = dict()

    for allele_name in alleles_to_read:
        per_allele_per_kfold_per_length_peptides[allele_name] = dict()
        if do_not_parse_alleles:
            target_allele_name = allele_name
        else:
            target_allele_name = mhcgnomes.parse(allele_name).compact_string(include_species=True).replace("*", "_")
        file_name = "/{}#{}#{}.csv".format(target_allele_name, data_type, dataset_type)
        total_df = pd.read_csv(data_path + file_name, sep=";")
        n_splits = list(total_df['split'].unique())
        lengths = list(total_df['length'].unique())
        for split_num in n_splits:
            per_allele_per_kfold_per_length_peptides[allele_name][split_num] = dict()
            for length in lengths:
                per_allele_per_kfold_per_length_peptides[allele_name][split_num][length] = \
                total_df[(total_df['length'] == length) &
                         (total_df['split'] == split_num)]['peptide'].unique()
    return per_allele_per_kfold_per_length_peptides


def get_train_test_data(
    experiment_params: ExperimentParams,
    clean: bool = False,
    collapse_to_dummy: bool = False,
):
    data_params: DataScenarioParams = experiment_params.data_scenario_params
    data_scenario = data_params.data_scenario
    DATA_PATH = data_params.input_data_path
    TOTAL_SPLITS = data_params.splits_to_read

    print(f"Will read files from the folder {DATA_PATH}")
    assert data_scenario in ["IEDB_preprocessed", "simulated", "simulated_preprocessed", "MixMHCpred"]

    additional_return = list()

    if isinstance(data_params, PreprocessedIEDBDataParams):
        ALLELES_ALL = get_available_alleles(DATA_PATH, do_not_parse_alleles=True)
        print(f"ALLELES_ALL {ALLELES_ALL}")
        # KEEP = {"HLA-DRB1_0101", "HLA-DRB1_0301", "HLA-DRB1_0401"}
        KEEP = {"HLA-DPA1_0202-DPB1_0501"}
        ALLELES = [a for a in ALLELES_ALL if a in KEEP]
        print("Keeping only alleles:", ALLELES)

        def filter_nested_dict_by_peptides(data_dict, allowed_peptides):
            filtered = {}

            for allele, kfold_dict in data_dict.items():
                filtered[allele] = {}

                for kfold, length_dict in kfold_dict.items():
                    filtered[allele][kfold] = {}

                    for peptide_length, values in length_dict.items():
                        arr = np.asarray(values).astype(str)
                        mask = np.isin(arr, list(allowed_peptides))
                        filtered[allele][kfold][peptide_length] = arr[mask]

            return filtered

        if collapse_to_dummy:
            train_raw = read_data(
                DATA_PATH, ALLELES, "train", "binders", do_not_parse_alleles=True
            )
            test_raw = read_data(
                DATA_PATH, ALLELES, "test", "binders", do_not_parse_alleles=True
            )

            if clean:
                non_clean_peptides_path = str(RESULTS_DIR / "score_tables" / "model_score_tables_IEDB_DRB1_0101_0301_0401" / "plot_df.csv")
                non_clean_df = pd.read_csv(non_clean_peptides_path)

                # Фильтруем по нужному значению plot_group
                filtered_df = non_clean_df[non_clean_df["plot_group"] == "not_in_markers"]
                print(f"filtered_df {filtered_df}")

                non_clean_peptides = set(filtered_df["peptide"].dropna().astype(str).unique())

                print(f"Loaded {len(non_clean_peptides)} non-clean peptides (plot_group=not_in_markers) from {non_clean_peptides_path}")

                train_raw = filter_nested_dict_by_peptides(train_raw, non_clean_peptides)
                test_raw = filter_nested_dict_by_peptides(test_raw, non_clean_peptides)

            train_raw = balance_peptides_preserve_length_distribution(train_raw, ALLELES, seed=42)

            # Сохраняем per-allele df до схлопывания
            per_allele_df_raw = join_dicts(train_raw)
            for allele_name in ALLELES:
                per_allele_df_raw[allele_name]["source_allele"] = allele_name

            additional_return.append(per_allele_df_raw)

            # Схлопываем все аллели в один dummy
            dummy = getattr(data_params, "dummy_allele_name", "dummy_allele")
            per_allele_per_kfold_per_length_binders_train = collapse_alleles_to_one(train_raw, dummy)
            per_allele_per_kfold_per_length_binders_test = collapse_alleles_to_one(test_raw, dummy)

            sample_allele = dummy

        else:
            # Текущая логика
            per_allele_per_kfold_per_length_binders_train = read_data(
                DATA_PATH, ALLELES, "train", "binders"
            )
            per_allele_per_kfold_per_length_binders_test = read_data(
                DATA_PATH, ALLELES, "test", "binders"
            )

            if clean:
                clean_peptides_path = str(PEPTIDES_DIR / "clean_peptides_by_cell_type.csv")
                clean_df = pd.read_csv(clean_peptides_path)
                clean_peptides = set(clean_df["peptide"].dropna().astype(str).unique())
                print(f"Loaded {len(clean_peptides)} clean peptides from {clean_peptides_path}")

                per_allele_per_kfold_per_length_binders_train = filter_nested_dict_by_peptides(
                    per_allele_per_kfold_per_length_binders_train,
                    clean_peptides,
                )
                per_allele_per_kfold_per_length_binders_test = filter_nested_dict_by_peptides(
                    per_allele_per_kfold_per_length_binders_test,
                    clean_peptides,
                )

            sample_allele = list(per_allele_per_kfold_per_length_binders_train.keys())[0]

            per_allele_df = join_dicts(per_allele_per_kfold_per_length_binders_train)
            for allele_name in ALLELES:
                per_allele_df[allele_name]["allele"] = allele_name

            additional_return.append(per_allele_df)

        assert len(per_allele_per_kfold_per_length_binders_train[sample_allele].keys()) >= TOTAL_SPLITS
    elif isinstance(data_params, SimulatedPreprocessedDataParams):
        ALLELES = get_available_alleles(DATA_PATH, do_not_parse_alleles=True)
        per_allele_per_kfold_per_length_binders_train = read_data(DATA_PATH, ALLELES, "train", "binders", do_not_parse_alleles=True)
        per_allele_per_kfold_per_length_binders_test = read_data(DATA_PATH, ALLELES, "test", "binders", do_not_parse_alleles=True)
        sample_allele = list(per_allele_per_kfold_per_length_binders_train.keys())[0]
        per_allele_df = join_dicts(per_allele_per_kfold_per_length_binders_train)
        for allele_name in ALLELES:
            per_allele_df[allele_name]['allele'] = allele_name
        assert len(per_allele_per_kfold_per_length_binders_train[
                       sample_allele].keys()) >= TOTAL_SPLITS  # check number of splits
        additional_return.append(per_allele_df)
    elif isinstance(data_params, SimulatedDataParams):
        simulated_exact_file = data_params.simulated_exact_file_name
        dummy_allele_name = data_params.dummy_allele_name
        simulated_scenario = data_params.simulated_scenario
        SIMULATED_DATA_PATH = f"{DATA_PATH}/{simulated_scenario}/{simulated_exact_file}"
        ALLELES = [dummy_allele_name]
        per_allele_df = dict()
        # For now just read the same data multiple times for alleles/splits
        for allele_name in ALLELES:
            allele_df = pd.read_csv(SIMULATED_DATA_PATH, sep=";")
            list_dfs = [allele_df.copy() for i in range(TOTAL_SPLITS)]
            for split_num, split_df in enumerate(list_dfs):
                split_df['split'] = split_num
                split_df['allele_name'] = allele_name
                result_allele_df = pd.concat(list_dfs)
            per_allele_df[allele_name] = result_allele_df
            result_allele_df['length'] = split_df.peptide.str.len()
            TARGET_LENGTHS = list(split_df['length'].unique())
        # split data into dicts
        per_allele_per_kfold_per_length_binders_train = split_to_dicts(per_allele_df,
                                                                       ALLELES=ALLELES,
                                                                       TARGET_LENGTHS=TARGET_LENGTHS,
                                                                       TOTAL_SPLITS=np.arange(TOTAL_SPLITS))
        per_allele_per_kfold_per_length_binders_test = split_to_dicts(per_allele_df,
                                                                      ALLELES=ALLELES,
                                                                      TARGET_LENGTHS=TARGET_LENGTHS,
                                                                      TOTAL_SPLITS=np.arange(TOTAL_SPLITS))
        additional_return.append(per_allele_df)
    elif isinstance(data_params, MixMHCpredDataParams):
        mixture_name = data_params.mixmhc_mixture_name
        dummy_allele_name = data_params.dummy_allele_name
        df = pd.read_csv(DATA_PATH, sep=';')
        print(df.columns)
        df = df.loc[
            df.Peptide.str.match("^[ACDEFGHIKLMNPQRSTVWY]+$")
        ]
        print("Total table length", len(df))
        # Filter out selected mixture
        df = df.loc[df.Sample_IDs.str.split(', ').apply(lambda x: mixture_name in x),]
        print("Filtered for given mixture", len(df))
        sample_data = pd.DataFrame(
            {"peptide": df.Peptide.values,
             "old_sample_id": df.Sample_IDs,
             "sample_id": mixture_name,
             "mixmhc_predicted_mixed_alleles": df.Allele.values})

        list_dfs = [sample_data.copy() for i in range(TOTAL_SPLITS)]
        per_allele_df = dict()
        allele_name = "d_" + dummy_allele_name
        ALLELES = [allele_name]
        for allele_name in ALLELES:
            for split_num, split_df in enumerate(list_dfs):
                split_df['split'] = split_num
                split_df['allele'] = allele_name
                split_df['length'] = split_df.peptide.str.len()
            result_allele_df = pd.concat(list_dfs)
            result_allele_df = result_allele_df.drop_duplicates(subset=['peptide'])
            TARGET_LENGTHS = list(split_df['length'].unique())
            per_allele_df[allele_name] = result_allele_df
        per_allele_per_kfold_per_length_binders_train = split_to_dicts(per_allele_df,
                                                                       ALLELES=ALLELES,
                                                                       TARGET_LENGTHS=TARGET_LENGTHS,
                                                                       TOTAL_SPLITS=np.arange(TOTAL_SPLITS))
        per_allele_per_kfold_per_length_binders_test = split_to_dicts(per_allele_df,
                                                                      ALLELES=ALLELES,
                                                                      TARGET_LENGTHS=TARGET_LENGTHS,
                                                                      TOTAL_SPLITS=np.arange(TOTAL_SPLITS))
        additional_return.append(per_allele_df)
        additional_return.append(df)

    return per_allele_per_kfold_per_length_binders_train, per_allele_per_kfold_per_length_binders_test, additional_return


## Methods to split and join the data


def merge_per_allele_sets_into_combination_sets(per_allele_per_kfold_per_length_binders_train,
                                                per_allele_per_kfold_per_length_binders_test,
                                                combination_size, selected_alelles, total_splits,
                                                list_of_transform_dicts=None, position_component=False):
    f"""
    works with result of {read_data}
    takes combination size and merges all datasets together in all possible wais using {itertools.combinations}
    :return:  per_allele_per_kfold_per_length_peptides, where allele is a combination of old alleles
    """
    per_allele_per_kfold_per_length_binders_train_new = dict()
    per_allele_per_kfold_per_length_binders_test_new = dict()
    OLD_ALLELES = selected_alelles
    NEW_ALELLES = list()
    TOTAL_SPLITS = total_splits

    for combination in itertools.combinations(OLD_ALLELES, combination_size):
        allele_name = "+".join(combination)
        NEW_ALELLES.append(allele_name)
        print(allele_name)
        per_allele_per_kfold_per_length_binders_train_new[allele_name] = dict()
        per_allele_per_kfold_per_length_binders_test_new[allele_name] = dict()
        for split_num in range(TOTAL_SPLITS):
            per_allele_per_kfold_per_length_binders_train_new[allele_name][split_num] = dict()
            per_allele_per_kfold_per_length_binders_test_new[allele_name][split_num] = dict()

            # for each split find intersected lengths
            list_of_length_sets = list()
            for old_allele in combination:
                lengths = set(per_allele_per_kfold_per_length_binders_train[old_allele][split_num].keys())
                list_of_length_sets.append(lengths)
            intersected_lengths = set.intersection(*list_of_length_sets)

            for target_length in intersected_lengths:
                data_arrays_train = list()
                data_arrays_test = list()
                if split_num == 0:
                    print(f"Length {target_length}:", end=" ")
                for old_allele in combination:
                    single_allele_data_array_train = \
                    per_allele_per_kfold_per_length_binders_train[old_allele][split_num][target_length]
                    if list_of_transform_dicts:
                        single_allele_data_array_train = np.array([np.array([tuple(
                            transform_dict[aa] for transform_dict in list_of_transform_dicts) for aa in peptide]) for
                                                                   peptide in single_allele_data_array_train])
                    if position_component:
                        single_allele_data_array_train = np.array([np.array([(aa, i) for i, aa in enumerate(peptide)])
                                                                   for peptide in single_allele_data_array_train])

                    if split_num == 0:
                        print(f"{old_allele} - {len(single_allele_data_array_train)}", end=" ")
                    data_arrays_train.append(single_allele_data_array_train)
                    single_allele_data_array_test = per_allele_per_kfold_per_length_binders_test[old_allele][split_num][
                        target_length]
                    if list_of_transform_dicts:
                        single_allele_data_array_test = np.array([np.array([tuple(
                            transform_dict[aa] for transform_dict in list_of_transform_dicts) for aa in peptide]) for
                                                                  peptide in single_allele_data_array_test])
                    if position_component:
                        single_allele_data_array_train = np.array([np.array([(aa, i) for i, aa in enumerate(peptide)])
                                                                   for peptide in single_allele_data_array_test])

                    if split_num == 0:
                        # print(f"{old_allele} - {len(single_allele_data_array_test)}", end= " ")
                        pass
                    data_arrays_test.append(single_allele_data_array_test)
                if split_num == 0:
                    print()

                per_allele_per_kfold_per_length_binders_train_new[allele_name][split_num][
                    target_length] = np.concatenate(data_arrays_train)
                per_allele_per_kfold_per_length_binders_test_new[allele_name][split_num][
                    target_length] = np.concatenate(data_arrays_test)
    return per_allele_per_kfold_per_length_binders_train_new, per_allele_per_kfold_per_length_binders_test_new


def transform_data_to_properties_and_join_alleles(per_allele_per_kfold_per_length_binders_train,
                                                  per_allele_per_kfold_per_length_binders_test,
                                                  model_training_params: ModelTrainingParams,
                                                  transform_dicts=None):
    ALLELES = model_training_params.alleles_to_use
    combination_size = model_training_params.combination_size
    aa_labels = model_training_params.aa_labels_training
    TOTAL_SPLITS = len(per_allele_per_kfold_per_length_binders_train[ALLELES[0]].keys())

    per_allele_per_kfold_per_length_binders_train_old = per_allele_per_kfold_per_length_binders_train
    per_allele_per_kfold_per_length_binders_test_old = per_allele_per_kfold_per_length_binders_test

    if aa_labels:
        transform_dicts = None
    per_allele_per_kfold_per_length_binders_train, per_allele_per_kfold_per_length_binders_test = merge_per_allele_sets_into_combination_sets(
        per_allele_per_kfold_per_length_binders_train, per_allele_per_kfold_per_length_binders_test,
        combination_size=combination_size,
        selected_alelles=ALLELES,
        total_splits=TOTAL_SPLITS,
        list_of_transform_dicts=transform_dicts,  # insert here map of amino acid to property
        position_component=model_training_params.position_component
    )
    OLD_ALLELES = ALLELES
    NEW_ALLELES = list(per_allele_per_kfold_per_length_binders_test.keys())
    ALLELES = NEW_ALLELES
    model_training_params.alleles_to_use = NEW_ALLELES
    if combination_size > 1:
        print("Alleles stored in model params are overwritten by new alleles")
    return per_allele_per_kfold_per_length_binders_train, \
        per_allele_per_kfold_per_length_binders_test, \
        ALLELES, \
        per_allele_per_kfold_per_length_binders_train_old, \
        per_allele_per_kfold_per_length_binders_test_old, \
        OLD_ALLELES


def extract_count_df(per_allele_per_kfold_per_length_binders, selected_alleles, total_splits):
    f"""
    support method for calculating weights for input data.
    takes result of {read_data} as input. 
    :return: dataframe with counts for all lengths inside each allele
    """
    per_split_intersected_lengths = dict()
    for split_num in range(total_splits):
        list_of_length_sets = list()
        for old_allele in selected_alleles:
            lengths = set(per_allele_per_kfold_per_length_binders[old_allele][split_num].keys())
            list_of_length_sets.append(lengths)
        intersected_lengths = set.intersection(*list_of_length_sets)
        per_split_intersected_lengths[split_num] = intersected_lengths

    per_kfold_per_length_per_allele_counts = {
        split_num: {
            target_length: {
                allele_name: len(per_allele_per_kfold_per_length_binders[allele_name][split_num][target_length])
                for allele_name in selected_alleles
            } for target_length in per_split_intersected_lengths[split_num]
        } for split_num in range(total_splits)
    }

    result_dict = {}
    for kfold, per_length_per_allele_counts in per_kfold_per_length_per_allele_counts.items():
        for length, per_allele_counts in per_length_per_allele_counts.items():
            for allele_name, count in per_allele_counts.items():
                result_dict[(allele_name, kfold, length)] = {'cnt': count}
    count_df = pd.DataFrame.from_dict(result_dict, orient="index")
    count_df.index.names = ['allele', 'split', 'length']
    return count_df


def calculate_weights_based_on_length_counts(original_peptide_data, experiment_params: ExperimentParams, verbose=False):
    use_weights = experiment_params.model_training_params.use_weights
    OLD_ALLELES = list(original_peptide_data.keys())
    total_splits = len(original_peptide_data[OLD_ALLELES[0]])
    count_df = extract_count_df(original_peptide_data,
                                selected_alleles=OLD_ALLELES,
                                total_splits=total_splits)
    # all possible alleles
    selected_lengths = experiment_params.model_training_params.lengths_to_use
    # here could be merged alleles
    selected_alleles = experiment_params.model_training_params.alleles_to_use

    per_allele_per_kfold_per_length_weights = dict()
    ALLELES = selected_alleles

    for combined_allele_name in ALLELES:
        per_allele_per_kfold_per_length_weights[combined_allele_name] = dict()
        old_alleles = combined_allele_name.split('+')
        max_weight = 0
        for split_num in range(total_splits):
            per_allele_per_kfold_per_length_weights[combined_allele_name][split_num] = dict()
            sub_count_df_train = count_df.loc[(count_df.index.get_level_values('split') == 0) &
                                              (count_df.index.get_level_values('allele').isin(old_alleles)) &
                                              (count_df.index.get_level_values('length').isin(selected_lengths))]
            total_data_length = sub_count_df_train['cnt'].sum()
            if verbose:
                print(f"Total data length for split {split_num} is {total_data_length}")
            for target_length in selected_lengths:
                current_weights_list = list()
                weights_dict = dict()
                for old_allele_name in old_alleles:
                    old_allele_cnt = count_df.loc[(old_allele_name, split_num, target_length)]['cnt']
                    current_weights_list.extend(
                        [total_data_length / old_allele_cnt if use_weights else 1 for i in range(old_allele_cnt)])
                    weights_dict[old_allele_name] = total_data_length / old_allele_cnt if use_weights else 1
                # if weights were in the data
                max_weight = max(max(current_weights_list), max_weight) if len(current_weights_list) > 0 else 1
                per_allele_per_kfold_per_length_weights[combined_allele_name][split_num][
                    target_length] = current_weights_list
                if verbose:
                    print(f"Weights for lentgh {target_length} are {weights_dict}")
            result_weight_dict = dict()
            for target_length in selected_lengths:
                current_weights_list = per_allele_per_kfold_per_length_weights[combined_allele_name][split_num][
                    target_length]
                current_weights_list = [item / max_weight for item in current_weights_list]
                per_allele_per_kfold_per_length_weights[combined_allele_name][split_num][
                    target_length] = current_weights_list
                result_weight_dict[target_length] = np.unique(current_weights_list)
            if verbose:
                print(f"Combination {combined_allele_name}, split {split_num}: {result_weight_dict}")

    return per_allele_per_kfold_per_length_weights


def remove_unused_lengths(per_allele_per_split_per_length_data, experiment_params: ExperimentParams):
    model_training_params: ModelTrainingParams = experiment_params.model_training_params
    TARGET_LENGTHS = model_training_params.lengths_to_use
    print("TARGET_LENGTHS:", TARGET_LENGTHS)
    per_allele_per_split_per_length_new_data = {}

    for allele_name, per_split_dict in per_allele_per_split_per_length_data.items():
        per_allele_per_split_per_length_new_data[allele_name] = {}

        for split_num, per_length_dict in per_split_dict.items():
            per_allele_per_split_per_length_new_data[allele_name][split_num] = {}

            for current_length in TARGET_LENGTHS:
                if current_length in per_length_dict:
                    per_allele_per_split_per_length_new_data[allele_name][split_num][current_length] = per_length_dict[current_length]
                else:
                    per_allele_per_split_per_length_new_data[allele_name][split_num][current_length] = np.empty(0, dtype=str)

    return per_allele_per_split_per_length_new_data