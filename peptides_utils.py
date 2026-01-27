import logomaker
import seaborn as sns
import mhcgnomes
from collections import defaultdict
import pandas as pd
import numpy as np

def defineClass(allele_name):
    if allele_name == 'dummy_allele':
        return 'II'
    if 'dummy_allele' in allele_name:
        return 'II'
    if allele_name.startswith('decoy'):
        return 'II'
    if allele_name.startswith("d_"):
        return 'II'
    parse_result = mhcgnomes.parse(allele_name)
    if parse_result.has_mhc_class:
        if parse_result.is_class1:
            return 'I'
        elif parse_result.is_class2:
            return 'II'
        else:
            print("Not class I or class II???!")
    else:
        print("Result without class")
    return None


def defineOrganism(allele_name):
    parse_result = mhcgnomes.parse(allele_name)
    if parse_result.is_mouse:
        return "Mouse"
    elif parse_result.is_human:
        return "Human"
    else:
        print("organism not found")
        return None

def make_logo_for_data(binders_array, name, ax=None):
    sns.set_theme(style="white")
    matrix = logomaker.alignment_to_matrix(binders_array)
    info_matrix = logomaker.transform_matrix(matrix, from_type='counts', to_type='information')
    prob_matrix = logomaker.transform_matrix(matrix, from_type='counts', to_type='probability')
    ww_logo = logomaker.Logo(info_matrix, font_name='DejaVu Sans', flip_below=True, ax=ax)
    ww_logo.ax.set_title(f"{name}, {len(binders_array)}")
    return ww_logo


def split_to_dicts(per_allele_df, ALLELES, TARGET_LENGTHS, TOTAL_SPLITS):
    per_allele_per_kfold_per_length_peptides = dict()
    for allele_name in ALLELES:
        per_allele_per_kfold_per_length_peptides[allele_name] = dict()
        total_df = per_allele_df[allele_name]
        for split_num in TOTAL_SPLITS:
            per_allele_per_kfold_per_length_peptides[allele_name][split_num] = dict()
            for length in TARGET_LENGTHS:
                per_allele_per_kfold_per_length_peptides[allele_name][split_num][length] = \
                    total_df[(total_df['length'] == length) &
                             (total_df['split'] == split_num)]['peptide'].values
    return per_allele_per_kfold_per_length_peptides

def join_dicts(per_allele_per_split_per_length_data):
    per_allele_list_of_dfs = defaultdict(list)
    for allele_name in per_allele_per_split_per_length_data.keys():
        for current_split in per_allele_per_split_per_length_data[allele_name].keys():
            for current_length in per_allele_per_split_per_length_data[allele_name][current_split].keys():
                peptides = per_allele_per_split_per_length_data[allele_name][current_split][current_length]
                df = pd.DataFrame({'peptide': peptides,'split': current_split, 'length':current_length})
                per_allele_list_of_dfs[allele_name].append(df)

    per_allele_data_df = dict()
    for allele_name in per_allele_per_split_per_length_data.keys():
        per_allele_data_df[allele_name] = pd.concat(per_allele_list_of_dfs[allele_name])
    return per_allele_data_df

def get_frequencies(number_of_positions):
    nat_freqs = {"A": 0.066470, "C": 0.021098, "D": 0.047650, "E": 0.070905, "F": 0.035178, "G": 0.062337, "H": 0.025244, \
           "I": 0.043243, "K": 0.057427, "L": 0.096161, "M": 0.048262, "N": 0.035963, "P": 0.059660, "Q": 0.046936, \
           "R": 0.054387, "S": 0.080709, "T": 0.052072, "V": 0.058573, "W": 0.011643, "Y": 0.026081}
    original_vector = list(nat_freqs.values())
    result_values = np.repeat(np.array([original_vector]), number_of_positions, axis=0)
    return pd.DataFrame(result_values, columns=list(nat_freqs.keys()))





