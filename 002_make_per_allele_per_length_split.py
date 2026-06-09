import os
import numpy as np
import pandas as pd
from os import listdir
from os.path import isfile, join
from mhcnames import normalize_allele_name
import mhcnames
import mhcgnomes



# ----------------------------------------
# # 1. Read data
# ----------------------------------------

DATA_PATH_BASE = '/Users/annaklimova/Desktop/Washu_project/data'
EXPERIMENT_NAME = 'simple_model_enrichment'
DATA_PATH = f'{DATA_PATH_BASE}/{EXPERIMENT_NAME}/'

onlyfiles = [f for f in listdir(DATA_PATH) if isfile(join(DATA_PATH, f))]

ALLELES = []
NOT_NORMALIZED = []
per_allele_train_df = dict()

skipped = []

for file_name in onlyfiles:
    # берём только CSV
    if not file_name.lower().endswith(".csv"):
        skipped.append((file_name, "not a .csv"))
        continue

    # файл должен содержать '#'
    if "#" not in file_name:
        skipped.append((file_name, "no # in filename"))
        continue

    # достаём часть после '#', отрезаем расширение
    allele_raw = file_name.split("#", 1)[1].rsplit(".", 1)[0].strip()
    if allele_raw == "":
        skipped.append((file_name, "empty allele after #"))
        continue

    NOT_NORMALIZED.append(allele_raw)

    # нормализация аллеля
    try:
        allele_name = mhcgnomes.parse(allele_raw).to_string()
    except Exception as e:
        skipped.append((file_name, f"mhcgnomes.parse failed: {e}"))
        continue

    # чтение CSV
    try:
        per_allele_train_df[allele_name] = pd.read_csv(join(DATA_PATH, file_name), sep=";")
        ALLELES.append(allele_name)
    except Exception as e:
        skipped.append((file_name, f"read_csv failed: {e}"))
        continue

print(f"Loaded alleles: {len(ALLELES)}")
if skipped:
    print("Skipped files:")
    for fn, reason in skipped:
        print(f"  - {fn}: {reason}")



# ----------------------------------------
# # 2. Make per length split
# ----------------------------------------

from sklearn.model_selection import KFold
MIN_LENGTH = 8
MAX_LENGTH = 30

#leave only binders
def getBindersPeptidesOnly(df):
    result_peptides = df[df['binder'] == 1]['peptide'].values
    return result_peptides
def getNonBindersPeptidesOnly(df):
    result_peptides = df[df['binder'] == 0]['peptide'].values
    return result_peptides

def make_per_length_dict(df, binder_flag):
    lengths = set(df['length'])
    lengths = set([length for length in lengths if length >= MIN_LENGTH and length <= MAX_LENGTH])
    per_length_dict = dict()
    for length in lengths:
        result_peptides = df[(df['binder'] == binder_flag) & (df['length'] == length)]['peptide'].values
        per_length_dict[length] = result_peptides
    return per_length_dict

kf = KFold(n_splits=5, shuffle=True)
def make_per_length_per_KFold_dict(per_length_dict):

    per_length_per_kfold_peptides_train = dict()
    per_length_per_kfold_peptides_test = dict()

    print("Length \t Train/Test for splits", end="")
    for target_length in per_length_dict.keys():
        print(f"\n{target_length}:", end="\t")
        peptides_array = per_length_dict[target_length]
        per_length_per_kfold_peptides_train[target_length] = dict()
        per_length_per_kfold_peptides_test[target_length] = dict()
        if len(peptides_array) > kf.n_splits:
            i = 0
            for train_indexes_b, test_indexes_b in kf.split(peptides_array):
                print("{}/{}\t".format(len(train_indexes_b), len(test_indexes_b)), end="")
                current_train = peptides_array[train_indexes_b]
                current_test = peptides_array[test_indexes_b]
                per_length_per_kfold_peptides_train[target_length][i] = current_train
                per_length_per_kfold_peptides_test[target_length][i] = current_test
                i = i + 1
        else:

            print(f"Not enough peptides of length {target_length}", end="")
    print()
    return per_length_per_kfold_peptides_train , per_length_per_kfold_peptides_test





# def addAuxColumns(per_allele_df):
#     for allele_name, df in per_allele_df.items():
#         df['length'] = df['peptide'].str.len()

per_allele_binders_train = dict()
per_allele_non_binders_train = dict()
per_allele_per_length_binders_train = dict()
per_allele_per_length_non_binders_train = dict()
per_allele_per_length_per_kfold_binders_train = dict()
per_allele_per_length_per_kfold_binders_test = dict()
per_allele_per_length_per_kfold_non_binders_train = dict()
per_allele_per_length_per_kfold_non_binders_test = dict()


for allele_name in ALLELES:
    print(f"Allele {allele_name}")
    df_train = per_allele_train_df[allele_name]
    df_train = df_train.sample(frac=1).reset_index(drop=True)
    per_allele_binders_train[allele_name] = getBindersPeptidesOnly(df_train)
    per_allele_non_binders_train[allele_name] = getNonBindersPeptidesOnly(df_train)
    per_allele_per_length_binders_train[allele_name] = make_per_length_dict(df_train, binder_flag=1)
    per_allele_per_length_non_binders_train[allele_name] = make_per_length_dict(df_train, binder_flag=0)
    print(f"Binders")
    per_allele_per_length_per_kfold_binders_train[allele_name],  per_allele_per_length_per_kfold_binders_test[allele_name] = make_per_length_per_KFold_dict(per_allele_per_length_binders_train[allele_name])
    print(f"Non-binders")
    per_allele_per_length_per_kfold_non_binders_train[allele_name],  per_allele_per_length_per_kfold_non_binders_test[allele_name] = make_per_length_per_KFold_dict(per_allele_per_length_non_binders_train[allele_name])


import seaborn as sns
import matplotlib.pyplot as plt
class1 = ['HLA-A*02:01',
 'HLA-A*03:01',
 'HLA-A*01:01',
 'HLA-A*24:02',
 'HLA-B*27:05',
 'HLA-B*07:02',
 'HLA-B*57:01',
 'HLA-B*15:02',
 'HLA-C*12:02',]

class2 = ['HLA-DRB1*01:01',
          'HLA-DRB1*04:01',
          'HLA-DRB1*07:01',
          'HLA-DRB1*04:04',
          'HLA-DPA1*01:03/DPB1*04:01',
          'HLA-DPA1*02:02/DPB1*05:01','HLA-DQA1*05:01/DQB1*02:01']

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 10))


for cur_ax, allele_name in zip(axes.flatten(), class1):
    df = per_allele_train_df[allele_name]
    ax = sns.countplot(df[df.length< 25], x='length', ax=cur_ax)
    ax.set(title=f'{allele_name}')
    ax.figure.suptitle(f'Peptide lengths for each allele')
plt.tight_layout()
# ax.set(yscale="log")

class2 = ['HLA-DRB1*01:01',
 'HLA-DRB1*04:01',
 'HLA-DRB1*07:01',
 'HLA-DRB1*04:04']

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))


for cur_ax, allele_name in zip(axes.flatten(), class2):
    df = per_allele_train_df[allele_name]
    ax = sns.countplot(df[df.length< 25], x='length', ax=cur_ax)
    ax.set(title=f'{allele_name}')
    ax.figure.suptitle(f'Peptide lengths for each allele')
plt.tight_layout()

import logomaker
LOGO_PATH = DATA_PATH + '/logos/'
if not os.path.exists(LOGO_PATH):
    os.makedirs(LOGO_PATH)

def make_logo_for_data(binders_array, name, save_path=None, ax=None):
    sns.set_theme(style="white")

    matrix = logomaker.alignment_to_matrix(binders_array)
    info_matrix = logomaker.transform_matrix(
        matrix, from_type='counts', to_type='information'
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 3))
    else:
        fig = ax.figure

    logo = logomaker.Logo(
        info_matrix,
        font_name='Arial Rounded MT Bold',
        flip_below=True,
        ax=ax
    )

    ax.set_title(f"{name}, n={len(binders_array)}")

    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    return fig, ax

target_allele_name = class1[0]
TARGET_LENGTHS = [8, 9, 10]

for target_length in TARGET_LENGTHS:
    peptides = per_allele_per_length_binders_train[target_allele_name].get(target_length)

    if peptides is None or len(peptides) == 0:
        continue

    allele_safe = target_allele_name.replace("*", "_").replace(":", "_").replace("/", "_")

    save_path = (
        f"{LOGO_PATH}"
        f"{allele_safe}_binders_length_{target_length}.png"
    )

    make_logo_for_data(
        peptides,
        f"{target_allele_name}, length {target_length}",
        save_path=save_path
    )

    plt.close()

target_allele_name = class2[0]
TARGET_LENGTHS = [14, 15, 16]

for target_length in TARGET_LENGTHS:
    peptides = per_allele_per_length_binders_train[target_allele_name].get(target_length)

    if peptides is None or len(peptides) == 0:
        continue

    allele_safe = target_allele_name.replace("*", "_").replace(":", "_").replace("/", "_")

    save_path = (
        f"{LOGO_PATH}"
        f"{allele_safe}_binders_length_{target_length}.png"
    )

    make_logo_for_data(
        peptides,
        f"{target_allele_name}, length {target_length}",
        save_path=save_path
    )

    plt.close()


# ----------------------------------------
# # 3. Save per_length_per_split to files
# ----------------------------------------

from collections import defaultdict

TARGET_PATH_TO_FILES = DATA_PATH  + '/per_length_per_kfold_split/'
if not os.path.exists(TARGET_PATH_TO_FILES):
    os.makedirs(TARGET_PATH_TO_FILES)

def save_data(per_allele_per_length_per_kfold_data, data_type=None, dataset_type=None):
    assert data_type in ["binders", "nonbinders"]
    assert dataset_type in ["train", "test"]

    per_allele_dfs = defaultdict(list)
    for allele_name in per_allele_per_length_per_kfold_data.keys():
        for length, per_k_fold_dict in per_allele_per_length_per_kfold_data[allele_name].items():
                for split_num, peptides in per_k_fold_dict.items():
                    df = pd.DataFrame({"peptide":peptides, "length":length, "split": split_num})
                    per_allele_dfs[allele_name].append(df)
        total_df= pd.concat(per_allele_dfs[allele_name])
        target_allele_name = mhcgnomes.parse(allele_name).compact_string(include_species=True).replace("*", "_")
        total_df.to_csv(f'{TARGET_PATH_TO_FILES}{target_allele_name}#{data_type}#{dataset_type}.csv',sep=";", index=False)


save_data(per_allele_per_length_per_kfold_binders_train, "binders", "train")
# save_data(per_allele_per_length_per_kfold_non_binders_train, "nonbinders", "train")
save_data(per_allele_per_length_per_kfold_binders_test, "binders", "test")
# save_data(per_allele_per_length_per_kfold_non_binders_test, "nonbinders", "test")

# allele_name

mhcgnomes.parse(mhcgnomes.parse(allele_name).compact_string(include_species=True).replace("*", "-"))





