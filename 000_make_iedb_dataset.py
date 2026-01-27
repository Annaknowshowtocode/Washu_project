import numpy as np
import pandas as pd
from mhcgnomes import parse
import mhcnames
import os
import dask.dataframe as dd

# ----------------------------------------
# # 1 Read total data
# ----------------------------------------

DATA_PATH = '/Users/annaklimova/Desktop/Washu_project/data'
EXPERIMENT_NAME = 'simple_model_enrichment'
DATA_PATH_BASE = '/Users/annaklimova/Desktop/Washu_project/'
DATA_PATH = f'{DATA_PATH_BASE}'

IEDB_DATA_PATH = DATA_PATH + 'data/mhc_ligand_full.csv'





headers = pd.read_csv(IEDB_DATA_PATH, sep=',', header=None, nrows=2)
mi = pd.MultiIndex.from_frame(headers.T)


usecols = [
    ('Epitope', 'Species'),
    ('Assay', 'Qualitative Measurement'),
    ('Epitope', 'Name'),
    ('Epitope', 'Reference Name'),
    ('Epitope', 'Object Type'),
    ('Assay', 'Units'),
    ('Assay', 'Quantitative measurement'),
    ('Assay', 'Measurement Inequality'),
    ('MHC Restriction', 'Name'),
    ('MHC Restriction', 'IRI'),
    ('MHC Restriction', 'Class'),
    ('Assay', 'Response measured'),
    ('Assay', 'Method'),
    ('Epitope', 'Molecule Parent IRI'),
    ('Epitope', 'Starting Position'),
    ('Epitope', 'Ending Position')
]


desired_index_values = [item for item in mi if item in usecols]
desired_index_indexes = [col_index  for col_index, col in enumerate(mi) if col in usecols]
dat_names = list(map(lambda x: f"{x[0]}_{x[1]}", desired_index_values ))
iedb_df = pd.read_csv(IEDB_DATA_PATH, sep=',', skiprows=2,header=None, usecols=desired_index_indexes, names=desired_index_values)

# Dataframes implement the Pandas API
df = dd.read_csv(IEDB_DATA_PATH, sep=',', skiprows=2,header=None, usecols=desired_index_indexes, names=desired_index_values)

chunks=pd.read_table(IEDB_DATA_PATH,chunksize=400000,sep=',',usecols=desired_index_indexes,
       names=desired_index_values,skiprows=2,header=None, dtype={12:str, 92:str, 95:str})

df=pd.DataFrame()
df=pd.concat(chunk for chunk in chunks)



def merge_column_name(column):
    return "_".join(column)



def filter_iedb_dataset(iedb_df, include_qualitative=True, include_mass_spec=True):

    QUALITATIVE_TO_AFFINITY_AND_INEQUALITY = {
    "Negative": (5000.0, ">"),
    "Positive": (500.0, "<"),  # used for mass-spec hits
    "Positive-High": (100.0, "<"),
    "Positive-Intermediate": (1000.0, "<"),
    "Positive-Low": (5000.0, "<"),
    }
    QUALITATIVE_TO_AFFINITY = dict(
        (key, value[0]) for (key, value)
        in QUALITATIVE_TO_AFFINITY_AND_INEQUALITY.items())
    QUALITATIVE_TO_INEQUALITY = dict(
        (key, value[1]) for (key, value)
        in QUALITATIVE_TO_AFFINITY_AND_INEQUALITY.items())



    print("Loaded iedb data: %s" % str(iedb_df.shape))
    filtered_df = iedb_df.copy()
    filtered_df = filtered_df[filtered_df[('Epitope', 'Object Type')] == "Linear peptide"]

    print("Linear peptides only, remaining: %d" % len(filtered_df))

    ## Clear bad data
    print("Subselecting to valid peptides (dropping peptides with PTMs). Starting with: %d" % len(filtered_df))
    filtered_df[('Epitope', 'Name')] = filtered_df[('Epitope', 'Name')].str.strip()
    filtered_df = filtered_df.loc[
        filtered_df[('Epitope', 'Name')].str.match("^[ACDEFGHIKLMNPQRSTVWY]+$")
    ]
    filtered_df = filtered_df.reset_index(drop=True)
    print("Now: %d" % len(filtered_df))


#     filtered_df = filtered_df[~filtered_df[('Epitope', 'Description')].str.contains("\+")]
#     filtered_df = filtered_df[~filtered_df[('Epitope', 'Description')].str.contains("X")]
#     filtered_df = filtered_df[~filtered_df[('Epitope', 'Description')].str.contains("B")]
#     filtered_df = filtered_df[~filtered_df[('Epitope', 'Description')].str.contains("Z")]
    filtered_df = filtered_df[[
    ('Epitope', 'Object Type'),
    ('Epitope', 'Name'),
    ('Epitope', 'Species'),
    ('Assay', 'Units'),
    ('Assay', 'Qualitative Measurement'),
    ('Assay', 'Quantitative measurement'),
     ('Assay', 'Measurement Inequality'),
    ('MHC Restriction', 'Name'),
    ('MHC Restriction', 'IRI'),
    ('MHC Restriction', 'Class'),
    ('Assay', 'Response measured'),
    ('Assay', 'Method'),
    ('Epitope', 'Molecule Parent IRI'),
    ('Epitope', 'Starting Position'),
    ('Epitope', 'Ending Position')
             ]]
    print("Subselecting assay groups as in Sarkizova. Starting with: %d" % len(filtered_df))

    assay_type1 = (filtered_df[('Assay', 'Response measured')] == 'half maximal effective concentration (EC50)') & \
                                              (filtered_df[('Assay', 'Method')] == 'purified MHC/direct/fluorescence')
    filtered_df = filtered_df[~assay_type1]
    assay_type2 = (filtered_df[('Assay', 'Response measured')] =='dissociation constant KD') & \
                                                  (filtered_df[('Assay', 'Method')] == 'purified MHC/direct/radioactivity')
    filtered_df = filtered_df[~assay_type2]
    assay_type3 = (filtered_df[('Assay', 'Response measured')] == 'half maximal effective concentration (EC50)') & \
                                                  (filtered_df[('Assay', 'Method')] == 'cellular MHC/direct/fluorescence')
    filtered_df = filtered_df[~assay_type3]
    filtered_df.columns = filtered_df.columns.map(merge_column_name)
    filtered_df = filtered_df.reset_index(drop=True)
    print("Now: %d" % len(filtered_df))


    quantitative = filtered_df.loc[filtered_df["Assay_Units"] == "nM"].copy()
    quantitative["measurement_type"] = "quantitative"
    quantitative = quantitative.rename(columns={"Assay_Measurement Inequality":"measurement_inequality"})
    quantitative['measurement_inequality'] = quantitative['measurement_inequality'].fillna("=").map(lambda s: {">=": ">", "<=": "<"}.get(s, s))
    print("Quantitative measurements: %d" % len(quantitative))
    # Ilya: remove quantitative measurement with no measurement values
    print("Dropping quantitative measurements with no measurement value")
    quantitative.dropna(subset = ["Assay_Quantitative measurement"], inplace=True)
    print("Quantitative measurements: %d" % len(quantitative))

    qualitative = filtered_df.loc[filtered_df["Assay_Units"] != "nM"].copy()
    qualitative["measurement_type"] = "qualitative"
    print("Qualitative measurements: %d" % len(qualitative))
    if not include_mass_spec:
        qualitative = qualitative.loc[
            (~qualitative["Assay_Method"].str.contains("mass spec"))
        ].copy()

    qualitative["Assay_Quantitative measurement"] = (
        qualitative["Assay_Qualitative Measurement"].map(QUALITATIVE_TO_AFFINITY))
    qualitative = qualitative.drop(columns=['Assay_Measurement Inequality'])
    qualitative["measurement_inequality"] = (qualitative["Assay_Qualitative Measurement"].map(QUALITATIVE_TO_INEQUALITY))
    print("Qualitative measurements (possibly after dropping MS): %d" % (
        len(qualitative)))

    filtered_df = pd.concat(
        (
            ([quantitative]) +
            ([qualitative] if include_qualitative else [])),
        ignore_index=True)
    print("Now: %d" % len(filtered_df))

    ## Mark binders
    binder_list = ['Positive', 'Positive-High', 'Positive-Intermediate', 'Positive-Low']
    non_binder_list = ['Negative']
    filtered_df.loc[filtered_df['Assay_Quantitative measurement'] <= 500, 'binder'] = 1
    filtered_df.loc[filtered_df['Assay_Quantitative measurement'] > 500, 'binder'] = 0
    #
    # filtered_df.loc[filtered_df['Qualitative Measure'].isin(binder_list),'binder'] = 1
    # filtered_df.loc[filtered_df['Qualitative Measure'].isin(non_binder_list),'binder'] = 0

    filtered_df.binder = filtered_df.binder.astype(int)
    filtered_df = filtered_df.drop(columns=[
     # 'Qualitative Measure',
     'Assay_Units',
     'Epitope_Object Type',
     'MHC Restriction_IRI'
     ])

    filtered_df = filtered_df.rename(columns={"MHC Restriction_Name":"allele",
                                              "Epitope_Name":"peptide",
                                              "Assay_Quantitative measurement":"measurement_value",
                                              'Assay_Qualitative Measurement': "qualitative_measurement",
                                              "Epitope_Molecule Parent IRI":"protein_id",
                                              "Epitope_Starting Position":"peptide_start",
                                              "Epitope_Ending Position": "peptide_end"})
    filtered_df['length'] = filtered_df['peptide'].str.len()

    #filtered_df['allele'] = filtered_df['original_allele'].map(parse).to_string()

    print("Take unique (peptide, allele, length, 'measurement_value', 'measurement_inequality','qualitative_measurement' 'measurement_type', binder). Starting with: %d" % len(filtered_df))

    filtered_df = filtered_df.drop_duplicates([
    'peptide',
    'allele',
    'length',
    'measurement_value',
    'measurement_inequality',
    'measurement_type',
    'qualitative_measurement',
    'binder',
    'protein_id',
    'peptide_start',
    'peptide_end']).reset_index(drop=True)
    print("Now: %d" % len(filtered_df))


    print("Allele_distribution")
    print(filtered_df['allele'].value_counts()[:30])



    assert 'peptide' in filtered_df.columns
    assert 'allele' in filtered_df.columns
    assert 'length' in filtered_df.columns
    assert 'binder' in filtered_df.columns
    return filtered_df

#test_df = filter_iedb_dataset(iedb_df)
iedb_df = filter_iedb_dataset(iedb_df)


iedb_df.allele.value_counts()[:40]

iedb_df[iedb_df.allele.str.contains('Q')].allele

ALLELES = [
    # 'H2-IAb',
    # 'H2-IAd',
    # 'H2-IAk',
    # 'H2-IAq',
    # 'H2-IAs',
    # 'H2-IAu',
    # 'H2-IEd',
    # 'H2-IAk',
     'H2-Kb',
     'H2-Db',
    'HLA-A*02:01',
    'HLA-A*03:01',
    'HLA-A*01:01',
    'HLA-A*24:02',
    'HLA-B*27:05',
    'HLA-B*07:02',
    'HLA-B*57:01',
    'HLA-B*15:02',
    'HLA-C*12:02',
    #'HLA-A*24:02',
    #'HLA-A*11:01',
    'HLA-DRB1*01:01',
    'HLA-DRB1*04:01',
    'HLA-DRB1*07:01',
    'HLA-DRB1*04:04',
    'HLA-DRB1*03:01',
    #'HLA-DRB1*04:01',
    #'HLA-DRB1*15:01',
    #'HLA-DRB3*01:01',
    #'HLA-DRB5*01:01',
    'HLA-DPA1*01:03/DPB1*04:01',
    'HLA-DPA1*02:02/DPB1*05:01'
]



target_df = iedb_df[iedb_df.allele.isin(ALLELES)]

target_df.allele.value_counts(dropna=False)

target_df.binder.value_counts(dropna=False)


# ----------------------------------------
# # Make per allele DF
# ----------------------------------------

def make_per_allele_df(source_df, ALLELES):
    per_allele_df = dict()
    for allele_name in ALLELES:
        allele_df = source_df[source_df['allele'] == allele_name]
        print(f"------Allele {allele_name} ----------")
        print("Binders_distribution:")
        print(allele_df['binder'].value_counts())
        print("Length_distribution:")
        print(allele_df['length'].value_counts().iloc[:10])
        per_allele_df[allele_name] = allele_df
    return per_allele_df


per_allele_df = make_per_allele_df(iedb_df, ALLELES)

df = iedb_df[iedb_df.allele.isin(ALLELES)]

per_allele_df[ALLELES[0]].length.value_counts()

import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(12,6))
#target_df = df[df['binder'] == 1]
ax = sns.barplot(data=df.groupby(['binder','allele'], as_index=False)['peptide'].count(), x='allele', y='peptide', hue='binder', orient='vertical')
for container in ax.containers:
    ax.bar_label(container)
#sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
for item in ax.get_xticklabels():
    item.set_rotation(45)

ax.figure.suptitle('Dataset size for each Allele')


# ----------------------------------------
# #  Save per allele DF
# ----------------------------------------

allele_name = "HLA-DRB1*0101"

import mhcnames
import mhcgnomes



TARGET_PATH_TO_FILES = DATA_PATH + EXPERIMENT_NAME + '/'


if not os.path.exists(TARGET_PATH_TO_FILES ):
    os.makedirs(TARGET_PATH_TO_FILES )


for allele_name in ALLELES:
    total_df = per_allele_df[allele_name]
    compact_name = mhcgnomes.parse(allele_name).compact_string(include_species=True).replace("*", "_")
    print(compact_name)

    #mhcnames.compact_allele_name(allele_name,)
    # allele_name = allele_name.replace('-', '_').replace('*', '_').replace(':', '_')
    total_df.to_csv("{}data#{}.csv".format(TARGET_PATH_TO_FILES,compact_name),sep=";", index=False)


# ----------------------------------------

# ----------------------------------------

print(total_df['binder'].value_counts())



