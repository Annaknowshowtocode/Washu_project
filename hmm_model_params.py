DUMMY_ALLELE_NAME = "dummy_allele"
DataParams = {
    "simple_model_class_2": {
        "ALLELES": sorted([
            # 'HLA-DQA1*05:01/DQB1*02:01'
            # 'HLA-DRB1*01:01'
            DUMMY_ALLELE_NAME
            #  'HLA-DPA1*02:02/DPB1*05:01'
        ]),
        "TARGET_LENGTHS": [12, 13, 14, 15, 16, 17, 18],
        'start_cycle': True,
        'end_cycle': True,
        'intermediate_cycles': False,
        'emission_pseudocount': 0.1,
        'transition_pseudocount': 1,
        'anchor_top_aas': 4,
        'anchor_default_pseudocount': 1,
        'in_cycle_tr_pseudocount': 0,
        'cycle_top_aas': 20,
        'cycle_default_pseudocount': 0.1
    },
    "class1": {
        "ALLELES": sorted(['HLA-A*01:01', 'HLA-A*03:01']),
        "TARGET_LENGTHS": [8, 9, 10, 11, 12],
        'start_cycle': False,
        'end_cycle': False,
        'intermediate_cycles': True,
        'emission_pseudocount': 50,
        'transition_pseudocount': 0,
        'cycle_top_aas': 20,
        'cycle_default_pseudocount': 0.1,
        'anchor_top_aas': 3,
        'anchor_default_pseudocount': 50
    },
    "class2": {
        "ALLELES": sorted([
            'HLA-DQA1*05:01/DQB1*02:01',
            #  'HLA-DRB1*01:01'
            # 'HLA-DPA1*02:02/DPB1*05:01'
        ]),
        "TARGET_LENGTHS": [12, 13, 14, 15, 16, 17, 18],
        'start_cycle': True,
        'end_cycle': True,
        'intermediate_cycles': False,
        'emission_pseudocount': 50,
        'transition_pseudocount': 50,

        'anchor_top_aas': 8,
        'anchor_default_pseudocount': 50,
        'in_cycle_tr_pseudocount': 300,
        'cycle_top_aas': 20
    },
    "sucessfull_MixMHCGA": {
        "ALLELES": sorted([
            # 'HLA-DQA1*05:01/DQB1*02:01'
            # 'HLA-DRB1*01:01'
            DUMMY_ALLELE_NAME
            #  'HLA-DPA1*02:02/DPB1*05:01'
        ]),
        #  "ALLELES":sorted(['HLA-DPA1*02:02/DPB1*05:01']),
        "TARGET_LENGTHS": [12, 13, 14, 15, 16, 17, 18],
        'start_cycle': True,
        'end_cycle': True,
        'intermediate_cycles': False,
        'emission_pseudocount': 0.1,
        'transition_pseudocount': 1,
        'anchor_top_aas': 4,
        'anchor_default_pseudocount': 1,
        'in_cycle_tr_pseudocount': 0,
        'cycle_top_aas': 20,
        'cycle_default_pseudocount': 0.1,
    },
    "class2_batches": {
        "ALLELES": sorted([
            # 'HLA-DQA1*05:01/DQB1*02:01'
            # 'HLA-DRB1*01:01'
            DUMMY_ALLELE_NAME
            #  'HLA-DPA1*02:02/DPB1*05:01'
        ]),
        #  "ALLELES":sorted(['HLA-DPA1*02:02/DPB1*05:01']),
        "TARGET_LENGTHS": [12, 13, 14, 15, 16, 17, 18],
        'start_cycle': True,
        'end_cycle': True,
        'intermediate_cycles': False,
        'emission_pseudocount': 0.1,
        'transition_pseudocount': 1,
        'anchor_top_aas': 4,
        'anchor_default_pseudocount': 1,
        'in_cycle_tr_pseudocount': 0,
        'cycle_top_aas': 20,
        'cycle_default_pseudocount': 0.1,
    },
    "class2_batches_but_more_states_per_group": {
        "ALLELES": sorted([
            # 'HLA-DQA1*05:01/DQB1*02:01'
            # 'HLA-DRB1*01:01'
            DUMMY_ALLELE_NAME
            #  'HLA-DPA1*02:02/DPB1*05:01'
        ]),
        #  "ALLELES":sorted(['HLA-DPA1*02:02/DPB1*05:01']),
        "TARGET_LENGTHS": [12, 13, 14, 15, 16, 17, 18],
        'start_cycle': True,
        'end_cycle': True,
        'intermediate_cycles': False,
        'emission_pseudocount': 1,
        'transition_pseudocount': 0,
        'anchor_top_aas': 4,
        'anchor_default_pseudocount': 1,
        'in_cycle_tr_pseudocount': 0,
        'cycle_top_aas': 20,
        'cycle_default_pseudocount': 0.1,
    },

    "anchors_for_real_data_should_be_on_place": {
        "ALLELES": sorted([
            # 'HLA-DQA1*05:01/DQB1*02:01'
            # 'HLA-DRB1*01:01'
            DUMMY_ALLELE_NAME
            #  'HLA-DPA1*02:02/DPB1*05:01'
        ]),
        #  "ALLELES":sorted(['HLA-DPA1*02:02/DPB1*05:01']),
        "TARGET_LENGTHS": [12, 13, 14, 15, 16, 17, 18],
        'start_cycle': True,
        'end_cycle': True,
        'intermediate_cycles': False,
        'emission_pseudocount': 0.5,
        'transition_pseudocount': 0,
        'anchor_top_aas': 4,
        'anchor_default_pseudocount': 0.5,
        'in_cycle_tr_pseudocount': 0,
        'cycle_top_aas': 20,
        'cycle_default_pseudocount': 0.1,
    },
    "class2_batches_complex": {
        "ALLELES": sorted([
            'HLA-DRB1*01:01'
            # ,'HLA-DQA1*05:01/DQB1*02:01'
        ]),
        #  "ALLELES":sorted(['HLA-DPA1*02:02/DPB1*05:01']),
        "TARGET_LENGTHS": [12, 13, 14, 15, 16, 17, 18],
        'start_cycle': True,
        'end_cycle': True,
        'intermediate_cycles': False,
        'emission_pseudocount': 0.5,
        'transition_pseudocount': 1,
        'anchor_top_aas': 5,
        'anchor_default_pseudocount': 0.5,
        'in_cycle_tr_pseudocount': 0,
        'cycle_top_aas': 20,
        'cycle_default_pseudocount': 0.01,
    },
    "class2_big_batches": {
        "ALLELES": sorted(['HLA-DQA1*05:01/DQB1*02:01']),
        #  "ALLELES":sorted(['HLA-DPA1*02:02/DPB1*05:01']),
        "TARGET_LENGTHS": [12, 13, 14, 15, 16, 17, 18],
        'start_cycle': True,
        'end_cycle': True,
        'intermediate_cycles': False,
        'emission_pseudocount': 10,
        'transition_pseudocount': 1,
        'anchor_top_aas': 4,
        'anchor_default_pseudocount': 0.1,
        'in_cycle_tr_pseudocount': 0,
        'cycle_top_aas': 20,
        'cycle_default_pseudocount': 0.01,
    }
}

ModelTrainingParams = {
    "simple_linear_model": {

    },
    "two_states_per_group": {
        'groups' : 7,
        'states_per_group' : 2,
        'st' : 14,

    }
}




params['st'] = 14
params['alg'] = algorithm
params['min_iters'] = 50
params['max_iters'] = 80
params['emission_pseudocount'] = DataParams[target_class]["emission_pseudocount"]
params['transition_pseudocount'] = DataParams[target_class]["transition_pseudocount"]
params['verbose'] = True
params['lr_decay'] = 0.0
params['edge_inertia'] = 0.4
params['distribution_inertia'] = 0.4
params['groups'] = 7
params['states_per_group'] = 2
params['model_complexity'] = 2
params['use_pseudocount'] = True

# Class 1 params
params['self_cycle'] = False
params['start_cycle'] = DataParams[target_class]["start_cycle"]
params['end_cycle'] = DataParams[target_class]["end_cycle"]
params['equal_probs_for_cycle'] = False
params['cycle_top_aas'] = DataParams[target_class]["cycle_top_aas"]
params['cycle_default_pseudocount'] = DataParams[target_class]["cycle_default_pseudocount"]

params['freeze_distr'] = False  # wether to freeze start and end cycle distributions
params['tie_cycle_states'] = False
params['in_cycle_tr_pseudocount'] = DataParams[target_class]["in_cycle_tr_pseudocount"]
params['anchor_top_aas'] = DataParams[target_class]["anchor_top_aas"]
params['anchor_states'] = True
params['anchor_default_pseudocount'] = DataParams[target_class]["anchor_default_pseudocount"]
params['tie_anchor_states'] = False
params['intermediate_cycles'] = DataParams[target_class]["intermediate_cycles"]

params['equal_probs_for_intermediate_cycle'] = False
params['tie_intermediate_cycles'] = False
params['freeze_int_cycle_distr'] = False
params['position_component'] = False
params['cycle_chain'] = False

params['minibatch_training'] = False

params['batches_per_epoch'] = 100
params['batch_size'] = 1

params['multiple_check_input'] = False
ALLELES = DataParams[target_class]["ALLELES"]
TARGET_LENGTHS = DataParams[target_class]["TARGET_LENGTHS"]

assert params['st'] % params['groups'] == 0
assert params['st'] % params['states_per_group'] == 0
assert params['st'] == params['groups'] * params['states_per_group']