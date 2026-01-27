import param
from peptides_utils import defineClass


class DataScenarioParams(param.Parameterized):
    data_scenario = param.Selector(objects=["preprocessed_IEDB"
                                            ])
    data_path_base = param.Foldername('/Users/annaklimova/Desktop/Washu_project/',
                                      check_exists=True)
    input_data_path = param.Foldername(check_exists = True)
    splits_to_read = param.Integer(default=1)
    extra = param.Dict(dict())


    def update_input_data_path(self, experiment_name):
        pass


class PreprocessedIEDBDataParams(DataScenarioParams):
    data_scenario = "preprocessed_IEDB"

    def update_input_data_path(self, experiment_name):
        self.input_data_path = f"{self.data_path_base}/{experiment_name}/per_length_per_kfold_split/"


class ModelTrainingParams(param.Parameterized):
    algorithm = param.Selector(default="viterbi", objects=["viterbi", "baum_welch"])
    num_runs = param.Integer(default=50, bounds=(1, None))
    combination_size = param.Integer(default=1, bounds=(1, None))

    alleles_to_use = param.List(item_type=str)
    lengths_to_use = param.List(default=[12, 13, 14, 15, 16,17, 18], item_type=int)

    # Modifications_in_architecture
    groups = param.Integer(default=7, bounds=(1, None))
    states_per_group = param.Integer(default=1, bounds=(1, None))
    st_num = param.Integer()
    model_complexity = param.Integer(default=1, bounds=(1, None))
    start_cycle = param.Boolean(default=True, doc="include start cycle state")
    end_cycle = param.Boolean(default=True, doc="include end cycle state")
    self_cycle = param.Boolean(default=False, doc="Make self transitions for all states")
    intermediate_cycle = param.Boolean(default=False, doc="include intermediate cycle")
    anchor_states = param.Boolean(default=True, doc="wether special anchor distribution in 'joiner' state")
    cycle_chain = param.Boolean(default=False, doc="replace cycles with set of chained distributions")
    cycle_chain_length = param.Integer(default=10, bounds=(1, None))

    # Modifications in distributions
    # turn off/on
    position_component = param.Boolean(default=False,
                                       doc="Experimental: wether to add additional position component as separate distribution for state. You will need provide numbers as weel with labels")
    freeze_start_end_cycle = param.Boolean(default=False, doc="wether to freeze start and end cycle distributions")
    freeze_intermediate_cycle = param.Boolean(default=False, doc="wether to freeze intermediate cycle distributions")
    tie_start_end_cycle_states = param.Boolean(default=False, doc="wether to tie together Start/End cycle states (set the same disribution)")
    tie_intermediate_cycle_states = param.Boolean(default=False, doc="wether to tie together intermediate cycle states (set the same disribution)")
    tie_anchor_states = param.Boolean(default=False, doc="wether to tie anchor states together (set the same disribution)")
    ## numbers
    emission_pseudocount = param.Number(default=0.1)
    transition_pseudocount = param.Number(default=1)
    anchor_default_pseudocount = param.Number(default=1)
    cycle_default_pseudocount = param.Number(default=0.1)
    in_cycle_tr_pseudocount = param.Number(default=0)
    anchor_top_aas = param.Integer(default=4, bounds=(1, 20))
    cycle_top_aas = param.Integer(default=20, bounds=(1, 20))
    equal_probs_for_cycle = param.Boolean(default=False)
    equal_probs_for_intermediate_cycle = param.Boolean(default = False)

    # Modifications in the training process
    lr_decay = param.Number(default=0.0, bounds=(0.0, 1))
    edge_inertia = param.Number(default=0.4, bounds=(0.0, 1))
    distribution_inertia = param.Number(default=0.4, bounds=(0.0, 1))
    min_iters = param.Integer(default=40, bounds=(1, None))
    maxiters = param.Integer(default=80, bounds=(1, None))
    use_pseudocounts = param.Boolean(default=True)
    use_weights = param.Boolean(default=False)
    minibatch_training = param.Boolean(default=True)
    batches_per_epoch = param.Integer(bounds=(1, None), default=100)
    batch_size = param.Integer(bounds=(1, None), default=1)
    aa_labels_training = param.Boolean(default=True,
                                       doc="Experimental: wether to keep amino acid labels or convert them to property values. Will change distributions completelly")
    stop_threshold = 2.5

    decrease_anchor_aas_steps = param.Integer(default=1, bounds=(1,20))

    # Splits
    trash_model_information_threshold = param.Number(default=0.35)
    same_model_distance_threshold = param.Number(default=0.05)
    data_split_strategy = param.Selector(default="median_score_of_best_models", objects=["best_model", "median_score_of_best_models", "single_average_model"])
    enrichment_steps = param.Integer(default=3, bounds=(1, None))
    split_decision_models_fraction = param.Number(default=0.4)
    enrichment_split_decision_models_fraction = param.Number(default=0.3)
    reassignment_decision_models_fraction = param.Number(default=0.4)


    statistics_estimate_model_fraction = param.Number(default=0.3)


    check_model_integrity = param.Boolean(default=False, doc="check for shifts")


    # Aux params
    verbose = param.Boolean(default=False)
    multiple_check_inpit = param.Boolean(default=True, doc="legacy param from pomegranate. needs to be true for now, but probably can speedup without it")
    initial_params = param.List(default=None)


    @param.depends('groups', 'states_per_group', watch=True, on_init=True)
    def _update_states_number(self):
        self.st_num = self.groups * self.states_per_group

    @param.depends('lengths_to_use', watch=True, on_init=True)
    def _update_stop_threshold_based_on_lengths(self):
        stop_criteria_for_lengths = [length * 0.2 for length in self.lengths_to_use]
        self.stop_threshold = sum(stop_criteria_for_lengths) / len(stop_criteria_for_lengths)

    def get_model_common_names(self):
        selected_params_dict = {
            "st": self.st_num,
            "alg": "v" if self.algorithm =="viterbi" else  self.algorithm,
            "min_iters": self.min_iters
        }
        params_string = "-".join([f"{value}-{key}" for key, value in selected_params_dict.items()])
        lengths_string = f"{min(self.lengths_to_use)}-{max(self.lengths_to_use)}"
        return "-".join([params_string, lengths_string])


class SimpleModelClassIIParams(ModelTrainingParams):
    #Data related
    lengths_to_use = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    #Model related
    start_cycle = True
    end_cycle = True
    intermediate_cycle = False
    emission_pseudocount = 0.1
    transition_pseudocount = 1
    anchor_top_aas = 4
    anchor_default_pseudocount = 1
    decrease_anchor_aas_steps = 1

    groups = 7
    states_per_group = 1
    model_complexity = 1
    edge_inertia = 0.4
    distribution_inertia = 0.4
    batches_per_epoch = 100
    batch_size = 1




    def __init__(self, **params):
        super().__init__(**params)

    @param.depends('alleles_to_use', watch=True, on_init=True)
    def _update_alleles(self):
        init_length = len(self.alleles_to_use)
        self.alleles_to_use = list(sorted([allele_name for allele_name in self.alleles_to_use if defineClass(allele_name) == 'II']))
        current_length = len(self.alleles_to_use)
        if init_length != current_length:
            print(f"{init_length-current_length} of alleles were removed since they are not class II alleles")

# class SimpleModelClassIIParamsMixMHC(SimpleModelClassIIParams):
#     enrichment_steps = 3
#     anchor_top_aas = 6
#     data_split_strategy = "median_score_of_best_models"
#     split_decision_models_fraction = 0.2
#     enrichment_split_decision_models_fraction = 0.2
#     statistics_estimate_model_fraction = 0.2
#
#
#     trash_model_information_threshold = 0.55
#     same_model_distance_threshold = 0.08
#     reassignment_decision_models_fraction = 0.2
#     def __init__(self, **params):
#         super().__init__(**params)


# class SimpleModelClassIIParamsMixMHCsmall(SimpleModelClassIIParams):
#     enrichment_steps = 3
#     anchor_top_aas = 4
#     data_split_strategy = "median_score_of_best_models"
#     split_decision_models_fraction = 0.2
#     enrichment_split_decision_models_fraction = 0.2
#     statistics_estimate_model_fraction = 0.2
#     emission_pseudocount = 0.1
#
#
#     trash_model_information_threshold = 0.55
#     same_model_distance_threshold = 0.08
#     reassignment_decision_models_fraction = 0.2
#     def __init__(self, **params):
#         super().__init__(**params)




class ComplexModelClassIIParams(ModelTrainingParams):
    #Data related
    lengths_to_use = [12, 13, 14, 15, 16, 17, 18, 19]
    #Model related
    start_cycle = True
    end_cycle = True
    intermediate_cycle = False
    emission_pseudocount = 0.1
    transition_pseudocount = 1
    anchor_top_aas = 4
    anchor_default_pseudocount = 1

    groups = 7
    states_per_group = 2
    model_complexity = 2
    edge_inertia = 0.4
    distribution_inertia = 0.4
    batches_per_epoch = 100
    batch_size = 1

    @param.depends('alleles_to_use', watch=True, on_init=True)
    def _update_alleles(self):
        init_length = len(self.alleles_to_use)
        self.alleles_to_use = [allele_name for allele_name in self.alleles_to_use if defineClass(allele_name) == 'II']
        current_length = len(self.alleles_to_use)
        if init_length != current_length:
            print(f"{init_length-current_length} of alleles were removed since they are not class II alleles")





class ExperimentParams(param.Parameterized):
    experiment_name = param.String()
    data_scenario_params = param.ClassSelector(default=PreprocessedIEDBDataParams(), class_=DataScenarioParams, is_instance=True, instantiate=True)
    #Data general params
    model_training_params = param.ClassSelector(default=SimpleModelClassIIParams(), class_=ModelTrainingParams, is_instance=True, instantiate=True)
    experiment_result_data_path = param.Foldername(check_exists=False)

    @param.depends('experiment_name', 'data_scenario_params', watch=True, on_init=True)
    def _update_experiment_path(self):
        self.data_scenario_params.update_input_data_path(experiment_name=self.experiment_name)
        self.experiment_result_data_path = f"experiment_results/{self.experiment_name}/{self.data_scenario_params.data_scenario}/"

