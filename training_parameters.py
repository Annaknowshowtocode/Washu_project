import param
from peptides_utils import defineClass


class DataScenarioParams(param.Parameterized):
    data_scenario = param.Selector(objects=["simulated",
                                            "simulated_preprocessed",
                                            "MixMHCpred",
                                            "preprocessed_IEDB"])
    data_path_base = param.Foldername('/Users/annaklimova/Desktop/Washu_project/data',
                                      check_exists=False)
    input_data_path = param.Foldername(check_exists = False)
    splits_to_read = param.Integer(default=1)
    extra = param.Dict(dict())

    def update_input_data_path(self, experiment_name):
        pass


class SimulatedPreprocessedDataParams(DataScenarioParams):
    data_scenario = "simulated_preprocessed"
    simulated_scenario = param.String("random_cores")
    simulated_exact_file_name = param.String("sim_data_motifs_2_ratio_(0.5, 0.5)_lengths_(8,)_noize_0.csv")

    def update_input_data_path(self, experiment_name):
        self.input_data_path = f"{self.data_path_base}/{experiment_name}/{self.data_scenario}/per_length_per_kfold_split/"


class SimulatedDataParams(DataScenarioParams):
    data_scenario = "simulated"
    simulated_scenario = param.String("five_different_motifs")
    simulated_exact_file_name = param.String(
        "sim_data_motifs_5_ratio_(0.15, 0.15, 0.2, 0.2, 0.3)_lengths_(8,)_noize_0.25.csv"
    )
    dummy_allele_name = param.String(default="dummy_allele")

    def update_input_data_path(self, experiment_name):
        self.input_data_path = f"{self.data_path_base}/{experiment_name}/simulated_data/16_02/{self.simulated_scenario}"
        # self.input_data_path = f"{self.data_path_base}/{experiment_name}/simulated_data/{self.simulated_scenario}"


class MixMHCpredDataParams(DataScenarioParams):
    data_scenario = "MixMHCpred"
    mixmhc_mixture_name = param.String(default='112')  # 3865_DM_DR  CD165_DR Maver_1_DR  RA957_D

    def __init__(self, **params):
        super().__init__(**params)
        self.extra['mono_samples'] = [
            "MAPTAC_expi293_3",
            "MAPTAC_expi293_31",
            "MAPTAC_expi293_11",
            "MAPTAC_expi293_17",
            "MAPTAC_expi293_18",
            "MAPTAC_expi293_21",
            "MAPTAC_expi293_25",
            "MAPTAC_expi293_27",
            "MAPTAC_expi293_34",
            "MAPTAC_expi293_32",
            "MAPTAC_expi293_35",
            "MAPTAC_expi293_31",
        ]
        self.extra['mixed_samples'] = [
            "9087_DR",
            "JESI_DR",
            "CD165_DR",
            "C1R_DR",
            "MCL030",
            "NeonDC_lung_1",
            "IHW09013",
        ]
        self.extra['mixed_samples_dict'] = {
            "9087_DR": "DRB1*03:01_DRB3*01:01",
            "JESI_DR": "DRB1*15:01_DRB1*10:01",
            "CD165_DR": "DRB3*02:02_DRB1*11:01",
            "C1R_DR": "DRB3*02:02_DRB1*12:01",
            "MCL030": "DRB3*01:01_DRB1*13:03",
            "NeonDC_lung_1": "DRB4*01:01_DRB1*07:01",
            "IHW09013": "DRB5*01:01_DRB1*15:01"
        }

        self.extra['mono_alleles_dict'] = {
            "MAPTAC_expi293_3": "DRB1*03:01",
            "MAPTAC_expi293_31": "DRB3*01:01",
            "MAPTAC_expi293_11": "DRB1*07:01",
            "MAPTAC_expi293_17": "DRB1*10:01",
            "MAPTAC_expi293_18": "DRB1*11:01",
            "MAPTAC_expi293_21": "DRB1*12:01",
            "MAPTAC_expi293_25": "DRB1*13:03",
            "MAPTAC_expi293_27": "DRB1*15:01",
            "MAPTAC_expi293_34": "DRB4*01:01",
            "MAPTAC_expi293_32": "DRB3*02:02",
            "MAPTAC_expi293_35": "DRB5*01:01",
            "MAPTAC_expi293_31": "DRB3*01:01"
        }

    if mixmhc_mixture_name in ['3865_DM_DR']:
        input_data_path = param.Filename(
            f"{DataScenarioParams.data_path_base}/MixMHC2pred_2023/train_data/current_study.csv")
    else:
        input_data_path = param.Filename(
            f"{DataScenarioParams.data_path_base}/MixMHC2pred_2023/train_data/external_studies.csv")

    dummy_allele_name = param.String(default="mixMHC_dummy_allele")

    @param.depends('mixmhc_mixture_name', watch=True, on_init=False)
    def update_mixture_name(self):
        if self.mixmhc_mixture_name in self.extra['mono_samples']:
            self.dummy_allele_name = "mixMHC_" + self.extra['mono_alleles_dict'][
                self.mixmhc_mixture_name].replace('-', '_').replace('*', '_').replace(':', '_')
        elif self.mixmhc_mixture_name in self.extra['mixed_samples']:
            self.dummy_allele_name = "mixMHC_" + self.extra['mixed_samples_dict'][
                self.mixmhc_mixture_name].replace('-', '_').replace('*', '_').replace(':', '_')
        else:
            self.dummy_allele_name = self.mixmhc_mixture_name


class PreprocessedIEDBDataParams(DataScenarioParams):
    data_scenario = "IEDB_preprocessed"

    def update_input_data_path(self, experiment_name):
        self.input_data_path = f"{self.data_path_base}/{experiment_name}/{self.data_scenario}/per_length_per_kfold_split/"


class ModelTrainingParams(param.Parameterized):
    algorithm = param.Selector(default="viterbi", objects=["viterbi", "baum_welch"])
    num_runs = param.Integer(default=20, bounds=(1, None))
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
    anchor_top_aas = param.Integer(default=4, bounds=(1, 20)) #сколько аминокислот главные в якорных позициях
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
    multiple_check_input = param.Boolean(default=True, doc="legacy param from pomegranate. needs to be true for now, but probably can speedup without it")
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
    lengths_to_use = [12, 13, 14, 15, 16, 17, 18]
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

class SimpleModelClassIIParamsMixMHC(SimpleModelClassIIParams):
    enrichment_steps = 3
    anchor_top_aas = 6
    data_split_strategy = "median_score_of_best_models"
    split_decision_models_fraction = 0.2
    enrichment_split_decision_models_fraction = 0.2
    statistics_estimate_model_fraction = 0.2


    trash_model_information_threshold = 0.55
    same_model_distance_threshold = 0.08
    reassignment_decision_models_fraction = 0.2
    def __init__(self, **params):
        super().__init__(**params)


class SimpleModelClassIIParamsMixMHCsmall(SimpleModelClassIIParams):
    enrichment_steps = 3
    anchor_top_aas = 4
    data_split_strategy = "median_score_of_best_models"
    split_decision_models_fraction = 0.2
    enrichment_split_decision_models_fraction = 0.2
    statistics_estimate_model_fraction = 0.2
    emission_pseudocount = 0.1


    trash_model_information_threshold = 0.55
    same_model_distance_threshold = 0.08
    reassignment_decision_models_fraction = 0.2
    def __init__(self, **params):
        super().__init__(**params)




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
        self.experiment_result_data_path = f"/Users/annaklimova/Desktop/Washu_project/{self.experiment_name}/IEDB_data/plots"