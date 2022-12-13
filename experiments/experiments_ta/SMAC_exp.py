# -*- coding: utf-8 -*-
import logging

logging.basicConfig(level=logging.INFO)

import sys
import os
import pickle
import numpy as np

import ConfigSpace as CS
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)

# from sklearn.datasets import load_digits
# from sklearn.exceptions import ConvergenceWarning
# from sklearn.model_selection import cross_val_score, StratifiedKFold
# from sklearn.neural_network import MLPClassifier


from smac.configspace import ConfigurationSpace, Configuration
from smac.facade.smac_mf_facade import SMAC4MF
from smac.intensification.successive_halving import SuccessiveHalving
from smac.facade.smac_ac_facade import SMAC4AC
from smac.scenario.scenario import Scenario
from smac.initial_design.latin_hypercube_design import LHDesign

from myBO.scripts.test_functions import test_functions
from myBO.scripts.myutils import myutils


__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"

# digits = load_digits()

if __name__ == "__main__":
    args = sys.argv
    data_name = args[1]
    initial_seed = int(args[2])

    #データの読み込み
    with open('../../scripts/test_functions/AutoML_data/cnn_'+data_name+'_data/cnn_'+data_name+'_data.pickle', 'rb') as f:
        data = pickle.load(f)

    best_value = 1 - np.max(data[data[:,-3]==20,-1])

    # Target Algorithm
    def cnn_from_cfg(cfg, seed, budget):
        """
        From dataset, a corresponding data is returned

        Parameters
        ----------
        cfg: Configuration
            configuration chosen by smac
        budget: float
            used to set max iterations for the CNN

        Returns
        -------
        float
        """

        # For deactivated parameters, the configuration stores None-values.
        # This is not accepted by the MLP, so we replace them with placeholder values.
        lr = cfg["learning_rate"]
        batch_size = cfg["batch_size"]
        ch1 = cfg["ch1"]
        ch2 = cfg["ch2"]
        drop_rate = cfg["drop_rate"]

        tmp_match_index = np.where( np.all(data[:,:6] == [lr, batch_size, ch1, ch2, drop_rate, np.ceil(budget)], axis=1) == True)[0]

        score = data[tmp_match_index][0][-1]

        return 1 - score

    def cnn_time_from_cfg(updated_cost, cfg, budget):
        # For deactivated parameters, the configuration stores None-values.
        # This is not accepted by the MLP, so we replace them with placeholder values.
        lr = cfg["learning_rate"]
        batch_size = cfg["batch_size"]
        ch1 = cfg["ch1"]
        ch2 = cfg["ch2"]
        drop_rate = cfg["drop_rate"]

        tmp_match_index = np.where( np.all(data[:,:6] == [lr, batch_size, ch1, ch2, drop_rate, np.ceil(budget)], axis=1) == True)[0]

        elapsed_time = updated_cost[tmp_match_index]

        return elapsed_time

    def reduce_time_from_cfg(updated_cost, cfg, ta_elaplsed_time):
        # For deactivated parameters, the configuration stores None-values.
        # This is not accepted by the MLP, so we replace them with placeholder values.
        lr = cfg["learning_rate"]
        batch_size = cfg["batch_size"]
        ch1 = cfg["ch1"]
        ch2 = cfg["ch2"]
        drop_rate = cfg["drop_rate"]

        tmp_match_index = np.where( np.all(data[:,:5] == [lr, batch_size, ch1, ch2, drop_rate], axis=1) == True)[0]

        updated_cost[tmp_match_index] = updated_cost[tmp_match_index] - ta_elaplsed_time
        updated_cost[updated_cost < 0] = 0
        return updated_cost


    # Build Configuration Space which defines all parameters and their ranges.
    # To illustrate different parameter types,
    # we use continuous, integer and categorical parameters.
    cs = ConfigurationSpace()

    learning_rate = UniformIntegerHyperparameter(
        "learning_rate", -4, 0, default_value=-2
    )
    batch_size = UniformIntegerHyperparameter("batch_size", 5, 9, default_value=5)

    ch1 = UniformIntegerHyperparameter("ch1", 2, 6, default_value=3)
    ch2 = UniformIntegerHyperparameter("ch2", 2, 6, default_value=3)

    drop_rate = UniformIntegerHyperparameter("drop_rate", -4, -1, default_value=-4)


    # Add all hyperparameters at once:
    cs.add_hyperparameters(
        [
            learning_rate,
            batch_size,
            ch1,
            ch2,
            drop_rate,
        ]
    )

    # SMAC scenario object
    scenario = Scenario(
        {
            "run_obj": "quality",  # we optimize quality (alternative to runtime)
            "wallclock-limit": 5,  # max duration to run the optimization (in seconds)
            "cs": cs,  # configuration space
            "deterministic": True,
            # Uses pynisher to limit memory and runtime
            # Alternatively, you can also disable this.
            # Then you should handle runtime and memory yourself in the TA
            # "limit_resources": False,
            # "cutoff": 30,  # runtime limit for target algorithm
            # "memory_limit": 3072,  # adapt this to reasonable value for your hardware
        }
    )

    # Max budget for hyperband can be anything. Here, we set it to maximum no. of epochs to train the MLP for
    max_epochs = 20

    # Intensifier parameters
    intensifier_kwargs = {"initial_budget": 5, "max_budget": max_epochs, "eta": 3, "min_chall": 1}

    #------------------------------------------------------------------------------------
    # make same initial inputs
    #------------------------------------------------------------------------------------
    # set the seed
    np.random.seed(initial_seed)

    FIRST_NUM = [4, 4, 4]
    if data_name=='MNIST':
        test_func = eval('test_functions.cnn_mnist')()
    elif data_name=='CIFAR10':
        test_func = eval('test_functions.cnn_cifar10')()

    M = test_func.M
    bounds = test_func.bounds
    input_dim = test_func.d
    pool_X = test_func.X.copy()
    X_all = pool_X[M-1].copy()
    training_input, pool_X = myutils.initial_design_pool(FIRST_NUM, input_dim, bounds, pool_X)

    initial_inputs = training_input[-1]

    initial_configs = []
    for i in range(4):
        tmp_dict = {
            "learning_rate": int(initial_inputs[i,0]),
            "batch_size": int(initial_inputs[i,1]),
            "ch1": int(initial_inputs[i,2]),
            "ch2": int(initial_inputs[i,3]),
            "drop_rate": int(initial_inputs[i,4])
        }
        print(tmp_dict)
        initial_configs.append(Configuration(cs, tmp_dict))
    # print(initial_configs)

    #------------------------------------------------------------------------------------

    # To optimize, we pass the function to the SMAC-object
    smac = SMAC4AC(
        scenario=scenario,
        rng=np.random.RandomState(42),
        tae_runner=cnn_from_cfg,
        initial_configurations=initial_configs,
        intensifier=SuccessiveHalving,
        intensifier_kwargs=intensifier_kwargs,
    )

    tae = smac.get_tae_runner()

    # Example call of the function with default values
    # It returns: Status, Cost, Runtime, Additional Infos
    def_value = tae.run(
        config=cs.get_default_configuration(),
        budget=max_epochs, seed=0
    )[1]

    print("Value for default configuration: %.4f" % def_value)

    # Start optimization
    try:
        incumbent = smac.optimize()
    finally:
        incumbent = smac.solver.incumbent

    # trajectory = smac.get_trajectory()
    # print(trajectory)


    cost_list = []
    SimReg_list = []

    wall_clock_time = 0
    SimReg = - np.inf
    updated_cost = data[:,-2].copy()

    rh = smac.get_runhistory()
    for (config_id, instance_id, seed, budget), (cost, time, status, starttime, endtime, additional_info) in rh.data.items():

        config = rh.ids_config[config_id]
        wall_clock_time += time
        target_eval_time = cnn_time_from_cfg(updated_cost, config, budget)
        updated_cost = reduce_time_from_cfg(updated_cost, config, target_eval_time)
        wall_clock_time += target_eval_time

        if budget==20:
            output = tae.run(config=config, budget=max_epochs, seed=0)[1]
            if 1 - SimReg > output:
                SimReg = 1 - output
        print(budget, wall_clock_time / 60., SimReg, config)

        cost_list.append(wall_clock_time / 60.)
        SimReg_list.append(SimReg)

    if data_name=='MNIST':
        results_path = 'cnn_mnist_results/SMAC3/MFGP_seed='+str(initial_seed)+'/'
    elif data_name=='CIFAR10':
        results_path = 'cnn_cifar10_results/SMAC3/MFGP_seed='+str(initial_seed)+'/'

    if not os.path.exists(results_path):
        os.makedirs(results_path)

    with open(results_path + 'cost.pickle', 'wb') as f:
        pickle.dump(np.array(cost_list), f)

    with open(results_path + 'SimReg.pickle', 'wb') as f:
        pickle.dump(np.array(SimReg_list), f)

    inc_value = tae.run(config=incumbent, budget=max_epochs, seed=0)[
        1
    ]

    print("Optimized Value: %.4f" % inc_value)
    print("Best Value: %.4f" % best_value)