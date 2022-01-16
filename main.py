import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys
import argparse
import pickle
import copy

import torch
from utils import DotDic
from utils import print_verbose
from utils import plot_different_confs
from utils import generate_colnames

from agent import PlayerMNIST
from agent import AgentNet
from agent import init_xavi_uniform
from environment import MNISTEnv


def create_agents(conf):
    agents = []
    cnet = AgentNet(conf)
    cnet.apply(init_xavi_uniform)
    cnet_target = copy.deepcopy(cnet)
    for i in range(conf.n_agents):
        agents.append(PlayerMNIST(i, conf, model=cnet, target=cnet_target))
        if not conf.model_know_share:
            cnet = AgentNet(conf)
            cnet.apply(init_xavi_uniform)
            cnet_target = copy.deepcopy(cnet)
    return agents


def read_conf(path):
    f = open(path, "r")
    conf = DotDic(json.load(f))
    f.close()
    conf.total_action_space = conf.n_actions + conf.comm_space
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.show_results = int(conf.n_episodes / 10)
    return conf


def generate_param_combinations(base_conf, parameter_id):
    """This function takes a base_conf dictionary, a single hyperparameter and
    returns a list with four different configurations changing only the given
    hyperparameter (parameter_id). If parameter_id is not "sigma" or "gamma",
    then automatically it is set to gamma. The values tested are hardcoded.
    """
    if parameter_id == "sigma":
        parameters = [0.25, 0.5, 1, 2]
    elif parameter_id == "gamma":
        parameters = [0, 0.5, 0.9, 1]
    else:
        parameter_id == "gamma"
        parameters = [0, 0.5, 0.9, 1]
    
    new_conf = copy.deepcopy(base_conf)
    param_comb = list()

    for param in parameters:
        new_conf[parameter_id] = param
        param_comb.append(copy.deepcopy(new_conf))
    
    return param_comb


def make_trials(conf, conf_info, return_test=False, seed=0):
    """This function performs the number of trials specified in conf. Each
    trial consist of a full training cycle of the number of epochs specified
    in conf. It returns the training data of all the trials. If return_test is 
    True it also returns the test data.
    """
    torch.manual_seed(seed)
    results = list()
    test_results = list()
    print(f"Running trials for configuration {conf_info}")
    for trial in range(conf.n_trials):
        environment = MNISTEnv(conf)
        agents = create_agents(conf)
        environment.train(agents)
        df = pd.DataFrame(environment.stats)
        dftest = pd.DataFrame(environment.test_stats)
        results.append(df)
        test_results.append(dftest)
    if return_test:
        return results, test_results

    return results, None


def test_different_configs(hparams_configurations, parameter, return_test=False, seed=0):
    """This function executes the number of trials specified in the
    configuration for each combination of hyper parameters in hparams_configurations.
    It returns the training data of all the trials for each configuration.
    If return_test is True it returns the test data as well.
    """
    all_results = list()
    for conf in hparams_configurations:
        conf_info = f"{parameter} = {conf[parameter]}"
        conf_results, test_results = make_trials(conf, conf_info, return_test=return_test, seed=seed)
        
        if return_test:
            all_results.append((conf, conf_results, test_results))
        else:
            all_results.append((conf, conf_results))
    
    return all_results


def select_data(all_results, return_test=False):
    """This function simply returns the data from all_results in a list
    that can be saved in a .pkl file (i.e formatting the DotDict to a simple 
    dict again).
    """
    if return_test:
        return list(map(lambda x: (dict(x[0]), x[1], x[2]), all_results))
    else:
        return list(map(lambda x: (dict(x[0]), x[1]), all_results))


def main(args, **kwargs):
    base_conf = read_conf(args.conf_file)
    hparams_configurations = generate_param_combinations(base_conf, args.parameter)

    all_results = test_different_configs(hparams_configurations, args.parameter, return_test=True)
    
    formatted_data = select_data(all_results, return_test=True)

    # save the stats to a file for later analisis
    with open('results.pkl', 'wb') as f:
        pickle.dump(formatted_data, f,  protocol=pickle.HIGHEST_PROTOCOL)

    # plot and save the plot
    window_size = int(max(1, base_conf.n_episodes/400))
    plot_different_confs(formatted_data, args.parameter, factor_size=1.2, Ws=window_size, with_test=True)
    plt.savefig("results.jpg", bbox_inches="tight", pad_inches=0.25)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", "--conf-file", type=str, help="The file in which the configuration for our experimets is stored.")
    parser.add_argument("-parameter", type=str, default="gamma", help="The parameter we want to iterate")
    parser.add_argument("-v", "--verbose", nargs='?', const=True, default=False)
    args = parser.parse_args()

    main(args)
