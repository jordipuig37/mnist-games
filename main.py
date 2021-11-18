import pandas as pd
import numpy as np
import json
import os
import sys
import argparse
import pickle

import torch
from utils.dotdic import DotDic
from utils.logtools import print_verbose
from agent import PlayerMNIST
from environment import MNISTEnv


def create_agents(conf) -> list:
    agents = list()
    for i in range(conf.n_agents):
        new = PlayerMNIST(i, conf)
        agents.append(new)
    
    return agents


def read_conf(path) -> DotDic:
    f = open(path, "r")
    conf = DotDic(json.load(f))
    f.close()
    conf.total_action_space = conf.n_actions + conf.comm_space 
    return conf


def save_stats(stats, path):
    converted_stats = list(map(lambda x: x.get_data(), stats))
    pd.DataFrame(converted_stats).to_csv(path)


def main(args, **kwargs):
    conf = read_conf(args.conf_file)
    environment = MNISTEnv(conf)
    agents = create_agents(conf)

    print_verbose("Agents and environment created", args.verbose)

    # run the different trials
    # TODO: set a seed
    #environment.reset()  # reset the environment
    #agents = list(map(lambda agent: agent.reset(), agents))  # reset the agents, should this be inplace?
    environment.train(agents, args.verbose)
    
    # save the stats to a file for later analisis
    save_stats(environment.stats, "stats.csv")

    for idx, agent in enumerate(agents):
        torch.save(agent.model.state_dict(), f"agent-{idx}-statedict.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-conf", "--conf-file", type=str, help="The file in which the configuration for our experimets is stored.")
    parser.add_argument("-v", "--verbose", nargs='?', const=True, default=False)
    args = parser.parse_args()

    main(args)