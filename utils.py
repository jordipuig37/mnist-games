import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import torch

def print_verbose(msg, verbose):
    if verbose:
        print(msg)


class DotDic(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


    def __deepcopy__(self, memo=None):
        return DotDic(copy.deepcopy(dict(self), memo=memo))


class DRU:
    """Discretize/Regularize Unit to use in testing or training respectively."""
    def __init__(self, sigma, comm_narrow=True, hard=False, device="cpu"):
        self.sigma = sigma
        self.comm_narrow = comm_narrow
        self.hard = hard
        self.device = device


    def regularize(self, m):    
        m_reg = m + torch.randn(m.size(), device=self.device) * self.sigma
        if self.comm_narrow:
            m_reg = torch.sigmoid(m_reg)
        else:
            m_reg = torch.softmax(m_reg, 0)
        return m_reg


    def discretize(self, m):
        if self.hard:
            if self.comm_narrow:
                return (m.gt(0.5).float() - 0.5).sign().float()
            else:
                m_ = torch.zeros_like(m)
                if m.dim() == 1:      
                    _, idx = m.max(0)
                    m_[idx] = 1.
                elif m.dim() == 2:      
                    _, idx = m.max(1)
                    for b in range(idx.size(0)):
                        m_[b, idx[b]] = 1.
                else:
                    raise ValueError('Wrong message shape: {}'.format(m.size()))
                return m_
        else:
            scale = 2 * 20
            if self.comm_narrow:
                return torch.sigmoid((m.gt(0.5).float() - 0.5) * scale)
            else:
                return torch.softmax(m * scale, -1)


    def forward(self, m, train_mode):
        if train_mode:
            return self.regularize(m)
        else:
            return self.discretize(m)


    def __call__(self, m, train_mode):
        return self.forward(m, train_mode)


def generate_colnames(base, n_trials):
    result = list()
    for i in range(n_trials):
        for name in base:
            result.append(f"{name}_trial_{i}")
    return result


def plot_trials_same_conf(all_results, conf, label="", Ws=10):
    all_df = pd.concat(all_results, axis=1)
    all_df.columns = generate_colnames(list(all_results[0].columns), conf["n_trials"])
    all_df = all_df.reset_index()
    rewards_columns = list(filter(lambda x: "norm_rewards" in x, all_df.columns))
    mean_rewards = all_df[rewards_columns].mean(axis=1)
    smoothed = np.convolve(mean_rewards, 1/Ws*np.ones(Ws), mode="valid")
    plt.plot(smoothed, label=label)


def plot_different_confs(all_results, parameter, Ws=10, factor_size=1.5, with_test=False):
    """This function plots an informative visualization about the performance
    obtained in all_results. It smoothes the line plots of the reward with a
    moving average of window size Ws. factor_size is the factor by which the
    visualization is scaled. 
    """
    base_conf = all_results[0][0]  # in general only one parameter will be different among configurations
    plt.figure(figsize=(6.4*factor_size, 4.8*factor_size)).suptitle(
        f"Smoothed normalized reward for different values of {parameter}",
        y=1.05, fontsize=12.5*factor_size)
    n = len(all_results[0][-1][0])
    if with_test:
        n_episodes = base_conf["n_episodes"]
        show_results = base_conf["show_results"]

        for conf, _, res in all_results:
            plot_trials_same_conf(res, conf, f"{parameter} = {conf[parameter]}", Ws)

        jumps = int(n_episodes/show_results / 10)
        ticks_label = list(range(0, n_episodes+1, show_results*jumps))
        ticks = list(range(0, int(n_episodes/show_results)+1, jumps))
        plt.xticks(ticks=ticks, labels=ticks_label)
    else:
        for conf, res in all_results:
            plot_trials_same_conf(res, f"{parameter} = {conf[parameter]}", Ws)

    plt.xlabel("Epoch")
    plt.ylabel(f"Smoothed Normalized reward\n(window size={Ws})")

    plt.hlines(1, 0, n, linestyles="dashed", colors="limegreen", label="Optimal")
    plt.hlines(1/10, 0, n, linestyles="dashed", colors="black", label="Random")

    plt.legend(loc='upper left', bbox_to_anchor=(0.025, 1.175), ncol=3,
              fontsize=10.5*factor_size)
    plt.ylim(0, 1.05)
