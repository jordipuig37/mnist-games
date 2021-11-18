import numpy as np
import numpy.random as random
import pandas as pd
from collections import defaultdict
from copy import deepcopy as dcopy

from utils.dotdic import DotDic
from utils.logtools import print_verbose
import torch


class EpisodeStats():
    """This class represents the saved information for a single episode. It is
    an easy way to create and manage episode data. Structure:
        it's a nested DotDic at the end
    
        step_records[step][agentidx][variable][batch]

        with variable: {qsa, qsc, q_act_target, q_com_target, rewards}
    
    TODO: fer el baseline sense communicació.
    TODO: fer tests amb menys digits. i veure com afecta a la convergència.
    TODO: interpretar el model simplement veient l'output de communicació. --> fer una taula per cada combinació.
    TODO: veure si els protocols son iguals entre agents.

    """
    def __init__(self, conf, states):
        self.conf = conf
        self.step_records = defaultdict(lambda: defaultdict(lambda: DotDic({})))        
        self.agents_input = defaultdict(lambda: defaultdict(lambda: DotDic({})))
        self.final_reward = torch.zeros(conf.bs)
        self.episode_loss = 0
        self.eps = 0


    def record_step(self, t, agent, step_dic):
        self.step_records[t][agent] = DotDic(step_dic)
        self.final_reward += step_dic["reward"]
        self.eps = step_dic["epsilon"]


    def record_input(self, t, agent, agent_input):
        self.agents_input[t][agent] = DotDic(agent_input)


    def get_data(self):
        """This function returns all the information of the episode in
        dictionary format
        """
        dictionary = {
            "rewards": self.final_reward.sum().item(),
            "norm_rewards": self.final_reward.sum().item() / self.conf.bs,
            "loss": self.episode_loss,
            "eps": self.eps
        }
        return dictionary


class MNISTEnv():
    """Docstring for MNISTEnv.
    The stats 
    """
    def __init__(self, conf, writer=None):
        self.conf = conf
        self.stats = []
        self.writer = writer
        

    def train(self, agents, verbose=False):
        """This function trains the given agents running the number of episodes
        defined in self.conf.
        """
        for n_episode in range(self.conf.n_episodes):
            episode_stats = self.run_episode(agents, n_episode)
            ep_loss = 0
            for idx, agent in enumerate(agents):
                ep_loss += agent.learn_from_episode(episode_stats, n_episode)
            episode_stats.episode_loss = ep_loss
            for agent in agents:
                agent.optimizer.step()  # this is ugly but it works
                if n_episode % self.conf.step_target == 0 and n_episode > 0:
                    agent.actualize_target_network()


            self.stats.append(episode_stats)
            if self.writer:
                self.writer.add_scalar(f"Trial/reward", episode_stats.final_reward.sum(), n_episode)

    
    def reset(self):
        """This function resets the stats and other conditions."""
        self.stats = []
    

    def get_stats(self):
        """This function returns the stats of the environement in a format
        that can be later saved and analyzed.
        """
        list_of_episodes = list(map(lambda x: x.get_data, self.stats))
        return list_of_episodes


    def generate_random_states(self) -> torch.Tensor:
        """This function randomly generates the sequence of the states recived
        by the agents. Returns a tensor of shape (batch_size, agents) that
        represents the state recieved by each agent at each episode of the
        batch.
        """
        states = torch.tensor(np.random.choice(self.conf.n_states,(self.conf.bs,self.conf.n_agents))).long()
        return states


    def get_reward(self, step, action, ground_truth):
        if step == self.conf.steps-1:
            return self.conf.right_digit_reward * (action == ground_truth)
        else:
            return torch.zeros(action.shape)


    def run_episode(self, agents, n_episode):
        """This function runs a single batch of episodes and records the
        tates, communications, outputs of the episode and returns this record.
        """
        states = self.generate_random_states()
        episode_stats = EpisodeStats(self.conf, states)

        # communications = torch.zeros(self.conf.nagents, self.conf.steps, dtype=torch.int8)
        for step in range(self.conf.steps):
            for idx, agent in enumerate(agents):
                agent_inputs = {
                    'current_state': states[:,idx].view(-1,1),
                    'comm_recived': episode_stats.step_records[step-1][(idx+1)%self.conf.n_agents].comm_vector,
                    'prev_action': episode_stats.step_records[step-1][idx].dial_action,
                    'hidden': episode_stats.step_records[step-1][idx].hidden
                }
                Q, hidden = agent.model(**agent_inputs)
                Qt, _    = agent.target(**agent_inputs)
                
                episode_epsilon = 1/max(1,n_episode)
                (action, action_value), (comm_vector, comm_action, comm_value) =\
                    agent.select_action_and_comm(Q, eps=episode_epsilon)

                reward = self.get_reward(step, action, states[:, idx-1])

                step_record = {
                    "action": action,
                    "action_value": action_value,
                    "comm_vector": comm_vector,
                    "comm_action": comm_action,
                    "comm_value": comm_value,
                    "reward": reward,
                    "epsilon": episode_epsilon,
                    "Q": Q,
                    "hidden": hidden,
                    "Qt": Qt
                }

                episode_stats.record_step(step, idx, step_record)
                episode_stats.record_input(step, idx, agent_inputs)

        return episode_stats

