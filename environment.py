import numpy as np
import numpy.random as random
import pandas as pd
from collections import defaultdict
from copy import deepcopy as dcopy

from utils import DotDic
import torch


class EpisodeStats():
    """This class represents the saved information for a batch of episodes.
    It is used to manage episode data.
    """
    def __init__(self, conf, states):
        self.conf = conf
        self.step_records = defaultdict(lambda: defaultdict(lambda: DotDic({})))        
        self.agents_input = defaultdict(lambda: defaultdict(lambda: DotDic({})))
        self.final_reward = torch.zeros(conf.bs, device=conf.device)
        self.total_reward = defaultdict(lambda: torch.zeros(conf.bs, device=conf.device))
        self.episode_loss = 0
        self.eps = 0


    def record_step(self, t, agent, step_dic):
        self.step_records[t][agent] = DotDic(step_dic)
        self.final_reward += step_dic["reward"]
        self.total_reward[t] += step_dic["reward"]
        self.eps = step_dic["epsilon"]


    def record_input(self, t, agent, agent_input):
        self.agents_input[t][agent] = DotDic(agent_input)


    def get_data(self):
        """This function returns all the information of the episode in a
        dictionary format. It returns the information that will be saved,
        thus, saving all the episode data is not needed.
        """
        dictionary = {
            "rewards": self.final_reward.sum().item(),
            "norm_rewards": self.final_reward.sum().item() / self.conf.bs,
            "loss": self.episode_loss,
            "eps": self.eps
        }
        return dictionary


class MNISTEnv():
    """This class represents the environment in which the experiments will take
    place. It records the train and test stats.
    """
    def __init__(self, conf, seed=1234): 
        np.random.seed(seed)
        self.conf = conf
        self.device = conf.device
        self.stats = []
        self.test_stats = []
        

    def train(self, agents):
        """This function trains the given agents running the number of episodes
        defined in self.conf. Also it saves the information for each episode;
        the variables that are saved are indicated in the get_data() function
        from the class EpisodeStats.
        """
        episode_stats = None
        for n_episode in range(self.conf.n_episodes):
            episode_stats = self.run_episode(agents)
            ep_loss = self.make_agents_learn(agents, episode_stats, n_episode)

            episode_stats.episode_loss = ep_loss

            self.stats.append(episode_stats.get_data())
            if (n_episode+1) % self.conf.test_freq == 0:
                test_episode = self.run_episode(agents, train_mode=False)
                self.test_stats.append(test_episode.get_data())
                if (n_episode+1) % self.conf.show_results == 0:
                    print(f"Mean Test Reward of {test_episode.final_reward.sum()/self.conf.bs:.3f} at episode {n_episode+1}")


    def make_agents_learn(self, agents, episode_stats, n_episode):
        """This auxiliar function performs the computation of the loss and 
        backpropagation, and finally the actualization of the network weights.
        The actualization is made inline and it returns the loss.
        """
        ep_loss = 0
        for idx, agent in enumerate(agents):
            # do this only once if model_know_share
            if (not self.conf.model_know_share) or (idx == 0):
                agent_loss = agent.learn_from_episode(episode_stats)
                ep_loss += agent_loss

        for idx, agent in enumerate(agents):
                # do this only once if model_know_share
                if (not self.conf.model_know_share) or (idx == 0):
                    agent.optimizer.step()  # this is ugly but it works
                    if n_episode % self.conf.step_target == 0 and n_episode > 0:
                        agent.actualize_target_network()
        
        return ep_loss

    
    def reset(self, seed=1234):
        """This function resets the stats and the seed."""
        self.stats = []
        self.test_stats = []
        np.random.seed(seed)


    def generate_random_states(self) -> torch.Tensor:
        """This function randomly generates the sequence of the states recived
        by the agents. Returns a tensor of shape (batch_size, agents) that
        represents the state recieved by each agent at each episode of the
        batch.
        """
        states = torch.tensor(np.random.choice(self.conf.n_states,(self.conf.bs,self.conf.n_agents))).long()
        return states


    def get_reward(self, step, action, ground_truth):
        """This function returns the reward of corresponding to the action
        vector with respect the ground_truth considering this is happening in
        the step indicated.
        """
        if step == self.conf.steps-1:
            return self.conf.right_digit_reward * (action == ground_truth)
        else:
            return torch.zeros(action.shape)


    def run_episode(self, agents, train_mode=True):
        """This function runs a single batch of episodes and records the
        states, communications, outputs of the episode and returns this record.
        """
        states = self.generate_random_states().to(self.device)
        episode_stats = EpisodeStats(self.conf, states)

        # communications = torch.zeros(self.conf.nagents, self.conf.steps, dtype=torch.int8)
        for step in range(self.conf.steps):
            for idx, agent in enumerate(agents):
                prev_action = episode_stats.step_records[step-1][idx].action
                agent_idx = (torch.ones(self.conf.bs, 1) * idx).long()
                agent_inputs = {
                    'current_state': states[:,idx].view(-1,1).to(self.device),
                    'comm_recived': episode_stats.step_records[step-1][(idx+1)%self.conf.n_agents].comm_vector,
                    'prev_action': prev_action if prev_action is None else prev_action.view(-1,1),
                    'hidden': episode_stats.step_records[step-1][idx].hidden,  # is already on device
                    'agent_idx': agent_idx.to(self.device)
                }
                Q, hidden = agent.model(**agent_inputs)
                Qt, _    = agent.target(**agent_inputs)
                
                #episode_epsilon = 1/max(1,n_episode)
                (action, action_value), (comm_vector, comm_action, comm_value) =\
                    agent.select_action_and_comm(Q, train_mode)

                reward = self.get_reward(step, action, states[:, idx-1]).to(self.device)
                
                step_record = {
                    "action": action.to(self.device),
                    "ground_truth": states[:, idx-1],  # the states of the other players
                    "action_value": action_value.to(self.device),
                    "comm_vector": comm_vector.to(self.device),
                    "comm_action": comm_action.to(self.device),
                    "comm_value": comm_value.to(self.device),
                    "reward": reward.to(self.device),
                    "epsilon": agent.eps,
                    "Q": Q,
                    "hidden": hidden,
                    "Qt": Qt
                }
                if train_mode:
                    agent.eps = agent.eps * self.conf.eps_decay

                episode_stats.record_step(step, idx, step_record)
                episode_stats.record_input(step, idx, agent_inputs)

        return episode_stats
