import numpy as np
import numpy.random as random
import copy
import pickle

from utils import DRU

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


def weight_reset(m):
    if isinstance(m, nn.Embedding) or isinstance(m, nn.RNN) or \
        isinstance(m, nn.Linear) or isinstance(m, nn.LSTM):
        m.reset_parameters()


def init_xavi_uniform(m):
    if isinstance(m, nn.Embedding) or \
        isinstance(m, nn.Linear) or isinstance(m, nn.LSTM):
            torch.nn.init.xavier_uniform_(m.weight)


class AgentNet(nn.Module):
    """This class represents the Network architecture of the agents."""
    def __init__(self, conf):
        super(AgentNet, self).__init__()
        self.conf = conf
        self.num_embedding = nn.Embedding(conf.n_states, conf.emb_dim)
        self.agent_lookup = nn.Embedding(conf.n_agents, conf.emb_dim)
        self.comm_embbeding = nn.Sequential(
            nn.BatchNorm1d(conf.comm_space),
            nn.Linear(conf.comm_space, conf.emb_dim),
            nn.ReLU()
        )
        self.prev_action_embbeding = nn.Embedding(conf.total_action_space, conf.emb_dim)

        self.rnn = nn.RNN(conf.emb_dim, conf.rnn_size, num_layers=conf.rnn_layers, batch_first=True)

        self.action = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(conf.rnn_size, conf.rnn_size),
            nn.BatchNorm1d(conf.rnn_size),
            nn.ReLU(),
            nn.Linear(conf.rnn_size, conf.total_action_space)
        )
            


    def forward(self, current_state, prev_action, comm_recived, hidden, agent_idx):
        n_emb = self.num_embedding(current_state)
        agent_emb = self.agent_lookup(agent_idx)
        z = n_emb + agent_emb
        if (not self.conf.no_comm) and (comm_recived is not None):
            c_emb = self.comm_embbeding(comm_recived)
            z += c_emb.view(-1, 1, self.conf.emb_dim)
        if self.conf.dial and (prev_action is not None):
            prev_emb = self.prev_action_embbeding(prev_action)
            z += prev_emb.view(-1, 1, self.conf.emb_dim)

        output, h_n = self.rnn(z, hidden)

        Q = self.action(output[:, -1, :].squeeze())
        # the output is the q values and the hidden state that will be fed for
        # the next activation.
        return Q, h_n


class PlayerMNIST():
    """This is the class definition of the agent that will interact with the
    environment.
    """
    def __init__(self, idx, conf, model=None, target=None):
        self.conf = conf
        self.idx = idx
        self.eps = conf.eps
        self.device = conf.device
        if model is None:
            self.model = AgentNet(conf).to(conf.device)
        else:
            self.model = model.to(conf.device)
        
        if target is None:
            self.target = AgentNet(conf).to(conf.device)
        else:
            self.target = target.to(conf.device)
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=conf.learningrate, momentum=conf.momentum)
        self.dru = DRU(conf.sigma, comm_narrow=conf.model_comm_narrow, device=conf.device)

        self.action_range = range(0, conf.n_actions)
        self.comm_range = range(conf.n_actions, conf.total_action_space)


    def reset(self):
        # reset the network
        self.model.apply(weight_reset)
        self.target.apply(weight_reset)
    

    def get_model_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    

    def actualize_target_network(self):
        self.target.load_state_dict(self.model.state_dict())
    

    def _random_choice(self, n):
        return torch.randint(0,high=n, size=(self.conf.bs,))
    

    def _eps_flip(self, eps):
        return random.random() < eps


    def select_action_and_comm(self, Q, eps=0, train_mode=True):
        """This function selects (following a epsilon greedy policy over Q)
        an action and a communication. It returns the action and comm as well
        as the Qvalue for that pair.
        """
        should_select_random_a = train_mode and self._eps_flip(self.eps)
        if should_select_random_a:
            action = self._random_choice(self.conf.n_actions).to(self.device)
            action_value = torch.take(Q[:,self.action_range], action)

            comm_action = self._random_choice(self.conf.comm_space).to(self.device)
            comm_value = torch.take(Q[:, self.comm_range], comm_action)
        
        else:
            action_value, action    = torch.max(Q[:,self.action_range], dim=1)
            comm_value, comm_action = torch.max(Q[:,self.comm_range],   dim=1)
        
        comm_vector = self.dru.forward(Q[:, self.comm_range], train_mode=train_mode) # apply DRU
        comm_vector = comm_vector.to(self.device)
        return (action.long(), action_value),\
               (comm_vector, comm_action.long(), comm_value)
    

    def episode_loss(self, episode):
        """This function returns a loss that can be backpropageted through the
        agents' networks.
        """
        # compute the loss for each element of the batch
        total_loss = torch.zeros(self.conf.bs, device=self.device)
        for agent_idx in range(self.conf.n_agents):
            for step in range(self.conf.steps):
                # L(.) = (r + gamma* max(Q_t(s+1, a+1)) - Q(s, a))**2
                r = episode.total_reward[step] 
                qsa = episode.step_records[step][agent_idx].action_value  # the q value for the action selected
                if step==self.conf.steps-1:
                    td_action = r - qsa
                else:
                    q_target = episode.step_records[step+1][agent_idx].Qt[:,self.action_range]
                    td_action = r + self.conf.gamma * q_target.max(dim=1)[0] - qsa

                total_loss = total_loss + (td_action**2)
        loss = total_loss.sum()
        loss = loss/float((self.conf.bs * self.conf.n_agents))
        return loss


    def learn_from_episode(self, episode_record):
        """This function computes the loss for a batch of epiosodes in 
        episode_record and actualizes the agent model wheights with it.
        """
        self.optimizer.zero_grad()
        loss = self.episode_loss(episode_record)
        loss.backward(retain_graph=not self.conf.model_know_share)  # retain graph because we don't share parameters
        clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
        # the optimization step will be done later

        return loss.item()