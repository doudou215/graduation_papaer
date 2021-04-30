# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch
from abc import ABC

import torch
import torch.nn as nn
import torch.nn.functional as F

class abstract_agent(nn.Module):
    def __init__(self):
        super(abstract_agent, self).__init__()
    
    def act(self, input):
        policy, value = self.forward(input) # flow the input through the nn
        return policy, value

class actor_agent(abstract_agent):
    def __init__(self, num_inputs, action_size, args):
        super(actor_agent, self).__init__()
        self.linear_a1 = nn.Linear(num_inputs, args.num_units_1)
        self.linear_a2 = nn.Linear(args.num_units_1, args.num_units_2)
        self.linear_a = nn.Linear(args.num_units_2, action_size)
        self.reset_parameters()
        # Activation func init
        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh= nn.Tanh()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        self.linear_a1.weight.data.mul_(gain)
        self.linear_a2.weight.data.mul_(gain)
        self.linear_a.weight.data.mul_(gain_tanh)
    
    def forward(self, input):
        """
        The forward func defines how the data flows through the graph(layers)
        """
        x = self.LReLU(self.linear_a1(input))
        x = self.LReLU(self.linear_a2(x))
        policy = self.tanh(self.linear_a(x))
        return policy 

class critic_agent(abstract_agent):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(critic_agent, self).__init__()
        self.linear_o_c1 = nn.Linear(obs_shape_n, args.num_units_1)
        self.linear_a_c1 = nn.Linear(action_shape_n, args.num_units_1)
        self.linear_c2 = nn.Linear(args.num_units_1*2, args.num_units_2)
        self.linear_c = nn.Linear(args.num_units_2, 1)
        self.reset_parameters()

        self.LReLU = nn.LeakyReLU(0.01)
        self.tanh= nn.Tanh()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        self.linear_o_c1.weight.data.mul_(gain)
        self.linear_a_c1.weight.data.mul_(gain)
        self.linear_c2.weight.data.mul_(gain)
        self.linear_c.weight.data.mul_(gain)

    def forward(self, obs_input, action_input):
        """
        input_g: input_global, input features of all agents
        """
        x_o = self.LReLU(self.linear_o_c1(obs_input))
        x_a = self.LReLU(self.linear_a_c1(action_input))
        x_cat = torch.cat([x_o, x_a], dim=1)
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value

class openai_critic(abstract_agent):
    def __init__(self, obs_shape_n, action_shape_n, args):
        super(openai_critic, self).__init__()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_c1 = nn.Linear(action_shape_n+obs_shape_n, args.num_units_openai)
        self.linear_c2 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_c = nn.Linear(args.num_units_openai, 1)

        self.reset_parameters()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_uniform_(self.linear_c1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_c.weight, gain=nn.init.calculate_gain('leaky_relu'))

    def forward(self, obs_input, action_input):
        """
        input_g: input_global, input features of all agents
        """
        x_cat = self.LReLU(self.linear_c1(torch.cat([obs_input, action_input], dim=1)))
        x = self.LReLU(self.linear_c2(x_cat))
        value = self.linear_c(x)
        return value

class openai_actor(abstract_agent):
    def __init__(self, num_inputs, action_size, args):
        super(openai_actor, self).__init__()
        self.tanh= nn.Tanh()
        self.LReLU = nn.LeakyReLU(0.01)
        self.linear_a1 = nn.Linear(num_inputs, args.num_units_openai)
        self.linear_a2 = nn.Linear(args.num_units_openai, args.num_units_openai)
        self.linear_a = nn.Linear(args.num_units_openai, action_size)

        self.reset_parameters()
        self.train()
    
    def reset_parameters(self):
        gain = nn.init.calculate_gain('leaky_relu')
        gain_tanh = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(self.linear_a1.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a2.weight, gain=nn.init.calculate_gain('leaky_relu'))
        nn.init.xavier_uniform_(self.linear_a.weight, gain=nn.init.calculate_gain('leaky_relu'))
    
    def forward(self, input, model_original_out=False):
        """
        The forward func defines how the data flows through the graph(layers)
        flag: 0 sigle input 1 batch input
        """
        x = self.LReLU(self.linear_a1(input))
        x = self.LReLU(self.linear_a2(x))
        model_out = self.linear_a(x)
        u = torch.rand_like(model_out)
        # 在列上softmax, model_out的维数batch_size * action_size
        policy = F.softmax(model_out - torch.log(-torch.log(u)), dim=-1)
        if model_original_out == True:   return model_out, policy # for model_out criterion
        return policy


class QMIXNet(nn.Module):
    def __init__(self, qmix_hidden_dim, state_shape, n_agents):
        super(QMIXNet, self).__init__()
        self.hyper_w1 = nn.Linear(state_shape, n_agents * qmix_hidden_dim)
        self.hyper_w2 = nn.Linear(state_shape, qmix_hidden_dim)
        self.hyper_b1 = nn.Linear(state_shape, qmix_hidden_dim)
        self.hyper_b2 = nn.Sequential(nn.Linear(state_shape, qmix_hidden_dim),
                                      nn.ReLU(),
                                      nn.Linear(qmix_hidden_dim, 1)
                                      )

    def forward(self, q_values, states, episode_num, n_agents):
        q_values = q_values.view(-1, 1, n_agents)  # (episode_num , 1, n_agents)
        # state_shape = 36
        # states = states.reshape(-1, state_shape)  # (episode_num * max_episode_len, state_shape)
        qmix_hidden_dim = 128
        # torch.abs保证了w1的参数全都是正的值
        w1 = torch.abs(self.hyper_w1(states))  # batch_size * (n_agents * qmix_hidden_dim)
        b1 = self.hyper_b1(states)  # (1920, 32)  # batch_size * qmix_hidden_dim

        w1 = w1.view(-1, n_agents, qmix_hidden_dim)  # batch_size, n_agents, qmix_hidden_dim
        b1 = b1.view(-1, 1, qmix_hidden_dim)  # batch_size 1 qmix_hidden_dim

        hidden = F.elu(torch.bmm(q_values, w1) + b1)  # batch_size 1 qmix_hidden_dim

        w2 = torch.abs(self.hyper_w2(states))  # (1920, 32)
        b2 = self.hyper_b2(states)  # (1920, 1)

        w2 = w2.view(-1, qmix_hidden_dim, 1)  # (1920, 32, 1)
        b2 = b2.view(-1, 1, 1)  # (1920, 1， 1)

        q_total = torch.bmm(hidden, w2) + b2  # (1920, 1, 1)
        q_total = q_total.view(episode_num, -1)  # (32, 60, 1)
        return q_total
