# Time: 2019-11-05
# Author: Zachary 
# Name: MADDPG_torch
# File func: main func
import os

import time
import torch
import pickle
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from arguments import parse_args
from replay_buffer import ReplayBuffer
import multiagent.scenarios as scenarios
from model import openai_actor, openai_critic, QMIXNet
from multiagent.environment import MultiAgentEnv
from tensorboardX import SummaryWriter

def make_env(scenario_name, arglist, benchmark=False):
    """ 
    create the environment from script 
    """
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    world = scenario.make_world()
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.getState) # simple_tag.observation
    return env

def get_trainers(env, num_adversaries, obs_shape_n, action_shape_n, state_shape, arglist):
    """
    init the trainers or load the old model
    """
    # env.n -= 2
    actors_cur = [None for _ in range(env.n)]
    critics_cur = [None for _ in range(env.n)]
    actors_tar = [None for _ in range(env.n)]
    critics_tar = [None for _ in range(env.n)]
    optimizers_c = [None for _ in range(env.n)]
    optimizers_a = [None for _ in range(env.n)]
    input_size_global = sum(obs_shape_n) + sum(action_shape_n)

    # obs_shape_n = [26, 26, ]
    # action_shape_n = [5, 5, ]
    if arglist.restore == True: # restore the model
        for idx in arglist.restore_idxs:
            trainers_cur[idx] = torch.load(arglist.old_model_name+'c_{}'.format(agent_idx))
            trainers_tar[idx] = torch.load(arglist.old_model_name+'t_{}'.format(agent_idx))

    # Note: if you need load old model, there should be a procedure for juding if the trainers[idx] is None
    for i in range(env.n):
        # actor 只根据自己观察的到的信息判断动作
        actors_cur[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        # critic 需要全部人的观察到的信息
        critics_cur[i] = openai_critic(sum(obs_shape_n), sum(action_shape_n), arglist).to(arglist.device)
        actors_tar[i] = openai_actor(obs_shape_n[i], action_shape_n[i], arglist).to(arglist.device)
        critics_tar[i] = openai_critic(sum(obs_shape_n), sum(action_shape_n), arglist).to(arglist.device)
        optimizers_a[i] = optim.Adam(actors_cur[i].parameters(), arglist.lr_a)
        optimizers_c[i] = optim.Adam(critics_cur[i].parameters(), arglist.lr_c)
    actors_tar = update_trainers(actors_cur, actors_tar, 1.0)  # update the target par using the cur
    critics_tar = update_trainers(critics_cur, critics_tar, 1.0)  # update the target par using the cur

    qmix_hidden_dim = 128
    global_critic_cur = QMIXNet(qmix_hidden_dim, state_shape, env.n)
    global_critic_tar = QMIXNet(qmix_hidden_dim, state_shape, env.n)
    optimizer_g = optim.Adam(global_critic_cur.parameters(), arglist.lr_c)
    global_critic_tar.load_state_dict(global_critic_cur.state_dict())
    return actors_cur, critics_cur, actors_tar, critics_tar, global_critic_cur, global_critic_tar, optimizers_a, optimizers_c, optimizer_g

def update_trainers(agents_cur, agents_tar, tao):
    """
    update the trainers_tar par using the trainers_cur
    This way is not the same as copy_, but the result is the same
    out:
    |agents_tar: the agents with new par updated towards agents_current
    """
    for agent_c, agent_t in zip(agents_cur, agents_tar):
        key_list = list(agent_c.state_dict().keys())
        state_dict_t = agent_t.state_dict()
        state_dict_c = agent_c.state_dict()
        for key in key_list:
            state_dict_t[key] = state_dict_c[key]*tao + \
                    (1-tao)*state_dict_t[key] 
        agent_t.load_state_dict(state_dict_t)
    return agents_tar

def update_global_trainers(agent_c, agent_t, tao):
    key_list = list(agent_c.state_dict().keys())
    state_dict_t = agent_t.state_dict()
    state_dict_c = agent_c.state_dict()
    for key in key_list:
        state_dict_t[key] = state_dict_c[key] * tao + \
                            (1 - tao) * state_dict_t[key]
    agent_t.load_state_dict(state_dict_t)
    return agent_t


def agents_train(arglist, game_step, update_cnt, memory, obs_size, action_size, \
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c, \
                 global_critic_cur, global_critic_tar, optimizer_g):
    # update all trainers, if not in display or benchmark mode
    if game_step > arglist.learning_start_step and \
        (game_step - arglist.learning_start_step) % arglist.learning_fre == 0:
        if update_cnt == 0:
            print('\r=start training ...'+' '*100)
        # update the target par using the cur
        update_cnt += 1
        _state_n, _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory.sample_all(arglist.batch_size)
        q_cur, q_tar = torch.zeros((arglist.batch_size, 1)), torch.zeros((arglist.batch_size, 1))
        flag = 1
        _done_n = torch.FloatTensor(_done_n)
        _rew_n = torch.FloatTensor(_rew_n)

        for agent_idx, (critic_c, critic_t) in enumerate(zip(critics_cur, critics_tar)):
            action_cur_o = torch.from_numpy(_action_n).to(arglist.device, torch.float)
            if agent_idx > 5:
                break
            obs_n_o = torch.from_numpy(_obs_n_o).to(arglist.device, torch.float)
            obs_n_n = torch.from_numpy(_obs_n_n).to(arglist.device, torch.float)

            action_tar = torch.cat([a_t(obs_n_n[:, obs_size[idx][0]:obs_size[idx][1]]).detach() \
                                        for idx, a_t in enumerate(actors_tar)], dim=1)
            q = critic_c(obs_n_o, action_cur_o).reshape(arglist.batch_size, -1)  # q.reshape(-1) 将当前值拍扁形状变为 1 * batch_size
            q_ = critic_t(obs_n_n, action_tar).reshape(arglist.batch_size, -1)  # q_  同上 1 * batch_size 所谓的action_tar就是下一步agent会采用的动作信息
            """
            if agent_idx > 5:
                flag = -1
            """
            q_cur += q
            q_tar = q_tar + torch.mul(q_, (1 - _done_n[:, agent_idx]).reshape(arglist.batch_size, -1)) + _rew_n[:, agent_idx].reshape(arglist.batch_size, -1)

        # q_cur: n_agents * batch_size, q_tar: n_agents * batch_size
        _state_n_n = _state_n[1:, ]
        tmp = np.zeros((1, 36))
        _state_n_n = np.append(_state_n_n, tmp, axis=0)
        _state_n_n = torch.FloatTensor(_state_n_n)
        _state_n = torch.FloatTensor(_state_n)

        q_target = q_tar * arglist.gamma
        loss_c = torch.nn.MSELoss()(q_cur, q_target.detach())

        for opt_c in optimizers_c:
            opt_c.zero_grad()
        loss_c.backward()
        for critic in critics_cur:
            nn.utils.clip_grad_norm_(critic.parameters(), arglist.max_grad_norm)
        for opt_cn in optimizers_c:
            opt_cn.step()
        n_agents = 8
        """
        q_total_eval = global_critic_cur(q_cur, _state_n, arglist.batch_size, n_agents)
        # q_total_target = global_critic_tar(q_tar, _state_n_n, arglist.batch_size, n_agents)
        targets = _rew_n.sum(dim=0) + q_total_target * arglist.gamma * (1 - _done_n)
        td_error = (q_total_eval - targets.detach())
        loss = (td_error ** 2).sum()
        optimizer_g.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(global_critic_cur.parameters(), arglist.max_grad_norm)
        optimizer_g.step()
        global_critic_tar = update_global_trainers(global_critic_cur, global_critic_tar, arglist.tao)
        """
        for agent_idx, (actor_c, actor_t, critic_c, critic_t, opt_a, opt_c) in \
            enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c)):
            if opt_c == None:
                continue  # jump to the next model update
            if agent_idx > 5:
                break
            # sample the experience
            # _obs_n_o 当前所有人的观察的状态， _obs_n_n下一步所有人观察到的状态， _action_n所有人的动作，由于每个人可以选择五个动作，一共八个人，所以是40维
            # _done_n, _rew_n是自己的信息
            _state_n, _obs_n_o, _action_n, _rew_n, _obs_n_n, _done_n = memory.sample( \
                arglist.batch_size, agent_idx)  # Note_The func is not the same as others

            # --use the date to update the CRITIC
            rew = torch.tensor(_rew_n, device=arglist.device, dtype=torch.float)   # set the rew to gpu
            done_n = torch.tensor(~_done_n, dtype=torch.float, device=arglist.device)  # set the rew to gpu
            action_cur_o = torch.from_numpy(_action_n).to(arglist.device, torch.float)
            obs_n_o = torch.from_numpy(_obs_n_o).to(arglist.device, torch.float)
            obs_n_n = torch.from_numpy(_obs_n_n).to(arglist.device, torch.float)
            # action_tar将原本所有的人动作信息转为只包含自己的动作信息
            # 这里obs_size[idx][0]:obs_size[idx][1]是为了取出对应agent的观察到的信息

            action_tar = torch.cat([a_t(obs_n_n[:, obs_size[idx][0]:obs_size[idx][1]]).detach() \
                for idx, a_t in enumerate(actors_tar)], dim=1)
            # q = critic_c(obs_n_o, action_cur_o).reshape(-1)  # q.reshape(-1) 将当前值拍扁
            # q_ = critic_t(obs_n_n, action_tar).reshape(-1)  # q_  所谓的action_tar就是下一步agent会采用的动作信息
            """
            tar_value = q_*arglist.gamma*done_n + rew # q_*gamma*done + reward
            loss_c = torch.nn.MSELoss()(q, tar_value) # bellman equation
            opt_c.zero_grad()
            loss_c.backward()
            nn.utils.clip_grad_norm_(critic_c.parameters(), arglist.max_grad_norm)
            opt_c.step()
            """
            # --use the data to update the ACTOR
            # There is no need to cal other agent's action
            model_out, policy_c_new = actor_c( \
                obs_n_o[:, obs_size[agent_idx][0]:obs_size[agent_idx][1]], model_original_out=True)
            # update the action of this agent
            # 这里为什么要更新action_cur_o 好奇怪, 可能要考虑到on-policy的缘故
            action_cur_o[:, action_size[agent_idx][0]:action_size[agent_idx][1]] = policy_c_new 
            loss_pse = torch.mean(torch.pow(model_out, 2))
            loss_a = torch.mul(-1, torch.mean(critic_c(obs_n_o, action_cur_o)))

            opt_a.zero_grad()
            (1e-3*loss_pse+loss_a).backward()
            nn.utils.clip_grad_norm_(actor_c.parameters(), arglist.max_grad_norm)
            opt_a.step()


        # save the model to the path_dir ---cnt by update number
        if update_cnt > arglist.start_save_model and update_cnt % arglist.fre4save_model == 0:
            time_now = time.strftime('%y%m_%d%H%M')
            print('=time:{} step:{}        save'.format(time_now, game_step))
            model_file_dir = os.path.join(arglist.save_dir, '{}_{}_{}'.format( \
                arglist.scenario_name, time_now, game_step))
            if not os.path.exists(model_file_dir): # make the path
                os.mkdir(model_file_dir)
            for agent_idx, (a_c, a_t, c_c, c_t) in \
                enumerate(zip(actors_cur, actors_tar, critics_cur, critics_tar)):
                torch.save(a_c, os.path.join(model_file_dir, 'a_c_{}.pt'.format(agent_idx)))
                torch.save(a_t, os.path.join(model_file_dir, 'a_t_{}.pt'.format(agent_idx)))
                torch.save(c_c, os.path.join(model_file_dir, 'c_c_{}.pt'.format(agent_idx)))
                torch.save(c_t, os.path.join(model_file_dir, 'c_t_{}.pt'.format(agent_idx)))

        # update the tar par
        actors_tar = update_trainers(actors_cur, actors_tar, arglist.tao) 
        critics_tar = update_trainers(critics_cur, critics_tar, arglist.tao)
    return update_cnt, actors_cur, actors_tar, critics_cur, critics_tar, global_critic_cur, global_critic_tar


def train(arglist):
    """
    init the env, agent and train the agents
    """
    """step1: create the environment """
    env = make_env(arglist.scenario_name, arglist, arglist.benchmark)

    print('=============================')
    print('=1 Env {} is right ...'.format(arglist.scenario_name))
    print('=============================')

    """step2: create agents"""
    state_dim = env.state_dim  # state_dim = 36
    obs_shape_n = [env.observation_space[i].shape[0] for i in range(env.n)]
    # 8个agent, 6个追, 2个躲. obs_shape_n = [26, 26, 26, 26, 26, 26, 24, 24]
    # 追的人比躲的人多两维信息，因为追的人能知道躲的人的速度，而正好躲的人有两个人
    action_shape_n = [env.action_space[i].n for i in range(env.n)]  # no need for stop bit
    # [5, 5, 5, 5, 5, 5, 5, 5]
    num_adversaries = min(env.n, arglist.num_adversaries)
    # num_adversaries = arglist.num_adversaries
    actors_cur, critics_cur, actors_tar, critics_tar, global_critic_cur, global_critic_tar, optimizers_a, optimizers_c, optimizers_g = \
        get_trainers(env, num_adversaries, obs_shape_n, action_shape_n, state_dim, arglist)
    #memory = Memory(num_adversaries, arglist)
    memory = ReplayBuffer(arglist.memory_size)

    '''
    dim1 = sum(obs_shape_n)
    dim2 = sum(action_shape_n)
    with SummaryWriter(comment="qmix") as w:
        w.add_graph(openai_critic(sum(obs_shape_n), sum(action_shape_n), arglist), (torch.zeros(1, dim1), torch.zeros(1, dim2)))
    '''
    print('=2 The {} agents are inited ...'.format(env.n))
    print('=============================')

    """step3: init the pars """
    obs_size = []
    action_size = []
    game_step = 0
    episode_cnt = 0
    update_cnt = 0
    t_start = time.time()
    rew_n_old = [0.0 for _ in range(env.n)] # set the init reward
    agent_info = [[[]]]  # placeholder for benchmarking info
    episode_rewards = [0.0]  # sum of rewards for all agents
    agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
    head_o, head_a, end_o, end_a = 0, 0, 0, 0
    for obs_shape, action_shape in zip(obs_shape_n, action_shape_n):
        end_o = end_o + obs_shape
        end_a = end_a + action_shape 
        range_o = (head_o, end_o)
        range_a = (head_a, end_a)
        obs_size.append(range_o)
        action_size.append(range_a)
        head_o = end_o
        head_a = end_a
    # obs_size = [(0, 26), (26, 52), (52, 78), (78, 104), (104, 130), (130, 156), (156, 180), (180, 204)]
    # action_size = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40)]
    print('=3 starting iterations ...')
    print('=============================')
    obs_n = env.reset()
    # env.reset() # 最终会调用simple_tag.py 的observation函数
    # max_episode = 150000
    x_data = []
    for episode_gone in range(arglist.max_episode):
        # cal the reward print the debug data
        x_data.append(episode_gone)
        if game_step > 1 and game_step % 2000 == 0:
            """
            mean_agents_r = [round(np.mean(agent_rewards[idx][-200:-1]), 2) for idx in range(env.n)]
            mean_ep_r = round(np.mean(episode_rewards[-200:-1]), 3)
            print(" "*43 + 'episode reward:{} agents mean reward:{}'.format(mean_ep_r, mean_agents_r), end='\r')
            """
            ln1, = plt.plot(x_data, episode_rewards, color='red')
            #ln2, = plt.plot(x_data, agent_rewards[1], color='black')
            #ln3, = plt.plot(x_data, agent_rewards[2], color='yellow')
            # ln4, = plt.plot(x_data, episode_rewards, color='purple')
            # plt.legend(handles=[ln1, ln2, ln3, ln4], labels=['agent1', 'agent2', 'agent3', 'total'])
            plt.legend(handles=[ln1] , labels=['reward'])
            plt.show()

        print('=Training: steps:{} episode:{}'.format(game_step, episode_gone), end='\r')

        # per_episode_max_len = 45
        for episode_cnt in range(arglist.per_episode_max_len):
            # get action
            # agent 是从 actors_cur挑选出来的网络， obs_n 是环境的反馈，它含有8个子数组，每个子数组的大小 [26, 26, 26, 26, 26, 26, 24, 24]
            # obs 是每个agent观察到的环境的数组

            # agent根据当前策略和观察选择一个动作， action_n是所有人选择的动作
            action_n = [agent(torch.from_numpy(obs).to(arglist.device, torch.float)).detach().cpu().numpy() \
                for agent, obs in zip(actors_cur, obs_n)]

            # interact with env
            # print(action_n)
            # shape 8 * 5
            state, new_obs_n, rew_n, done_n, info_n = env.step(action_n)
            # action_n 一共有env.n个元素，每个元素的大小都是5， 所以是 8 * 5
            # env.render()
            # save the experience
            # np.concatenate(action_n) 将action_n 变为 40 * 1
            # rew_n shape 1 * 8
            # done_n shape 1 * 8
            memory.add(state, obs_n, np.concatenate(action_n), rew_n, new_obs_n, done_n)
            episode_rewards[-1] += np.sum(rew_n)
            for i, rew in enumerate(rew_n):
                agent_rewards[i][-1] += rew

            # train our agents 
            update_cnt, actors_cur, actors_tar, critics_cur, critics_tar, global_critic_cur, global_critic_tar = agents_train(\
                arglist, game_step, update_cnt, memory, obs_size, action_size, \
                actors_cur, actors_tar, critics_cur, critics_tar, optimizers_a, optimizers_c,\
                global_critic_cur, global_critic_tar, optimizers_g)

            # update the obs_n
            game_step += 1
            # print(game_step)
            obs_n = new_obs_n
            done = all(done_n)
            terminal = (episode_cnt >= arglist.per_episode_max_len-1)
            if done or terminal:
                episode_step = 0
                obs_n = env.reset()
                agent_info.append([[]])
                # print('agent reward is {}\n{}\n{}'.format(agent_rewards[0], agent_rewards[1], agent_rewards[2]))
                # print('total reward is {}'.format(episode_rewards[-1]))
                episode_rewards.append(0)
                # print(agent_rewards[0][-1], agent_rewards[1][-1])

                for i in range(env.n):
                    agent_rewards[i].append(0)
                # print(agent_rewards[1][-1])
                # continue
    print(agent_rewards[0][-1])
    print(agent_rewards[1][-1])
if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
