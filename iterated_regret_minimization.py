import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
from tensorboardX import SummaryWriter
import copy
import matplotlib.pyplot as plt

mem = dict()
gamma = 0.9

def choose_action(Qtable, state, epsilon):
    n = 101

    Qmax1, Qmax2 = -10000, -10000
    x, y = -1, -1
    for i in range(2, n):
        for j in range(2, n):
            if i not in state[0]:
                if Qtable[i][j][0] > Qmax1:
                    x = i
            if j not in state[1]:
                if Qtable[i][j][1] > Qmax2:
                    y = j
    return x, y


def learn(a1, a2, r1, r2, Qtable):
    regret = find_minimum_regret(a1, a2, state)
    # Qtable[a1][a2][0] = r1 + gamma *


def find_minimum_regret(a1, a2, state):
    n = 101
    regret_table = np.zeros((n, n, 2))
    state[0] = sorted(state[0])
    state[1] = sorted(state[1])
    min_stratey = state[0][0]
    for i in state[0]:
        for j in state[1]:
            if i > j:
                actual = j - 2
                best = j + 1 if j - 1 >= min_stratey else min_stratey
                regret_table[i][j][0] = best - actual
                regret_table[i][j][1] = i - j - 1
            elif i == j:
                if i == min_stratey:
                    regret_table[i][j][0] = 0
                    regret_table[i][j][1] = 0
                    continue
                regret_table[i][j][0] = 1
                regret_table[i][j][1] = 1
            else:
                actual = i - 2
                best = i + 1 if i - 1 >= min_stratey else min_stratey
                regret_table[i][j][1] = best - actual
                regret_table[i][j][0] = j - i - 1
            # print(i, j, regret_table[i][j][0], regret_table[i][j][1])
    # print(regret_table)
    regret_x = np.full(n, 1000)
    # print(regret_x)
    flag = 1
    min_regret = 1000
    for i in state[0]:
        tmp = -1000
        for j in state[1]:
            # print(i, j, regret_table[i][j][0])
            tmp = max(regret_table[i][j][0], tmp)
        regret_x[i] = tmp
        # print(tmp)
        if tmp != -1000:
            min_regret = min(tmp, min_regret)
    # print(min_regret)
    state_x = copy.deepcopy(state[0])
    for i in state[0]:
        # print(i)
        if regret_x[i] != min_regret:
            # print(i)
            state_x.remove(i)
            flag = 0
    print(min_regret)
    state[0] = state_x
    state[1] = state[0]
    """
    min_regret = 1000
    regret_y = np.full(n, 1000)

    for j in range(2, n):
        tmp = -1000
        for i in range(2, n):
            if i in state[0] or j in state[1]:
                continue
            tmp = max(regret_table[i][j][1], tmp)
        regret_y[j] = tmp
        # print(tmp)
        if tmp != -1000:
            min_regret = min(tmp, min_regret)
    for j in range(2, n):
        if regret_x[j] != min_regret and j not in state[1]:
            state[1].append(j)
            flag = 0
    """
    return state, flag



def main():
    Qtable = np.zeros((101, 101, 2))
    reward = np.zeros((101, 101, 2))
    n = 100

    state = [[], []]
    for i in range(2, 101):
        state[0].append(i)
        state[1].append(i)
    for i in range(100, 1, -1):
        for j in range(100, 1, -1):
            if i == j:
                for k in range(2):
                    reward[i][j][k] = i
                    reward[i][j][k] = i
            elif i > j:
                reward[i][j][0] = j - 2
                reward[i][j][1] = j + 2
            else:
                reward[i][j][0] = i + 2
                reward[i][j][1] = i - 2
            # print(i, j, reward[i][j][0], reward[i][j][1])

    episode = 10000

    """
    for i in range(episode):
        a1, a2 = choose_aciton(Qtable)
        r1, r2 = reward[a1][a2][0], reward[a1][a2][1]
        learn(a1, a2, r1, r2, Qtable)
    """
    cnt = 0
    while 1:
        print("{0} round survival strategy: {1}".format(cnt, state))
        state, flag = find_minimum_regret(0, 0, state)
        cnt += 1
        if flag == 1:
            break


if __name__ == '__main__':
    main()
