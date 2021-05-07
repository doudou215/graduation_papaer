import numpy as np
import copy
from random import choice

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
    regret_x = np.full(n, 1000)
    flag = 1
    min_regret = 1000
    for i in state[0]:
        tmp = -1000
        for j in state[1]:
            tmp = max(regret_table[i][j][0], tmp)
        regret_x[i] = tmp
        if tmp != -1000:
            min_regret = min(tmp, min_regret)
    state_x = copy.deepcopy(state[0])
    retx = min_regret
    for i in state[0]:
        if regret_x[i] != min_regret:
            state_x.remove(i)
            flag = 0

    regret_y = np.full(n, 1000)
    min_regret = 1000
    for j in state[1]:
        tmp = -1000
        for i in state[0]:
            tmp = max(regret_table[i][j][1], tmp)
        regret_y[j] = tmp
        if tmp != -1000:
            min_regret = min(tmp, min_regret)
    rety = min_regret
    state_y = copy.deepcopy(state[1])
    for j in state[1]:
        if regret_y[j] != min_regret:
            state_y.remove(j)
            flag = 0
    state[0] = state_x
    state[1] = state_y
    print(state)
    return state, flag, retx, rety


def choose_action(agent1, agent2, states):
    state1 = "".join(str(i) for i in states[0])
    state2 = "".join(str(i) for i in states[1])
    next_states, done, ret_x, ret_y = find_minimum_regret(agent1, agent2, states)

    action1, action2 = choice(next_states[0]), choice(next_states[1])
    next_state = ["", ""]
    next_state[0] = "".join(str(i) for i in next_states[0])
    next_state[1] = "".join(str(i) for i in next_states[1])
    if next_state[0] not in agent1.Qtable:
        agent1.Qtable[next_state[0]] = 0
    if next_state[1] not in agent2.Qtable:
        agent2.Qtable[next_state[1]] = 0
    predict_Q = agent1.Qtable[state1]
    if done:
        target_Q = agent1.reward[action1, action2] - ret_x
    else:
        target_Q = agent1.reward[action1, action2] - ret_x + agent1.gamma * agent1.Qtable[next_state[0]]
    agent1.Qtable[state1] = predict_Q + agent1.lr * (target_Q - predict_Q)

    predict_Q = agent2.Qtable[state2]
    if done:
        target_Q = agent2.reward[action2, action1] - ret_x
    else:
        target_Q = agent2.reward[action2, action1] - ret_x + agent2.gamma * agent2.Qtable[next_state[1]]
    agent1.Qtable[state2] = predict_Q + agent2.lr * (target_Q - predict_Q)

    return done, next_states


class Agent(object):
    def __init__(self):
        # self.state = np.zeros((1, 101))
        self.reward = np.zeros((101, 101))
        # self.Qtable = dict()
        self.Q = np.zeros(101)
        self.lr = 0.1
        self.gamma = 0.9
        for i in range(2, 101):
            for j in range(2, 101):
                if i == j:
                    self.reward[i][j] = i
                elif i > j:
                    self.reward[i][j] = j - 2
                else:
                    self.reward[i][j] = i + 2


def find_max_index(Q):
    max = -1000
    index = -1
    for i in range(2, 101):
        if Q[i] > max:
            index = i
            max = Q[i]
    return index

if __name__ == '__main__':
    agent1 = Agent()
    agent2 = Agent()
    """
    state = [[], []]
    for i in range(2, 101):
        state[0].append(i)
        state[1].append(i)
    agent1.Qtable["".join(str(i) for i in state[0])] = 0
    agent2.Qtable["".join(str(i) for i in state[1])] = 0
    """
    cnt = 0
    flag = 1
    # print(state)
    episode = 0
    len = 500000
    epsilon = 0.8

    while episode < len:
        p = np.random.rand()
        # print(p)
        if p < epsilon:
            action1 = np.random.randint(2, 101)
            action2 = np.random.randint(2, 101)
        else:
            action1 = find_max_index(agent1.Q)
            action2 = find_max_index(agent2.Q)
        reward1 = agent1.reward[action1, action2]
        reward2 = agent2.reward[action2, action1]
        if action1 == action2 and action1 == 97:
            reward1 *= 100
            reward2 *= 100
        # print(action1, action2, reward1, reward2)
        if action1 == 97:
            regret_x = 5 / 3
        elif action1 >= 96 and action1 <= 100:
            regret_x = 5 / 2
        else:
            regret_x = 3

        if action2 == 97:
            regret_y = 5 / 3
        elif action2 >= 96 and action2 <= 100:
            regret_y = 5 / 2
        else:
            regret_y = 3

        predict_Q = agent1.Q[action1]
        target_Q = reward1 - regret_x
        agent1.Q[action1] = predict_Q + agent1.lr * (target_Q - predict_Q)

        predict_Q = agent2.Q[action2]
        target_Q = reward2 - regret_y
        agent2.Q[action2] = predict_Q + agent2.lr * (target_Q - predict_Q)

        epsilon -= 0.0000016
        episode += 1
        # print(reward1, reward2)

    print(agent1.Q, np.argmax(agent1.Q))
    print(agent2.Q, np.argmax(agent2.Q))



