import numpy as np


class Agent(object):
    def __init__(self):
        self.Q = np.zeros(9)
        self.lr = 0.1
        self.gamma = 0.9
        self.pos = -1


dir = dict()
dir[0] = [1, 3]
dir[1] = [0, 4, 2]
dir[2] = [1, 5]
dir[3] = [0, 6, 4]
dir[4] = [1, 3, 5, 7]
dir[5] = [2, 4, 8]
dir[6] = [3, 7]
dir[7] = [6, 4, 8]
dir[8] = [7, 5]


def find_max_index(agent):
    avail_actions = dir[agent.pos]
    index = -1
    max = -1000

    for i in avail_actions:
        if agent.Q[i] > max:
            index = i
            max = agent.Q[i]
    return index


def move_forward(agent1, agent2, action1, action2):
    if action1 == action2:
        return -10, -10, False
    if action1 == 6 and action2 == 8 or action1 == 8 and action2 == 6:
        return 100, 100, True
    elif action1 == 6 or action2 == 6 or action2 == 8 or action1 == 8:
        return 0, 0, True
    else:
        agent1.pos = action1
        agent2.pos = action2
        return -1, -1, False



def get_reward(action1, action2):
    if action1 == action2:
        return -10, -10
    if action1 == 6 and action2 == 8 or action1 == 8 and action2 == 6:
        return 100, 100
    elif action1 == 6 or action2 == 6 or action2 == 8 or action1 == 8:
        return 0, 0
    return -1, -1, False


def compute_regret(old_pos_x, old_pos_y, reward1, reward2):
    avail_x = dir[old_pos_x]
    avail_y = dir[old_pos_y]

    regret_x = 100000
    for i in avail_x:
        for j in avail_y:
            r1, r2 = get_reward(i, j)





if __name__ == '__main__':
    agent1 = Agent()
    agent2 = Agent()
    agent1.pos = 0
    agent2.pos = 2

    grid_word = np.zeros(9)
    grid_word[6] = 1
    grid_word[8] = 1

    episode = 0
    rewards, episodes = [], []
    len = 100000
    reward = 0
    epsilon = 0.6

    while episode < len:
        p = np.random.rand()
        if p < epsilon:
            action1 = np.random.randint(2, 101)
            action2 = np.random.randint(2, 101)
        else:
            action1 = find_max_index(agent1)
            action2 = find_max_index(agent2)
        old_pos_x = agent1.pos
        old_pos_y = agent2.pos

        reward1, reward2, done = move_forward(agent1, agent2, action1, action2)
        reward = reward1 + reward2 + reward
        regret_x, regret_y = compute_regret(old_pos_x, old_pos_y, action1, action2, reward1, reward2)
