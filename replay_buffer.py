import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = int(size)
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def clear(self):
        self._storage = []
        self._next_idx = 0

    def add(self, state, obs_t, action, reward, obs_tp1, done):
        data = (state, obs_t, action, reward, obs_tp1, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes, agent_idx):
        states, obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]  # 从索引中取出i条数据
            state, obs_t, action, reward, obs_tp1, done = data
            states.append(state)
            obses_t.append(np.concatenate(obs_t[:]))  # 将obs_t拍扁，压缩为一维
            actions.append(action)
            rewards.append(reward[agent_idx])  # 除了奖励和结束标志是选择自己的数据外，obs, next_obs, action 都是选取所有人的数据
            obses_tp1.append(np.concatenate(obs_tp1[:]))
            dones.append(done[agent_idx])
        return np.array(states), np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def _encode_sample_all(self, idxes):
        states, obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            state, obs_t, action, reward, obs_tp1, done = data
            states.append(state)
            obses_t.append(np.concatenate(obs_t[:]))
            actions.append(action)
            rewards.append(reward)
            obses_tp1.append(np.concatenate(obs_tp1[:]))
            dones.append(done)
        return np.array(states), np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):
        return [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) % self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size, agent_idx):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        # make_index是从buffer中随机选择一批数量为batch_size的索引
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample(idxes, agent_idx)

    def sample_all(self, batch_size):
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, len(self._storage))
        return self._encode_sample_all(idxes)

    def collect(self):
        return self.sample(-1)
