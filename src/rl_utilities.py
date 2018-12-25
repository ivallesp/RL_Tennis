import pandas as pd
import numpy as np
import random
from collections import deque
import torch
import matplotlib.pyplot as plt
from ipdb import set_trace

def ewma(x, span=100):
    """
    Weighted average of an iterable
    :param x:  input for which we want to calculate the average
    :param span: Number of samples to take into account
    :return: wrighted average (pandas series)
    """
    return pd.Series(x).ewm(span=span).mean()


def ewmsd(x, span=100):
    """
    Weighted standard deviation of an iterable
    :param x:  input for which we want to calculate the average
    :param span: Number of samples to take into account
    :return: wrighted average (pandas series)
    """
    return pd.Series(x).ewm(span=span).std()


def plot_smoothed_return(scores, span=100, title=""):
    """
    Function in charge of genetrating the figures for visualization
    :param scores:
    :param span:
    :return:
    """
    means = ewma(scores, span)
    stds = ewmsd(scores, span)

    plt.grid()

    plt.scatter(range(len(scores)), scores, alpha=1, s=1, color="grey")
    plt.fill_between(range(len(means)), means + stds, means - stds,
                     color='#1f77b4', alpha=.3)
    plt.plot(means, color='#1f77b4')
    plt.title(title)
    plt.xlabel("# of episodes")
    plt.ylabel("Cummulative reward")
    plt.legend(["Average return", "Individual returns", "Standard deviation"])


class ExperienceReplay():
    def __init__(self, size=int(1e5)):
        """
        Experience replay buffer
        :param size: size of the batch
        """
        self.size = size
        self.reset()


    @property
    def length(self):
        """
        Returns the current stte of the experienced replay buffer
        :return:
        """
        return len(self.buffer)

    def reset(self):
        """
        Resets the experience replay buffer
        :return:
        """
        self.buffer = deque(maxlen=self.size)

    def append(self, observation):
        """
        Appends a new observation to the experience replay buffer
        :param observation: the observation to be appended (iterable
        :return: None
        """
        self.buffer.append(observation)

    def draw_sample(self, sample_size):
        """
        Draws a completely random sample for training the predictive models
        :param sample_size: batch size of the sample to draw(int)
        :return: list of iterables (state, reward, erc)
        """
        buffer_sample = random.choices(self.buffer, k=sample_size)
        #set_trace()
        states, actions, rewards, next_states, dones = zip(*buffer_sample)
        states = [torch.from_numpy(np.array(s)).float() for s in zip(*states)]
        actions = [torch.from_numpy(np.array(a)).float() for a in zip(*actions)]
        rewards = torch.from_numpy(np.array(rewards)).float()
        next_states = [torch.from_numpy(np.array(ns)).float() for ns in zip(*next_states)]
        dones = torch.from_numpy(np.array(dones)+0).float()
        return states, actions, rewards, next_states, dones