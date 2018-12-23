import copy
from src.rl_utilities import ExperienceReplay
import random
import numpy as np
from torch import optim
import torch
import torch.nn.functional as F
from ipdb import set_trace


class DDPGAgent:
    def __init__(self, critic_arch, actor_arch, state_size, action_size, tau, gamma, replay_size, batch_size,
                 n_agents=1, n_batches_train=1, alpha=0, random_seed=655321):
        """
        Agent implementing DDPG algorithm. More info here: https://arxiv.org/abs/1509.02971
        :param critic_arch: pytorch neural network implementing a critic function (s, a -> Q), located in the
        src.models module (pytorch model object)
        :param actor_arch: pytorch neural network implementing a actor function (s -> P(a|s)), located in the
        src.models module (pytorch model object)
        :param state_size: size of the state space (int)
        :param action_size: size of the action space (int)
        :param tau: constant controling the rate of the soft update of the target networks from the local
        networks (float)
        :param gamma: discount factor (float)
        :param replay_size: size of the experience replay buffer (int)
        :param batch_size: size of the batches which are going to be used to train the neural networks (int)
        :param n_agents: number of agents (in case of MADDPG, the critic network should implement a centralized
         action-value function) (int)
        :param n_batches_train: number of batches to train in each agent step (int)
        :param alpha: effort punishment (experiment) (float)
        :param random_seed: random seed for numpy and pytorch (int)
        """
        np.random.seed(random_seed)
        self.critic_local = critic_arch(state_size=state_size, action_size=action_size,
                                        n_agents=n_agents, random_seed=random_seed)
        self.critic_target = critic_arch(state_size=state_size, action_size=action_size,
                                         n_agents=n_agents, random_seed=random_seed)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=1e-4)

        self.actor_local = actor_arch(state_size=state_size, action_size=action_size,
                                      random_seed=random_seed)
        self.actor_target = actor_arch(state_size=state_size, action_size=action_size,
                                       random_seed=random_seed)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=1e-4)

        # Equalize target and local networks
        self._soft_target_update(tau=1)

        # Noise
        self.noise = OUNoise(action_dimension=action_size, scale=1.0)

        # Experience replay buffer
        self.replay_buffer = ExperienceReplay(int(replay_size))

        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_batches_train = n_batches_train
        self.alpha = alpha
        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size

    def step(self, states, actions, rewards, next_states, dones):
        """
        Update the experience replay buffer and trains the networks
        :param states: observation variables of the MDP (iterable)
        :param actions: actions taken at each step (iteranble)
        :param rewards: rewards achieved (iterable)
        :param next_states: next observation where the agent lead to (iterable)
        :param dones: did the environment done? (iterable)
        :return: None
        """
        # Update replay buffer
        for s, a, r, ns, d in zip(states, actions, rewards, next_states, dones):
            self.replay_buffer.append([s, a, r, ns, d])

        for _ in range(self.n_batches_train):
            # Sample a batch of experiences
            states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = \
                self.replay_buffer.draw_sample(self.batch_size)

            effort = torch.sqrt((actions_batch**2).sum(dim=1))
            effort = torch.unsqueeze(effort, 1)
            rewards_batch -= self.alpha*effort

            # Train
            if self.replay_buffer.length > self.batch_size:
                self.update(states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch)

    def update(self, states, actions, rewards, next_states, dones):
        """
        Updates the networks using the data provided
        :param states: observation variables of the MDP (iterable)
        :param actions: actions taken at each step (iteranble)
        :param rewards: rewards achieved (iterable)
        :param next_states: next observation where the agent lead to (iterable)
        :param dones: did the environment done? (iterable)
        :return: None
        """
        self._update_critic(states, actions, rewards, next_states, dones)
        self._update_actor(states)
        self._soft_target_update()

    def reset(self):
        """
        Performs the agent related tasks required when reseting the environment.
        :return: None
        """
        self.noise.reset()

    def act(self, states, epsilon=1, use_target_model=False):
        """
        Calculates the next action (given the experience) (iterable)
        :param states: observations of the MDP (iterable)
        :param epsilon: exploration rate (float)
        :param use_target_model: indicates if target model should be used instead of local model (bool)
        :return: actions to take (iterable)
        """
        model = self.actor_local if not use_target_model else self.actor_target
        states = torch.from_numpy(states).float()
        if states.dim==1:
            states = torch.unsqueeze(states, 0)
        model.eval()
        with torch.no_grad():
            actions = model.forward(states).cpu().data.numpy()
        model.train()
        actions += self.noise.sample()*epsilon
        actions = np.clip(actions, -1, 1)
        return actions

    def _soft_target_update(self, tau=None):
        """
        Updates the target networks a step towards the critic ones
        :param tau: Update rate
        :return: None
        """
        if tau is None:
            tau = self.tau
        self.critic_target.copy_weights_from(self.critic_local, tau)
        self.actor_target.copy_weights_from(self.actor_local, tau)

    def _update_critic(self, states, actions, rewards, next_states, dones):
        """
        Updates the critic network
        :param states: observation variables of the MDP (iterable)
        :param actions: actions taken at each step (iteranble)
        :param rewards: rewards achieved (iterable)
        :param next_states: next observation where the agent lead to (iterable)
        :param dones: did the environment done? (iterable)
        :return: None
        """
        # Calculate td target
        next_actions = self.actor_target.forward(next_states)

        q_value_next_max = self.critic_target.forward(next_states, next_actions)
        q_value_target = rewards + self.gamma * q_value_next_max * (1 - dones)
        # q_value_target = torch.from_numpy(q_value_target).float()
        # Calculate the loss
        q_value_current = self.critic_local.forward(states, actions)

        loss = F.mse_loss(q_value_current, q_value_target)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

    def _update_actor(self, states):
        """
        Updates the actor network
        :param states: observation variables of the MDP (iterable)
        :return: None
        """
        actions_predicted = self.actor_local.forward(states)
        critic_action_values = self.critic_local.forward(states, actions_predicted)
        # Calculate the loss
        loss = -critic_action_values.mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()


class OUNoise:

    def __init__(self, action_dimension, scale=0.1, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = scale
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return torch.tensor(self.state * self.scale).float()



class MADDPGAgent:
    def __init__(self, critic_arch, actor_arch, state_size, action_size, tau, gamma, replay_size, batch_size,
                 n_agents=1, n_batches_train=1, alpha=0, random_seed=655321):
        # Initialize agents
        self.agents = [DDPGAgent(critic_arch=critic_arch,
                                 actor_arch=actor_arch,
                                 state_size=state_size,
                                 action_size=action_size,
                                 tau=tau,
                                 gamma=gamma,
                                 replay_size=replay_size,
                                 batch_size=batch_size,
                                 n_agents=n_agents,
                                 n_batches_train=n_batches_train,
                                 alpha=alpha,
                                 random_seed=random_seed+i)
                       for i in range(n_agents)]

        # General experience replay buffer
        self.replay_buffer = ExperienceReplay(int(replay_size))

        self.n_agents = n_agents
        self.state_size = state_size
        self.action_size = action_size
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.n_batches_train = n_batches_train

    def act(self, states, epsilon, use_target_model=False):
        assert states.shape[0] == self.n_agents
        actions = []
        for i, agent in enumerate(self.agents):
            actions.append(agent.act(states=states[[i], :], epsilon=epsilon,
                                     use_target_model=use_target_model))
        actions = np.array(actions).squeeze()
        return actions

    def step(self, states, actions, rewards, next_states, dones):
        self.replay_buffer.append([states, actions, rewards, next_states, dones])

        for _ in range(self.n_batches_train):
            # Sample a batch of experiences
            states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch = \
                self.replay_buffer.draw_sample(self.batch_size)

            # Train
            if self.replay_buffer.length > self.batch_size:
                self.update(states_batch, actions_batch, rewards_batch, next_states_batch, dones_batch)

    def update(self, states, actions, rewards, next_states, dones):
        for i, agent in enumerate(self.agents):
            self._update_critics(states=states, actions=actions, rewards=rewards, next_states=next_states,
                                 dones=dones, agent_number=i)
            self._update_actors(states, agent_number=i)
            agent._soft_target_update()

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def _update_critics(self, states, actions, rewards, next_states, dones, agent_number):
        agent = self.agents[agent_number]
        # Calculate next actions centralized using actor target network
        next_actions_centralized = []
        for i_critic, agent_critic in enumerate(self.agents):
            next_actions_centralized.append(agent_critic.actor_target.forward(next_states[:, i_critic, :]))
        next_actions_centralized = torch.cat(next_actions_centralized, 1)

        # Calculate the centralized versions of the states, next states and actions
        states_centralized = states.view(-1, self.state_size*self.n_agents)
        next_states_centralized = next_states.view(-1, self.state_size*self.n_agents)
        actions_centralized = actions.view(-1, self.action_size*self.n_agents)

        # Calculate the q_value future to calculate the target
        with torch.no_grad():
            q_value_next_max = agent.critic_target.forward(next_states_centralized, next_actions_centralized).detach()
        q_value_target = rewards.squeeze()[:,[agent_number]] + agent.gamma * q_value_next_max * (1 - dones.squeeze()[:,[agent_number]])

        # Calculate the current q value
        q_value_current = agent.critic_local.forward(states_centralized, actions_centralized)

        # Calculate the loss
        loss = F.mse_loss(q_value_current, q_value_target.detach())

        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()

    def _update_actors(self, states, agent_number):
        states_centralized = states.view(-1, self.state_size * self.n_agents)
        agent = self.agents[agent_number]
        actions_pred_centralized = []
        for i_critic, agent_critic in enumerate(self.agents):
            action_pred = agent_critic.actor_local.forward(states[:, i_critic, :])
            action_pred = action_pred.detach() if agent_number!=i_critic else action_pred
            actions_pred_centralized.append(action_pred)

        actions_pred_centralized = torch.cat(actions_pred_centralized, 1)
        critic_action_values = agent.critic_local.forward(states_centralized, actions_pred_centralized)
        # Calculate the loss
        loss = -critic_action_values.mean()
        #set_trace()
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        agent.actor_optimizer.step()
