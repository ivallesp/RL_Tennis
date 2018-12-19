import copy
from src.models import *
from src.rl_utilities import ExperienceReplay
import random
from ipdb import set_trace
from torch import optim


class MADDPGAgent():
    def __init__(self, critic_arch, actor_arch, n_agents, state_size, action_size, tau, gamma, replay_size, batch_size,
                 n_batches_train, alpha=0, random_seed=655321):
        """
        Agent implementing MADDPG algorithm. More info here: https://arxiv.org/abs/1706.02275

        :param critic_arch: pytorch neural network implementing a critic function (s, a -> Q), located in the
        src.models module (pytorch model object)
        :param actor_arch: pytorch neural network implementing a actor function (s -> P(a|s)), located in the
        src.models module (pytorch model object)
        :param n_agents: number of agents to train in the multiagent setting (int)
        :param state_size: size of the state space (int)
        :param action_size: size of the action space (int)
        :param tau: constant controling the rate of the soft update of the target networks from the local
        networks (float)
        :param gamma: discount factor (float)
        :param replay_size: size of the experience replay buffer (int)
        :param batch_size: size of the batches which are going to be used to train the neural networks (int)
        :param n_batches_train: number of batches to train in each agent step (int)
        :param alpha: effort punishment (experiment) (float)
        :param random_seed: random seed for numpy and pytorch (int)
        """
        np.random.seed(random_seed)
        self.critic_local = critic_arch(state_size*n_agents, action_size, n_agents, random_seed)
        self.critic_target = critic_arch(state_size*n_agents, action_size, n_agents, random_seed)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=2e-4)
        
        self.actor_locals = []
        self.actor_targets = []
        self.actor_optimizers = []
        
        for i in range(n_agents):
            self.actor_locals.append(actor_arch(state_size, action_size, random_seed+i))
            self.actor_targets.append(actor_arch(state_size, action_size, random_seed+i))
            self.actor_optimizers.append(optim.Adam(self.actor_locals[i].parameters(), lr=2e-4))

        # Equalize target and local networks
        self._soft_target_update(tau=1)

        # Noise
        self.noise = [OUNoise(size=action_size, seed=random_seed) for _ in range(n_agents)]

        # Experience replay buffer
        self.replay_buffer = ExperienceReplay(int(replay_size))

        self.tau = tau
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_batches_train = n_batches_train
        self.alpha = alpha
        self.n_agents = n_agents
        self.action_size = action_size
        self.state_size = state_size

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
        # Update replay buffer<
        self.replay_buffer.append([states, actions, rewards, next_states, dones])

        for _ in range(self.n_batches_train):
            # Sample a batch of experiences
            states_batch, \
            actions_batch, \
            rewards_batch, \
            next_states_batch, \
            dones_batch = self.replay_buffer.draw_sample(self.batch_size)

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
        
        self._update_actors(states)
        self._soft_target_update()

    def reset(self):
        """
        Performs the agent related tasks required when reseting the environment.
        :return: None
        """
        for n in self.noise:
            n.reset()

    def act(self, states, epsilon=1):
        """
        Calculates the next action (given the experience) (iterable)
        :param states: observations of the MDP (iterable)
        :param epsilon: exploration rate (float)
        :return: actions to take (iterable)
        """
        states = torch.from_numpy(states).float()
        if states.dim==1:
            states = torch.unsqueeze(states, 0)
        
        actions_agents = []
        with torch.no_grad():
            for i in range(self.n_agents):
                state_local = states[[i],]
                actor = self.actor_locals[i]
                actor.eval()
                actions_agents.append(actor.forward(state_local).cpu().data.numpy())
                actor.train()
        actions = np.row_stack(actions_agents)
        actions += np.row_stack([n.sample() for n in self.noise])*epsilon
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
        for actor_target, actor_local in zip(self.actor_targets, self.actor_locals):
            actor_target.copy_weights_from(actor_local, tau)

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
        next_actions = []
        for i in range(self.n_agents):
            next_state_local = next_states[:,i,:]
            next_actions.append(torch.unsqueeze(self.actor_targets[i].forward(next_state_local),1))
        next_actions = torch.cat(next_actions, 1).view(-1, self.action_size*self.n_agents)
        actions = actions.view(-1, self.action_size*self.n_agents)
        next_states = next_states.view(-1, self.state_size*self.n_agents)
        states = states.view(-1, self.state_size*self.n_agents)
        
        q_value_next_max = self.critic_target.forward(next_states, next_actions)
        q_value_target = rewards.squeeze() + self.gamma * q_value_next_max * (1 - dones.squeeze())
        # q_value_target = torch.from_numpy(q_value_target).float()
        # Calculate the loss
        q_value_current = self.critic_local.forward(states, actions)

        loss = F.mse_loss(q_value_current, q_value_target)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

    def _update_actors(self, states):
        """
        Updates the actor network
        :param states: observation variables of the MDP (iterable)
        :return: None
        """

        
        for i in range(self.n_agents):
            actions_predicted = []
            for i in range(self.n_agents):
                state_local = states[:,i,:]
                actions_predicted.append(torch.unsqueeze(self.actor_locals[i].forward(state_local), 1))
            actions_predicted = torch.cat(actions_predicted, 1)
            
            critic_action_values = self.critic_local.forward(states.view(-1, self.state_size*self.n_agents), 
                                                             actions_predicted.view(-1, self.action_size*self.n_agents))[:,[i]]
            #set_trace()
            # Calculate the loss
            loss = -critic_action_values.mean()
            # Minimize the loss
            self.actor_optimizers[i].zero_grad()
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
            self.actor_optimizers[i].step()


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state