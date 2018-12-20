from unityagents import UnityEnvironment
import numpy as np
from src.agents import MADDPGAgent
from src.models import CriticArchitecture, ActorArchitecture
import matplotlib.pyplot as plt
from src.rl_utilities import plot_smoothed_return
from IPython.display import clear_output
import torch

env = UnityEnvironment(file_name="./envs/Tennis_Windows_x86_64/Tennis.exe")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


batch_size = 128  # Size of the batch to train the neural networks
n_episodes = 100000 # Number of episodes to run when training the agent
n_batches_train = 1 # Number of times to train for each time step
n_agents = 2 # Number of agents
exp_replay_buffer_size = int(1e5) # Experience replay buffer size
epsilon_decay =  1 # Decay of the exploration constant
epsilon = 1 # Initial value of the exploration constant
epsilon_final = 0.1 # Final value of the exploration constant
plot_every_n = 10# # Period to update the rewards chart
save_every_n = 100 # Period to save the model if an improvement has been found
tau = 0.001 # Parameter that controls how fast the local networks update the target networks
gamma = 0.99 # Discount factor


agent = MADDPGAgent(CriticArchitecture, ActorArchitecture, state_size=state_size, action_size=action_size, n_agents=n_agents,
                  tau=tau, gamma=gamma, batch_size=batch_size, replay_size = exp_replay_buffer_size,
                  n_batches_train=n_batches_train, random_seed=655321)
scores = []
epsilons = []
buffer_sizes = []
max_score = 0

for episode in range(n_episodes):
    epsilons.append(epsilon)
    epsilon = epsilon_decay * epsilon + (1 - epsilon_decay) * epsilon_final
    env_info = env.reset(train_mode=True)[brain_name]
    agent.reset()
    state = env_info.vector_observations
    score = []
    done = [False]
    c = 0
    if episode > 250:
        pass
    while not any(done):
        # take random action
        action = agent.act(state, epsilon=epsilon)
        env_info = env.step(action)[brain_name]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += [reward]
        c += 1

    scores.append(np.max(np.sum(score, axis=0)))
    buffer_sizes.append(agent.replay_buffer.length)

    #if (episode + 1) % plot_every_n == 0:
    #    clear_output(True)
    #    plt.figure(figsize=(15, 4))
    #    plt.subplot(1, 3, 1)
    #    plot_smoothed_return(scores)
    #    plt.subplot(1, 3, 2)
    #    plt.grid()
    #    plt.plot(epsilons)
    #    plt.xlabel("# of episodes")
    #    plt.title("Epsilon")
    #    plt.subplot(1, 3, 3)
    #    plt.grid()
    #    plt.plot(buffer_sizes)
    #    plt.xlabel("# of episodes")
    #    plt.title("Replay buffer size")
    #    plt.show()
    # if (episode + 1) % save_every_n == 0:
    #    if max_score < np.mean(scores[-100:]):
    #        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_local.pth')
    #        torch.save(agent.critic_target.state_dict(), 'checkpoint_critic_target.pth')
    #        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_local.pth')
    #        torch.save(agent.actor_target.state_dict(), 'checkpoint_actor_target.pth')
    #        max_score = np.mean(scores[-100:])