from unityagents import UnityEnvironment
from unity_env_wrapper import EnvMultipleWrapper
import numpy as np

unity_env = UnityEnvironment(file_name="Tennis.app")
env = EnvMultipleWrapper(env=unity_env, train_mode=True)

action_size = env.action_size
state_size = env.state_size
num_agents = env.num_agents

from p3_maddpg import MADDPG
from ReplayBuffer import ReplayBuffer
from collections import deque
import pickle as pkl

seed = 1
GAMMA = 0.99  # discount factor
UPDATE_EVERY = 1  # update once after every UPDATE_EVERY stpes
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 256  # batch size
TAU = 1e-3  # for soft update of target parameters

def maddpg_train():
    np.random.seed(seed)
    torch.manual_seed(seed)

    num_agents = 2

    num_episodes = 5000

    print_every = 100

    # initialize ReplayBuffer
    buffer = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, seed)

    # initialize policy and critic
    maddpg = MADDPG(gamma=GAMMA, tau=TAU, state_size=state_size, action_size=action_size, seed=seed)

    scores_window = deque(maxlen=100)
    scores = []

    for e_ in range(1, num_episodes + 1):
        # Initialize time step (for updating every UPDATE_EVERY steps)
        t_step = 0

        # record undiscounted rewards of 1 episode
        undiscounted_episode_reward = np.zeros((num_agents))

        states = env.reset()  # reset the environment
        maddpg.reset()

        while True:
            # get actions with noise
            actions = maddpg.act_local(states)
            # send all actions to tne environment
            next_states, rewards, dones = env.step(actions)
            # get next state (for each agent)
            states_full = np.concatenate((states[0], states[1]))
            next_states_full = np.concatenate((next_states[0], next_states[1]))

            buffer.add(states, states_full, actions, rewards, next_states, next_states_full, dones)

            # update once after every UPDATE_EVERY stpes
            t_step = (t_step + 1) % UPDATE_EVERY
            if t_step == 0:
                if len(buffer) > BATCH_SIZE:
                    for agent_idx in range(num_agents):
                        experiences = buffer.sample()
                        maddpg.learn(experiences, agent_idx)
                    # soft update
                    for agent in maddpg.maddpg_agent:
                        agent.update()

            # accumulate undiscounted rewards of two agent over every time step of whole episode
            undiscounted_episode_reward += rewards

            states = next_states

            if np.any(dones):
                break

        # take the maximum as the rewards of this episode
        max_reward = np.max(undiscounted_episode_reward)
        scores_window.append(max_reward)
        scores.append(max_reward)

        print('\rEpisode {}\tAverage Score: {:.4f}'.format(e_, np.mean(scores_window)), end="")

        # print every 100 episodes
        if e_ % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.4f}'.format(e_, np.mean(scores_window)))
            torch.save(maddpg.maddpg_agent[0].actor_local.state_dict(), 'agent_0_checkpoint_actor.pth')
            torch.save(maddpg.maddpg_agent[0].critic_local.state_dict(), 'agent_0_checkpoint_critic.pth')

            torch.save(maddpg.maddpg_agent[1].actor_local.state_dict(), 'agent_1_checkpoint_actor.pth')
            torch.save(maddpg.maddpg_agent[1].critic_local.state_dict(), 'agent_1_checkpoint_critic.pth')
            with open('training_scores_curve.pickle', 'wb') as f:
                pkl.dump(scores, f)

        # break if average score in the window is bigger than certain number
        if np.mean(scores_window) >= 0.5:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.4f}'.format(e_ - 100,
                                                                                         np.mean(scores_window)))
            torch.save(maddpg.maddpg_agent[0].actor_local.state_dict(), 'agent_0_checkpoint_actor.pth')
            torch.save(maddpg.maddpg_agent[0].critic_local.state_dict(), 'agent_0_checkpoint_critic.pth')

            torch.save(maddpg.maddpg_agent[1].actor_local.state_dict(), 'agent_1_checkpoint_actor.pth')
            torch.save(maddpg.maddpg_agent[1].critic_local.state_dict(), 'agent_1_checkpoint_critic.pth')
            with open('training_scores_curve.pickle', 'wb') as f:
                pkl.dump(scores, f)

            break
    return scores

import torch
import matplotlib.pyplot as plt

scores = maddpg_train()

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.savefig("scores_curve.jpg", format='jpg')
plt.show()