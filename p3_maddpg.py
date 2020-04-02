from p3_ddpg import DDPG_Agent
import torch
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPSILON_DECAY = 0.99    # decay rate for noise process

class MADDPG():
    """ Mlti-agent DDPG """

    def __init__(self, gamma=0.99, tau=0.02, state_size=None, action_size=None, seed=1):
        super(MADDPG, self).__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.maddpg_agent = [DDPG_Agent(state_size, action_size, seed, tau=tau),
                             DDPG_Agent(state_size, action_size, seed, tau=tau)
                             ]

        self.num_agents = len(self.maddpg_agent)
        self.gamma = gamma
        self.iter = 0

    def reset(self):
        for agent in self.maddpg_agent:
            agent.reset()

    # def get_loacal_actors(self):
    #     """get actors of all the agents in the MADDPG object"""
    #     actors = [agent.actor_local for agent in self.maddpg_agent]
    #     return actors
    #
    # def get_target_actors(self):
    #     """get target_actors of all the agents in the MADDPG object"""
    #     actors = [agent.actor_target for agent in self.maddpg_agent]
    #     return actors

    def act_local(self, states):
        """get actions from all agents in the MADDPG object"""
        actions = []
        for ii, (agent, state) in enumerate(zip(self.maddpg_agent, states)):
            if agent.epsilon > 0.01:
                agent.epsilon *= EPSILON_DECAY
            actions.append(agent.act_local(state).squeeze())
        return actions

    def act_target(self, states):
        """get target actions from all agents in the MADDPG object"""
        actions = []
        for ii, (agent, state) in enumerate(zip(self.maddpg_agent, states)):
            # if agent.epsilon > 0.01:
            #     agent.epsilon *= EPSILON_DECAY
            actions.append(agent.act_target(state).squeeze())
        return actions

    # def add(self, agents, states, states_full, actions, rewards, next_states, next_states_full, dones):
    #     """ add experience to each agents """
    #     for ii, agent in enumerate(agents):
    #         agent.memory.add(states, states_full, actions, rewards[ii], next_states, next_states_full, dones)

    def learn(self, samples, agent_idx):
        """update the critics and actors of all the agents """
        states, states_full, actions, rewards, next_states, next_states_full, dones = samples
        # states, next_states: [batch_size, num_agents, 24]
        # states_full, next_states_full: [batch_size, 48]
        # actions: [batch_size, num_agents, 2]
        # dones, rewards: [batch_size, 2]

        # critic = self.maddpg_agent[0] # use the critic of the first agent as the common critic
        agent = self.maddpg_agent[agent_idx]

        #---------------------------------------- update critic ----------------------------------------
        # get the action of each agent given corresponding state
        target_action_next = self.act_target(next_states.transpose(0, 1))
        # target Q value of next state
        cat_target_actions = torch.from_numpy(
            np.concatenate([action for action in target_action_next], axis=1)).float().to(device)

        with torch.no_grad():
            Q_target_next = agent.critic_target(next_states_full,
                                                 cat_target_actions)
        # y_i
        y = rewards[:, agent_idx] + self.gamma * Q_target_next.squeeze() * (1 - dones[:, agent_idx])
        # predicted Q value of current state
        cat_actions = torch.cat([action for action in actions.transpose(0, 1)], dim=1).float().to(device)

        Q_local_current = agent.critic_local(states_full, cat_actions).squeeze()
        # loss of critic
        # detach makes involved params from target network not trainable
        critic_loss = F.mse_loss(Q_local_current, y.detach())
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1.0)
        agent.critic_optimizer.step()

        #---------------------------------------- update actor ----------------------------------------
        #update actor network using policy gradient
        action = agent.actor_local(states[:, agent_idx])
        actions_clone = actions.clone()
        actions_clone[:, agent_idx] = action

        # get the policy gradient
        # For agent_{i}, its Q is calculated by full states observed by all agents
        # and all actions from experiences, except action a_{i}
        actor_loss = -agent.critic_local(states_full, torch.cat([a for a in actions_clone.transpose(0, 1)], dim=1)).mean()
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.actor_local.parameters(), 1.0)
        agent.actor_optimizer.step()

        # # ---------------------------- update noise ---------------------------- #
        # for agent in self.maddpg_agent:
        #     agent.epsilon -= EPSILON_DECAY
        # agent.noise.reset()

    # def update_targets(self, agent, agent_critic):
    #     """soft update targets
    #     Params
    #     ======
    #     agent_critic: the agent whose critic network is used as common critic
    #     """
    #     self.iter += 1
    #     self.soft_update(agent.actor_local, agent.actor_target, self.tau)
    #     self.soft_update(agent_critic.critic_local, agent_critic.critic_target, self.tau)
    #
    #

