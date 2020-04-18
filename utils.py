from unityagents import UnityEnvironment

import numpy as np
import torch

from PPOAgent import Agent, HyperParameter
from Model import PPOPolicyNetwork




def initEnvironment():
    # env = UnityEnvironment(file_name='Reacher_1Agent')
    env = UnityEnvironment(file_name='Reacher')
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
    print(env_info)
    print(states)
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    return env, brain_name, num_agents, state_size, action_size

def createAgent(action_size, state_size, num_agents):
    device = "cuda:0"
    seed = 10

    params = HyperParameter()
    print(params)
    policy = PPOPolicyNetwork(state_size, action_size, params.hidden_size,  device, seed)
    agent = Agent(num_agents, state_size, action_size, device,policy, hyperparameter=params)
    return agent


def playOneEpisode(environment, agent, brain_name, train_mode=False):
    state = environment.reset(train_mode=train_mode)[brain_name].vector_observations
    num_agents = len(state)
    total_r = np.zeros(num_agents)
    timestep = 0
    while True:
        timestep += 1
        action = agent.act(state)
        env_info = environment.step(action)[brain_name]  # send the action to the environment
        next_state = env_info.vector_observations  # get the next state
        reward = env_info.rewards  # get the reward
        done = env_info.local_done  # see if episode has finished
        state = next_state
        total_r += reward
        if np.any(done):
            break
    return total_r


def movingaverage(values, window):
    weights = np.repeat(1.0, window) / window
    sma = np.convolve(values, weights, 'valid')
    return sma


def plotScores(scores, desired_score, ma_window = 100, show_window = False, filename="scores.png"):
    import matplotlib.pyplot as plt
    ma_scores = movingaverage(scores, ma_window)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.plot(np.arange(len(ma_scores)) + ma_window, ma_scores)
    plt.plot([0, len(scores)], [desired_score, desired_score])
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(filename)
    if show_window:
        plt.show()

