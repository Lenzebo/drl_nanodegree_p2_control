from utils import *


def main():
    env, brain_name, num_agents, state_size, action_size = initEnvironment()
    agent = createAgent(action_size, state_size, num_agents)
    agent.load()

    scores = playOneEpisode(env, brain_name, agent)

    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

    env.close()


if __name__ == "__main__":
    main()
