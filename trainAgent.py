from utils import *
from collections import deque

import tqdm # used for progress bar


desired_score = 30
ma_window = 100

def trainAgent(agent, brain_name, n_episodes=5000):
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=ma_window)  # last 100 scores
    t = tqdm.tqdm(range(1, n_episodes + 1))
    min_episodes = 500
    try:
        for i_episode in t:
            agent.train_step(env, brain_name)

            total_r = playOneEpisode(env, agent, brain_name, train_mode=True)
            scores_window.append(np.mean(total_r))  # save most recent score
            scores.append(np.mean(total_r))  # s|ave most recent score

            t.set_postfix(score=np.mean(total_r), avg_score=np.mean(scores_window))
            if i_episode % ma_window == 0:
                agent.save('model/model_i_' + str(i_episode) + '.pth')
            if np.mean(scores_window) >= desired_score and i_episode > min_episodes:
                break
        print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))

    except KeyboardInterrupt:
        print("Training interrupted....")
    agent.save()
    return scores



env, brain_name, num_agents, state_size, action_size = initEnvironment()
agent = createAgent(action_size, state_size, num_agents)
scores = trainAgent(agent, brain_name, n_episodes=5000)

plotScores(scores, desired_score, ma_window, show_window = True)

env.close()