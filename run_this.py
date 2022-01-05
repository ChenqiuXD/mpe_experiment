from maze_env import Maze
from RL_brain import DeepQNetwork


def run_maze():
    step = 0
    record_return = []
    for episode in range(600):
        # initial observation
        observation = env.reset()
        # print(episode)

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            RL.store_transition(observation, action, reward, observation_)

            if (step > 200) and (step % 5 == 0):
                RL.learn()

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break
            step += 1

        # Test the expected return of current Q-network
        if episode % 25 == 0:
            exp_return = 0
            for i in range(5):
                observation = env.reset()
                while True:
                    action = RL.choose_action(observation)
                    observation_, reward, done = env.step(action)
                    exp_return += RL.gamma * reward
                    observation = observation_
                    if done:
                        break
            exp_return /= 5
            record_return.append(exp_return)

    # end of game
    print('game over')
    env.destroy()


if __name__ == "__main__":
    # maze game
    env = Maze()
    RL = DeepQNetwork(env.n_actions, env.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      e_greedy_increment=0.0
                      # output_graph=True
                      )
    env.after(100, run_maze)
    env.mainloop()
    RL.plot_cost()