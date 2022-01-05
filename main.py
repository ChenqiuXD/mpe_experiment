import time
import make_env
import numpy as np
from RL_brain import DeepQNetwork
import pickle
import tensorflow as tf

# Constant variables Reference: https://github.com/AI4Finance-Foundation/ElegantRL/blob/master/elegantrl/train/config.py
MAX_EPISODE = 100
MAX_STEP = 200
LR_RATE = 0.0001            # roughly 1e-4 ~ 5e-5
REWARD_DECAY = 0.99         # 1/(1-gamma) = steps (which is the expected steps to consider)
E_GREEDY = 0.9
REPLACE_TAR_ITER = 200
MEMORY_SIZE = 2**12         # Bigger the better
E_GREEDY_INCRE = 0.0
BATCH_SIZE = 32             # 2*dim_net = 2*16

# Try out for different parameters
# LR_RATE_LIST = np.arange(0.01, 0.05, 0.02)
# REPLACE_TAR_ITER_LIST = np.arange(100, 201, 50)
# BATCH_SIZE_LIST = [8, 16, 32, 64]
recorded_return_dict = {}

# Environment# New environment
env = make_env.make_env('simple_spread')
obs = env.reset()
env.discrete_action_input = True

# Establish the individual agent brain (i.e. Q-network)
Q_network = []
for i, agent in enumerate(env.agents):
    Q_network.append(
        DeepQNetwork(
            env.action_space[i].n, env.observation_space[i].shape[0]-4, # Since communication has no effect
            learning_rate=LR_RATE,
            reward_decay=REWARD_DECAY,
            e_greedy=E_GREEDY,
            replace_target_iter=REPLACE_TAR_ITER,
            memory_size=MEMORY_SIZE,
            e_greedy_increment=E_GREEDY_INCRE,
            batch_size=BATCH_SIZE,
            index=i
            # output_graph=True
        )
    )

# Tryout for best parameters
iter_index = 0
# while iter_index < len(LR_RATE_LIST)*len(REPLACE_TAR_ITER_LIST)*len(BATCH_SIZE_LIST):
while iter_index < 1:
    # Assign parameters
    # LR_RATE = LR_RATE_LIST[int(iter_index/len(BATCH_SIZE_LIST)/len(REPLACE_TAR_ITER_LIST)) % len(LR_RATE_LIST)]
    # REPLACE_TAR_ITER = REPLACE_TAR_ITER_LIST[int(iter_index/len(BATCH_SIZE_LIST)) % len(REPLACE_TAR_ITER_LIST)]
    # BATCH_SIZE = BATCH_SIZE_LIST[iter_index % len(BATCH_SIZE_LIST)]
    # print("This is iteration count: " + str(iter_index) + " with parameters: " +
    #       str(LR_RATE) + ' ' +
    #       str(REPLACE_TAR_ITER) + ' ' +
    #       str(BATCH_SIZE))

    # Adjust parameters
    for i, agent in enumerate(env.agents):
        Q_network[i].lr = LR_RATE
        Q_network[i].replace_target_iter = REPLACE_TAR_ITER
        Q_network[i].batch_size = BATCH_SIZE
        Q_network[i].init_op()

    # Train agents
    time.sleep(1)
    return_list = []    # Used to record the experimental return procedure
    for episode in range(MAX_EPISODE):
        # Initial observi]ation
        observation = env.reset()
        # Delete the last four observations since it is the communication and has no effect
        for i, agent in enumerate(env.agents):
            observation[i] = observation[i][:-4]

        # Start episode
        step = 0
        exp_return = 0 # accumulated return
        while step < MAX_STEP:
            # Fresh env
            env.render(mode='other_than_human')
            # This parameters originally was 'human', adapted since it would output messaged passed between agents.

            # RL choose actions based on observations
            actions = []
            for i, agent in enumerate(env.agents):
                action = Q_network[i].choose_action(observation[i])
                actions.append(action)

            # Take action and get next observation and reward
            observation_n, reward_n, done_n, info_n = env.step(actions)
            for i, agent in enumerate(env.agents):
                observation_n[i] = observation_n[i][:-4]
            # Calculate experiment accumulated reward
            exp_return += pow(REWARD_DECAY, step)*reward_n[0]

            # Store transition and train the agents
            for i, agent in enumerate(env.agents):
                Q_network[i].store_transition(observation[i], actions[i], reward_n[i], observation_n[i])

                if (episode != 0 or step > 32) and step % 5 == 0:
                    Q_network[i].learn()

            # Swap observation
            observation = observation_n

            # Break if ended
            for done in done_n:
                if done:
                    break
            step += 1
        print("At episode " + str(episode) + ", the experiment return is: " + str(exp_return))
        return_list.append(exp_return)
    recorded_return_dict["lr: " + str(LR_RATE) +
                         " replace target: " + str(REPLACE_TAR_ITER) +
                         " batch size: " + str(BATCH_SIZE)] = return_list
    iter_index += 1

# Save the recorded experiment results
result_file = open('record.pickle', 'wb')
pickle.dump(recorded_return_dict, result_file)


# End of game
print("Training over")
env.close()




# for _ in range(50):
#     actions = []
#     for i, agent in enumerate(env.agents):
#         action = env.action_space[i].sample()
#         actions.append(action)
#     obs_n, reward_n, done_n, info_n = env.step(actions)
#     env.render()
#     time.sleep(0.1)
# env.close()
