import gym
import numpy as np

# Initialize the environment
env = gym.make('CartPole-v1')

# Initialize the parameters
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# Initialize the Q-Table
Q = np.zeros((n_states, n_actions))

# Set the number of episodes
n_episodes = 1000

# Train the agent
for episode in range(n_episodes):
    # Reset the environment
    state = env.reset()
    done = False
    while not done:
        # Choose an action based on the current state
        action = np.argmax(Q[state, :] + np.random.randn(1, n_actions) * (1. / (episode + 1)))

        # Take the action and observe the new state, reward, and whether the episode is done
        new_state, reward, done, _ = env.step(action)

        # Update the Q-Table
        Q[state, action] = reward + np.max(Q[new_state, :])
        state = new_state

# Test the agent
state = env.reset()
done = False
while not done:
    # Choose the action with the maximum value
    action = np.argmax(Q[state, :])

    # Take the action and observe the new state, reward, and whether the episode is done
    new_state, reward, done, _ = env.step(action)
    state = new_state

print("Episode finished after {} timesteps".format(t + 1))
