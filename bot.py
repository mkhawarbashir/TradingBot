import numpy as np
import pandas as pd
import tensorflow as tf
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam


class TradingEnvironment:
    def __init__(self, data):
        self.data = data
        self.action_space = [0, 1] # 0=hold, 1=buy
        self.state_size = data.shape[1]

    def reset(self):
        self.current_step = 0
        self.balance = 1000
        self.shares_held = 0
        self.net_worth = self.balance + self.shares_held * self.data[0][0] # Assuming first value is price
        self.history = deque(maxlen=200)

        return self.data[self.current_step]

    def step(self, action):
        # Buy if action = 1 and hold if action = 0
        if action == 1:
            shares_bought = self.balance / self.data[self.current_step][0] # Assuming first value is price
            self.shares_held += shares_bought
            self.balance -= shares_bought * self.data[self.current_step][0] # Assuming first value is price

        # Advance to next time step
        self.current_step += 1
        self.net_worth = self.balance + self.shares_held * self.data[self.current_step][0] # Assuming first value is price

        # Calculate reward based on net worth change
        reward = self.net_worth - self.history[-1][0]

        # Save current net worth to history buffer
        self.history.append((self.net_worth, action))

        # Check if episode is done
        done = self.net_worth <= 0 or self.current_step >= len(self.data) - 1

        # Return new state, reward, and done flag
        return self.data[self.current_step], reward, done


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95 # Discount factor
        self.epsilon = 1.0 # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.model.predict(state)[0])

    def replay(self, batch_size):
        minibatch = np.array(random.sample(self.memory, batch_size))
        states = np.array(minibatch[:, 0].tolist())
        actions = np.array(minibatch[:, 1].tolist())
        rewards = np.array(minibatch[:, 2].tolist())
        next_states = np.array(minibatch[:, 3].tolist())
        dones = np.array(minibatch[:, 4].tolist())

        targets = self.model.predict(states)
        q_values_next = self.model.predict(next_states)

        for i in range(len(minibatch)):
            target = rewards[i]

            if not dones[i]:
                target += self.gamma * np.amax(q_values_next[i])

            targets[i][actions[i]] = target

        self.model.fit(states, targets, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    # Load historical data into pandas dataframe
    df = pd.read_csv('historical_data.csv')

    # Create trading environment
    env = TradingEnvironment(df.values)

    # Create DQN agent
    agent = DQNAgent(env.state_size, len(env.action_space))

    # Train agent
    batch_size = 32
    episodes = 100
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, env.state_size])
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, env.state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        print(f"Episode {e + 1}/{episodes}, Balance: {env.balance}, Shares held: {env.shares_held}, Net worth: {env.net_worth}")

    # Save trained agent
    agent.save("trained_agent.h5")

    # Test trained agent
    agent.load("trained_agent.h5")
    state = env.reset()
    state = np.reshape(state, [1, env.state_size])
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        next_state = np.reshape(next_state, [1, env.state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state

    print(f"Final Balance: {env.balance}, Shares held: {env.shares_held}, Net worth: {env.net_worth}")
