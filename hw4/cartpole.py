import numpy as np
import gym
import os
from tqdm import tqdm

total_reward = []
episode = 3000
decay = 0.045


class Agent():
    def __init__(self, env, epsilon=0.05, learning_rate=0.5, GAMMA=0.97, num_bins=7):
        """
        The agent learning how to control the action of the cart pole.
        Hyperparameters:
            epsilon: Determines the explore/expliot rate of the agent
            learning_rate: Determines the step size while moving toward a minimum of a loss function
            GAMMA: The discount factor (tradeoff between immediate rewards and future rewards)
            num_bins: Number of part that the continuous space is to be sliced into.
        """
        self.env = env

        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA

        self.num_bins = num_bins
        self.qtable = np.zeros((self.num_bins, self.num_bins,
                               self.num_bins, self.num_bins, self.env.action_space.n))

        # init_bins() is your work to implement.
        self.bins = [
            self.init_bins(-2.4, 2.4, self.num_bins),  # cart position
            self.init_bins(-3.0, 3.0, self.num_bins),  # cart velocity
            self.init_bins(-0.5, 0.5, self.num_bins),  # pole angle
            self.init_bins(-2.0, 2.0, self.num_bins)  # tip velocity
        ]

    def init_bins(self, lower_bound, upper_bound, num_bins):
        """
        Slice the interval into #num_bins parts.
        Parameters:
            lower_bound: The lower bound of the interval.
            upper_bound: The upper bound of the interval.
            num_bins: Number of parts to be sliced.
        Returns:
            a numpy array of #num_bins - 1 quantiles.
        Example:
            Let's say that we want to slice [0, 10] into five parts,
            that means we need 4 quantiles that divide [0, 10].
            Thus the return of init_bins(0, 10, 5) should be [2. 4. 6. 8.].
        Hints:
            1. This can be done with a numpy function.
        """
        # Begin your code
        """
        This function slices the interval by lower_bound and upper_bound into num_bins 
        parts and returns an array of quantiles. It uses NumPy's linspace function to 
        divide the interval, excluding the upper bound, and then selects all elements 
        except the first one, as the first element is essentially the lower bound.
        """
        return np.linspace(lower_bound, upper_bound, num_bins, endpoint=False)[1:]
        #raise NotImplementedError("Not implemented yet.")
        # End your code

    def discretize_value(self, value, bins):
        """
        Discretize the value with given bins.
        Parameters:
            value: The value to be discretized.
            bins: A numpy array of quantiles
        returns:
            The discretized value.
        Example:
            With given bins [2. 4. 6. 8.] and "5" being the value we're going to discretize.
            The return value of discretize_value(5, [2. 4. 6. 8.]) should be 2, since 4 <= 5 < 6 where [4, 6) is the 3rd bin.
        Hints:
            1. This can be done with a numpy function.
        """
        # Begin your code
        """
        This function discretizes a given value using the provided bins array. 
        It uses NumPy's digitize function, which returns the indices of the bins 
        to which each value in the input array belongs.
        """
        return np.digitize(value, bins)
        #raise NotImplementedError("Not implemented yet.")
        # End your code

    def discretize_observation(self, observation):
        """
        Discretize the observation which we observed from a continuous state space.
        Parameters:
            observation: The observation to be discretized, which is a list of 4 features:
                1. cart position.
                2. cart velocity.
                3. pole angle.
                4. tip velocity.
        Returns:
            state: A list of 4 discretized features which represents the state.
        Hints:
            1. All 4 features are in continuous space.
            2. You need to implement discretize_value() and init_bins() first
            3. You might find something useful in Agent.__init__()
        """
        # Begin your code
        """
        This function discretizes a continuous observation into a discrete state representation. 
        It iterates over each feature of the observation and discretizes it using the corresponding 
        pre-defined bins. Then return a tuple containing the discretized values of all four features,
        representing the discretized state.
        """
        discretized_state = []
        for obs, bin in zip(observation, self.bins):
            discretized_value = self.discretize_value(obs, bin)
            discretized_state.append(discretized_value)
        return tuple(discretized_state)
        #raise NotImplementedError("Not implemented yet.")
        # End your code

    def choose_action(self, state):
        """
        Choose the best action with given state and epsilon.
        Parameters:
            state: A representation of the current state of the enviornment.
            epsilon: Determines the explore/expliot rate of the agent.
        Returns:
            action: The action to be evaluated.
        """
        # Begin your code
        """
        I randomly select a number between [0,1], and if this number is greater
        than epsilon, it chooses the action with the highest Q-value for exploitation. 
        Otherwise, it selects a random action exploration.
        """
        if (np.random.uniform(0, 1) > self.epsilon): 
            return np.argmax(self.qtable[state])
        else:
            return self.env.action_space.sample()
        #raise NotImplementedError("Not implemented yet.")
        # End your code

    def learn(self, state, action, reward, next_state, done):
        """
        Calculate the new q-value base on the reward and state transformation observered after taking the action.
        Parameters:
            state: The state of the enviornment before taking the action.
            action: The exacuted action.
            reward: Obtained from the enviornment after taking the action.
            next_state: The state of the enviornment after taking the action.
            done: A boolean indicates whether the episode is done.
        Returns:
            None (Don't need to return anything)
        """
        # Begin your code
        """
        This function implements the Q-learning algorithm to update the Q-values based on the reward obtained from 
        the environment and the observed state transitions. It calculates the maximum Q-value for the next state, 
        and if the episode is done, it sets the maximum next Q-value to 0. Then, it updates the Q-value for the current 
        state-action pair using the Q-learning formula. Finally, if the episode is done, it saves the Q-table to a file.
        """
        max_nxt = np.max(self.qtable[next_state])
        if done:
            max_nxt = 0
        self.qtable[state + (action,)] = (1 - self.learning_rate) * self.qtable[state + (action,)] +self.learning_rate * (reward + self.gamma * max_nxt)
        if done:
            np.save("./Tables/cartpole_table.npy", self.qtable)
        #raise NotImplementedError("Not implemented yet.")
        # End your code

    def check_max_Q(self):
        """
        - Implement the function calculating the max Q value of initial state(self.env.reset()).
        - Check the max Q value of initial state
        Parameter:
            self: the agent itself.
            (Don't pass additional parameters to the function.)
            (All you need have been initialized in the constructor.)
        Return:
            max_q: the max Q value of initial state(self.env.reset())
        """
        # Begin your code
        """
        This function calculates the maximum Q-value for the initial state of the environment. It resets 
        the environment using self.env.reset() to get the initial observation, then discretizes it to 
        obtain the initial state. After that, it retrieves the Q-values corresponding to the initial state 
        from the Q-table and calculates the maximum Q-value. Finally, it returns this maximum Q-value.
        """
        initial_state = tuple(self.discretize_observation(self.env.reset()))
        max_q = np.max(self.qtable[initial_state])
        return max_q
        #raise NotImplementedError("Not implemented yet.")
        # End your code


def train(env):
    """
    Train the agent on the given environment.
    Paramenters:
        env: the given environment.
    Returns:
        None (Don't need to return anything)
    """
    training_agent = Agent(env)
    rewards = []
    for ep in tqdm(range(episode)):
        state = training_agent.discretize_observation(env.reset())
        done = False

        count = 0
        while True:
            count += 1
            action = training_agent.choose_action(state)
            next_observation, reward, done, _ = env.step(action)

            next_state = training_agent.discretize_observation(next_observation)
            training_agent.learn(state, action, reward, next_state, done)

            if done:
                rewards.append(count)
                break
            state = next_state

        if (ep + 1) % 500 == 0:
            training_agent.learning_rate -= decay
        np.save("./Tables/cartpole_table.npy", training_agent.qtable)
    total_reward.append(rewards)


def test(env):
    """
    Test the agent on the given environment.
    Paramenters:
        env: the given environment.
    Returns:
        None (Don't need to return anything)
    """
    testing_agent = Agent(env)

    testing_agent.qtable = np.load("./Tables/cartpole_table.npy")
    rewards = []

    for _ in range(100):
        state = testing_agent.discretize_observation(testing_agent.env.reset())
        count = 0
        while True:
            # env.render() #Uncomment for visualizing the process
            count += 1
            action = np.argmax(testing_agent.qtable[tuple(state)])
            next_observation, _, done, _ = testing_agent.env.step(action)

            if done == True:
                rewards.append(count)
                break

            next_state = testing_agent.discretize_observation(next_observation)
            state = next_state

    print(f"average reward: {np.mean(rewards)}")
    print(f"max Q:{testing_agent.check_max_Q()}")


if __name__ == "__main__":
    env = gym.make('CartPole-v0')
    os.makedirs("./Tables", exist_ok=True)

   # training section:
    for i in range(5):
        print(f"#{i + 1} training progress")
        train(env)
    # testing section:
    test(env)
    env.close()

    os.makedirs("./Rewards", exist_ok=True)
    np.save("./Rewards/cartpole_rewards.npy", np.array(total_reward))
