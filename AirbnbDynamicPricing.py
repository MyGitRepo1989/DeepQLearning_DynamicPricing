import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
#from stable_baselines3.common.vec_env import make_vec_env


# Custom Airbnb Environment
class AirbnbEnv(gym.Env):
    
    def __init__(self,*args, **kwargs):
        super(AirbnbEnv, self).__init__()
        self.price_range = (120, 350)  # Price range in dollars
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(150)  # 100 discrete price levels
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32) 
        # [inventory, price_level, demand, competitor_price]
        self.reset()

    def reset(self, seed=None,options=None):
            # Seed the environment for reproducibility
            if seed is not None:
                np.random.seed(seed)

            # Reset environment state to random initial values
            self.state = np.array([
                np.random.uniform(0.5, 1.0),  # Inventory
                0.5,  # Price level (normalized)
                np.random.uniform(0.5, 1.0),  # Demand
                np.random.uniform(150, 300) / 300  # Competitor price (normalized to [0, 1])
            ], dtype=np.float32)
            
            return self.state, {}

    def step(self, action):
        # Convert action to actual price that has some locig applied to current inventory and action
        price = self.price_range[0] + action * (self.price_range[1] - self.price_range[0]) / (self.action_space.n - 1)
        inventory, price_level, demand, competitor_price = self.state

        # Calculate reward: Demand and closeness to competitor's price
        if demand < 0.3 and price > competitor_price * 300:
            penalty = -50  # Penalty for overpricing in low demand
        else:
            penalty = 0

        competitor_match_reward = 0.3 * (1 - abs(price - competitor_price * 300) / 300)  # Scaled match reward
        #eg-  0.3 * (1 - (150-200*300/300))
        reward = 0.7 * demand * price + competitor_match_reward + penalty
        # accumulative reward =  0.7 *0.1 * 150 + 15.299999999999999 + 0.2

        # Ensure reward is bounded
        reward = max(-100, min(reward, 100))
        #eg- max(-100, min(25, 100))

        # Update state
        self.state = [
            max(0, inventory - np.random.uniform(0.1, 0.2)),  # Decrease inventory
            action / (self.action_space.n - 1),  # New price level
            np.random.uniform(0, 1),  # New random demand
            np.random.uniform(150, 300) / 300  # New competitor price (normalized)
        ]

        # Check if episode is done (inventory exhausted)
        done = self.state[0] < 0.1
        
        #step Return Tuple: Now includes terminated and truncated flags, as required by Gym and SB3.
        #terminated: Indicates if the episode ended due to a terminal condition.
        #truncated: Indicates if the episode ended due to time or step limits (set to False unless explicitly implemented).
        
        terminated = self.state[0] < 0.1
        truncated = False
        
        
        #all DQN env need 4 things returned , states , reward, done , and anything else that we want to pass
        #return np.array(self.state), reward, done, {}
        return np.array(self.state), reward, terminated, truncated, {}



if __name__ == "__main__":    
    # Initialize the environment
    #env = DummyVecEnv([lambda: AirbnbEnv()])  # Wrap environment for compatibility with SB3
    env = AirbnbEnv()
    
    #Initialize DQN model
    #PARAMETERS EXPLANATION and Notes 
    #learn rate can be 0.01 to 0.0001 , gamma can be 0-1 mostly 0.9 
    #exploration_fraction is how much % time model can exploit a random space for new oppertunities eg 10% ,20% randome
    #exploration_final_eps is in end of training can model still exploit random spaces typically low at end 
    #if you are getting random values use small exploration_final_eps 

    model = DQN('MlpPolicy', env, verbose=1, learning_rate=0.01, gamma=0.50, exploration_fraction=0.05, exploration_final_eps=0.01)

    # Train the model and collect rewards
    reward_list = []  # To store rewards for each step to chack at end 

    #custom call back will collect rewards
    def custom_train_callback(locals_, globals_):
        # Access environment and rewards during training
        reward_list.append(locals_['rewards'][0])
        return True

    # Train the model
    model.learn(total_timesteps=90000, callback=custom_train_callback)
    
    # Plot the rewards after training
    plt.figure(figsize=(13, 6))
    plt.plot(np.convolve(reward_list, np.ones(200)/200, mode='valid')) #rounded 
    #plt.plot(reward_list)
    # Smoothed reward plot
    plt.title('Smoothed Rewards Over Training')
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.grid()
    plt.show()


    # Example state: [inventory, price_level, demand, competitor_price]
    test_state = [0.7, 0.5, 0.5, 210 / 300]  # Example normalized state

    # Predict the action for the test state
    action, _ = model.predict(test_state, deterministic=True)

    # Convert the action to the actual price
    price = 150 + action * (300 - 150) / (model.action_space.n - 1)

    print(f"Test State: {test_state}")
    print(f"Predicted Action: {action}")
    print(f"Suggested Price: ${price:.2f}")