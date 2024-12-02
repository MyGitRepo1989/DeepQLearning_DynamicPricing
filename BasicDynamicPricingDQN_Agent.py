import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.env_checker import check_env



class DynamicPricingEnv(gym.Env):
    def __init__(self):
        super(DynamicPricingEnv, self).__init__()
        
        # Define state: [remaining_inventory, competitor_price, demand_factor]
        self.observation_space = spaces.Box(low=0, high=1, shape=(3,), dtype=np.float32)
        
        # Define action: price levels (discrete prices: $5, $10, $15)
        self.action_space = spaces.Discrete(3)
        
        # Inventory and demand parameters
        self.total_inventory = 100
        self.remaining_inventory = self.total_inventory
        self.demand_factor = np.random.uniform(0.5, 1.5)  # Fluctuating demand
        self.competitor_price = np.random.uniform(5, 15)  # Simulated competitor price

    def step(self, action):
        # Convert action to price
        price = [5, 10, 15][action]
        
        # Simulate demand
        demand = self.demand_factor * (15 - price + np.random.uniform(-1, 1))
        units_sold = min(demand, self.remaining_inventory)
        
        # Calculate revenue
        revenue = units_sold * price
        
        # Encourage revenue maximization but discourage inventory leftovers
        # Adjust to reward both revenue and minimize unsold inventory
        
        
        if self.remaining_inventory == 0:
            reward += 100  # Bonus for selling all inventory

        
        reward = revenue - 0.1 * self.remaining_inventory  # Penalty for excess inventory


        
        # Update inventory
        self.remaining_inventory -= units_sold
        
        # Create next state
        self.demand_factor = np.random.uniform(0.5, 1.5)
        self.competitor_price = np.random.uniform(5, 15)
        state = np.array([
            self.remaining_inventory / self.total_inventory,
            self.competitor_price / 15,
            self.demand_factor / 1.5,
        ], dtype=np.float32)  # Cast to float32
        
        print(state)
        
        # Check if the episode is done
        terminated = self.remaining_inventory <= 0  # True if inventory is depleted
        truncated = False  # No external truncation logic applied here

        # Return the step output as required
        return state, revenue, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        self.remaining_inventory = self.total_inventory
        self.demand_factor = np.random.uniform(0.5, 1.5)
        self.competitor_price = np.random.uniform(5, 15)
        return np.array([
            self.remaining_inventory / self.total_inventory,
            self.competitor_price / 15,
            self.demand_factor / 1.5,
        ], dtype=np.float32), {}
        
        

if __name__ == "__main__":        
    # Initialize the environment
    env = DynamicPricingEnv()

    # Check the environment
    check_env(env)

    # Train the model
    model = DQN(
        "MlpPolicy", env, verbose=1,
        exploration_fraction=0.3,  # 30% of the timesteps for exploration
        exploration_initial_eps=1.0,  # Fully random at the start
        exploration_final_eps=0.05    # Gradually reduce exploration to 5%
    )
    model.learn(total_timesteps=20000)
    
    # Save and load the trained model this is a zip file
    #model.save("dynamic_pricing_dqn_v1")   
    
     # Load the trained model
    #dynamic_price_model = DQN.load("dynamic_pricing_dqn_v1") 
    
    # Test Input: Define a specific state
    # Example: Remaining inventory = 50, Competitor price = 10, Demand factor = 1.2
    test_state = np.array([
        50 / 100,  # Normalize remaining inventory by total inventory
        10 / 15,   # Normalize competitor price by maximum price
        1.2 / 1.5  # Normalize demand factor by its maximum value
    ], dtype=np.float32)

    # Get the action (price level) predicted by the model
    action, _ = model.predict(test_state)

    # Map action to actual price (ensure alignment with training)
    price_levels = [5, 10, 15]  # Same as in the environment
    predicted_price = price_levels[action]

    # Print the result
    print(f"Predicted Action: {action}, Predicted Price: ${predicted_price}")
    
    