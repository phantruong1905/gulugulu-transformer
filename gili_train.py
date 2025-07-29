import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from collections import deque, namedtuple
import pandas as pd
import pickle
from model.ddqn_rl import *

# Training function
def train_trading_agent(train_samples, test_samples=None, epochs=30):
    """Train improved DDQN trading agent"""

    if not train_samples or len(train_samples) == 0:
        print("No training samples provided")
        return None, None, None, None

    # Get state size
    state_size = len(train_samples[0]['state'])
    print(f"State size: {state_size}")

    # Split data if needed
    if test_samples is None:
        train_size = int(0.8 * len(train_samples))
        test_samples = train_samples[train_size:]
        train_samples = train_samples[:train_size]

    print(f"Training samples: {len(train_samples)}")
    print(f"Test samples: {len(test_samples)}")

    # Create improved agent
    agent = DDQNTradingAgent(
        state_size=state_size,
        learning_rate=1e-5,  # Lower learning rate
        gamma=0.95,  # Slightly lower discount
        epsilon_start=0.9,  # Higher initial exploration
        epsilon_end=0.05,  # Higher final exploration
        epsilon_decay=0.98,  # Faster decay
        batch_size=64,  # Smaller batches
        target_update=1000  # More frequent updates
    )

    # Train
    print("\nStarting training...")
    losses = agent.train_on_data(train_samples, epochs=epochs)

    # Evaluate
    print("\nEvaluating on test data...")
    test_results = agent.evaluate(test_samples)

    return agent, losses, None, test_results


# Example usage with your prepared data
if __name__ == "__main__":
    # Load your prepared data
    try:
        with open("C:/Users/PC/PycharmProjects/GILIGILI_RL/data/train_rl_data.pkl", "rb") as f:
            train_data = pickle.load(f)

        with open("C:/Users/PC/PycharmProjects/GILIGILI_RL/data/test_rl_data.pkl", "rb") as f:
            test_data = pickle.load(f)

        print(f"Loaded {len(train_data)} training samples and {len(test_data)} test samples")

        # Train the agent
        agent, losses, train_results, test_results = train_trading_agent(
            train_samples=train_data,
            test_samples=test_data,
            epochs=100
        )

        # Save the trained model
        if agent:
            agent.save_model("C:/Users/PC/PycharmProjects/GILIGILI_RL/ddqn_trading_agent.pth")

    except FileNotFoundError as e:
        print(f"Data files not found: {e}")
        print("Please run the data preparation script first")