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

# Experience tuple for replay buffer
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class TradingDataset(Dataset):
    """Custom Dataset for trading experiences"""

    def __init__(self, experiences):
        self.experiences = experiences

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        exp = self.experiences[idx]
        return {
            'state': torch.FloatTensor(exp.state),
            'action': torch.LongTensor([exp.action]),
            'reward': torch.FloatTensor([exp.reward]),
            'next_state': torch.FloatTensor(exp.next_state),
            'done': torch.BoolTensor([exp.done])
        }


class DQN(nn.Module):
    """Deep Q-Network with improved architecture - Fixed BatchNorm issue"""

    def __init__(self, state_size, action_size=3, hidden_sizes=[256, 128, 64]):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Build network layers - Use LayerNorm instead of BatchNorm for single sample compatibility
        layers = []
        input_size = state_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.LayerNorm(hidden_size),  # Changed from BatchNorm1d to LayerNorm
                nn.Dropout(0.2)
            ])
            input_size = hidden_size

        # Output layer
        layers.append(nn.Linear(input_size, action_size))
        self.network = nn.Sequential(*layers)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0)

    def forward(self, state):
        return self.network(state)


class ReplayBuffer:
    """Experience replay buffer for DDQN"""

    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


class DDQNTradingAgent:
    """Double Deep Q-Network Trading Agent with Position Management"""

    def __init__(self, state_size, action_size=3, learning_rate=5e-5, gamma=0.95,
                 epsilon_start=0.9, epsilon_end=0.05, epsilon_decay=0.9995,
                 memory_size=50000, batch_size=32, target_update=500):

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # Position tracking - KEY IMPROVEMENT
        self.current_position = 0  # -1: Short, 0: Hold, 1: Long

        # NEW: Track when we last bought (step counter)
        self.last_buy_step = -999  # Initialize to allow immediate buying
        self.current_step = 0  # Track current step

        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Networks
        self.q_network = DQN(state_size, action_size).to(self.device)
        self.target_network = DQN(state_size, action_size).to(self.device)

        # Optimizer with lower learning rate
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.9)

        # Initialize target network
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Replay buffer
        self.memory = ReplayBuffer(memory_size)

        # Training metrics
        self.step_count = 0
        self.losses = []
        self.episode_rewards = []

        # Action mapping: 0=SELL, 1=HOLD, 2=BUY
        self.action_names = {0: "SELL", 1: "HOLD", 2: "BUY"}

        print(f"Initialized agent on {self.device}")

    def get_action_with_position_management(self, state, training=True):
        """
        Get action with proper position management logic (like Table 1 in paper)
        """
        # Get raw action from neural network
        if training and random.random() < self.epsilon:
            raw_action = random.randint(0, self.action_size - 1)
        else:
            if isinstance(state, np.ndarray):
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
            else:
                state_tensor = state.unsqueeze(0) if state.dim() == 1 else state

            state_tensor = state_tensor.to(self.device, non_blocking=True)

            # Set network to eval mode for inference to handle single samples
            self.q_network.eval()
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
                raw_action = q_values.argmax().item()

            # Set back to train mode if training
            if training:
                self.q_network.train()

        # Apply position management logic (from paper's Table 1)
        actual_action = self._apply_position_logic(raw_action)

        return actual_action, raw_action

    def _apply_position_logic(self, signal):
        """
        Apply position management logic from the paper's Table 1 with 2-day hold constraint
        signal: 0=SELL, 1=HOLD, 2=BUY
        Returns: actual action to execute
        """
        pos = self.current_position

        # Position management logic from paper
        if pos == 0:  # Currently holding cash
            if signal == 0:  # SELL signal -> No action
                return 1  # HOLD
            elif signal == 1:  # HOLD signal -> Hold
                return 1  # HOLD
            elif signal == 2:  # BUY signal -> Open long
                return 2  # BUY

        elif pos == 1:  # Currently long
            if signal == 0:  # SELL signal -> Check 2-day constraint
                # NEW: Check if 2 days have passed since last buy
                if self.current_step - self.last_buy_step >= 2:
                    return 0  # SELL (close long)
                else:
                    return 1  # HOLD (constraint violation)
            elif signal == 1:  # HOLD signal -> Hold long
                return 1  # HOLD
            elif signal == 2:  # BUY signal -> Hold long (can't buy more)
                return 1  # HOLD

        elif pos == -1:  # Currently short
            if signal == 0:  # SELL signal -> Hold short (can't sell more)
                return 1  # HOLD
            elif signal == 1:  # HOLD signal -> Hold short
                return 1  # HOLD
            elif signal == 2:  # BUY signal -> Close short
                return 2  # BUY

        return 1  # Default to HOLD

    def update_position(self, action):
        """Update current position based on executed action"""
        if action == 0:  # SELL
            if self.current_position == 1:  # Close long
                self.current_position = 0
            elif self.current_position == 0:  # Open short
                self.current_position = -1

        elif action == 2:  # BUY
            if self.current_position == -1:  # Close short
                self.current_position = 0
            elif self.current_position == 0:  # Open long
                self.current_position = 1
                # NEW: Track when we bought
                self.last_buy_step = self.current_step
        # action == 1 (HOLD) doesn't change position

        # NEW: Increment step counter
        self.current_step += 1

    def calculate_reward(self, action, current_price, next_price, y_pred=None, future_returns=None):
        """
        Reward calculation with position awareness and reduced overtrading penalty
        """
        # Convert action to trading signal: 0=SELL(-1), 1=HOLD(0), 2=BUY(+1)
        action_value = action - 1  # Maps 0->-1, 1->0, 2->1

        # Basic return calculation
        price_return = next_price  # Assuming this is already percentage change

        # Base reward
        base_reward = action_value * price_return

        # Anti-overtrading penalty - penalize excessive trading
        overtrading_penalty = 0
        if action != 1:  # If not HOLD
            overtrading_penalty = -0.001  # Small penalty for trading

        # Position consistency bonus
        position_bonus = 0
        if action == 1:  # Reward holding positions
            position_bonus = 0.0005

        # Directional accuracy bonus
        directional_bonus = 0
        if y_pred is not None and future_returns is not None:
            if len(y_pred) >= 3 and len(future_returns) >= 3:
                matches = sum(
                    np.sign(y_pred[i]) == np.sign(future_returns[i])
                    for i in range(min(3, len(y_pred), len(future_returns)))
                    if not (np.isnan(y_pred[i]) or np.isnan(future_returns[i]))
                )
                directional_bonus = 0.05 * (matches / 3.0)

        final_reward = base_reward + overtrading_penalty + position_bonus + directional_bonus
        return final_reward

    def store_experience(self, rl_sample):
        """Store experience with position management"""
        # Get action with position management
        action, raw_action = self.get_action_with_position_management(rl_sample['state'], training=True)

        # Calculate reward
        reward = self.calculate_reward(
            action,
            rl_sample['current_price'],
            rl_sample['next_price'],
            rl_sample.get('y_pred'),
            rl_sample.get('future_returns')
        )

        # Update position
        self.update_position(action)

        # Create experience with raw_action for training
        experience = Experience(
            state=rl_sample['state'],
            action=raw_action,  # Use raw action for network training
            reward=reward,
            next_state=rl_sample['next_state'],
            done=rl_sample['done']
        )

        self.memory.push(experience)
        return action, reward

    def train_step(self):
        """Single training step"""
        if len(self.memory) < self.batch_size:
            return 0

        # Sample batch
        experiences = self.memory.sample(self.batch_size)

        # Prepare batch
        states = torch.FloatTensor([exp.state for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        next_states = torch.FloatTensor([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in experiences]).to(self.device)

        # Ensure network is in training mode for batch processing
        self.q_network.train()

        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # Next Q values (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))

        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        # Update target network
        self.step_count += 1
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()

    def train_on_data(self, rl_samples, epochs=None):
        """Train agent on RL samples"""
        print(f"Training on {len(rl_samples)} samples for {epochs} epochs...")

        # Reset position for training
        self.current_position = 0
        # NEW: Reset step counters for training
        self.last_buy_step = -999
        self.current_step = 0

        # Store all experiences
        actions_taken = []
        rewards_earned = []

        for sample in rl_samples:
            action, reward = self.store_experience(sample)
            actions_taken.append(action)
            rewards_earned.append(reward)

        print(f"Stored {len(self.memory)} experiences")
        print(
            f"Action distribution: SELL: {actions_taken.count(0)}, HOLD: {actions_taken.count(1)}, BUY: {actions_taken.count(2)}")
        print(f"Average reward: {np.mean(rewards_earned):.4f}")

        # Training loop
        total_losses = []
        self.q_network.train()

        for epoch in range(epochs):
            epoch_losses = []

            # Multiple training steps per epoch
            for _ in range(len(rl_samples) // self.batch_size):
                loss = self.train_step()
                if loss > 0:
                    epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses) if epoch_losses else 0
            total_losses.append(avg_loss)

            # Update learning rate
            self.scheduler.step()

            # ADD THIS: Decay epsilon once per epoch instead of per step
            if self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Epsilon: {self.epsilon:.3f}")

        return total_losses

    def evaluate(self, rl_samples):
        """Evaluate agent performance and include Q-values for each action in action_log"""
        self.q_network.eval()

        # Reset position for evaluation
        self.current_position = 0
        self.last_buy_step = -999
        self.current_step = 0

        total_reward = 0
        actions_taken = []
        action_log = []
        constraint_violations = 0

        with torch.no_grad():
            for sample in rl_samples:
                # Convert state to tensor
                state = torch.FloatTensor(sample['state']).unsqueeze(0).to(self.device)

                # Compute Q-values for all actions
                q_values = self.q_network(state).cpu().numpy().flatten()  # Shape: [3] for SELL, HOLD, BUY

                # Get action with position management
                action, raw_action = self.get_action_with_position_management(sample['state'], training=False)

                # Check if this was a constraint violation
                if (raw_action == 0 and self.current_position == 1 and
                        self.current_step - self.last_buy_step < 2):
                    constraint_violations += 1

                # Calculate reward
                reward = self.calculate_reward(
                    action,
                    sample['current_price'],
                    sample['next_price'],
                    sample.get('y_pred'),
                    sample.get('future_returns')
                )

                # Update position
                self.update_position(action)

                total_reward += reward
                actions_taken.append(action)

                # Log action and Q-values
                action_log.append({
                    'date': sample.get('date'),
                    'symbol': sample.get('symbol'),
                    'action': action,
                    'action_name': self.action_names[action],
                    'position': self.current_position,
                    'price': sample.get('current_price'),
                    'reward': reward,
                    'raw_action': raw_action,
                    'days_since_buy': self.current_step - self.last_buy_step,
                    'q_values': q_values.tolist(),  # Q-values for SELL, HOLD, BUY
                    'estimated_q_value': q_values[raw_action]  # Q-value of the raw action
                })

        avg_reward = total_reward / len(rl_samples) if rl_samples else 0.0

        action_dist = {
            'SELL': actions_taken.count(0),
            'HOLD': actions_taken.count(1),
            'BUY': actions_taken.count(2)
        }

        print(f"Evaluation Results:")
        print(f"Total Reward: {total_reward:.4f}")
        print(f"Average Reward: {avg_reward:.4f}")
        print(f"Action Distribution: {action_dist}")
        print(f"Final Position: {self.current_position}")
        print(f"Constraint Violations: {constraint_violations}")

        return {
            'total_reward': total_reward,
            'average_reward': avg_reward,
            'action_distribution': action_dist,
            'action_log': action_log,
            'final_position': self.current_position,
            'constraint_violations': constraint_violations
        }

    def save_model(self, filepath):
        """Save model state"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'current_position': self.current_position,
            'last_buy_step': self.last_buy_step,  # NEW
            'current_step': self.current_step  # NEW
        }, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load model state"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.current_position = checkpoint.get('current_position', 0)
        self.last_buy_step = checkpoint.get('last_buy_step', -999)  # NEW
        self.current_step = checkpoint.get('current_step', 0)  # NEW
        print(f"Model loaded from {filepath}")