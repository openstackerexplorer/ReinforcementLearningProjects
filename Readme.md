# Reinforcement  Learning Game Project

## Project Overview
This is a reinforcement learning game project that demonstrates Q-Learning through interactive gameplay. The project consists 1 file:
- A Tic-tac-toe implementation with Q-Learning (`tictactoe.py`)

![Tic Tac Toe Game](game.gif)

## Core Architecture

### Q-Learning Implementation
The project uses a consistent Q-Learning pattern across games:
- Q-tables store state-action mappings as dictionaries
- States are represented as tuples of game-specific information
- Actions are mapped to numeric indices
- The Bellman equation is used for value updates

Key example from `tictactoe.py`:
```python
# Q-table structure: {(state): {action: value}}
new_value = old_value + alpha * (reward + gamma * next_max - old_value)
```

### Game Components
1. **Main Predator-Prey Game** (`tictactoe.py`):
   - Uses 4*4 grid for Tic-tac-toe
   - Grid-based movement system (4x4 cells)
   - Used AI agent to play against human player
   - AI agent (red) learns to chase play tic-tac-toe

2. **Q-Learning Agent** (`QLearningAgent` class):
   - Learning rate (alpha) = 0.1
   - Discount factor (gamma) = 0.9
   - Exploration rate (epsilon) = 0.1
   - State representation: relative position between player and AI agent

## Development Workflow

### Running the Games
1. 4 X 4 Tic-tac-toe Game:
```bash
python tictactoe.py
```

### Game Constants
- Provide grid index as input (number between 1-16)


### State Management
When extending the project, follow these patterns:
1. Define states as tuples of relevant game information
2. Use relative positions when possible for better generalization
3. Store Q-tables as nested dictionaries with string-converted state tuples

## Integration Points



### Data Persistence
The project supports Q-table persistence (currently commented out):
```python
# Save Q-table as pickle
def save_q_table(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, file_name):
        with open(file_name, 'rb') as f:
            self.q_table = pickle.load(f)
```



## Key Files
- `tictactoe.py` - Tic-tac-toe game with Q-Learning (in progress)
- `Readme.md` - This file
