import random
import pickle

'''
This is an experimental work to understand how RL can be used to train AI to plan 4 X 4 Tic Tac Toe game

'''

# --- The AI Brain ---
class QLearningAgent:
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        alpha (learning_rate): How much we update our Q-value.
        gamma (discount_factor): How much we value future rewards.
        epsilon (exploration_rate): How often we take a random action.
        """
        self.q_table = {}  # The "cheat sheet" dictionary: {state: {action: value}}
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def get_state(self, board):
        """Converts the board list into a unique, hashable tuple to use as a dictionary key."""

        return tuple(board)

    def choose_action(self, board, available_moves):
        """Decides whether to explore (random) or exploit (use Q-table)."""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(available_moves)  # Explore
        else:
            # Exploit
            state = self.get_state(board)
            if state not in self.q_table:
                return random.choice(available_moves) # No info yet, act randomly
            
            # Choose the action with the highest Q-value for this state
            q_values = self.q_table.get(state, {})
            max_q = -float('inf')
            best_action = random.choice(available_moves) # Default to random if no q-values

            # Find the best action among available moves
            for move in available_moves:
                move_q = q_values.get(move, 0) # Default to 0 if move not seen
                if move_q > max_q:
                    max_q = move_q
                    best_action = move
            return best_action

    def update_q_table(self, state, action, reward, next_state, next_available_moves):
        """The core RL formula (Bellman Equation) for learning."""
        
        # Get the current Q-value for the (state, action) pair
        current_q = self.q_table.get(state, {}).get(action, 0)
        
        # Get the max Q-value for the *next* state (what's the best we can do from here?)
        next_q_values = self.q_table.get(next_state, {})
        max_next_q = 0
        if next_available_moves: # Only if game is not over
            max_next_q = max([next_q_values.get(move, 0) for move in next_available_moves], default=0)

        # The Q-Learning update rule
        # new_q = (current_q) + learning_rate * ( (reward + discount * max_future_reward) - current_q )
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

        # Update the Q-table
        if state not in self.q_table:
            self.q_table[state] = {}
        self.q_table[state][action] = new_q
        
    def save_q_table(self, file_name):
        with open(file_name, 'wb') as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, file_name):
        with open(file_name, 'rb') as f:
            self.q_table = pickle.load(f)

# --- The Game Environment ---
class TicTacToe:
    def __init__(self):
        self.board = [' '] * 16
        self.current_winner = None

    def print_board(self):
        print("-----------------")
        for i in range(4):
            print(f"| {self.board[i*4]} | {self.board[i*4+1]} | {self.board[i*4+2]} | {self.board[i*4+3]} |")
            print("-----------------")

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def make_move(self, square, letter):
        if self.board[square] == ' ':
            self.board[square] = letter
            if self.check_winner(letter):
                self.current_winner = letter
            return True
        return False

    def check_winner(self, letter):
        # Check rows, columns, and diagonals
        # lines = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]
        lines = [[0, 1, 2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [0,4,8,12], [1,5,9,13], [2,6,10,14], [3,7,11,15],[0,5,10,15],[12,9,6,3]]
        for line in lines:
            if all(self.board[i] == letter for i in line):
                return True
        return False
    
    def is_board_full(self):
        return ' ' not in self.board

# --- The Training Loop ---
def train_agent(episodes=20000):
    agent_O = QLearningAgent(epsilon=0.1) # AI plays 'O'
    agent_X = QLearningAgent(epsilon=0.3) # We'll train a second AI 'X' to be the opponent
    
    for episode in range(episodes):
        game = TicTacToe()
        turn = 'X'
        
        # Store moves to update Q-tables at the end of the game
        history_X = []
        history_O = []
        
        while not game.current_winner and not game.is_board_full():
            available = game.available_moves()
            state = agent_O.get_state(game.board) # Use O's perspective for state

            if turn == 'X':
                action = agent_X.choose_action(game.board, available)
                game.make_move(action, 'X')
                history_X.append((state, action))
                turn = 'O'
            else:
                action = agent_O.choose_action(game.board, available)
                game.make_move(action, 'O')
                history_O.append((state, action))
                turn = 'X'
        
        # Game over, assign rewards and learn
        next_state = agent_O.get_state(game.board)
        next_available = game.available_moves()
        
        if game.current_winner == 'O':
            # 'O' won, 'X' lost
            reward_O = 1
            reward_X = -1
        elif game.current_winner == 'X':
            # 'X' won, 'O' lost
            reward_O = -1
            reward_X = 1
        else:
            # Draw
            reward_O = 0.5
            reward_X = 0.5
            
        # Update 'O' agent's Q-table
        for (state, action) in reversed(history_O):
            agent_O.update_q_table(state, action, reward_O, next_state, next_available)
            next_state = state # The next_state for the *previous* move is the current state
            reward_O = reward_O * agent_O.gamma # Discount the reward for earlier moves
            
        # Update 'X' agent's Q-table
        for (state, action) in reversed(history_X):
            agent_X.update_q_table(state, action, reward_X, next_state, next_available)
            next_state = state
            reward_X = reward_X * agent_X.gamma

        if (episode + 1) % 2000 == 0:
            print(f"Training... Episode {episode + 1}/{episodes} done.")
            
    print("Training complete! AI 'O' is ready.")
    agent_O.save_q_table('tictactoe_q_table.pkl')
    return agent_O

# --- Play against the Trained AI ---
def play_human_vs_ai(agent):
    game = TicTacToe()
    agent.epsilon = 0 # Turn off exploration, AI will only use its Q-table
    
    turn = 'X' # Human plays 'X'
    
    while not game.current_winner and not game.is_board_full():
        game.print_board()
        available = game.available_moves()
        
        if turn == 'X':
            try:
                move = int(input(f"Your turn (X). Choose a spot (1-9): ") ) - 1
                if move not in available:
                    print("Invalid move. Try again.")
                    continue
            except ValueError:
                print("Invalid input. Enter a number (0-8).")
                continue
        else:
            # AI's turn ('O')
            print("AI 'O' is thinking...")
            move = agent.choose_action(game.board, available)
            print(f"AI chose spot {move}")
            
        game.make_move(move, turn)
        turn = 'O' if turn == 'X' else 'X'
    
    game.print_board()
    if game.current_winner:
        print(f"Game over! Winner: {game.current_winner}")
    else:
        print("Game over! It's a draw.")


# --- Main execution ---
if __name__ == "__main__":
    # 1. Train the AI
    # This will take 10-20 seconds. It's playing 20,000 games.
    trained_agent = train_agent(episodes=20000)
    
    # 2. Play against the trained AI
    play_human_vs_ai(trained_agent)