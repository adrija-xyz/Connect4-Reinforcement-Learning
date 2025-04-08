#URL for Connect 4 code used: https://github.com/KeithGalli/Connect4-Python

import random
import numpy as np
import pygame
import sys
import math
import time

BLUE = (0,0,255)
BLACK = (0,0,0)
RED = (255,0,0)
YELLOW = (255,255,0)

ROW_COUNT = 6
COLUMN_COUNT = 7

def create_board():
	board = np.zeros((ROW_COUNT,COLUMN_COUNT))
	return board

def drop_piece(board, row, col, piece):
	board[row][col] = piece

def is_valid_location(board, col):
	return board[ROW_COUNT-1][col] == 0

def get_next_open_row(board, col):
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r
    return None

def print_board(board):
    print(np.flip(board, 0))

def winning_move(board, piece):
    # Check horizontal locations for win
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if board[r][c] == piece and board[r][c + 1] == piece and board[r][c + 2] == piece and board[r][
                c + 3] == piece:
                return True

    # Check vertical locations for win
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c] == piece and board[r + 2][c] == piece and board[r + 3][
                c] == piece:
                return True

    # Check positively sloped diaganols
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if board[r][c] == piece and board[r + 1][c + 1] == piece and board[r + 2][c + 2] == piece and board[r + 3][
                c + 3] == piece:
                return True

    # Check negatively sloped diaganols
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):
            if board[r][c] == piece and board[r - 1][c + 1] == piece and board[r - 2][c + 2] == piece and board[r - 3][
                c + 3] == piece:
                return True

def draw_board(board):
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            pygame.draw.rect(screen, BLUE, (c * SQUARESIZE, r * SQUARESIZE + SQUARESIZE, SQUARESIZE, SQUARESIZE))
            pygame.draw.circle(screen, BLACK, (
            int(c * SQUARESIZE + SQUARESIZE / 2), int(r * SQUARESIZE + SQUARESIZE + SQUARESIZE / 2)), RADIUS)

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            if board[r][c] == 1:
                pygame.draw.circle(screen, RED, (
                int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
            elif board[r][c] == 2:
                pygame.draw.circle(screen, YELLOW, (
                int(c * SQUARESIZE + SQUARESIZE / 2), height - int(r * SQUARESIZE + SQUARESIZE / 2)), RADIUS)
    pygame.display.update()

def is_last_move(board):
    return winning_move(board,1) or winning_move(board,2) or len(get_valid_locations(board)) == 0

def get_valid_locations(board):
    valid_locations=[]
    
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_locations.append(col)
            
    return valid_locations

def window_evaluation(window,piece):
    score=0

    #Initializaing the Opponent Piece
    opponent_piece=1
    
    if piece==1:
        opponent_piece=2

    #More score is rewarded for 4 pieces in window
    if window.count(piece)==4:
        score+=100

    #Score rewarded for 3 pieces in window
    if window.count(piece)==3 and window.count(0)==1:
        score+=20

    #Score rewarded for 2 pieces in window
    if window.count(piece)==2 and window.count(0)==2:
        score+=10

    #Score rewarded for 1 piece in window
    if window.count(piece)==1 and window.count(0)==3:
        score+=1

    #Applying Defensive logic i.e. the opponent has an upper hand

    #Penalty if opponent has 4 pieces in a row
    if window.count(opponent_piece)==4:
        score-=50

    #Penalty for 3 opponent pieces in window
    if window.count(opponent_piece)==3 and window.count(0)==1:
        score-=30

    #Penalty for 2 opponent pieces in window
    if window.count(opponent_piece)==2 and window.count(0)==2:
        score-=10

    return score

def heuristic_score(board,piece):
    score=0
    piece_count=0

    #Counting the number of pieces in the centre column
    centre_column=COLUMN_COUNT//2
    for row in range(ROW_COUNT):
        if board[row][centre_column]==piece:
            piece_count+=1

    #Giving bonus to pieces in the centre column
    score=piece_count*5

    #Horizontal Scores
    for row in range(ROW_COUNT):
        row_array=[int(i) for i in list(board[row, :])] 
        for col in range(COLUMN_COUNT-3):
            row_window=row_array[col:col + 4]
            score+=window_evaluation(row_window,piece)

    #Vertical Scores
    for col in range(COLUMN_COUNT):
        col_array = [int(board[row][col]) for row in range(ROW_COUNT)]
        for row in range(ROW_COUNT - 3):
            col_window = col_array[row:row + 4]
            score += window_evaluation(col_window, piece)

    #Positive Diagonal Scores
    for row in range(ROW_COUNT - 3):
        for col in range(COLUMN_COUNT - 3):
            window = [board[row + i][col + i] for i in range(4)]
            score += window_evaluation(window, piece)

    #Negative Diagonal Scores
    for row in range(3,ROW_COUNT):
        for col in range(COLUMN_COUNT - 3):
            window = [board[row - i][col + i] for i in range(4)]
            score += window_evaluation(window, piece)

    return score

def minmax_connect4(board, depth, is_maximizing, piece):
    opponent_piece = 1

    if piece == 1:
        opponent_piece = 2

    is_last = is_last_move(board)
    valid_locations = get_valid_locations(board)

    # Base Case if the game cannot be played further
    if is_last or depth == 0:
        if is_last:
            if winning_move(board, piece):
                return (None, 100000)
            elif winning_move(board, opponent_piece):
                return (None, -100000)
            else:
                return (None, 0)
        else:
            return (None, heuristic_score(board, piece))

    # If piece "1" is having next turn - Maximizing Player
    if is_maximizing:

        # Since "1" aims to maximize the score, the initial best score is set to negative infinity
        best_score = float('-inf')

        # Randomly choosing the Best Column before any move
        column = random.choice(valid_locations)

        # Exploring all valid columns for next move
        for col in valid_locations:
            row = get_next_open_row(board, col)

            # Cloning the board to drop the piece
            board_clone = board.copy()
            drop_piece(board_clone, row, col, piece)

            # Find the best score for the corresponding action taken
            _, score = minmax_connect4(board_clone, depth - 1, False, piece)

            if score > best_score:
                best_score = score
                column = col

        return column, best_score

    # If piece "2" is having next turn - Minimizing Player
    else:

        # Since "2" aims to minimize the score, the initial best score is set to infinity
        best_score = float('inf')

        # Randomly choosing the Best Column before any move
        column = random.choice(valid_locations)

        # Exploring all valid columns for next move
        for col in valid_locations:
            row = get_next_open_row(board, col)

            # Cloning the board to drop the piece
            board_clone = board.copy()
            drop_piece(board_clone, row, col, opponent_piece)

            # Find the best score for the corresponding action taken
            _, score = minmax_connect4(board_clone, depth - 1, True, piece)

            if score < best_score:
                best_score = score
                column = col

        return column, best_score

def minmax_abpruning_connect4(board, depth, alpha, beta, is_maximizing, piece):
    opponent_piece = 1

    if piece == 1:
        opponent_piece = 2

    is_last = is_last_move(board)
    valid_locations = get_valid_locations(board)

    # Base Case if the game cannot be played further
    if is_last or depth == 0:
        if is_last:
            if winning_move(board, piece):
                return (None, 100000)
            elif winning_move(board, opponent_piece):
                return (None, -100000)
            else:
                return (None, 0)
        else:
            return (None, heuristic_score(board, piece))

    # If piece "1" is having next turn - Maximizing Player
    if is_maximizing:

        # Since "1" aims to maximize the score, the initial best score is set to negative infinity
        best_score = float('-inf')

        # Randomly choosing the Best Column before any move
        column = random.choice(valid_locations)

        # Exploring all valid columns for next move
        for col in valid_locations:
            row = get_next_open_row(board, col)

            # Cloning the board to drop the piece
            board_clone = board.copy()
            drop_piece(board_clone, row, col, piece)

            # Find the best score for the corresponding action taken
            _, score = minmax_abpruning_connect4(board_clone, depth - 1, alpha, beta, False, piece)

            if score > best_score:
                best_score = score
                column = col
            alpha = max(alpha, best_score)
            if alpha >= beta:
                break

        return column, best_score

    # If piece "2" is having next turn - Minimizing Player
    else:

        # Since "2" aims to minimize the score, the initial best score is set to infinity
        best_score = float('inf')

        # Randomly choosing the Best Column before any move
        column = random.choice(valid_locations)

        # Exploring all valid columns for next move
        for col in valid_locations:
            row = get_next_open_row(board, col)

            # Cloning the board to drop the piece
            board_clone = board.copy()
            drop_piece(board_clone, row, col, opponent_piece)

            # Find the best score for the corresponding action taken
            _, score = minmax_abpruning_connect4(board_clone, depth - 1, alpha, beta, True, piece)

            if score < best_score:
                best_score = score
                column = col
            beta = min(beta, best_score)
            if alpha >= beta:
                break

        return column, best_score

#State Representation of the Board
def curr_state(board):
    board_list = []
    for row in range(ROW_COUNT):
        for col in range(COLUMN_COUNT):
            if board[row][col]!=0:
                board_list.append(int(board[row][col]))
            else:
                board_list.append(0)
    return tuple(board_list)

#Action Space
def get_valid_actions(board):
    valid_actions = []
    for col in range(COLUMN_COUNT):
        row = get_next_open_row(board, col)
        if row is not None:
            valid_actions.append(col)
    return valid_actions

# Dictionary to store quality of each move - - Q[(state, action)] = value
q_table = {}

# Hyperparameters
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Check!
min_epsilon = 0.01
epsilon_decay = 0.999


# Ensure Q-table has entry - CHECK!!!
def ensure_state_in_q_table(state, valid_actions):
    if state not in q_table:
        q_table[state] = {a: 0.0 for a in valid_actions}

# Exploration vs Exploitation
def choose_action(state, valid_actions, epsilon):
    ensure_state_in_q_table(state, valid_actions)

    # Agent to explore a random move epsilon% of time
    if random.random() < epsilon:
        return random.choice(valid_actions)

    # Agent to follow Q-values for valid actions
    q_values = q_table[state]
    highest_q = float('-inf')
    best_actions = []

    for action in valid_actions:
        if q_values[action] > highest_q:
            highest_q = q_values[action]
            best_actions = [action]

        elif q_values[action] == highest_q:
            best_actions.append(action)

    return random.choice(best_actions)

def play_game():
    global epsilon

    # Initialize a new board
    board = create_board()

    # Initialize the state of the game
    state = curr_state(board)

    # Tracking history of the game
    game_history = []

    player = 1

    game_over = False

    # Starting the game
    while not game_over:
        # Getting all the legal actions at the current state of the board
        actions = get_valid_actions(board)

        # Choosing an action from all the legal actions set
        action_taken = choose_action(state, actions, epsilon)
        row = get_next_open_row(board, action_taken)

        # Performing the selected action on the board
        drop_piece(board, row, action_taken, player)

        next_state = curr_state(board)
        game_history.append((state, action_taken, player, next_state))

        if winning_move(board, player):
            winner = player
            game_over = True
        elif len(get_valid_actions(board)) == 0:
            winner = 0
            game_over = True
        else:
            state = next_state
            player = 2 if player == 1 else 1

    # Scores = {1: 1, 2: -1, Draw: 0}
    if winner == 1:
        reward = 1
    elif winner == 2:
        reward = -1
    else:
        reward = 0

    # Updating Q-Values
    reversed_history = game_history[::-1]

    for state, action, player_id, next_state in reversed_history:
        valid_actions = get_valid_actions(board)
        ensure_state_in_q_table(state, [action])
        ensure_state_in_q_table(next_state, valid_actions)

        old_q = q_table[state][action]
        future_q = max(q_table[next_state].values()) if valid_actions else 0
        q_table[state][action] = old_q + alpha * (reward + gamma * future_q - old_q)

        reward *= -1

    # Decaying Epsilon
    epsilon = max(min_epsilon, epsilon * epsilon_decay)

def q_agent_move(board, piece):
    state = curr_state(board)
    valid_actions = get_valid_actions(board)
    action = choose_action(state, valid_actions, epsilon)
    return action

def default_opponent(board,piece):
    opponent_piece=1
    if piece==1:
        opponent_piece = 2

    valid_locations=get_valid_locations(board)

    #Opponent trying to win
    for col in valid_locations:
        temp_board=board.copy()
        row = get_next_open_row(temp_board, col)
        drop_piece(temp_board, row, col, piece)
        if winning_move(temp_board, piece):
            return col

    #Opponent blocking moves
    for col in valid_locations:
        temp_board=board.copy()
        row = get_next_open_row(temp_board, col)
        drop_piece(temp_board, row, col, opponent_piece)
        if winning_move(temp_board, opponent_piece):
            return col

    # center = COLUMN_COUNT // 2
    # if center in valid_locations:
    #     return center

    return random.choice(valid_locations)

def random_opponent(board, piece):
    valid_columns = get_valid_locations(board)
    return random.choice(valid_columns) if valid_columns else None

TRAINING_EPISODES = 50000  # Number of games to train for
DISPLAY_EVERY = 10000  # Display progress every N games

# Training Function
def train_q_agent():
    print("Training Q-learning agent...")
    for episode in range(1, TRAINING_EPISODES + 1):
        play_game()  # This updates the Q-table

        if episode % DISPLAY_EVERY == 0:
            print(f"Episode {episode}, Epsilon: {epsilon:.4f}")
    print("Training completed!")

pygame.init()
SQUARESIZE = 100
width = COLUMN_COUNT * SQUARESIZE
height = (ROW_COUNT+1) * SQUARESIZE
size = (width, height)
RADIUS = int(SQUARESIZE/2 - 5)
screen = pygame.display.set_mode(size)
myfont = pygame.font.SysFont("monospace", 75)

board = create_board()
draw_board(board)
pygame.display.update()

game_over = False
turn = 0 # 0 = Random (Player 1), 1 = Minimax (Player 2)

def play_q_vs_default(q_player_first=True, display=False):
    board = create_board()
    state = curr_state(board)
    player = 1 if q_player_first else 2
    game_over = False

    while not game_over:
        actions = get_valid_actions(board)

        if player == 1:  # Q-learning agent
            col = q_agent_move(board, 1) if q_player_first else default_opponent(board, 1)
        else:  # Default opponent
            col = default_opponent(board, 2) if q_player_first else q_agent_move(board, 2)

        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, player)

            if winning_move(board, player):
                return player  # 1 or 2 wins
            elif len(get_valid_actions(board)) == 0:
                return 0  # Draw

            player = 2 if player == 1 else 1

def play_minimax_vs_default(minimax_player_first=True, ab_pruning=False, display=False):
    board = create_board()
    player = 1 if minimax_player_first else 2
    game_over = False

    while not game_over:
        if player == 1:  # Minimax
            if minimax_player_first:
                col, _ = minmax_abpruning_connect4(board, 4, float('-inf'), float('inf'), True, 1) if ab_pruning else minmax_connect4(board, 4, True, 1)
            else:
                col = default_opponent(board, 1)
        else:  # Default Opponent
            if minimax_player_first:
                col = default_opponent(board, 2)
            else:
                col, _ = minmax_abpruning_connect4(board, 4, float('-inf'), float('inf'), True, 2) if ab_pruning else minmax_connect4(board, 4, True, 2)

        if is_valid_location(board, col):
            row = get_next_open_row(board, col)
            drop_piece(board, row, col, player)

            if winning_move(board, player):
                return player  # 1 or 2 wins
            elif len(get_valid_actions(board)) == 0:
                return 0  # Draw

            player = 2 if player == 1 else 1

def evaluate_agent(agent='q', num_games=100, ab_pruning=True):
    q_wins = 0
    default_wins = 0
    draws = 0
    total_time = 0

    for i in range(num_games):
        start_time = time.time()
        if agent == 'q':
            result = play_q_vs_default(q_player_first=(i % 2 == 0))
        elif agent == 'minimax':
            result = play_minimax_vs_default(minimax_player_first=(i % 2 == 0), ab_pruning=ab_pruning)
        else:
            raise ValueError("Invalid agent type. Use 'q' or 'minimax'.")

        elapsed_time = time.time() - start_time
        total_time += elapsed_time

        if result == 1:
            q_wins += 1
        elif result == 2:
            default_wins += 1
        else:
            draws += 1

    avg_time = total_time / num_games

    print(f"\nResults after {num_games} games:")
    if agent == 'q':
        print(f"Q-agent wins: {q_wins}")
    else:
        print(f"Minimax wins: {q_wins}")
    print(f"Default Opponent wins: {default_wins}")
    print(f"Draws: {draws}")
    print(f"Average execution time per game: {avg_time:.4f} seconds")

if __name__ == "__main__":
    print("Play Connect 4!")
    print("Choose agent types for both players:")
    print("1. Q-Learning")
    print("2. Minimax")
    print("3. Minimax with Alpha-Beta Pruning")
    print("4. Default Opponent")
    print("5. Random Opponent")

    player1_choice = input("Select Player 1 (X) agent (1/2/3/4/5): ").strip()
    player2_choice = input("Select Player 2 (O) agent (1/2/3/4/5): ").strip()

    def get_agent_func(choice):
        if choice == "1":
            train_q_agent()
            return lambda board, piece: q_agent_move(board, piece)
        elif choice == "2":
            return lambda board, piece: minmax_connect4(board, 4, True, piece)[0]
        elif choice == "3":
            return lambda board, piece: minmax_abpruning_connect4(board, 4, float('-inf'), float('inf'), True, piece)[0]
        elif choice == "4":
            return lambda board, piece: default_opponent(board, piece)
        elif choice == "5":
            return lambda board, piece: random_opponent(board, piece)
        else:
            raise ValueError("Invalid choice! Please select 1 to 5.")

    def play_custom_agents(agent1_func, agent2_func, num_games=100):
        p1_wins, p2_wins, draws, total_time = 0, 0, 0, 0.0

        for _ in range(num_games):
            board = create_board()
            game_over = False
            player = 1
            start_time = time.time()

            while not game_over:
                col = agent1_func(board, 1) if player == 1 else agent2_func(board, 2)
                if col is not None and is_valid_location(board, col):
                    row = get_next_open_row(board, col)
                    drop_piece(board, row, col, player)

                    if winning_move(board, player):
                        if player == 1:
                            p1_wins += 1
                        else:
                            p2_wins += 1
                        game_over = True
                    elif len(get_valid_locations(board)) == 0:
                        draws += 1
                        game_over = True
                    else:
                        player = 2 if player == 1 else 1

                total_time += time.time() - start_time

        avg_time = total_time / num_games
        print(f"\nResults after {num_games} games:")
        print(f"Player 1 (X) wins: {p1_wins}")
        print(f"Player 2 (O) wins: {p2_wins}")
        print(f"Draws: {draws}")
        print(f"Average execution time per game: {avg_time:.6f} seconds")

    agent1_func = get_agent_func(player1_choice)
    agent2_func = get_agent_func(player2_choice)
    play_custom_agents(agent1_func, agent2_func)