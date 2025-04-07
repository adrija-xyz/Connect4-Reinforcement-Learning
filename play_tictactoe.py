#URL for TicTacToe code used: https://github.com/ImKennyYip/tictactoe-python

import random

def set_tile(row, column):
    print(f"Clicked on row {row}, column {column}")
    global curr_player

    if game_over:
        return

    if board[row][column]["text"] != "":
        return

    board[row][column]["text"] = curr_player

    if curr_player == playerO:
        curr_player = playerX
    else:
        curr_player = playerO

    label["text"] = curr_player + "'s turn"

    check_winner()

def check_winner():
    global turns, game_over
    turns += 1

    # horizontally
    for row in range(3):
        if (board[row][0]["text"] == board[row][1]["text"] == board[row][2]["text"]
                and board[row][0]["text"] != ""):
            label.config(text=board[row][0]["text"] + " is the winner!", foreground=color_yellow)
            for column in range(3):
                board[row][column].config(foreground=color_yellow, background=color_light_gray)
            game_over = True
            return

    # vertically
    for column in range(3):
        if (board[0][column]["text"] == board[1][column]["text"] == board[2][column]["text"]
                and board[0][column]["text"] != ""):
            label.config(text=board[0][column]["text"] + " is the winner!", foreground=color_yellow)
            for row in range(3):
                board[row][column].config(foreground=color_yellow, background=color_light_gray)
            game_over = True
            return

    # diagonally
    if (board[0][0]["text"] == board[1][1]["text"] == board[2][2]["text"]
            and board[0][0]["text"] != ""):
        label.config(text=board[0][0]["text"] + " is the winner!", foreground=color_yellow)
        for i in range(3):
            board[i][i].config(foreground=color_yellow, background=color_light_gray)
        game_over = True
        return

    # anti-diagonally
    if (board[0][2]["text"] == board[1][1]["text"] == board[2][0]["text"]
            and board[0][2]["text"] != ""):
        label.config(text=board[0][2]["text"] + " is the winner!", foreground=color_yellow)
        board[0][2].config(foreground=color_yellow, background=color_light_gray)
        board[1][1].config(foreground=color_yellow, background=color_light_gray)
        board[2][0].config(foreground=color_yellow, background=color_light_gray)
        game_over = True
        return

    # tie
    if turns == 9:
        game_over = True
        label.config(text="Tie!", foreground=color_yellow)

def new_game():
    global turns, game_over, curr_player

    turns = 0
    game_over = False
    curr_player = playerX

    label.config(text=curr_player + "'s turn", foreground="white")

    for row in range(3):
        for column in range(3):
            board[row][column].config(text="", foreground=color_blue, background=color_gray)

# game setup
playerX = "X"
playerO = "O"
curr_player = playerX
board = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]

color_blue = "#4584b6"
color_yellow = "#ffde57"
color_gray = "#343434"
color_light_gray = "#646464"

turns = 0
game_over = False

scores = {"x": 1, "o": -1, "tie": 0}

def winner_tictactoe(board):
    # Checking rows to select the winner
    for row in board:
        if row[0] == row[1] == row[2] != "":
            return row[0]

    # Checking columns to select the winner
    for col in range(len(board[0])):
        if board[0][col] == board[1][col] == board[2][col] != "":
            return board[0][col]

    # Checking diagonals to select the winner
    if board[0][0] == board[1][1] == board[2][2] != "" or board[0][2] == board[1][1] == board[2][0] != "":
        return board[1][1]

    # Checking if the game still can be played
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col] == "":
                return None

    # Game returns a tie if it cannot be played anymore
    return "tie"

def minmax_tictactoe(board, is_maximizing):
    winner = winner_tictactoe(board)

    # Base case if the game cannot be played further
    if winner is not None:
        return scores[winner]

    # If "x" is having next turn - Maximizing Player
    if is_maximizing:

        # Since "x" aims to maximize the score, the initial best score is set to negative infinity
        best_score = float('-inf')

        # Iterating through the board to check a vacant cell
        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] == "":
                    # Selecting the vacant cell for next move
                    board[row][col] = "x"

                    # Find the best score for the corresponding action taken
                    score = minmax_tictactoe(board, False)

                    # Undo the action to check alternate move
                    board[row][col] = ""

                    best_score = max(best_score, score)

        return best_score

    # If 'o' is having next turn - Minimizing Player
    else:

        # Since "o" aims to minimize the score, the initial best score is set to infinity
        best_score = float('inf')

        # Iterating through the board to check a vacant cell
        for row in range(len(board)):
            for col in range(len(board[0])):

                # Selecting the vacant cell for next move
                if board[row][col] == "":
                    # Selecting the vacant cell for next move
                    board[row][col] = "o"

                    # Find the best score for the corresponding action taken
                    score = minmax_tictactoe(board, True)
                    best_score = min(best_score, score)

                    # Undo the action to check alternate move
                    board[row][col] = ""

        return best_score

def minmax_abpruning_tictactoe(board, alpha, beta, is_maximizing):
    winner = winner_tictactoe(board)

    # Base case if the game cannot be played further
    if winner is not None:
        return scores[winner]

    # If "x" is having next turn - Maximizing Player
    if is_maximizing:

        # Since "x" aims to maximize the score, the initial best score is set to negative infinity
        best_score = float('-inf')

        # Iterating through the board to check a vacant cell
        for row in range(len(board)):
            for col in range(len(board[0])):
                if board[row][col] == "":

                    # Selecting the vacant cell for next move
                    board[row][col] = "x"

                    # Find the best score for the corresponding action taken
                    score = minmax_abpruning_tictactoe(board,alpha, beta,False)

                    # Undo the action to check alternate move
                    board[row][col] = ""

                    best_score = max(best_score, score)

                    # Alpha tracks the best score of max
                    alpha = max(alpha, best_score)

                    if alpha >= beta:
                        break

        return best_score

    # If 'o' is having next turn - Minimizing Player
    else:

        # Since "o" aims to minimize the score, the initial best score is set to infinity
        best_score = float('inf')

        # Iterating through the board to check a vacant cell
        for row in range(len(board)):
            for col in range(len(board[0])):

                # Selecting the vacant cell for next move
                if board[row][col] == "":

                    # Selecting the vacant cell for next move
                    board[row][col] = "o"

                    # Find the best score for the corresponding action taken
                    score = minmax_abpruning_tictactoe(board, alpha, beta, True)

                    # Undo the action to check alternate move
                    board[row][col] = ""

                    best_score = min(best_score, score)

                    beta = min(beta, best_score)
                    if beta <= alpha:
                        break

        return best_score

#Dictionary to store quality of each move
q_table = {}

#Hyperparameters
alpha=0.1
gamma=0.9
epsilon=0.1

def curr_state(board):
    board_list = []
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col]!="":
                board_list.append(board[row][col])
            else:
                board_list.append("_")
    return tuple(board_list)

def get_valid_actions(board):
    valid_actions=[]
    for row in range(len(board)):
        for col in range(len(board[0])):
            if board[row][col]=="":
                valid_actions.append((row,col))
    return valid_actions

#Implementing Epsilon Greedy Strategy
def choose_action(state,valid_actions,epsilon):
    #Initializing Q values for unseen state
    if state not in q_table:
        q_table[state] = {action: 0 for action in valid_actions}

    #Agent to explore a random move epsilon% of time
    if random.random() < epsilon:
        return random.choice(valid_actions)

    #Agent to follow Q-values for valid actions
    else:
        q_values=q_table[state]
        highest_q=float('-inf')
        best_actions=[]

        for action in valid_actions:
            if q_values[action]>highest_q:
                highest_q = q_values[action]
                best_actions = [action]

            elif q_values[action]==highest_q:
                best_actions.append(action)

        return random.choice(best_actions)

def get_valid_actions_from_state(state):
    return [(i // 3, i % 3) for i, v in enumerate(state) if v == "_"]

def default_opponent_move(board, player):
    opponent = "o" if player == "x" else "x"
    actions = get_valid_actions(board)

    # Try to win
    for row, col in actions:
        board[row][col] = player
        if winner_tictactoe(board) == player:
            board[row][col] = ""  # reset
            return (row, col)
        board[row][col] = ""  # reset

    # Try to block opponent's winning move
    for row, col in actions:
        board[row][col] = opponent
        if winner_tictactoe(board) == opponent:
            board[row][col] = ""  # reset
            return (row, col)
        board[row][col] = ""  # reset

    # Else, choose a random move
    return random.choice(actions)

def random_opponent_move(board,player):
    valid_moves = get_valid_actions(board)
    return random.choice(valid_moves) if valid_moves else None

def play_game():
    # Separate board for logic (not GUI buttons)
    logic_board = [["" for _ in range(3)] for _ in range(3)]

    # Initialize state
    state = curr_state(logic_board)

    # Game history to track transitions
    game_history = []

    player = "x"

    while True:
        actions = get_valid_actions(logic_board)
        action_taken = choose_action(state, actions, epsilon)

        # Apply action
        row, col = action_taken
        logic_board[row][col] = player

        # Get new state
        new_state = curr_state(logic_board)

        # Save to history
        game_history.append((state, action_taken, player))

        # Check for winner
        winner = winner_tictactoe(logic_board)
        if winner is not None:
            break

        # Switch state and player
        state = new_state
        player = "o" if player == "x" else "x"

    # Assign reward
    reward = scores[winner]  # winner is "x", "o", or "tie"

    # Reverse history for backpropagation
    reversed_history = game_history[::-1]

    for idx, (state, action, player) in enumerate(reversed_history):
        if state not in q_table:
            q_table[state] = {a: 0 for a in get_valid_actions_from_state(state)}

        if idx > 0:
            next_state = reversed_history[idx - 1][0]
        else:
            next_state = None

        future_q = 0
        if next_state and next_state in q_table and q_table[next_state]:
            future_q = max(q_table[next_state].values())

        old_q = q_table[state][action]
        q_table[state][action] = old_q + alpha * (reward + gamma * future_q - old_q)

        # Alternate reward for the opponent
        reward = -reward
import time

TRAINING_EPISODES = 50000  # Number of games to train for
DISPLAY_EVERY = 10000       # Display progress every N games

def train_q_agent():
    print("Training Q-learning agent...")
    for episode in range(1, TRAINING_EPISODES + 1):
        play_game()  # This updates the Q-table
        if episode % DISPLAY_EVERY == 0:
            print(f"Episode {episode}, Epsilon: {epsilon:.4f}")
    print("Training completed!")

def q_table_action(board, player):
    state = curr_state(board)
    valid_actions = get_valid_actions(board)

    if state in q_table:
        best_q = float('-inf')
        best_actions = []
        for action in valid_actions:
            q_val = q_table[state].get(action, 0)
            if q_val > best_q:
                best_q = q_val
                best_actions = [action]
            elif q_val == best_q:
                best_actions.append(action)
        return random.choice(best_actions)
    else:
        return random.choice(valid_actions)

def play_q_vs_default(q_player_first=True):
    board = [["" for _ in range(3)] for _ in range(3)]
    player = "x" if q_player_first else "o"

    while True:
        if player == "x":
            row, col = q_table_action(board, "x")
            board[row][col] = "x"
        else:
            row, col = default_opponent_move(board, "o")
            board[row][col] = "o"

        winner = winner_tictactoe(board)
        if winner is not None:
            break
        player = "o" if player == "x" else "x"

    if winner == "x":
        return 1 if q_player_first else 2
    elif winner == "o":
        return 2 if q_player_first else 1
    else:
        return 0  # draw

def play_minimax_vs_default(minimax_player_first=True, ab_pruning=True):
    board = [["" for _ in range(3)] for _ in range(3)]
    player = "x" if minimax_player_first else "o"

    while True:
        if player == "x":
            best_score = float('-inf')
            best_move = None
            for row, col in get_valid_actions(board):
                board[row][col] = "x"
                if ab_pruning:
                    score = minmax_abpruning_tictactoe(board, float('-inf'), float('inf'), False)
                else:
                    score = minmax_tictactoe(board, False)
                board[row][col] = ""
                if score > best_score:
                    best_score = score
                    best_move = (row, col)
            row, col = best_move
            board[row][col] = "x"
        else:
            row, col = default_opponent_move(board, "o")
            board[row][col] = "o"

        winner = winner_tictactoe(board)
        if winner is not None:
            break
        player = "o" if player == "x" else "x"

    if winner == "x":
        return 1 if minimax_player_first else 2
    elif winner == "o":
        return 2 if minimax_player_first else 1
    else:
        return 0  # draw

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
    print(f"Average execution time per game: {avg_time:.6f} seconds")

if __name__ == "__main__":
    print("Play Tic Tac Toe!")
    print("Choose player types:")
    print("1. Q-Learning")
    print("2. Minimax")
    print("3. Minimax with Alpha-Beta Pruning")
    print("4. Default Opponent")
    print("5. Random Opponent")

    player1_choice = input("Select Player 1 (X) agent (1/2/3/4): ").strip()
    player2_choice = input("Select Player 2 (O) agent (1/2/3/4): ").strip()

    def get_agent_func(choice, is_first_player):
        if choice == "1":
            train_q_agent()
            return lambda: q_table_action
        elif choice == "2":
            return lambda: lambda b, p: minimax_action(b, p, ab_pruning=False)
        elif choice == "3":
            return lambda: lambda b, p: minimax_action(b, p, ab_pruning=True)
        elif choice == "4":
            return lambda: default_opponent_move
        elif choice == "5":
            return lambda: random_opponent_move
        else:
            raise ValueError("Invalid agent choice!")

    def minimax_action(board, player, ab_pruning):
        best_score = float('-inf') if player == "x" else float('inf')
        best_move = None
        for row, col in get_valid_actions(board):
            board[row][col] = player
            score = (minmax_abpruning_tictactoe(board, float('-inf'), float('inf'), player == "x")
                     if ab_pruning else
                     minmax_tictactoe(board, player == "x"))
            board[row][col] = ""
            if (player == "x" and score > best_score) or (player == "o" and score < best_score):
                best_score = score
                best_move = (row, col)
        return best_move

    player1_func = get_agent_func(player1_choice, True)()
    player2_func = get_agent_func(player2_choice, False)()


    def play_custom_agents(player1_func, player2_func, num_games=100):
        player1_wins = 0
        player2_wins = 0
        draws = 0
        total_time = 0

        for _ in range(num_games):
            board = [["" for _ in range(3)] for _ in range(3)]
            player = "x"
            winner = None

            start_time = time.time()

            while True:
                if player == "x":
                    row, col = player1_func(board, "x")
                    board[row][col] = "x"
                else:
                    row, col = player2_func(board, "o")
                    board[row][col] = "o"

                winner = winner_tictactoe(board)
                if winner is not None:
                    break
                player = "o" if player == "x" else "x"

            elapsed_time = time.time() - start_time
            total_time += elapsed_time

            if winner == "x":
                player1_wins += 1
            elif winner == "o":
                player2_wins += 1
            else:
                draws += 1

        avg_time = total_time / num_games
        print(f"\nResults after {num_games} games:")
        print(f"Player 1 (x) wins: {player1_wins}")
        print(f"Player 2 (o) wins: {player2_wins}")
        print(f"Draws: {draws}")
        print(f"Average execution time per game: {avg_time:.6f} seconds")

    play_custom_agents(player1_func, player2_func,num_games=100)