#!/usr/bin/python3
#  agent.py
#  Nine-Board Tic-Tac-Toe Agent starter code
#  COMP3411/9814 Artificial Intelligence
#  CSE, UNSW

import socket
import sys
import numpy as np

# a board cell can hold:
#   0 - Empty
#   1 - We played here
#   2 - Opponent played here

# the boards are of size 10 because index 0 isn't used
boards = np.zeros((10, 10), dtype="int8")
s = [".","X","O"]
curr = 0 # this is the current board to play in

# print a row
def print_board_row(bd, a, b, c, i, j, k):
    print(" "+s[bd[a][i]]+" "+s[bd[a][j]]+" "+s[bd[a][k]]+" | " \
             +s[bd[b][i]]+" "+s[bd[b][j]]+" "+s[bd[b][k]]+" | " \
             +s[bd[c][i]]+" "+s[bd[c][j]]+" "+s[bd[c][k]])

# Print the entire board
def print_board(board):
    print_board_row(board, 1,2,3,1,2,3)
    print_board_row(board, 1,2,3,4,5,6)
    print_board_row(board, 1,2,3,7,8,9)
    print(" ------+-------+------")
    print_board_row(board, 4,5,6,1,2,3)
    print_board_row(board, 4,5,6,4,5,6)
    print_board_row(board, 4,5,6,7,8,9)
    print(" ------+-------+------")
    print_board_row(board, 7,8,9,1,2,3)
    print_board_row(board, 7,8,9,4,5,6)
    print_board_row(board, 7,8,9,7,8,9)
    print()

def evaluate_mini_board(mini_board, player):
    score = 0
    winning_score = 1000  # A large number representing a winning state for a mini-board
    losing_score = -1000  # A large negative number representing a losing state for a mini-board
    win_patterns = [
        [1, 2, 3], [4, 5, 6], [7, 8, 9],  # Horizontal
        [1, 4, 7], [2, 5, 8], [3, 6, 9],  # Vertical
        [1, 5, 9], [3, 5, 7]              # Diagonal
    ]
    corners = [1, 3, 7, 9]
    # Check if the game is won or lost on the mini-board
    for pattern in win_patterns:
        if all(mini_board[idx] == player for idx in pattern):
            return winning_score
        elif all(mini_board[idx] == opponent(player) for idx in pattern):
            return losing_score

    # Check if the entire mini-board is empty and if so, prioritize corner strategy
    if is_board_empty(mini_board):
        for corner in corners:
            if mini_board[corner] == 0:
                score += 250 
    # Evaluate based on two-in-a-line or one-in-a-line
    for pattern in win_patterns:
        cells = [mini_board[idx] for idx in pattern]
        if cells.count(player) == 2 and cells.count(0) == 1:
            score += 50  # Two-in-a-line is a strong position
        elif cells.count(player) == 1 and cells.count(0) == 2:
            score += 10  # One-in-a-line with two empty spaces is an opportunity

    # Deduct points for opponent's potential wins
    for pattern in win_patterns:
        cells = [mini_board[idx] for idx in pattern]
        if cells.count(opponent(player)) == 2 and cells.count(0) == 1:
            score -= 70  # Opponent's two-in-a-line is a threat

    return score

def is_board_empty(mini_board):
    return all(cell == 0 for cell in mini_board[1:])

def evaluate_board_state(boards, player, last_move):
    overall_score = 0
    current_board = last_move % 9 if last_move else 0  # Determine the current mini-board based on the last move

    # Evaluate each mini-board and add its score to the overall score
    for board_num in range(1, 10):  # Assuming board_num is 1-indexed
        mini_board_score = evaluate_mini_board(boards[board_num], player)
        if mini_board_score != 0:
            # If a winning or losing state is found, it dominates the evaluation
            return mini_board_score
        overall_score += mini_board_score

    return overall_score

def opponent(player):
    return 1 if player == 0 else 1

def alphabeta(depth, alpha, beta, player, is_maximizing):
    global curr, boards  # Ensure global access to curr and boards

    if is_terminal(boards, curr) or depth == 0:
        return evaluate_board_state(boards, player, curr)

    if is_maximizing:
        max_eval = -float('inf')
        possible_moves = get_possible_moves(boards, curr)
        for move in possible_moves:
            place(curr, move, player)  # Assuming 'place' updates the board and curr
            previous_curr = curr
            curr = move % 9
            eval = alphabeta(depth - 1, alpha, beta, opponent(player), False)
            boards[previous_curr][move] = 0  # Assuming 0 is empty
            curr = previous_curr
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        possible_moves = get_possible_moves(boards, curr)
        for move in possible_moves:
            place(curr, move, player)
            previous_curr = curr
            curr = move % 9
            eval = alphabeta(depth - 1, alpha, beta, opponent(player), True)
            boards[previous_curr][move] = 0
            curr = previous_curr
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if alpha >= beta:
                break
        return min_eval


# Helper function to determine if the current state is terminal
def is_terminal(boards, current_board):
    # Winning positions for a 3x3 Tic-Tac-Toe board
    winning_positions = [
        [1, 2, 3],  # Top row
        [4, 5, 6],  # Middle row
        [7, 8, 9],  # Bottom row
        [1, 4, 7],  # Left column
        [2, 5, 8],  # Center column
        [3, 6, 9],  # Right column
        [1, 5, 9],  # Diagonal top-left to bottom-right
        [3, 5, 7]   # Diagonal top-right to bottom-left
    ]

    board = boards[current_board]  # Get the specific mini-board to check

    # Check for a win
    for positions in winning_positions:
        if board[positions[0]] != 0 and \
           board[positions[0]] == board[positions[1]] == board[positions[2]]:
            return True  # Terminal because there's a winner

    # Check for a draw (i.e., no empty spots left and no winner)
    if all(board[i] != 0 for i in range(1, 10)):
        return True  # Terminal because it's a draw

    return False  # Not terminal if there's no winner and empty spots are available

def get_possible_moves(boards, current_board):
    """Return a list of valid moves on the current mini-board."""
    possible_moves = []
    for cell in range(1, 10): 
        if boards[current_board][cell] == 0:
            possible_moves.append(cell)
    return possible_moves

# Play function using alphabeta pruning
def play():
    global boards, curr
    best_score = -float('inf')
    best_move = None
    alpha = -float('inf')
    beta = float('inf')
    depth = 3  # Define the search depth for Alpha-Beta pruning

    # Get all possible moves for the current mini-board
    possible_moves = get_possible_moves(boards, curr)
    for move in possible_moves:
        # Simulate the move for the AI player (assuming AI is player 1)
        place(curr, move, 1)
        previous_curr = curr  # Store current board index before recursive call changes it
        # Use Alpha-Beta pruning to evaluate the move
        score = alphabeta(depth, alpha, beta, 0, False)  # Evaluate as opponent's move next
        # Undo the simulated move
        boards[curr][move] = 0  # Reset the move; assumes '0' means empty
        curr = previous_curr  # Restore current board index

        # Update best move based on evaluation
        if score > best_score:
            best_score = score
            best_move = move

    # Apply the best move found
    if best_move is not None:
        place(curr, best_move, 1)

    return best_move



# place a move in the global boards
def place( board, num, player ):
    global curr
    curr = num
    boards[board][num] = player

# read what the server sent us and
# parse only the strings that are necessary
def parse(string):
    if "(" in string:
        command, args = string.split("(")
        args = args.split(")")[0]
        args = args.split(",")
    else:
        command, args = string, []

    # init tells us that a new game is about to begin.
    # start(x) or start(o) tell us whether we will be playing first (x)
    # or second (o); we might be able to ignore start if we internally
    # use 'X' for *our* moves and 'O' for *opponent* moves.

    # second_move(K,L) means that the (randomly generated)
    # first move was into square L of sub-board K,
    # and we are expected to return the second move.
    if command == "second_move":
        # place the first move (randomly generated for opponent)
        place(int(args[0]), int(args[1]), 2)
        return play()  # choose and return the second move

    # third_move(K,L,M) means that the first and second move were
    # in square L of sub-board K, and square M of sub-board L,
    # and we are expected to return the third move.
    elif command == "third_move":
        # place the first move (randomly generated for us)
        place(int(args[0]), int(args[1]), 1)
        # place the second move (chosen by opponent)
        place(curr, int(args[2]), 2)
        return play() # choose and return the third move

    # nex_move(M) means that the previous move was into
    # square M of the designated sub-board,
    # and we are expected to return the next move.
    elif command == "next_move":
        # place the previous move (chosen by opponent)
        place(curr, int(args[0]), 2)
        return play() # choose and return our next move

    elif command == "win":
        print("Yay!! We win!! :)")
        return -1

    elif command == "loss":
        print("We lost :(")
        return -1

    return 0

# connect to socket
def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    port = int(sys.argv[2]) # Usage: ./agent.py -p (port)

    s.connect(('localhost', port))
    while True:
        text = s.recv(1024).decode()
        if not text:
            continue
        for line in text.split("\n"):
            response = parse(line)
            if response == -1:
                s.close()
                return
            elif response > 0:
                s.sendall((str(response) + "\n").encode())

if __name__ == "__main__":
    main()
