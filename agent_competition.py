import random
import sys
import time

# You can use the functions in othello_shared to write your AI for competition
from othello_shared import find_lines, get_possible_moves, get_score, play_move

# If you choose to try MCTS, you can make use of the code below
class MCTS_state():
    """
            This sample code gives you a idea of how to store records for each node
            in the tree. However, you are welcome to modify this part or define your own
            class.
    """
    def __init__(self, ID, parent, child, reward, total, board):
        self.ID = ID
        self.parent = parent    # a list of states
        self.child = child      # a list of states
        self.reward = reward    # number of win
        self.total = total      # number of simulation for self and (grand*)children
        self.board = board
        self.visited = 0        # 0 -> not visited yet, 1 -> already visited


def select_move_MCTS(board, color, limit):
    """
               You can add additional help functions as long as this function will return a position tuple
    """
    initial_state = MCTS_state(0, [], [], 0, 0, board) # this is just an example. delete it when you start to code.
    pass


def run_ai():
    """
        Please do not modify this part.
        """
    print("Othello AI")  # First line is the name of this AI
    arguments = input().split(",")

    color = int(arguments[0])  # Player color: 1 for dark (goes first), 2 for light.
    limit = int(arguments[1])  # Iteration limit
    minimax = int(arguments[2])  # not used here
    caching = int(arguments[3])  # not used here
    ordering = int(arguments[4])  # not used here

    if (limit == -1):
        eprint("Iteration Limit is OFF")
    else:
        eprint("Iteration Limit is ", limit)

    if (minimax == 1 and ordering == 1): eprint("Node Ordering should have no impact on Minimax")

    while True:  # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL":  # Game is over.
            print
        else:
            board = eval(input())  # Read in the input and turn it into a Python
            # object. The format is a list of rows. The
            # squares in each row are represented by
            # 0 : empty square
            # 1 : dark disk (player 1)
            # 2 : light disk (player 2)

            #Uncomment the line below if you choose to use MCTS
            #movei, movej = select_move_MCTS(board, color, limit)

            #Otherwise, use whatever formulation you like! e.g.:
            #movei, movej = select_move_minimax(board, color, limit, caching)
            #movei, movej = select_move_alphabeta(board, color, limit, caching, ordering)            

            print("{} {}".format(movei, movej))