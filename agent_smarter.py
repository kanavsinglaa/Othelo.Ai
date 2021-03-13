"""
An AI player for Othello. 
"""

import random
import sys
import time
import copy

# You can use the functions in othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move


# for caching
cache = {}
weight_matrix = []

def eprint(*args, **kwargs): #you can use this for debugging, as it will print to sterr and not stdout
    print(*args, file=sys.stderr, **kwargs)
    
# Method to compute utility value of terminal state
# calculated as the number of disks of the number of the player’s color minus the number of disks of the opponent
def compute_utility(board, color):
    # color 1 - dark
    # color 2 - light
    #IMPLEMENT
    dark_score, light_score = get_score(board)
    # default assuming color 1
    utility = dark_score - light_score
    if color == 2:
        utility = -utility
    return utility

# Better heuristic value of board
def compute_heuristic(board, color): #not implemented, optional
    #IMPLEMENT
    # 1. count mobility
    mobilitiy = compute_mobility(board,color)
    #eprint(mobilitiy)
    # 2. normal utility
    utility = compute_utility(board, color)
    #eprint(utility)
    # 3. compute weight score
    weight_score = compute_weight_score(board,color)
    final_value = mobilitiy+utility+weight_score
    #eprint(cornerCount)
    return final_value


def compute_weight_score(board,color):
    global weight_matrix
    dark = 0
    light = 0
    for i in range(len(weight_matrix)):
        for j in range(len(weight_matrix)):
            if board[i][j] == 1:
                dark += weight_matrix[i][j]
            elif board[i][j] == 2:
                light += weight_matrix[i][j]
    weight_score = dark-light
    if color == 2:
        weight_score = -weight_score
    return weight_score

def compute_mobility(board, color):
    dark, light = len(get_possible_moves(board, 1)), len(get_possible_moves(board,2))
    mobility = dark-light
    if color == 2:
        mobility = -mobility
    return mobility

def compute_corners(board,color):
    corners = [board[0][0], board[0][-1], board[-1][-1], board[-1][0]]
    dark, light = len([x for x in corners if x == 1]), len([x for x in corners if x == 2])
    corner_score = dark-light
    if color == 2:
        corner_score = -corner_score
    return corner_score


# for more complicated heuristics
def initialize_weight_matrix(board):
    n = len(board)
    global weight_matrix
    weight_matrix = [[1 for _ in range(n)] for _ in range(n)]
    # update edge first
    for i in range(n):
        weight_matrix[0][i] = 10
        weight_matrix[i][0] = 10
        weight_matrix[i][-1] = 10
        weight_matrix[-1][i] = 10
    # update corner
    weight_matrix[0][0], weight_matrix[-1][0], weight_matrix[0][-1], weight_matrix[-1][-1] = 1000, 1000, 1000, 1000
    # update trap (second last to corner)
    weight_matrix[1][0], weight_matrix[0][1], weight_matrix[0][-2], weight_matrix[1][-1], weight_matrix[-2][0], weight_matrix[-1][1], \
    weight_matrix[-1][-2], weight_matrix[-2][-1] = -10, -10, -10, -10, -10, -10, -10, -10



############ MINIMAX ###############################
def minimax_min_node(board, color, limit, caching = 0): #returns lowest possible utility
    #IMPLEMENT
    # find terminal state
    # terminal state
    # stand on the position of AI player
    global cache
    #eprint(("caching:", cache))
    state = color, board
    # caching
    if caching == 1 and state in cache:
        return cache[state]
    if limit == 0:
        return None, compute_utility(board,color)
    # compute possible moves of opponents
    possible_moves = get_possible_moves(board, 3-color)
    if not possible_moves:
        # terminal status, return utility value
        res =  None, compute_utility(board, color)
        if caching == 1:
            cache[state] = res
        return res
    best_score = float('inf')
    best_move = None
    for move in possible_moves:
        i, j = move
        successor_board = play_move(board, 3-color, i,j)
        # minimum of other player's gain
        # Player is Min, now player switch to min, use minimax_min_node with color MAX
        tmp_mov, score = minimax_max_node(successor_board, color, limit-1, caching)
        if score < best_score:
            best_score = score
            best_move = move
    res = best_move, best_score
    if caching == 1:
        cache[state] = res
    return res


def minimax_max_node(board, color, limit, caching = 0): #returns highest possible utility
    # color 1 is the max player
    # color 2 is the min player
    # max player always plays a move to change the state to the highest value child
    # player is always max in this case
    global cache
    #eprint(("caching:", cache))
    state = color, board
    if caching == 1 and state in cache:
        return cache[state]
    if limit == 0:
        # compute heuristic
        return None, compute_utility(board,color)
    possible_moves = get_possible_moves(board, color)
    if not possible_moves:
        # terminal status, return utility value
        res =  None, compute_utility(board, color)
        if caching == 1:
            cache[state] = res
        return res
    best_score = -float('inf')
    best_move = None
    for move in possible_moves:
        i,j = move
        successor_board = play_move(board,color, i,j)
        #Player is MAX, now player switch to min, use minimax_min_node with color MIN
        # always pass in AI's color in min
        tmp_mov, score = minimax_min_node(successor_board, color, limit-1, caching)
        if score > best_score:
            best_score = score
            best_move = move
    res = best_move, best_score
    if caching == 1:
        cache[state] = res
    return res


def select_move_minimax(board, color, limit, caching = 0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.

    1. Max always plays a move to change the state to the highest valued child.
    2. Min always plays a move to change the state to the lowest valued child.
   """
    #IMPLEMENT
    #eprint(("minmax",MAX,MIN))
    # take player's color as MAX node, opponent as MIN node
    # initialize weight matrix
    # initialize_weight_matrix(board)
    move, utility = minimax_max_node(board, color, limit, caching)
    # add delay to show the process
    # time.sleep(0.5)
    return move


############ ALPHA-BETA PRUNING #####################
def alphabeta_min_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    #IMPLEMENT
    # current player is MIN
    global cache
    state = color, board
    if caching == 1 and state in cache:
        return cache[state]
    if limit == 0:
        return None, compute_heuristic(board,color)
    # name ordering heuristics
    possible_moves = name_ordering_heuristic(board,get_possible_moves(board, 3-color), 3-color, color,ordering,False)
    if not possible_moves:
        # terminal status, return utility value
        res =  None, compute_heuristic(board, color)
        if caching == 1:
            cache[state] = res
        return res
    best_score = float('inf')
    best_move = None
    for move in possible_moves:
        i, j = move
        # Player is MIN
        successor_board = play_move(board, 3-color, i, j)
        tmp_move, score = alphabeta_max_node(successor_board, color, alpha, beta, limit-1, caching)
        if best_score > score:
            best_score = score
            best_move = move
        if beta > best_score:
            beta = best_score
            if beta <= alpha:
                break
    res = best_move, best_score
    if caching == 1:
        cache[state] = res
    return res

def alphabeta_max_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    #IMPLEMENT
    # current player is MIN
    global cache
    state = color, board
    if caching == 1 and state in cache:
        return cache[state]
    if limit == 0:
        return None, compute_heuristic(board,color)
    # name ordering heuristics
    possible_moves = name_ordering_heuristic(board, get_possible_moves(board, color), color, color, ordering,True)
    if not possible_moves:
        # terminal status, return utility value
        res =  None, compute_heuristic(board, color)
        if caching == 1:
            cache[state] = res
        return res
    best_score = -float('inf')
    best_move = None
    for move in possible_moves:
        i, j = move
        # Player is MAX
        successor_board = play_move(board, color, i, j)
        tmp_move, score = alphabeta_min_node(successor_board, color, alpha, beta, limit-1, caching)
        if best_score < score:
            best_score = score
            best_move = move
        if alpha < best_score:
            alpha = best_score
            if beta <= alpha:
                break
    res = best_move, best_score
    if caching == 1:
        cache[state] = res
    return res


def select_move_alphabeta(board, color, limit, caching = 0, ordering = 0):
    """
    Given a board and a player color, decide on a move. 
    The return value is a tuple of integers (i,j), where
    i is the column and j is the row on the board.  

    Note that other parameters are accepted by this function:
    If limit is a positive integer, your code should enfoce a depth limit that is equal to the value of the parameter.
    Search only to nodes at a depth-limit equal to the limit.  If nodes at this level are non-terminal return a heuristic 
    value (see compute_utility)
    If caching is ON (i.e. 1), use state caching to reduce the number of state evaluations.
    If caching is OFF (i.e. 0), do NOT use state caching to reduce the number of state evaluations.    
    If ordering is ON (i.e. 1), use node ordering to expedite pruning and reduce the number of state evaluations. 
    If ordering is OFF (i.e. 0), do NOT use node ordering to expedite pruning and reduce the number of state evaluations. 
    """
    #IMPLEMENT
    alpha = -float('inf')
    beta = float('inf')
    initialize_weight_matrix(board)
    move, utility = alphabeta_max_node(board, color,alpha, beta, limit, caching, ordering )
    # add delay to show the process
    #time.
    # (0.5)
    #update_weight_matrix(move[0],move[1])
    return move

def name_ordering_heuristic(board, possible_moves, color, ai_color, ordering, reverse):
    if ordering:
        return sorted(possible_moves, key= lambda x: compute_heuristic(play_move(board, color, x[0],x[1]),ai_color), reverse=reverse)
    return possible_moves



####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Othello Heuristic AI") # First line is the name of this AI
    arguments = input().split(",")
    
    color = int(arguments[0]) #Player color: 1 for dark (goes first), 2 for light. 
    limit = int(arguments[1]) #Depth limit
    minimax = int(arguments[2]) #Minimax or alpha beta
    caching = int(arguments[3]) #Caching 
    ordering = int(arguments[4]) #Node-ordering (for alpha-beta only)
   
    if (minimax == 1): eprint("Running MINIMAX")
    else: eprint("Running ALPHA-BETA")

    if (caching == 1): eprint("State Caching is ON")
    else: eprint("State Caching is OFF")

    if (ordering == 1): eprint("Node Ordering is ON")
    else: eprint("Node Ordering is OFF")

    if (limit == -1): eprint("Depth Limit is OFF")
    else: eprint("Depth Limit is ", limit)

    if (minimax == 1 and ordering == 1): eprint("Node Ordering should have no impact on Minimax")

    while True: # This is the main loop
        # Read in the current game status, for example:
        # "SCORE 2 2" or "FINAL 33 31" if the game is over.
        # The first number is the score for player 1 (dark), the second for player 2 (light)
        next_input = input()
        status, dark_score_s, light_score_s = next_input.strip().split()
        dark_score = int(dark_score_s)
        light_score = int(light_score_s)

        if status == "FINAL": # Game is over.
            print
        else:
            board = eval(input()) # Read in the input and turn it into a Python
                                  # object. The format is a list of rows. The
                                  # squares in each row are represented by
                                  # 0 : empty square
                                  # 1 : dark disk (player 1)
                                  # 2 : light disk (player 2)

            # Select the move and send it to the manager
            if (minimax == 1): #run this if the minimax flag is given
                movei, movej = select_move_minimax(board, color, limit, caching)
            else: #else run alphabeta
                movei, movej = select_move_alphabeta(board, color, limit, caching, ordering)
            
            print("{} {}".format(movei, movej))

if __name__ == "__main__":
    run_ai()
