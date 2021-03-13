"""
An AI player for Othello. 
"""

import random
import sys
import time
import copy
#my agent can play properly with depth level 5 on an 8 by 8 (uptil 12 by 12) boards agianst other agents with odering wihtout timming out  
#Look at the comments in my_heuristic and its subsequent functions for a description of my heurisitc

from othello_shared import find_lines, get_possible_moves, get_score, play_move

# initializing caching and improved_heuristics value 
cache={}
weights=[]

def eprint(*args, **kwargs): #you can use this for debugging, as it will print to sterr and not stdout
    print(*args, file=sys.stderr, **kwargs)
    
def compute_utility(board, color):
    score= get_score(board)
    #printing the utility based on the color of the player 
    if color==1: return score[0]-score[1]
    elif color==2: return score[1]-score[0]

# Better heuristic function for our board 
def my_heuristic(board, color): 
# we are adding the effects of three different things to compute the final utility value for our board 
    # 1. count how many moves each player has, more the number of subsequent possible moves better the move 
    movabilitiy = compute_movability(board,color)

    # 2. normal utility
    utility = compute_utility(board, color)

    # 3. compute weight score
    weight_score = compute_weight_score(board,color)

    final_value = movabilitiy + utility+ weight_score

    return final_value


def compute_weight_score(board,color):
#Statigically weighing different positions on the board differently based on different advantages that certian positions might provide 
    global weights
    dark = 0
    light = 0
    for i in range(len(weights)):
        for j in range(len(weights)):
            if board[i][j] == 1:
                dark += weights[i][j]
            elif board[i][j] == 2:
                light += weights[i][j]
    if color == 1:
        return dark-light
    elif color ==2:
        return light-dark

    

def compute_movability(board, color):
    #this function computes movability as a subtraction in the number of possible moves of the players 
    dark, light = len(get_possible_moves(board, 1)), len(get_possible_moves(board,2))

    if color == 1:
        return dark-light
    else:
        return light-dark




# for more complicated heuristics
def initializing_weights(board):
    n = len(board)
    global weights
    weights = [[1 for _ in range(n)] for _ in range(n)]
    # Setting the weight of the edge pieces including the corners to 10
    for i in range(n):
        weights[0][i] = 10
        weights[i][0] = 10
        weights[i][-1] = 10
        weights[-1][i] = 10
    # Correcting and updating the corner weights to higher values 
    weights[0][0], weights[-1][0], weights[0][-1], weights[-1][-1] = 1000, 1000, 1000, 1000
    # updating the bad weights for bad positions (second last to each corner values so in total 8 positions). These positions would be traps where the oponent can easily attain the corner positions
    weights[1][0], weights[0][1], weights[0][-2], weights[1][-1], weights[-2][0], weights[-1][1], weights[-1][-2], weights[-2][-1] = -10, -10, -10, -10, -10, -10, -10, -10



############ MINIMAX ###############################
def minimax_min_node(board, color, limit, caching = 0): #returns lowest possible utility
    #IMPLEMENT
    # find terminal state
    # terminal state
    # stand on the position of AI player
    #eprint(("caching:", cache))
    global cache
    if color==1: other_color=2
    else: other_color=1
    state = board
    if caching==1:
        if state in cache:
            return cache[state]
    # compute possible moves of opponents

    if limit==0:
        return None,compute_utility(board,color)
    if not get_possible_moves(board,other_color):
        res= None,compute_utility(board,color)
        if caching==1:
            cache[state]= res
        return res

    best_score = float('Inf')

    best_move = None
    limit-=1
    for move in get_possible_moves(board,other_color):
 
        # minimum of other player's gain
        # Player is Min, now player switch to min, use minimax_min_node with color MAX
        temp_mov, score = minimax_max_node(play_move(board, other_color, move[0],move[1]), color, limit, caching)
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
    if color==1: other_color=2
    else: other_color=1
    state = board
    if caching==1:
        if state in cache:
            return cache[state]
    # compute possible moves of opponent
    if limit==0:
        return None,compute_utility(board,color)
    if not get_possible_moves(board,color):
        res= None,compute_utility(board,color)
        if caching==1:
            cache[state]= res
        return res

    best_score = float('-Inf')
    best_move = None
    limit-=1
    for move in get_possible_moves(board,color):
 
        # minimum of other player's gain
        # Player is Min, now player switch to min, use minimax_min_node with color MAX
        temp_mov, score = minimax_min_node(play_move(board, color, move[0],move[1]), color, limit, caching)
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

    move, utility = minimax_max_node(board, color, limit, caching)
  
    return move


############ ALPHA-BETA PRUNING #####################
def alphabeta_min_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    #IMPLEMENT
    # current player is MIN
    global cache
    if color==1: other_color=2
    else: other_color=1
    state = board
    if caching==1:
        if state in cache:
            return cache[state]
    if limit == 0:
        return None, compute_utility(board,color)
    # name ordering heuristics
    if not ordering_heuristic(board,get_possible_moves(board,other_color),other_color,color,ordering,False):
        # terminal status, return utility value
        res =  None, compute_utility(board, color)
        if caching == 1:
            cache[state] = res
        return res
    best_score = float('Inf')
    best_move = None
    limit-=1
    for move in ordering_heuristic(board,get_possible_moves(board,other_color),other_color,color,ordering,False):

        tmp_move, score = alphabeta_max_node(play_move(board, other_color, move[0],move[1]), color, alpha, beta, limit, caching)
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
    if color==1: other_color=2
    else: other_color=1
    state = board
    if caching==1:
        if state in cache:
            return cache[state]
    if limit == 0:
        return None, compute_utility(board,color)
    # name ordering heuristics
    if not ordering_heuristic(board,get_possible_moves(board,color),other_color,color,ordering,True):
        # terminal status, return utility value
        res =  None, compute_utility(board, color)
        if caching == 1:
            cache[state] = res
        return res
    best_score = float('-Inf')
    best_move = None
    limit-=1
    for move in ordering_heuristic(board,get_possible_moves(board,color),color,color,ordering,True):
        tmp_move, score = alphabeta_min_node(play_move(board,color, move[0],move[1]), color, alpha, beta, limit, caching)
        if best_score < score:
            best_score = score
            best_move = move
        if beta < best_score:
            beta = best_score
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
    alpha = float('-Inf')
    beta = float('Inf')
    #initializing weights for my_heuristics (uncomment the following line when using my_heuristics)
    #initializing_weights(board)
    move, utility = alphabeta_max_node(board, color,alpha, beta, limit, caching, ordering )
 
    return move

def ordering_heuristic(board, possible_moves, color, ai_color, ordering, reverse):
    if ordering==1:
        return sorted(possible_moves, key= lambda x: compute_utility(play_move(board, color, x[0],x[1]),ai_color), reverse=reverse)
    return possible_moves



####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Othello AI") # First line is the name of this AI
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
