"""
An AI player for Othello. 
"""
#### Answers #####
# With depth limit enforced, we can implement a max board size of 5 with a max depth limit of 6 when running the Alpha Beta Prunning Algo 
#
import random
import sys
import time

# You can use the functions in othello_shared to write your AI
from othello_shared import find_lines, get_possible_moves, get_score, play_move
#hash table for cach values
cache={}
weights_matrix=[]
def eprint(*args, **kwargs): #you can use this for debugging, as it will print to sterr and not stdout
    print(*args, file=sys.stderr, **kwargs)
    
# Method to compute utility value of terminal state
def compute_utility(board, color):
    #IMPLEMENT
    score= get_score(board)
    #printing the utility based on the color of the player 
    if color==1: return score[0]-score[1]
    elif color==2: return score[1]-score[0]

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
def minimax_min_node(board, color, limit, caching = 0):
    #IMPLEMENT (and replace the line below)
    #Return: The lowest possible utility node 
    global cache
    if color==1: other_color=2
    else: other_color=1
    limit-=1
    if caching==1:
        if board in cache:
            return cache[board]
    #iniitlazing a very high value for the node
    best_node = float("Inf")

    #checking if we are at the end leaf
    if not get_possible_moves(board,color) or limit==0:
        return None,compute_utility(board,color)
    for move in get_possible_moves(board,color):
        best_node = min(best_node,minimax_max_node(play_move(board,color,move[0],move[1]),opponent,limit,caching))
        
    return best_node

def minimax_max_node(board, color, limit, caching = 0): #returns highest possible utility
    #IMPLEMENT (and replace the line belowS
    global cache 
    if color==1: other_color=2
    else: other_color=1
    limit-=1
    if caching==1:
        if board in cache:
            return cache[board]
    #iniitlazing a very low value for the node
    best_node = float("-Inf")
    #checking if we are at the end leaf
    if not get_possible_moves(board,color) or limit==0:
        return compute_utility(board,color)
    for move in get_possible_moves(board,color):
        best_node= max(best_node,minimax_min_node(play_move(board,color,move[0],move[1]),opponent,lim))
    return best_node

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
    """
    #IMPLEMENT (and replace the line below)
    suggestedMove=[]
    for move in get_possible_moves(board,color):
        next_move= play_move(board,color,move[0],move[1])
        ourUtility=minimax_max_node(next_move,color,limit,caching)
        if next_move not in cache:
            cache[next_move] = ourUtility
        suggestedMove.append([(move[0],move[1]),ourUtility])
    #now sorting the values 
    
    return sorted(suggestedMove, key=lambda x: x[1])[0]

############ ALPHA-BETA PRUNING #####################

def alphabeta_min_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    #IMPLEMENT (and replace the line below)
    if color == 1 : opponent = 2
    else: opponent = 1
    limit-=1
    if caching==1:
        if board in cache:
            return cache[board]
    if not get_possible_moves(board,color) or limit==0:
        return compute_heuristic(board,color)
        
    node_val = float("Inf")
    for move in get_possible_moves(board,color):
        new_move = play_move(board, color, move[0], move[1])
        node_val = min(node_val,alphabeta_max_node(new_move,opponent,alpha,beta,limit,caching,ordering))
        if node_val <= alpha : 
            return node_val
        beta = min(beta,node_val)
    return node_val


def alphabeta_max_node(board, color, alpha, beta, limit, caching = 0, ordering = 0):
    #IMPLEMENT (and replace the line below)
    if color == 1 : opponent = 2
    else: opponent = 1
    limit-=1
    if caching==1:
        if board in cache:
            return cache[board]
    if not get_possible_moves(board,color) or limit == 0:
        return compute_heuristic(board,color)
        
    node_val = float("-Inf")
    for move in get_possible_moves(board,color):
        new_move = play_move(board, color, move[0], move[1])
        node_val = max(node_val,alphabeta_min_node(new_move,opponent,alpha,beta,limit,caching,ordering))
        if node_val <= alpha : return node_val
        alpha = max(alpha,node_val)

    return node_val

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
    #IMPLEMENT (and replace the line below)
    moves = []
    node_ordering = []
    alpha = float("-Inf")
    beta = float("Inf")
    initialize_weight_matrix(board)

    for option in get_possible_moves(board,color):
        node_ordering.append([(option[0],option[1]),compute_heuristic(play_move(board,color,option[0],option[1]),color)])
    #odering in descing order of utility values     
    if ordering==1:
        node_ordering = sorted(node_ordering, key = lambda x : x[1], reverse = True)
    for plays in node_ordering:
        new_move = play_move(board, color, plays[0][0], plays[0][1])
        utility = alphabeta_max_node(new_move,color,alpha,beta, limit, caching, ordering)
        if caching ==1:
            if new_move not in cache:
                cache[new_move] = utility
        moves.append([(plays[0][0], plays[0][1]), utility])
    sorted_options = sorted(moves, key = lambda x : x[1])

    return sorted_options[0][0]
 

####################################################
def run_ai():
    """
    This function establishes communication with the game manager.
    It first introduces itself and receives its color.
    Then it repeatedly receives the current score and current board state
    until the game is over.
    """
    print("Othello My AI") # First line is the name of this AI
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
