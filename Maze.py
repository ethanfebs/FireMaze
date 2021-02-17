import random
import copy # Allows us to deep copy a 2D array
from collections import deque # Importing a simple queue since pop(0) on a list is O(n) time where n is the size of the list

# These offsets allow us to define all squares that potentially can be reached from a 'current' square
nearby_offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)]

#========================================#
# Some helper functions


def is_valid(square: tuple, n: int):
    """
    Determines whether or not a given square is within the maze square matrix
    """
    square_i, square_j = square
    if (square_i < n and square_i >= 0 and square_j < n and square_j >= 0):
        return True
    return False


def print_maze(maze: list):
    """
    Prints our maze in 2D format for readability

    maze - a square 2D array
    """
    n = len(maze)
    for i in range(n):
        print(maze[i])

#========================================#


def gen_maze(dim, p):
    """Generate a maze with a given dimension and obstacle density p"""
    maze = []
    for i in range(dim):
        maze.append([])
        for j in range(dim):
            if(random.uniform(0, 1) < p):
                maze[i].append(1)
            else:
                maze[i].append(0)

    maze[0][0] = 0
    maze[dim - 1][dim - 1] = 0
    return maze


def gen_fire_maze(dim, p):
    """Generate a maze with one empty cell on fire"""
    maze_f = gen_maze(dim, p)
    num_empty = 0

    # count the number of empty cells in the maze
    for i in maze_f:
        num_empty += (dim-sum(i))

    # chose an empty cell to set on fire
    fire_spawn = random.randrange(num_empty)

    # iterate over chosen number of empty cells before setting one on fire
    for i in range(dim):
        for j in range(dim):
            if(maze_f[i][j] == 0 and fire_spawn == 0):
                maze_f[i][j] = 2  # set cell to be on fire
                return maze_f
            elif(maze_f[i][j] == 0):
                fire_spawn -= 1  # decrement counter

    # function should always return before loop is completed
    return -1


def advance_fire_one_step(maze, q):
    """ 
    Spread fire through maze using the following criteria
        1. cells on fire stay on fire
        2. empty cells with no adjacent cells on fire stay empty
        3. blocked cells cannot be set on fire
        4. empty cells with k adjacent cells on fire are set on fire with probability 1-(1-q)^k
    """

    n = len(maze)  # get dimension of maze
    next_maze = maze.copy()  # create a copy of previous maze

    # iterate over all cells in the maze
    for i in range(n):
        for j in range(n):
            # check if cell is empty
            if(next_maze[i][j] == 0):
                k = 0
                # count number of adjacent cells that are on fire
                for x in range(len(nearby_offsets)):
                    offset_i, offset_j = nearby_offsets[x]
                    possible = (i + offset_i, j + offset_j)

                    if(is_valid(possible, n) and next_maze[possible[0]][possible[1]] == 2):
                        k += 1
                # use random variable to determine whether cells should be set on fire
                if(random.uniform(0, 1) < 1-((1-q) ** k)):
                    # represent cell to be set on fire differently than fire cell to avoid affecting future calculations
                    next_maze[i][j] = 3

    # iterate over all cells in the maze
    for i in range(n):
        for j in range(n):
            # if cell should be set on fire, set it on fire
            if(next_maze[i][j] == 3):
                next_maze[i][j] = 2

    # return maze with fire advanced on step
    return next_maze


def reachable(maze: list, start: tuple, goal: tuple):
    """ 
    Determines whether or not there exists a path 
    between the start square and the goal square.

    maze - a square 2D array
    start - an ordered pair of the indices representing the start square
    goal - an ordered pair of the indices representing the goal square
    """
    n = len(maze)  # Get the dimension of the maze

    #========================================#
    # Some data checking statements

    if (not is_valid(start, n)):
        print("reachable: Start indices outside maze dimensions")
        return False
    elif (not is_valid(goal, n)):
        print("reachable: Goal indices outside maze dimensions")
        return False

    # End data checking statements
    #========================================#

    visited = copy.deepcopy(maze) # We can use a copy of the maze to keep track of visited squares (Considered using a set here, thought that time efficiency was important)
    # visited = list(map(list, maze)) # Alternative to using copy.deepcopy
    stack = [] # Define our stack of "fringe" squares
    stack.append(start) # Push the start square onto our stack
    visited[start[0]][start[1]] = 1 # Set our start to visited

    while (len(stack)):  # While there exists items in the stack
        current = stack.pop()  # Pop the last element

        if (current == goal):
            return True  # If current is the goal, we found it!

        current_i, current_j = current  # Unpack the current pair
        # If this square has not been visited yet
        if (visited[current_i][current_j] == False):
            visited[current_i][current_j] = True  # Set this square to visited


        current_i, current_j = current # Unpack the current pair
        
        # Now we want to add all unvisited squares that are possible to get to from the current square
        for i in range(len(nearby_offsets)):
            offset_i, offset_j = nearby_offsets[i]
            possible = (current_i + offset_i, current_j + offset_j)
            # print(f"Current possible: {possible_i} {possible_j}") # DEBUG
            if (is_valid(possible, n)):  # If the calculated square is within the maze matrix
                if (not visited[possible[0]][possible[1]]):
                    stack.append(possible)
                    visited[possible[0]][possible[1]] = 1
    return False # If the while loop goes out, and the stack is empty, then there is no possible path
            
def BFS(maze: list, start: tuple, goal: tuple):
    """ 
    Determines the shortest path (if it exists) between
    a start square and an end square using BFS (dijkstra's).

    maze - a square 2D array
    start - an ordered pair of the indices representing the start square
    goal - an ordered pair of the indices representing the goal square
    """
    n = len(maze) # Get the dimension of the maze

    #========================================#
    # Some data checking statements

    if (not is_valid(start, n)):
        print("reachable: Start indices outside maze dimensions")
        return False
    elif (not is_valid(goal, n)):
        print("reachable: Goal indices outside maze dimensions")
        return False

    # End data checking statements
    #========================================#

    visited = copy.deepcopy(maze) # We can use a copy of the maze to keep track of visited squares (Considered using a set here, thought that time efficiency was important)
    # visited = list(map(list, maze)) # Alternative to using copy.deepcopy

    distance = [[float('inf') for i in range(n)] for j in range(n)] # Initialize a matrix of the same size as maze where each value is infinity.
    previous = [[None for i in range(n)] for j in range(n)] # Initialize a matrix of the same size as maze where each value is None.

    queue = deque() # Define our queue of "fringe" squares
    queue.append(start) # Push the start square into our queue
    visited[start[0]][start[1]] = 1 # Set our start to visited
    distance[start[0]][start[1]] = 0 # Set our start distance to 0

    while (len(queue)): # While there exists items in the queue
        current = queue.popleft() # Pop the square at index 0

        if (current == goal):
            return True  # If current is the goal, we found it!

        current_i, current_j = current  # Unpack the current pair
        # If this square has not been visited yet
        if (visited[current_i][current_j] == False):
            visited[current_i][current_j] = True  # Set this square to visited

        current_i, current_j = current # Unpack the current pair
        
        # Now we want to add all unvisited squares that are possible to get to from the current square
        for i in range(len(nearby_offsets)):
            offset_i, offset_j = nearby_offsets[i]
            possible = (current_i + offset_i, current_j + offset_j)
            # print(f"Current possible: {possible_i} {possible_j}") # DEBUG
            if (is_valid(possible, n)): # If the calculated square is within the maze matrix
                if (not visited[possible[0]][possible[1]]):
                    queue.append(possible)
                    visited[possible[0]][possible[1]] = 1
    return False # If the while loop goes out, and the queue is empty, then there is no possible path

maze = gen_fire_maze(6, 0.3)
print_maze(maze)
#print(reachable(maze, (0, 0), (5, 5)))

for i in range(10):
    maze = advance_fire_one_step(maze, 0.5)
    print("ITERATION ", i)
    print_maze(maze)
