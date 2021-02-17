import random
import copy # Allows us to deep copy a 2D array
from collections import deque # Importing a simple queue since pop(0) on a list is O(n) time where n is the size of the list

nearby_offsets = [(-1, 0), (0, 1), (1, 0), (0, -1)] # These offsets allow us to define all squares that potentially can be reached from a 'current' square

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

def reachable(maze: list, start: tuple, goal: tuple):
    """ 
    Determines whether or not there exists a path 
    between the start square and the goal square.

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
    stack = [] # Define our stack of "fringe" squares
    stack.append(start) # Push the start square onto our stack
    visited[start[0]][start[1]] = 1 # Set our start to visited

    while (len(stack)): # While there exists items in the stack
        current = stack.pop() # Pop the last element

        if (current == goal):
            return True # If current is the goal, we found it!

        current_i, current_j = current # Unpack the current pair
        
        # Now we want to add all unvisited squares that are possible to get to from the current square
        for i in range(len(nearby_offsets)):
            offset_i, offset_j = nearby_offsets[i]
            possible = (current_i + offset_i, current_j + offset_j)
            # print(f"Current possible: {possible_i} {possible_j}") # DEBUG
            if (is_valid(possible, n)): # If the calculated square is within the maze matrix
                if (not visited[possible[0]][possible[1]]):
                    stack.append(possible)
                    visited[possible[0]][possible[1]] = 1
    return False # If the while loop goes out, and the stack is empty, then there is no possible path
            
def BFS(maze: list, start: tuple, goal: tuple):
    """ 
    Determines the shortest path (if it exists) between
    a start square and an end square using BFS.

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
    queue = deque() # Define our queue of "fringe" squares
    queue.append(start) # Push the start square into our queue
    visited[start[0]][start[1]] = 1 # Set our start to visited

    while (len(queue)): # While there exists items in the queue
        current = queue.popleft() # Pop the square at index 0

        if (current == goal):
            return True # If current is the goal, we found it!

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

maze = gen_maze(6, 0.3)
print_maze(maze)
print(reachable(maze, (0, 0), (5, 5)))
print_maze(maze)
print(BFS(maze, (0, 0), (5, 5)))
