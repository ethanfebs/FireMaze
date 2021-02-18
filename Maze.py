import random
import math # Math functions
import copy # Allows us to deep copy a 2D array
from collections import deque # Importing a simple queue since pop(0) on a list is O(n) time where n is the size of the list
import heapq # Importing functions to treat lists as heaps/prioirty queues

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

def euclidean_distance(square_one: tuple, square_two: tuple):
    """
    Find the euclidean distance between two squares
    """
    return math.sqrt((square_one[0] - square_two[0]) ** 2 + (square_one[1] - square_two[1]) ** 2)

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

    returns - an ordered triple, with the first element either True or False, 
              representing whether or not it is possible to form a path. 
              The second element is a list of ordered pairs representing 
              (one of) the shortest path(s).
              The third element is the number of nodes visited.
    """
    n = len(maze) # Get the dimension of the maze

    #========================================#
    # Some data checking statements

    if (not is_valid(start, n)):
        print("BFS: Start indices outside maze dimensions")
        return False
    elif (not is_valid(goal, n)):
        print("BFS: Goal indices outside maze dimensions")
        return False

    # End data checking statements
    #========================================#

    number_of_nodes_visited = 0
    visited = copy.deepcopy(maze) # We can use a copy of the maze to keep track of visited squares (Considered using a set here, thought that time efficiency was important)
    # visited = list(map(list, maze)) # Alternative to using copy.deepcopy

    previous = [[None for i in range(n)] for j in range(n)] # Initialize a matrix of the same size as maze where each value is None.

    queue = deque() # Define our queue of "fringe" squares
    queue.append(start) # Push the start square into our queue
    visited[start[0]][start[1]] = 1 # Set our start to visited

    while (len(queue)): # While there exists items in the queue
        current = queue.popleft() # Pop the square at index 0
        number_of_nodes_visited += 1 # Increase number of nodes visited

        if (current == goal): # If current is the goal, we found it!
            # We now want to traverse back to make a path using our 'previous' matrix
            path = []
            while (current != None):
                path.append(current)
                current = previous[current[0]][current[1]]
            path.reverse()
            return (True, path, number_of_nodes_visited)

        current_i, current_j = current  # Unpack the current pair
        
        # Now we want to add all unvisited squares that are possible to get to from the current square
        for i in range(len(nearby_offsets)):
            offset_i, offset_j = nearby_offsets[i]
            possible = (current_i + offset_i, current_j + offset_j)
            # print(f"Current possible: {possible_i} {possible_j}") # DEBUG
            if (is_valid(possible, n)): # If the calculated square is within the maze matrix
                if (not visited[possible[0]][possible[1]]): # If possible has not been visited yet
                    queue.append(possible) # Add possible to our queue
                    visited[possible[0]][possible[1]] = 1 # Set possible to visited
                    previous[possible[0]][possible[1]] = current # Set the previous square for possible to the current square
    return (False, [], number_of_nodes_visited) # If the while loop goes out, and the queue is empty, then there is no possible path

def AStar(maze: list, start: tuple, goal: tuple):
    """ 
    Determines the shortest path (if it exists) between
    a start square and an end square using A* algorithm.

    maze - a square 2D array
    start - an ordered pair of the indices representing the start square
    goal - an ordered pair of the indices representing the goal square

    returns - an ordered triple, with the first element either True or False, 
              representing whether or not it is possible to form a path. 
              The second element is a list of ordered pairs representing 
              (one of) the shortest path(s).
              The third element is the number of nodes visited.
    """
    n = len(maze) # Get the dimension of the maze

    #========================================#
    # Some data checking statements

    if (not is_valid(start, n)):
        print("AStar: Start indices outside maze dimensions")
        return False
    elif (not is_valid(goal, n)):
        print("AStar: Goal indices outside maze dimensions")
        return False

    # End data checking statements
    #========================================#

    number_of_nodes_visited = 0
    # visited = copy.deepcopy(maze) # We can use a copy of the maze to keep track of visited squares (Considered using a set here, thought that time efficiency was important)
    # visited = list(map(list, maze)) # Alternative to using copy.deepcopy

    g_cost = [[float('inf') for i in range(n)] for j in range(n)] # Initialize a matrix of the same size as maze where each value is 'infinity'.
    # f_cost = [[float('inf') for i in range(n)] for j in range(n)] # Initialize a matrix of the same size as maze where each value is 'infinity'.
    previous = [[None for i in range(n)] for j in range(n)] # Initialize a matrix of the same size as maze where each value is None.

    heap = [] # Define our 'heap' which is just a list, but all pushes and pops will be through the heapq library.
    
    heapq.heappush(heap, (0, start)) # Push our start onto the heap. It's ok for this to have 0 'f' value since it'll be immediately popped off anyway.
    g_cost[start[0]][start[1]] = 0
    # f_cost[start[0]][start[1]] = euclidean_distance(start, goal)

    while (len(heap)): # While there exists items in the queue
        min_value = heapq.heappop(heap) # Pop the square with lowest 'f' value from our heap.
        number_of_nodes_visited += 1 # Increase number of nodes visited

        # if (visited[current[0]][current[1]] == False): # If we have not visited this node
        #     visited[start[0]][start[1]] = 1 # Set it to visited

        current_f, current = min_value

        if (current == goal): # If current is the goal, we found it!
            # We now want to traverse back to make a path using our 'previous' matrix
            path = []
            while (current != None):
                path.append(current)
                current = previous[current[0]][current[1]]
            path.reverse()
            return (True, path, number_of_nodes_visited)

        current_i, current_j = current  # Unpack the current pair
        
        # Now we want to add all unvisited squares that are possible to get to from the current square
        for i in range(len(nearby_offsets)):
            offset_i, offset_j = nearby_offsets[i]
            possible = (current_i + offset_i, current_j + offset_j)
            # print(f"Current possible: {possible_i} {possible_j}") # DEBUG
            if (is_valid(possible, n)): # If the calculated square is within the maze matrix
                if (maze[possible[0]][possible[1]]): # If there is something there
                    continue
                # Check to see if this path is better (just need to check g_cost since h_cost is always the same)
                possible_g_cost = g_cost[current[0]][current[1]] + 1
                if (possible_g_cost <  g_cost[possible[0]][possible[1]]): # If the cost is indeed less
                    previous[possible[0]][possible[1]] = current
                    g_cost[possible[0]][possible[1]] = possible_g_cost
                    # Check to see if the node is in the heap, and if it is not, put it in.
                    found = False
                    for (f_cost, (square_i, square_j)) in heap:
                        if (square_i == possible[0] and square_j == possible[1]):
                            found = True
                            break
                    if (not found):
                        heapq.heappush(heap, (possible_g_cost + euclidean_distance(possible, goal), possible))

                # if (visited[possible[0]][possible[1]]): # If this node has already been visited
                #     # Check to see if this path is better (just need to check g_cost since h_cost is always the same)
                #     if (f_cost[possible[0]][possible[1]] > possible_f_cost):
                #         heapq.heappush(heap, (possible_f_cost, possible)) # Push this back onto the heap for re-examination
                #         f_cost[possible[0]][possible[1]] = possible_f_cost # Assign the new f-cost
                #         previous[possible[0]][possible[1]] = current # Update previous
                # else
    return (False, [], number_of_nodes_visited) # If the while loop goes out, and the queue is empty, then there is no possible path

if __name__ == "__main__":
    n = 19
    maze = gen_fire_maze(n + 1, 0.3)
    print(maze)
    print_maze(maze)
    print(reachable(maze, (0, 0), (n, n)))
    BFS_blah, BFS_result, BFS_number_of_nodes_visited = BFS(maze, (0, 0), (n, n))
    AStar_blah, AStar_result, AStar_number_of_nodes_visited = AStar(maze, (0, 0), (n, n))

    print(f"BFS with length of: {len(BFS_result)} and # of nodes visited: {BFS_number_of_nodes_visited}\n{BFS_result}")
    print(f"AStar with length of: {len(AStar_result)} and # of nodes visited: {AStar_number_of_nodes_visited}\n{AStar_result}")
    # for i in range(10):
    #     maze = advance_fire_one_step(maze, 0.5)
    #     print("ITERATION ", i)
    #     print_maze(maze)
