import random
import copy  # Allows us to deep copy a 2D array
# Importing a simple queue since pop(0) on a list is O(n) time where n is the size of the list
from collections import deque

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


def gen_fire_maze(maze):
    """Generate a maze with one empty cell on fire"""
    maze_f = copy.deepcopy(maze)
    dim = len(maze)
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
    next_maze = copy.deepcopy(maze)  # create a copy of previous maze

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

    # We can use a copy of the maze to keep track of visited squares (Considered using a set here, thought that time efficiency was important)
    visited = copy.deepcopy(maze)
    # visited = list(map(list, maze)) # Alternative to using copy.deepcopy
    stack = []  # Define our stack of "fringe" squares
    stack.append(start)  # Push the start square onto our stack
    visited[start[0]][start[1]] = 1  # Set our start to visited

    while (len(stack)):  # While there exists items in the stack
        current = stack.pop()  # Pop the last element

        if (current == goal):
            return True  # If current is the goal, we found it!

        current_i, current_j = current  # Unpack the current pair
        # If this square has not been visited yet
        if (visited[current_i][current_j] == False):
            visited[current_i][current_j] = True  # Set this square to visited

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
    return False  # If the while loop goes out, and the stack is empty, then there is no possible path


def BFS(maze: list, start: tuple, goal: tuple):
    """
    Determines the shortest path (if it exists) between
    a start square and an end square using BFS (dijkstra's).

    maze - a square 2D array
    start - an ordered pair of the indices representing the start square
    goal - an ordered pair of the indices representing the goal square

    returns - an ordered pair, with the first element either True or False,
              representing whether or not it is possible to form a path.
              The second element is a list of ordered pairs representing
              (one of) the shortest path(s).
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

    # We can use a copy of the maze to keep track of visited squares (Considered using a set here, thought that time efficiency was important)
    visited = copy.deepcopy(maze)
    # visited = list(map(list, maze)) # Alternative to using copy.deepcopy

    # Initialize a matrix of the same size as maze where each value is None.
    previous = [[None for i in range(n)] for j in range(n)]

    queue = deque()  # Define our queue of "fringe" squares
    queue.append(start)  # Push the start square into our queue
    visited[start[0]][start[1]] = 1  # Set our start to visited

    while (len(queue)):  # While there exists items in the queue
        current = queue.popleft()  # Pop the square at index 0

        if (current == goal):  # If current is the goal, we found it!
            # We now want to traverse back to make a path using our 'previous' matrix
            path = []
            while (current != None):
                path.append(current)
                current = previous[current[0]][current[1]]
            path.reverse()
            return (True, path)

        current_i, current_j = current  # Unpack the current pair
        # If this square has not been visited yet
        if (visited[current_i][current_j] == False):
            visited[current_i][current_j] = True  # Set this square to visited

        current_i, current_j = current  # Unpack the current pair

        # Now we want to add all unvisited squares that are possible to get to from the current square
        for i in range(len(nearby_offsets)):
            offset_i, offset_j = nearby_offsets[i]
            possible = (current_i + offset_i, current_j + offset_j)
            # print(f"Current possible: {possible_i} {possible_j}") # DEBUG
            if (is_valid(possible, n)):  # If the calculated square is within the maze matrix
                # If possible has not been visited yet
                if (not visited[possible[0]][possible[1]]):
                    queue.append(possible)  # Add possible to our queue
                    # Set possible to visited
                    visited[possible[0]][possible[1]] = 1
                    # Set the previous square for possible to the current square
                    previous[possible[0]][possible[1]] = current
    # If the while loop goes out, and the queue is empty, then there is no possible path
    return (False, [])


def fire_strategy_1(maze, q):
    """
    Calculate the shortest path through a fire maze before any fire has spread
    Progress agent through maze as fire spreads
    Return false if agent touches fire cell on path
    Returns true otherwise
    """

    n = len(maze)
    path = BFS(maze, (0, 0), (n-1, n-1))

    # End if no path exists from start to goal
    if(not path[0]):
        return False

    route = path[1]

    timestep = 0
    agent_pos = route[timestep]
    maze_f = copy.deepcopy(maze)

    while(timestep < len(route)):

        timestep += 1  # increase timer by 1
        agent_pos = route[timestep]  # update to new position

        # if agent moves into fire, report failure
        if(maze_f[agent_pos[0]][agent_pos[1]] != 0):
            # print_maze(maze_f)
            #print("timestep ", timestep)
            return False

        # if agent has reached goal, report success
        if(timestep == len(route)-1):
            # print_maze(maze_f)
            #print("timestep ", timestep)
            return True

        maze_f = advance_fire_one_step(maze_f, q)  # advance fire

        # if fire spread into agent, report failure
        if(maze_f[agent_pos[0]][agent_pos[1]] != 0):
            # print_maze(maze_f)
            #print("timestep ", timestep)
            return False

    # function should always return before while loop is completed
    return False


def fire_strategy_2(maze, q):
    """
    Recalculate the shortest path through the fire at each timestep
    Return false if agent touches fire cell on path
    Returns true otherwise
    """

    n = len(maze)

    timestep = 0
    maze_f = copy.deepcopy(maze)
    agent_pos = (0, 0)

    while(agent_pos != (n-1, n-1)):

        path = BFS(maze_f, agent_pos, (n-1, n-1))

        # End if no path exists from start to goal
        if(not path[0]):
            return False

        route = path[1]
        timestep += 1  # increase timer by 1
        agent_pos = route[1]  # update to new position

        # if agent moves into fire, report failure
        if(maze_f[agent_pos[0]][agent_pos[1]] != 0):
            return False

        # if agent has reached goal, report success
        if(agent_pos == (n-1, n-1)):
            return True

        maze_f = advance_fire_one_step(maze_f, q)  # advance fire

        # if fire spread into agent, report failure
        if(maze_f[agent_pos[0]][agent_pos[1]] != 0):
            return False

    # function should always return before while loop is completed
    return False


maze = gen_maze(100, 0.3)

maze_f = gen_fire_maze(maze)
print_maze(maze_f)
print('####')
print(fire_strategy_2(maze_f, 0.3))


# Questions to resolve later
#   should we allow fire at the start and/or goal?
#   should we implement option for player to "stay in same place"?
