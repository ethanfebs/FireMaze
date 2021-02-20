import matplotlib.pyplot as plt # Plotting library to generate graphs
import numpy # Handy library to perform calculations on our data
from numpy.polynomial import Polynomial
import Maze # Import our Maze functions

# Constants
figure_save_path = "graphs/"
figure_size = (16 * 2/3, 6)

def generate_reachable_plot(iterations: int, maze_size: int, start: tuple = (), goal: tuple = (), p_start: float = 0, p_stop: float = 1.0, p_step: float = .05):
    """
    Generates a plot showing the probability that for a given p, there exists a path through the room.
    Each probability is calculated by using the basic definition of probability:
                      number of success
                    ---------------------
                      number of samples
    Plots and saves this graph as reachable.png

    iterations - the amomunt of iterations to perform at each p. The essential 'total sample size' for a given p.
    maze_size - the size of the mazes generated to test.
    start - the start square represented as an ordered pair. Default is (0, 0).
    goal - the goal square represented as an ordered pair. Default is (n - 1, n - 1) where n = maze_size.
    p_start - the smallest p value to test (inclusive). Default is 0.
    p_stop - the largest p value to test (inclusive). Default is 1.
    p_step - the 'step' between each p-value. Default is .05.
    """
    if (not start): # If start is not provided
        start = (0, 0)
    if (not goal): # If goal is not provided
        goal = (maze_size - 1, maze_size - 1)

    p_values = numpy.arange(p_start, p_stop + p_step, p_step) # Generate p values
    prob_success = [] # Create a list to store the success outcomes
    for p in p_values: # Loop through p values
        number_successes = 0 # Keep track of how many succeed
        for i in range(iterations): # Loop through iterations
            success = Maze.reachable(Maze.gen_maze(maze_size, p), start, goal) # Either True or False here
            if (success): # Increase total successes if it was successful
                number_successes += 1
        prob_success.append(number_successes / iterations) # Add this value to our list
    
    # Now that we have data, we plot and save the figure
    # Scatter plot
    plt.figure(figsize = figure_size)
    plt.scatter(p_values, prob_success, label = "Reachable", color = "indigo")
    plt.xlabel("p-values")
    plt.ylabel("Simulated Success Probability")
    plt.locator_params(nbins = 20)
    plt.title("Probability of a Path as Obstacles Increase")
    plt.legend(loc = "best")
    plt.grid()
    plt.savefig(f"{figure_save_path}reachable_scatter.png")
    plt.show()

    # Line plot (These look weird when the amount of x's is high)
    # plt.figure(figsize = figure_size)
    # plt.plot(p_values, prob_success, label = "Reachable", color = "indigo")
    # plt.xlabel("p-values")
    # plt.ylabel("Simulated Success Probability")
    # plt.title("Probability of a Path as Obstacles Increase")
    # plt.legend(loc = "best")
    # plt.grid()
    # plt.savefig(f"{figure_save_path}reachable.png")
    # plt.show()

def generate_BFS_AStar_plot(iterations: int, maze_size: int, start: tuple = (), goal: tuple = (), p_start: float = 0, p_stop: float = 1.0, p_step: float = .05):
    """
    Generates a plot showing the average difference of nodes explored for BFS and A* for a given p.
    Average is simply calculated as such:
                    diff1 + diff2 + ... + diff3
                  --------------------------------
                            iterations
    Since AStar should always compare less or equal amount of nodes than BFS, we will always subtract AStar from BFS.
    Plots and saves this graph as BFS_AStar.png


    iterations - the amomunt of iterations to perform at each p. The essential 'total sample size' for a given p.
    maze_size - the size of the mazes generated to test.
    start - the start square represented as an ordered pair. Default is (0, 0).
    goal - the goal square represented as an ordered pair. Default is (n - 1, n - 1) where n = maze_size.
    p_start - the smallest p value to test (inclusive). Default is 0.
    p_stop - the largest p value to test (inclusive). Default is 1.
    p_step - the 'step' between each p-value. Default is .05.
    """
    if (not start): # If start is not provided
        start = (0, 0)
    if (not goal): # If goal is not provided
        goal = (maze_size - 1, maze_size - 1)
    
    p_values = numpy.arange(p_start, p_stop + p_step, p_step) # Generate p values
    average_differences = [] # Create a list to store the average differences
    for p in p_values: # Loop through p values
        total_difference = 0 # Kepp track of total difference
        for i in range(iterations): # Loop through iterations
            maze = Maze.gen_maze(maze_size, p) # Generate a maze since we need to use it for both BFS and AStar
            BFS_valid, BFS_result, BFS_number_of_nodes_visited = Maze.BFS(maze, start, goal) # Perform BFS
            AStar_valid, AStar_result, AStar_number_of_nodes_visited = Maze.AStar(maze, start, goal) # Perform AStar
            total_difference += BFS_number_of_nodes_visited - AStar_number_of_nodes_visited # Add the difference, which should always be >= 0, so no need to use absolute value
        average_differences.append(total_difference / iterations) # Add the average to our list
    
    # Now that we have data, we plot and save the figure
    # Scatter plot
    plt.figure(figsize = figure_size)
    plt.scatter(p_values, average_differences, label = "Difference", color = "deepskyblue")
    plt.xlabel("p-values")
    plt.ylabel("Average Difference in BFS and A*")
    plt.locator_params(nbins = 20)
    plt.title("Difference in Nodes Explored for BFS and A*")
    plt.legend(loc = "best")
    plt.grid()
    plt.savefig(f"{figure_save_path}BFS_AStar_scatter.png")
    plt.show()

    # Line plot (These look weird when the amount of x's is high)
    # plt.figure(figsize = figure_size)
    # plt.plot(p_values, average_differences, label = "Difference", color = "deepskyblue")
    # plt.xlabel("p-values")
    # plt.ylabel("Average Difference in BFS and A*")
    # plt.title("Difference in Nodes Explored for BFS and A*")
    # plt.legend(loc = "best")
    # plt.grid()
    # plt.savefig(f"{figure_save_path}BFS_AStar.png")
    # plt.show()

def generate_strategy_1_plot(iterations: int, maze_size: int, start: tuple = (), goal: tuple = (), p: float = 0.3, q_start: float = 0, q_stop: float = 1.0, q_step: float = .05):
    """
    Generates a plot showing the average success of this strategy for differeing q's.
    Average is simply calculated as such:
                      total number of successes
                  --------------------------------
                       iterations x iterations

    iterations - for each 'iteration', we generate a valid maze. Then for each valid maze, we generate 'iteration' amount of fire mazes.
    maze_size - the size of the mazes generated to test.
    start - the start square represented as an ordered pair. Default is (0, 0).
    goal - the goal square represented as an ordered pair. Default is (n - 1, n - 1) where n = maze_size.
    p - the obstacle density. Default is 0.3.
    q_start - the smallest p value to test (inclusive). Default is 0.
    q_stop - the largest p value to test (inclusive). Default is 1.
    q_step - the 'step' between each p-value. Default is .05.
    """
    if (not start): # If start is not provided
        start = (0, 0)
    if (not goal): # If goal is not provided
        goal = (maze_size - 1, maze_size - 1)
    
    q_values = numpy.arange(q_start, q_stop + q_step, q_step) # Generate q values
    prob_successes = [] # Create a list to store the average differences
    for q in q_values: # Loop through q values
        total_successes = 0 # Keep track of number of successes
        for i in range(iterations):
            maze = Maze.gen_maze(maze_size, p)
            while (not Maze.reachable(maze, start, goal)):
                maze = Maze.gen_maze(maze_size, p)
            for j in range(iterations):
                (f_maze, fire_start) = Maze.gen_fire_maze(maze)
                while (not Maze.reachable(maze, start, fire_start)):
                    (f_maze, fire_start) = Maze.gen_fire_maze(maze)
                # print(f"{i}, {j}")
                result = Maze.fire_strategy_1(f_maze, q)
                if (result):
                    total_successes += 1

        prob_successes.append(total_successes / (iterations * iterations))
    
    # Now that we have data, we plot and save the figure
    # Scatter plot
    plt.figure(figsize = figure_size)
    plt.scatter(q_values, prob_successes, label = "Average Successes", color = "deeppink")
    plt.xlabel("q-values")
    plt.ylabel("Average Successes")
    plt.locator_params(nbins = 20)
    plt.title("Average Successes of Strategy 1 as q Increases")
    plt.legend(loc = "best")
    plt.grid()
    plt.savefig(f"{figure_save_path}strategy_1_scatter.png")
    plt.show()

def generate_strategy_2_plot(iterations: int, maze_size: int, start: tuple = (), goal: tuple = (), p: float = 0.3, q_start: float = 0, q_stop: float = 1.0, q_step: float = .05):
    """
    Generates a plot showing the average success of this strategy for differeing q's.
    Average is simply calculated as such:
                      total number of successes
                  --------------------------------
                       iterations x iterations

    iterations - for each 'iteration', we generate a valid maze. Then for each valid maze, we generate 'iteration' amount of fire mazes.
    maze_size - the size of the mazes generated to test.
    start - the start square represented as an ordered pair. Default is (0, 0).
    goal - the goal square represented as an ordered pair. Default is (n - 1, n - 1) where n = maze_size.
    p - the obstacle density. Default is 0.3.
    q_start - the smallest p value to test (inclusive). Default is 0.
    q_stop - the largest p value to test (inclusive). Default is 1.
    q_step - the 'step' between each p-value. Default is .05.
    """
    if (not start): # If start is not provided
        start = (0, 0)
    if (not goal): # If goal is not provided
        goal = (maze_size - 1, maze_size - 1)
    
    q_values = numpy.arange(q_start, q_stop + q_step, q_step) # Generate q values
    prob_successes = [] # Create a list to store the average differences
    for q in q_values: # Loop through q values
        total_successes = 0 # Keep track of number of successes
        for i in range(iterations):
            maze = Maze.gen_maze(maze_size, p)
            while (not Maze.reachable(maze, start, goal)):
                maze = Maze.gen_maze(maze_size, p)
            for j in range(iterations):
                (f_maze, fire_start) = Maze.gen_fire_maze(maze)
                while (not Maze.reachable(maze, start, fire_start)):
                    (f_maze, fire_start) = Maze.gen_fire_maze(maze)
                # print(f"{i}, {j}")
                result = Maze.fire_strategy_2(f_maze, q)
                if (result):
                    total_successes += 1

        prob_successes.append(total_successes / (iterations * iterations))
    
    # Now that we have data, we plot and save the figure
    # Scatter plot
    plt.figure(figsize = figure_size)
    plt.scatter(q_values, prob_successes, label = "Average Successes", color = "darkorange")
    plt.xlabel("q-values")
    plt.ylabel("Average Successes")
    plt.locator_params(nbins = 20)
    plt.title("Average Successes of Strategy 2 as q Increases")
    plt.legend(loc = "best")
    plt.grid()
    plt.savefig(f"{figure_save_path}strategy_2_scatter.png")
    plt.show()

if __name__ == "__main__":
    """ These functions overwrite the existing graphs. Run them with caution. """
    # generate_reachable_plot(100, 100, p_step = 0.01)
    # generate_BFS_AStar_plot(100, 100, p_step = 0.01)
    # generate_strategy_2_plot(10, 100)