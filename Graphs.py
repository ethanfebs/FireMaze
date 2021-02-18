import matplotlib.pyplot as plt # Plotting library to generate graphs
import numpy # Handy library to perform calculations on our data
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
    plt.figure(figsize = figure_size)
    plt.plot(p_values, prob_success, label = "Reachable", color = "indigo")
    plt.xlabel("p-values")
    plt.ylabel("Simulated Success Probability")
    plt.title("Probability of a Path as Obstacles Increase")
    plt.legend(loc = "best")
    plt.grid()
    plt.savefig(f"{figure_save_path}reachable.png")
    plt.show()

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
    plt.figure(figsize = figure_size)
    plt.plot(p_values, average_differences, label = "Difference", color = "deepskyblue")
    plt.xlabel("p-values")
    plt.ylabel("Average Difference in BFS and A*")
    plt.title("Difference in Nodes Explored for BFS and A*")
    plt.legend(loc = "best")
    plt.grid()
    plt.savefig(f"{figure_save_path}BFS_AStar.png")
    plt.show()

if __name__ == "__main__":
    generate_reachable_plot(1000, 25, p_step = 0.025)
    generate_BFS_AStar_plot(1000, 25, p_step = 0.025)