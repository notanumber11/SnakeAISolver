import matplotlib.pyplot as plt
import numpy as np

from game.game_provider import GameProvider
from solvers.hamilton_solver import HamiltonSolver


def plot_games(games, title):
    x = [i for i in range(len(games))]
    y = [len(game.game_statuses[-1].snake) for game in games]
    avg_val = sum(y) / (len(y))
    plt.title(title)
    plt.plot(x, y, label="score")
    plt.plot(x, [avg_val] * len(x), label="avg", linestyle="--")
    plt.ylabel("snake size")
    plt.ylim([0, board_size ** 2 + 1])
    plt.xlim(0, len(games) + 1)
    plt.legend(loc="upper left")
    result_lst = [len(game.game_statuses[-1].snake) for game in games]
    result_lst = sorted(result_lst, key=lambda x: x)

    results = "avg={:.0F}   std={:.0F}   p50={:.0F}   p75={:.0F}   p90={:.0F}".format(np.average(result_lst),
                                                                                      np.std(result_lst),
                                                                                      np.percentile(result_lst, 50),
                                                                                      np.percentile(result_lst, 75),
                                                                                      np.percentile(result_lst, 90))
    plt.xlabel("games \n\n {}".format(results))
    filename = "{}   {}".format(title, results)

    plt.savefig("plot_results/" + filename + ".png", bbox_inches="tight")
    plt.show()
    plt.close()


game_provider = GameProvider()

solvers = [
    # game_provider.random_solver,
    # game_provider.survival_random_solver,
    # game_provider.dfs_solver,
    # game_provider.reward_based_dnn_solver,
    # game_provider.hamilton_solver,
    # game_provider.short_vision_genetic,
    # game_provider.distance_vision_genetic,
    game_provider.distance_vision_genetic_with_fallback
]

for solver in solvers:
    board_size = 9
    number_of_games = 20
    snake_size = 2
    if type(solver) == HamiltonSolver:
        number_of_games = 1
    games = game_provider.get_random_games(solver, number_of_games, board_size=board_size,
                                           snake_size=snake_size)
    if type(solver) == HamiltonSolver:
        games = [games[0] for i in range(100)]

    # for game in games:
    #     if len(game.game_statuses[-1].snake) == snake_size:
    #         result = solver.solve(game.game_statuses[0])[-1]
    #         if len(result.snake) != snake_size:
    #             print("***************************Error")
    #         else:
    #             print("Same result")

    print("Games resolved correctly")
    title = "Solver={}   Board size={}   Initial Snake size={}".format(solver.__str__(), board_size ** 2, snake_size)
    plot_games(games, title)
