from typing import List

import numpy as np

from game.game import Game
from game.game_status import GameStatus
from solvers.genetic.model_genetic_evaluated import ModelGeneticEvaluated


def evaluate_population(population_genetic: List, game_status_seeds: List[GameStatus], model, training_generator) \
        -> List[ModelGeneticEvaluated]:
    population_evaluated = []
    # For each game genetic we need to perform artificial games and see the performance
    for model_genetic in population_genetic:
        model_evaluated = evaluate_model(game_status_seeds, model, model_genetic, training_generator)
        population_evaluated.append(model_evaluated)
    return population_evaluated


def set_model_weights(model, model_genetic):
    weights = model.get_weights()
    for i in range(0, len(model_genetic)):
        weights[i] = model_genetic[i]
    model.set_weights(weights)


def evaluate_model(game_status_seeds: List[GameStatus], model, model_genetic,
                   training_generator) -> ModelGeneticEvaluated:
    set_model_weights(model, model_genetic)
    games = []
    for game_status in game_status_seeds:
        game = play_one_game(game_status, model, training_generator)
        games.append(game)

    advance_model = ModelGeneticEvaluated(games, model_genetic)
    for game in games:
        del game.game_statuses
    return advance_model


def play_one_game(current_game_status: GameStatus, model, training_generator):
    game_statuses = [current_game_status]
    movements_left = current_game_status.get_movements_left()
    while current_game_status.is_valid_game() and movements_left > 0:
        _input = [training_generator.get_input_from_game_status(current_game_status)]
        _dir = get_best_movement(_input, model)
        new_game_status = current_game_status.move(_dir)
        game_statuses.append(new_game_status)
        # Continue iteration
        movements_left -= 1
        if current_game_status.apple != new_game_status.apple:
            movements_left = current_game_status.get_movements_left()
        current_game_status = new_game_status

    # The game was in a loop
    is_loop = True if movements_left == 0 and current_game_status.is_full_finished() is False else False
    game = Game(game_statuses, is_loop)
    return game


def get_best_movement(_input, model):
    test_predictions = model.__call__(np.array(_input))
    max_index = np.argmax(test_predictions[0])
    result = GameStatus.DIRS[max_index]
    return result
