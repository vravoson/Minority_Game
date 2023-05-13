import pandas as pd
import numpy as np
import random as rd
import tqdm
from utils import bin2dec, strategy

class Naive_Minority_Game():

    """
    Naive minority game with non-strategic players and stationary strategies (no update)
    Rule: Reward = +1 if minority choice, 1 if majority choice
    """

    def __init__(self, memory_size, n_players, n_simulations = 1000, verbose = True):
        self.memory_size = memory_size
        self.n_players = n_players
        self.n_simulations = n_simulations
        self.verbose = verbose

    def simulate_game(self):
        players_strats = [strategy(self.memory_size) for _ in range(self.n_players)]
        players_profit = []
        game_history, _ = self._random_start(players_strats)
        iterator_sim = tqdm.trange(self.n_simulations) if self.verbose else range(self.n_simulations)
        for _ in iterator_sim:
            past_Mbits = game_history[-self.memory_size:]
            game_output, player_profit = self._game_step_sim(past_Mbits, players_strats)
            game_history.append(game_output)
            players_profit.append(player_profit)

        return game_history, players_profit
    
    def _game_step_output(self, strat_outputs):
        sum_yes = np.sum(strat_outputs)
        return int(sum_yes > self.n_players/2)
    
    def _random_start(self, players_strats):
        first_outputs, player_profit = [], []
        for _ in range(self.memory_size):
            strats_output = [rd.sample(list(player_strat),1) for player_strat in players_strats] 
            game_output = self._game_step_output(strats_output)
            profit = [-1 if x == game_output else 1 for x in strats_output]
            player_profit.append(profit)
            first_outputs.append(game_output)
        return first_outputs, player_profit
    
    def  _game_step_sim(self, past_Mbits, players_strats):
        strat_id = bin2dec(past_Mbits)
        strats_output = [player_strat[strat_id] for player_strat in players_strats]
        game_output = self._game_step_output(strats_output)
        player_profit = [-1 if x == game_output else 1 for x in strats_output]
        return game_output, player_profit
