from enum import Enum
from tqdm import tqdm
import numpy as np
import random

class Sign(Enum):
    ROCK = 0
    PAPER = 1
    SCISSORS = 2

strategies_list = [Sign.ROCK, Sign.PAPER, Sign.SCISSORS]


class RPS:

    def __init__(self):
        self.p1_games_won = 0
        self.p2_games_won = 0
        self.ties = 0
    
    def reset_count(self):
        self.p1_games_won = 0
        self.p2_games_won = 0
        self.ties = 0
    
    def get_scores(self):
        return (self.p1_games_won, self.p2_games_won, self.ties)

    def play(self, p1, p2):
        """Returns the number of the player that won (1 or 2) or 0 if they tied"""
        if p1 == p2: return 0
        if p1 == Sign.ROCK:
            if p2 == Sign.SCISSORS: return 1 
            else: return 2 
        elif p1 == Sign.PAPER:
            if p2 == Sign.ROCK: return 1 
            else: return 2 
        else:
            if p2 == Sign.PAPER: return 1 
            else: return 2  

    def play_cnt(self, p1, p2):
        f = self.play(p1, p2)
        match f:
            case 0:
                self.ties += 1 
                return 0 
            case 1:
                self.p1_games_won += 1 
                return 1 
            case 2: 
                self.p2_games_won += 1 
                return 2 

class Player:

    def __init__(self):
        self.regrets_sum = np.zeros(len(Sign))
        self.strategy = np.zeros(len(Sign))
        self.strategy_sum = np.zeros(len(Sign))

    def get_action(self, strategy):
        r = random.random()
        cum_prob = 0
        for i in range(len(Sign) - 1):
            cum_prob += strategy[i]
            if r < cum_prob: return strategies_list[i]
        
        return strategies_list[-1]
    
    def normalize(self, v):
        v = np.maximum(v, 0)
        total = np.sum(v)
        if total > 0:
            return v / total 
        else:
            return np.ones_like(v) / len(v)

    def get_strategy(self):
        normalizing_sum = 0
        for i in range(len(Sign)):
            self.strategy[i] = max(self.regrets_sum[i], 0)
            normalizing_sum += self.strategy[i]

        for i in range(len(Sign)):
            if normalizing_sum > 0:
                self.strategy[i] /= normalizing_sum
            else:
                self.strategy[i] = 1. / len(Sign)
            self.strategy_sum[i] += self.strategy[i]

        return self.strategy
 
    def update_strategy(self):
        positive_regrets = np.maximum(self.regrets_sum, 0)
        self.strategy = self.normalize(positive_regrets)
        self.strategy_sum += self.strategy
    
    def get_avg_strategy(self):
        avg_strategy = np.zeros(len(Sign))
        normalizing_sum = 0
        for i in range(len(Sign)):
            normalizing_sum += self.strategy_sum[i]
        for i in range(len(Sign)):
            if normalizing_sum > 0:
                avg_strategy[i] = self.strategy_sum[i] / normalizing_sum
            else:
                avg_strategy[i] = 1. / len(Sign)
        return avg_strategy

    def train(self, iterations, opponent_strategy):
        actionUtility = np.zeros(len(Sign))
        for i in tqdm(range(iterations)):
            # get regret-matched mixed-strategy actions 
            strategy = self.get_strategy()
            my_action = self.get_action(strategy)
            other_action = self.get_action(opponent_strategy)
            # compute action utilities 
            actionUtility[other_action.value] = 0
            actionUtility[0 if other_action.value == len(Sign) - 1 else other_action.value + 1] = 1 
            actionUtility[len(Sign) - 1 if other_action.value == 0 else other_action.value - 1] = -1
            # accumulate action regrets 
            for i in range(len(Sign)):
                self.regrets_sum[i] += actionUtility[i] - actionUtility[my_action.value]

    def update_learning(self, self_action, opponent_action):
        actionUtility = np.zeros(len(Sign))
        actionUtility[opponent_action.value] = 0
        actionUtility[0 if opponent_action.value == len(Sign) - 1 else opponent_action.value + 1] = 1 
        actionUtility[len(Sign) - 1 if opponent_action.value == 0 else opponent_action.value - 1] = -1
        
        for i in range(len(Sign)):
            self.regrets_sum[i] += actionUtility[i] - actionUtility[self_action.value]

class TwoPlayer:

    def __init__(self, p1, p2):
        self.p1 = p1 
        self.p2 = p2 

    def iterate(self):
        p1_action = self.p1.get_action(self.p1.get_strategy())
        p2_action = self.p2.get_action(self.p2.get_strategy())
        self.p1.update_learning(p1_action, p2_action)
        self.p2.update_learning(p2_action, p1_action)
    
    def train(self, iterations):
        for i in tqdm(range(iterations)):
            self.iterate()

        print("P1 Strategy: ", self.p1.get_avg_strategy())
        print("P2 Strategy: ", self.p2.get_avg_strategy())


game = TwoPlayer(Player(), Player())
game.train(10000)
