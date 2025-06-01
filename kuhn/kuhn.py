import numpy as np
import random
from enum import Enum

class Actions(Enum):
    PASS = 0
    RAISE = 1

class KuhnPoker:
   
    def __init__(self):
        self.cards = np.array([1, 2, 3])
        self.p1_card = None
        self.p2_card = None
    
    def reset(self):
        np.random.shuffle(self.cards)
        self.p1_card = None
        self.p2_card = None

    def begin_play(self):
        # we haven't dealt yet
        if self.p1_card == None:
            # first shuffle 
            self.reset()
            # now deal 
            self.p1_card = self.cards[0]
            self.p2_card = self.cards[1]

    

class KuhnNode:
    
    def __init__(self):
        self.infoset = ""
        self.regret_sum = np.zeros(len(Actions))
        self.strategy = np.zeros(len(Actions))
        self.strategy_sum = np.zeros(len(Actions))

    def get_strategy(self, realization_weight):
        normalizing_sum = 0.
        for a in range(len(Actions)):
            self.strategy[a] = max(0, self.regret_sum[a])
            normalizing_sum += self.strategy[a]
        for a in range(len(Actions)):
            if normalizing_sum > 0:
                self.strategy[a] /= normalizing_sum
            else: 
                self.strategy[a] = 1.0 / len(Actions)

        return self.strategy
    
    def get_average_strategy(self):
        avg_strategy = np.zeros(len(Actions))
        normalizing_sum = 0.
        for a in range(len(Actions)):
            normalizing_sum += self.strategy_sum[a]
        for a in range(len(Actions)):
            if normalizing_sum > 0:
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
            else: 
                avg_strategy[a] = 1.0 / len(Actions)

        return avg_strategy
    
    def to_string(self):
        return f"{self.infoset}, {self.get_average_strategy()}"
