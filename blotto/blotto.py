import numpy as np
from tqdm import tqdm
import copy
import random
from blotto_plus import compute_exploitability

def generate_strategies_loop(state, idx, troops_left, master_list):
    new_state = copy.deepcopy(state)
    if idx == len(state) - 1:
            new_state[idx] = troops_left
            master_list.append(new_state)
    else:
            for n in range(troops_left+1):
                new_state[idx] = n 
                generate_strategies_loop(new_state, idx+1, troops_left - n, master_list)
            
def generate_strategies(num_soldiers, num_battlefields):
    master = []
    initial_state = np.zeros(num_battlefields)
    generate_strategies_loop(initial_state, 0, num_soldiers, master)
    return master


class Blotto:

    def __init__(self, num_soldiers, num_battlefields):
        self.num_soldiers = num_soldiers
        self.num_battlefields = num_battlefields
        self.land1 = np.zeros(num_battlefields)
        self.land2 = np.zeros(num_battlefields)

    def reset_battlefields(self):
        self.land1 = np.zeros(num_battlefields)
        self.land2 = np.zeros(num_battlefields)

    def allocate_soldiers(self, battlefields1, battlefields2):
        self.land1 = np.array(battlefields1)
        self.land2 = np.array(battlefields2)

    def evaluate_game(self, land1, land2):
        """Returns 1 for a win, -1 for a loss, and 0 for a tie"""
        p1_wins = 0
        p2_wins = 0
        ties = 0
        for i, tile in enumerate(land1):
            if tile > land2[i]:
                p1_wins += 1 
            elif tile < land2[i]:
                p2_wins += 1 
            else: 
                ties += 1
        if p1_wins > p2_wins and p1_wins > ties:
            return 1 
        elif p2_wins > p1_wins and p2_wins > ties:
            return -1
        else: return 0

    def self_eval(self):
        return self.evaluate_game(self.land1, self.land2)

class Player:

    def __init__(self, number_soldiers, number_battlefields):
        self.num_soldiers = number_soldiers
        self.num_battlefields = number_battlefields
        self.strategies = generate_strategies(number_soldiers, number_battlefields)
        self.regret_sum = np.zeros(len(self.strategies))
        self.strategy = np.zeros(len(self.strategies))
        self.strategy_sum = np.zeros(len(self.strategies))

    def update_strategy(self):
        normalizing_sum = 0
        size = len(self.strategies)
        for a in range(size):
            self.strategy[a] = max(0, self.regret_sum[a])
            normalizing_sum += self.strategy[a]
        for a in range(size):
            if normalizing_sum > 0:
                self.strategy[a] /= normalizing_sum
            else:
                self.strategy[a] = 1.0 / size
            self.strategy_sum[a] += self.strategy[a] 
    
    def select_action_idx(self):
        r = random.random()
        a = 0 
        cum_prob = 0. 
        while(a < len(self.strategies) - 1):
            cum_prob += self.strategy[a]
            if r < cum_prob:
                break
            a += 1 
        return a

    def select_action(self):
        return self.strategies[self.select_action_idx()]
    
    def update_learning(self, game, self_action, opponent_action):
        my_action_utility = game.evaluate_game(self_action, opponent_action)
        for a in range(len(self.strategies)):
            self.regret_sum[a] += game.evaluate_game(self.strategies[a], opponent_action) - my_action_utility
         
    
    def get_average_strategy(self):
        size = len(self.strategies)
        avg_strategy = np.zeros(size)
        normalizing_sum = 0 
        for a in range(size):
            normalizing_sum += self.strategy_sum[a]
        for a in range(size):
            if normalizing_sum > 0:
                avg_strategy[a] = self.strategy_sum[a] / normalizing_sum
            else:
                avg_strategy[a] = 1.0 / size 

        return avg_strategy

    
class Trainer:

    def __init__(self, num_soldiers, num_battlefields):
        self.p1 = Player(num_soldiers, num_battlefields)
        self.p2 = Player(num_soldiers, num_battlefields)
        self.game = Blotto(num_soldiers, num_battlefields)
        self.num_soldiers = num_soldiers
        self.num_battlefields = num_battlefields

    def iterate(self):
        self.p1.update_strategy()
        self.p2.update_strategy()
        p1_action = self.p1.select_action()
        p2_action = self.p2.select_action()
        self.p1.update_learning(self.game, p1_action, p2_action)
        self.p2.update_learning(self.game, p2_action, p1_action)
    
    def compute_exploitability(self):
        p1_strategies = self.p1.strategies
        avg_strat = self.p1.get_average_strategy()

        def eval_game(s1, s2):
            return self.game.evaluate_game(s1, s2)

        return compute_exploitability(p1_strategies, avg_strat, eval_game)

    def train(self, iterations):
        for i in tqdm(range(iterations)):
            self.iterate()
        
        prob_sums = 0
        for prob, strat in zip(self.p1.get_average_strategy(), self.p1.strategies):
            print(f"{strat} -> {prob:.4f}")
            prob_sums += prob 
        print(f"Exploitability: {self.compute_exploitability()}")
        #print(f"Sum of all probabilities: {prob_sums}")
        # print(self.p1.get_average_strategy())
        # print(self.p2.get_average_strategy())
        
if __name__ == "__main__":
    trainer = Trainer(6, 4)
    trainer.train(100000)
