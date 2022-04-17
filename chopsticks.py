import copy
import random

import numpy

from functools import partial
import functools

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

from inspect import isclass


class ChopsticksSimulator():
    def __init__(self, max_moves=100, opp_strategy=None):
        self.hands = [[1, 1], [1, 1]]
        self.max_moves = max_moves
        self.num_moves = 0
        self.debug = False
        if opp_strategy is None:
            self.opp_strategy = [self.random_move]
        else:
            self.opp_strategy = opp_strategy

    def reset(self):
        self.hands = [[1, 1], [1, 1]]
        self.num_moves = 0

    def is_finished(self):
        return sum(self.hands[0]) == 0 or sum(self.hands[1]) == 0 or self.num_moves >= self.max_moves

    def winner(self):
        if sum(self.hands[1]) > 0:
            winner = 1
        else:
            winner = 0
        return (winner, self.num_moves / self.max_moves)

    def set_opp_strat(self, opp_strategy):
        self.opp_strategy = opp_strategy

    def random_move(self):
        move = random.choice(self.valid_moves())
        if (self.debug):
            print(move)
        move()

    def heuristic_random_move(self):
        me = self.num_moves % 2
        opp = (self.num_moves + 1) % 2
        if self.hands[me][0] + self.hands[opp][0] >= 5:
            self.do_attack(0, 0)
        elif self.hands[me][0] + self.hands[opp][1] >= 5:
            self.do_attack(0, 1)
        elif self.hands[me][1] + self.hands[opp][0] >= 5:
            self.do_attack(1, 0)
        elif self.hands[me][1] + self.hands[opp][1] >= 5:
            self.do_attack(1, 1)
        elif ((self.hands[me][0] == 0 or self.hands[me][1] == 0)
              and sum(self.hands[me]) >= 2):
            total = sum(self.hands[self.num_moves % 2])
            left = total // 2
            self.do_transfer(left, total - left)
        else:
            self.random_move()

        if sum(self.hands[me]) == 0:
            print("BUG IN YOUR AGENT")

    def do_attack(self, fro, to):
        if self.hands[self.num_moves % 2][fro] == 0 or self.hands[(self.num_moves + 1) % 2][to] == 0:
            self.hands[self.num_moves % 2] = [0, 0]
            return

        self.hands[(self.num_moves + 1) % 2][to] += self.hands[self.num_moves % 2][fro]
        if self.hands[(self.num_moves + 1) % 2][to] >= 5:
            self.hands[(self.num_moves + 1) % 2][to] = 0
        self.num_moves += 1

    def attack(self, fro, to):
        return lambda: self.do_attack(fro, to)

    def do_transfer(self, left, right):
        if sum(self.hands[self.num_moves % 2]) != (left + right) or left == 0 or right == 0 or left >= 5 or right >= 5:
            self.hands[self.num_moves % 2] = [0, 0]
            return
        self.hands[self.num_moves % 2] = [left, right]
        self.num_moves += 1

    def transfer(self, left, right):
        return lambda: self.do_transfer(left, right)

    def valid_moves(self):
        me = self.num_moves % 2
        opp = (self.num_moves + 1) % 2
        attacks = []
        if self.hands[me][0] > 0:
            attacks += [self.attack(0, to) for to in range(2) if self.hands[opp][to] > 0]
        if self.hands[me][1] > 0:
            attacks += [self.attack(1, to) for to in range(2) if self.hands[opp][to] > 0]

        transfers = []
        my_sticks = sum(self.hands[me])
        if self.hands[me][0] == 0 and my_sticks > 1:
            transfers += [self.transfer(left, my_sticks - left) for left in range(1, my_sticks)]
        if self.hands[me][1] == 0 and my_sticks > 1:
            transfers += [self.transfer(my_sticks - right, right) for right in range(1, my_sticks)]
        if self.hands[me][0] > 0 and self.hands[me][1] > 0:
            transfers += [self.transfer(left, my_sticks - left)
                          for left in range(1, min(my_sticks, 5))
                          if left != self.hands[me][0]
                          and left != self.hands[me][1]
                          and my_sticks - left < 5]

        return attacks + transfers

    def run(self, strat):
        self.reset()
        opp_turn = random.choice(self.opp_strategy)
        while not self.is_finished():
            strat()
            if not self.is_finished():
                opp_turn()
        return self.winner()

    def debug_run(self, strat):
        self.reset()
        self.debug = True
        print("FFT Hands: ", self.hands[0])
        print("RAND Hands: ", self.hands[1])
        while not self.is_finished():
            print("FFT MOVE")
            print("FFT Pre: ", self.hands[self.num_moves % 2])
            strat()
            print("FFT Hands: ", self.hands[0])
            print("RAND Hands: ", self.hands[1])
            if not self.is_finished():
                print("RAND MOVE")
                print("RAND Pre: ", self.hands[self.num_moves % 2])
                self.opp_strategy()
        print("FFT Hands: ", self.hands[0])
        print("RAND Hands: ", self.hands[1])
        self.debug = False
        return self.winner()

    def hand_greater(self, hand, num):
        return lambda: len([1 for h in self.hands[self.num_moves % 2] if h > num]) > 0

    def opp_hand_greater(self, hand, num):
        return lambda: len([1 for h in self.hands[(self.num_moves + 1) % 2] if h > num]) > 0

    def hand_less(self, hand, num):
        return lambda: len([1 for h in self.hands[self.num_moves % 2] if h < num]) > 0

    def opp_hand_less(self, hand, num):
        return lambda: len([1 for h in self.hands[(self.num_moves + 1) % 2] if h < num]) > 0

    def hand_equal(self, hand, num):
        return lambda: len([1 for h in self.hands[self.num_moves % 2] if h == num]) > 0

    def opp_hand_equal(self, hand, num):
        return lambda: len([1 for h in self.hands[(self.num_moves + 1) % 2] if h == num]) > 0

    def generate_transfer(self, left):
        my_sticks = sum(self.hands[self.num_moves % 2])
        return self.transfer(left, my_sticks - left)
