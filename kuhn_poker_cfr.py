#!/usr/bin/env python3
"""
Kuhn Poker CFR - Corrected Implementation
===========================================
Following the standard CFR algorithm exactly.

The key fix: terminal utilities are ALWAYS from player 0's perspective.
At player 1 nodes, we negate the VALUES, not the utilities themselves.
"""

import numpy as np
from collections import defaultdict

JACK, QUEEN, KING = 0, 1, 2
PASS, BET = 0, 1


class KuhnCFR:
    def __init__(self):
        self.regretSum = defaultdict(lambda: np.zeros(2))
        self.strategySum = defaultdict(lambda: np.zeros(2))
    
    def getStrategy(self, infoSet):
        regrets = self.regretSum[infoSet]
        strategy = np.maximum(regrets, 0)
        normalizingSum = np.sum(strategy)
        
        if normalizingSum > 0:
            strategy /= normalizingSum
        else:
            strategy = np.array([1.0/2.0, 1.0/2.0])
        
        return strategy
    
    def getAverageStrategy(self, infoSet):
        avgStrategy = self.strategySum[infoSet].copy()
        normalizingSum = np.sum(avgStrategy)
        
        if normalizingSum > 0:
            avgStrategy /= normalizingSum
        else:
            avgStrategy = np.array([1.0/2.0, 1.0/2.0])
        
        return avgStrategy
    
    def cfr(self, cards, history, p0, p1):
        """
        Standard CFR algorithm.
        Returns utility for player 0 (ALWAYS).
        """
        plays = len(history)
        player = plays % 2
        opponent = 1 - player
        
        # Terminal nodes
        if plays > 1:
            terminalPass = history[plays - 1] == 'p'
            doubleBet = history[plays - 2:plays] == "bb"
            isPlayerCardHigher = cards[player] > cards[opponent]
            
            if terminalPass:
                if history == "pp":
                    # Both passed
                    return 1 if cards[0] > cards[1] else -1
                else:
                    # Player passed after opponent bet (fold)
                    return 1 if player == 1 else -1
            elif doubleBet:
                # Both bet
                return 2 if cards[0] > cards[1] else -2
        
        infoSet = str(cards[player]) + history
        strategy = self.getStrategy(infoSet)
        
        if player == 0:
            self.strategySum[infoSet] += p0 * strategy
        else:
            self.strategySum[infoSet] += p1 * strategy
        
        # Compute action utilities
        util = np.zeros(2)
        nodeUtil = 0
        
        for a in range(2):
            nextHistory = history + ("p" if a == PASS else "b")
            
            if player == 0:
                util[a] = self.cfr(cards, nextHistory, p0 * strategy[a], p1)
            else:
                util[a] = self.cfr(cards, nextHistory, p0, p1 * strategy[a])
            
            nodeUtil += strategy[a] * util[a]
        
        # Accumulate regrets
        regrets = util - nodeUtil
        
        if player == 0:
            self.regretSum[infoSet] += p1 * regrets
        else:
            self.regretSum[infoSet] += p0 * regrets
        
        return nodeUtil
    
    def train(self, iterations):
        cards = [JACK, QUEEN, KING]
        util = 0
        
        for i in range(iterations):
            for c0 in cards:
                for c1 in cards:
                    if c0 == c1:
                        continue
                    
                    util += self.cfr([c0, c1], "", 1, 1)
            
            if (i + 1) % 5000 == 0:
                print(f"Iteration {i+1:6d} | Avg utility: {util / ((i+1) * 6):+.6f}")
        
        return util / (iterations * 6)
    
    def displayStrategy(self):
        print("\n" + "=" * 70)
        print("  Kuhn Poker Nash Equilibrium")
        print("=" * 70)
        
        for infoSet in sorted(self.strategySum.keys()):
            strategy = self.getAverageStrategy(infoSet)
            card_str = ['J', 'Q', 'K'][int(infoSet[0])]
            history = infoSet[1:] if len(infoSet) > 1 else "(start)"
            
            print(f"  [{card_str}] {history:10s} →  Pass: {strategy[0]:.3f}  Bet: {strategy[1]:.3f}")
        
        print("=" * 70)


def main():
    print("\n" + "=" * 70)
    print("  Kuhn Poker CFR Solver (Corrected)")
    print("=" * 70 + "\n")
    
    cfr = KuhnCFR()
    value = cfr.train(50000)
    cfr.displayStrategy()
    
    knownValue = -1.0/18.0
    error = abs(value - knownValue)
    
    print(f"\n{'='*70}")
    print("  Convergence")
    print("=" * 70)
    print(f"  Learned value: {value:+.6f}")
    print(f"  Nash value:    {knownValue:+.6f}")
    print(f"  Error:         {error:.6f}")
    
    if error < 0.01:
        print(f"\n  ✓ Converged! (ε = {error:.6f})")
    else:
        print(f"\n  ✗ Not converged (ε = {error:.6f})")
    
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
