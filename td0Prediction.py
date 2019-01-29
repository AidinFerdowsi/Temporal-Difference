# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 01:03:41 2019

@author: Aidin
"""

import numpy as np
import matplotlib.pyplot as plt
from gridWorld import standardGrid, negativeGrid
from iterativePolicyEvaluation import printPolicy, printValues

EPSILON = 10e-4
GAMMA = 0.9
ALPHA = 0.1
ACTIONS = {'U','D','L','R'}

def randomAction(a, grid, eps = 0.1):
    
    
    p = np.random.random()
    
    if p < (1 - eps):
        return a
    else:
        return np.random.choice(list(ACTIONS))
#        return np.random.choice(list(grid.actions[grid.currentState()]))
    
    
    
def playGame(grid,policy):
    s = (2,0)
    grid.setState(s)   
    
    statesRewards = [(s,0)]
    while not grid.gameOver():
        a = randomAction(policy[s],grid)
        r = grid.move(a)
        s = grid.currentState()
        statesRewards.append((s,r))
    return statesRewards

if __name__ == '__main__':
#    grid = standardGrid()
    
    grid = negativeGrid(stepCost = - 0.1)
    
    print ("Rewards:")
    printValues(grid.rewards,grid)
    
    policy = {
            (2,0): "U",
            (1,0): "U",
            (0,0): "R",
            (0,1): "R",
            (0,2): "R",
            (1,2): "R",
            (2,1): "R",
            (2,2): "R",
            (2,3): "U",
    } 
    
    V = {}
    states = grid.allStates()
    for s in states:
        V[s] = 0
        
    
    for iters in range(1000):
        statesRewards = playGame(grid,policy)
        
        for t in range(len(statesRewards)-1):
            s = statesRewards[t][0]
            sprime, r = statesRewards[t+1]
            
            V[s] = V[s] + ALPHA*(r+GAMMA*V[sprime]-V[s])
            
    print("Policy")
    printPolicy(policy,grid)
    
        
    print("Values")
    printValues(V,grid)