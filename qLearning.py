# -*- coding: utf-8 -*-
"""
Created on Tue Jan 29 19:12:42 2019

@author: Aidin
"""

import numpy as np
import matplotlib.pyplot as plt
from gridWorld import standardGrid, negativeGrid
from iterativePolicyEvaluation import printPolicy, printValues
from monteCarloControl import argMax
from td0Prediction import randomAction


GAMMA = 0.9
ALPHA = 0.1
ACTIONS = {'U','D','L','R'}

if __name__ == '__main__':
    grid = negativeGrid(stepCost = -0.1)
    
    print ("Rewards:")
    printValues(grid.rewards,grid)
    
    Q = {}
    
    states = grid.allStates()
    
    for s in states:
        Q[s] = {}
        for a in ACTIONS:
            Q[s][a] = 0
    
    
    updateCountsSarsa = {}
    
    for s in states:
        updateCountsSarsa[s] = {}
        for a in ACTIONS:
            updateCountsSarsa[s][a] = 1
            
    
    
    t = 1
    deltas = []
    
    for iterations in range(10000):
        if iterations % 100 == 0 :
            t += 10e-3
        if iterations % 2000 == 0:
            print('Iteration:',iterations)
            
        s = (2,0)
        grid.setState(s)
        
        a = argMax(Q[s])[0]
        biggestChange = 0
        
        while not grid.gameOver():
            a = randomAction(argMax(Q[s])[0],grid,eps = 0.5/t)
            r = grid.move(a)
            s2 = grid.currentState()
            
            a2,maxQValue = argMax(Q[s2])
            
            
            alpha = ALPHA/updateCountsSarsa[s][a]
            updateCountsSarsa[s][a] += 0.005
            oldQ = Q[s][a]
            Q[s][a] = Q[s][a] + alpha * (r + GAMMA * maxQValue - Q[s][a])
            biggestChange = max(biggestChange, np.abs(oldQ - Q[s][a]))
            
            a = a2
            s = s2
        deltas.append(biggestChange)
    plt.plot(deltas)
    
    policy = {}
    V = {}
    for s in grid.actions.keys():
        policy[s],V[s] = argMax(Q[s])
        
    
    
    print("Values:")
    printValues(V,grid)
    
    print("Policy:")
    printPolicy(policy,grid)
        