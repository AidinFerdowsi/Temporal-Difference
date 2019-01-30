# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:28:38 2019

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
    
    Q1 = {}
    Q2 = {}
    
    states = grid.allStates()
    
    for s in states:
        Q1[s] = {}
        Q2[s] = {}
        for a in ACTIONS:
            Q1[s][a] = 0
            Q2[s][a] = 0
    
    
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
        
        biggestChange = 0
        
        while not grid.gameOver():
            p =  np.random.random()
            if p > 0.5:
                aChosen = argMax(Q1[s])[0]
            else:
                aChosen = argMax(Q2[s])[0]    
            a = randomAction(aChosen,grid,eps = 0.5/t)
            r = grid.move(a)
            s2 = grid.currentState()
            alpha = ALPHA/updateCountsSarsa[s][a]
            updateCountsSarsa[s][a] += 0.005
            if p < 0.5:
                a2,maxQValue = argMax(Q1[s2])
                oldQ = Q2[s][a]
                Q2[s][a] = Q2[s][a] + alpha * (r + GAMMA * maxQValue - Q2[s][a])
                biggestChange = max(biggestChange, np.abs(oldQ - Q2[s][a]))
            else:
                a2,maxQValue = argMax(Q2[s2])
                oldQ = Q1[s][a]
                Q1[s][a] = Q1[s][a] + alpha * (r + GAMMA * maxQValue - Q1[s][a])
                biggestChange = max(biggestChange, np.abs(oldQ - Q1[s][a]))
            
            a = a2
            s = s2
        deltas.append(biggestChange)
    plt.plot(deltas)
    
    policy = {}
    V = {}
    for s in grid.actions.keys():
        policy[s],V[s] = argMax(Q1[s])
        
    
    
    print("Values:")
    printValues(V,grid)
    
    print("Policy:")
    printPolicy(policy,grid)