import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from scipy.signal import convolve2d

#construct gridworld
gridworld = -1. * np.ones((9,9))
stars = [(0,0), (0,1), (0,2), (0,7), (0,8), (1,8), \
         (3,2), (3,3), (3,4), (3,5), (4,2), \
         (7,1), (7,2), (7,3), (7,4), (7,5), (7,6), \
         (8,1), (8,2), (8,3), (8,5), (8,6)]
for cell in stars:
    gridworld[cell] = 15.

obstacles = [(1,5), (1,7), (2,1), (2,2), (2,3), (2,4), (2,5), (2,7),\
             (3,1), (3,6), (4,3), (4,4), (4,5), (5,7), \
             (6,1), (6,2), (6,3), (6,4), (6,5), (6,6), \
             (7,8), (8,4), (8,8)]

for cell in obstacles:
    gridworld[cell] = -30.

terminalStates = [(3,8)]
for cell in terminalStates:
    gridworld[cell] = 150.

states = [(y,x) for y in range(len(gridworld[0])) for x in range(len(gridworld))]

###construct kernels
kernels_conv = []
blankKernel = np.zeros((3,3))    
for action_conv in [8, 7, 6, 3, 0, 1, 2, 5]: #[SE, S, SW, W, NW, N, NE, E], this is [NW, N, NE, E, SE, S, SW, W] rotated 180°
    kernel_conv = blankKernel.copy()
    kernel_conv[np.unravel_index(action_conv, kernel_conv.shape)] = 1.0
    kernels_conv.append(kernel_conv)

#convolve grid with action kernels
rewards = []
for actionKernel in kernels_conv:
    actionRewards = convolve2d(gridworld, actionKernel, fillvalue=-30)
    actionRewards = actionRewards[1:(len(actionRewards)-1),:][:,1:(len(actionRewards[0])-1)]
    rewards.append(actionRewards)
rewards = np.array(rewards)

def getNextStatesRewardsAndProbabilities(state, action, rewards=rewards, deviation = True):
    coordinateActions = [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]
                        #[NW,      N,      NE,     E,     SE,    S,     SW,     W]
    nextStatesAndProbabilities = []
    if deviation:
        actions = [(action + wind)%8 for wind in [-1, 0, 1]] #45° to the left, none, 45° to the right deviations
        probabilities = [0.2, 0.7, 0.1]                    #deviation probabilities defined on the assignment sheet
    else:
        actions = [action]                                   #the agent always executes the selected action correctly
        probabilities = [1.]
    for i,a in enumerate(actions):
        R = rewards[a][state]                                #check which reward the action would yield
        if R < -1.:                                          #if the reward is -1, or -30, s' remains s 
            nextState = state
        else:
            nextState = tuple(map(sum, zip(state, coordinateActions[a]))) #else, actually move to a different s'
        nextStatesAndProbabilities.append([nextState, R, probabilities[i]])
    return nextStatesAndProbabilities


def drawPolicy(pi, cells = states, terminalStates = terminalStates, obstacles = obstacles, stars = stars):
    ticks = np.arange(0,9)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([0,9,0,9])
    #ax.axis('equal')
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    fig.gca().set_aspect('equal')
    fig.gca().invert_yaxis()
    ax.grid(True,color='k')
    for axi in (ax.xaxis, ax.yaxis):
        for tic in axi.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
    for cell in cells:
        if cell in terminalStates:
            rect = Rectangle((cell[1],cell[0]),1,1,color='palegreen')
            ax.add_patch(rect)
        elif cell in obstacles:
            rect = Rectangle((cell[1],cell[0]),1,1,color='lightsalmon')
            ax.add_patch(rect)
        else:
            if cell in stars:
                rect = Rectangle((cell[1],cell[0]),1,1,color='cornflowerblue')
                ax.add_patch(rect)
            if pi[cell][0] == 1.:
                ax.arrow(cell[1]+0.85,cell[0]+0.85,-0.7,-0.7,head_width=0.2, head_length=0.2,length_includes_head=True,
                        fill = True)
            elif pi[cell][1] == 1.:
                ax.arrow(cell[1]+0.5,cell[0]+0.9,0,-0.8,head_width=0.2, head_length=0.2,length_includes_head=True,
                        fill = True)
            elif pi[cell][2] == 1.:
                ax.arrow(cell[1]+0.15,cell[0]+0.85,0.7,-0.7,head_width=0.2, head_length=0.2,length_includes_head=True,
                        fill = True)
            elif pi[cell][3] == 1.:
                ax.arrow(cell[1]+0.1,cell[0]+0.5,0.8,0,head_width=0.2, head_length=0.2,length_includes_head=True,
                        fill = True)
            elif pi[cell][4] == 1.:
                ax.arrow(cell[1]+0.15,cell[0]+0.15,0.7,0.7,head_width=0.2, head_length=0.2,length_includes_head=True,
                        fill = True)
            elif pi[cell][5] == 1.:
                ax.arrow(cell[1]+0.5,cell[0]+0.1,0,0.8,head_width=0.2, head_length=0.2,length_includes_head=True,
                        fill = True)
            elif pi[cell][6] == 1.:
                ax.arrow(cell[1]+0.85,cell[0]+0.15,-0.7,0.7,head_width=0.2, head_length=0.2,length_includes_head=True,
                        fill = True)
            else:
                ax.arrow(cell[1]+0.9,cell[0]+0.5,-0.8,0,head_width=0.2, head_length=0.2,length_includes_head=True,
                        fill = True)
    ax.tick_params(
    axis='x',          
    which='both',      
    bottom=False,      
    top=False)
    ax.tick_params(
    axis='y',          
    which='both',      
    left=False,      
    right=False)

    plt.show()
    
    
def drawGrid(cells = states, terminalStates = terminalStates, obstacles = obstacles, stars = stars):
    ticks = np.arange(0,9)
    fig = plt.figure(figsize=(11,11))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([0,9,0,9])
    #ax.axis('equal')
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    fig.gca().set_aspect('equal')
    fig.gca().invert_yaxis()
    ax.grid(True,color='k')
    for axi in (ax.xaxis, ax.yaxis):
        for tic in axi.get_major_ticks():
            tic.tick1On = tic.tick2On = False
            tic.label1On = tic.label2On = False
    for cell in cells:
        if cell in terminalStates:
            rect = Rectangle((cell[1],cell[0]),1,1,color='palegreen')
            ax.add_patch(rect)
        elif cell in obstacles:
            rect = Rectangle((cell[1],cell[0]),1,1,color='lightsalmon')
            ax.add_patch(rect)
        else:
            if cell in stars:
                rect = Rectangle((cell[1],cell[0]),1,1,color='cornflowerblue')
                ax.add_patch(rect)
    ax.tick_params(
    axis='x',          
    which='both',      
    bottom=False,      
    top=False)
    ax.tick_params(
    axis='y',          
    which='both',      
    left=False,      
    right=False)

    red_patch = Rectangle((cell[1],cell[0]),1,1,color='lightsalmon', label='Obstacle')
    blue_patch = Rectangle((cell[1],cell[0]),1,1,color='cornflowerblue', label='Snack')
    green_patch = Rectangle((cell[1],cell[0]),1,1,color='palegreen', label='Goal')

    plt.legend(handlelength=2, handleheight=2.7,handles=[red_patch, blue_patch, green_patch],  loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show()
