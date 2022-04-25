from helpers.utils import *
import numpy as np
import skimage.draw
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle, Circle

racetrack = -15. * np.ones((20,24))

circuit =   [(1,x) for x in np.arange(19,23)] + \
            [(2,x) for x in np.arange(19,23)] + \
            [(3,x) for x in np.arange(19,22)] + \
            [(4,x) for x in np.arange(19,22)] + \
            [(5,x) for x in np.arange(19,22)] + \
            [(6,x) for x in np.arange(19,22)] + \
            [(7,x) for x in np.arange(19,22)] + \
            [(8,x) for x in np.arange(18,22)] + \
            [(9,x) for x in np.arange(17,22)] + \
            [(10,x) for x in np.arange(9,22)] + \
            [(11,x) for x in np.arange(8,21)] + \
            [(12,x) for x in np.arange(7,19)] + \
            [(13,x) for x in np.arange(6,10)] + \
            [(14,x) for x in np.arange(5,9)] + \
            [(15,x) for x in np.arange(4,8)] + \
            [(16,x) for x in np.arange(3,7)] + \
            [(17,x) for x in np.arange(2,6)] + \
            [(18,x) for x in np.arange(1,5)] + \
            [(x,20-x) for x in np.arange(1,10)] + \
            [(x,21-x) for x in np.arange(1,10)]
            

for cell in circuit:
    racetrack[cell] = 0.1

finishLine = [(0,x) for x in np.arange(19,23)]

for cell in finishLine:
    racetrack[cell] = 15.

startingGrid = [(19,x) for x in np.arange(1,5)]

for cell in startingGrid:
    racetrack[cell] = 0.1
    
def velocity_to_ind(vel):
    return vel.reshape(-1,2) @ 6 ** np.arange(0, 2, 1)

def ind_to_velocity(num):
    return ((np.array(num).reshape((-1,1)) // (6 ** np.arange(0, 2, 1))) % 6).reshape((-1,2))

coordinateActions = np.array([[0, 0], [-1,-1], [-1,0], [-1,1], [0,1], [1,1], [1,0], [1,-1], [0,-1]])
                             #[=u|=v,  -u|-v,  -u|=v,  -u|+v,  =u|+v, +u|+v, +u|=v, +u|-v,  =u|-v]
    
allVelocities = ind_to_velocity(np.arange(0,36))

accelerationResult = coordinateActions + allVelocities[:,None]

#get the sum of both component velocities , get the minimum component velocity, stack these
velocityFeatures = np.dstack([accelerationResult.sum(axis = -1), accelerationResult.min(axis = -1)])

#check if the does not exceed 6, check if the sum is at least 1, check if both components are positive
permittedAcceleration = np.dstack([velocityFeatures[:,:,0] < 7, velocityFeatures[:,:,0] > 0, velocityFeatures[:,:,1] > -1])
#the action is permitted for a given velocity if all of these 3 criteria hold
permittedAcceleration = permittedAcceleration.all(-1)

#lets wrap this up in a helper function
def getAvailableActions(velocity, permittedActions = permittedAcceleration):
    if velocity.shape[-1] == 1: ###check if we (accidentally) already passed an index
        velocityIndex = velocity #no conversion necessary
    else:
        velocityIndex = velocity_to_ind(velocity) ###if the velocity is of the form [u,v], convert to index
    return np.argwhere(permittedActions[velocityIndex] == True)[:,1]

states = [(y,x) for y in range(racetrack.shape[0]) for x in range(racetrack.shape[1])]

def drawTrajectory(moves, actions, velocities, cells = states, circuit = circuit, finishLine = finishLine, startingGrid = startingGrid):
    yticks = np.arange(0,20)
    xticks = np.arange(0,24)
    fig = plt.figure(figsize = (24,20))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis([0,24,0,20])
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    fig.gca().set_aspect('equal')
    fig.gca().invert_yaxis()
    ax.grid(True,color='k')
    
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
    
    
    for cell in cells:
        if cell in circuit:
            rect = Rectangle((cell[1],cell[0]),1,1,color='palegreen')
            ax.add_patch(rect)
        elif cell in finishLine:
            rect = Rectangle((cell[1],cell[0]),1,1,color='cornflowerblue')
            ax.add_patch(rect)
        elif cell in startingGrid:
            rect = Rectangle((cell[1],cell[0]),1,1,color='dimgray')
            ax.add_patch(rect)
        else:
            rect = Rectangle((cell[1],cell[0]),1,1,color='lightsalmon')
            ax.add_patch(rect)
            
    for i in range(len(actions)):
        for x, y in moves[i]:
            rect = Rectangle((y,x),1,1,color='blue')
            ax.add_patch(rect)
        y, x = moves[i][0]
        a = actions[i]
        if a == 3:
            ax.arrow(x+0.85,y+0.85,-0.7,-0.7,head_width=0.2, head_length=0.2,length_includes_head=True,
                    fill = True)
        elif a == 4:
            ax.arrow(x+0.5,y+0.9,0,-0.8,head_width=0.2, head_length=0.2,length_includes_head=True,
                    fill = True)
        elif a == 5:
            ax.arrow(x+0.15,y+0.85,0.7,-0.7,head_width=0.2, head_length=0.2,length_includes_head=True,
                    fill = True)
        elif a == 6:
            ax.arrow(x+0.1,y+0.5,0.8,0,head_width=0.2, head_length=0.2,length_includes_head=True,
                    fill = True)
        elif a == 7:
            ax.arrow(x+0.15,y+0.15,0.7,0.7,head_width=0.2, head_length=0.2,length_includes_head=True,
                    fill = True)
        elif a == 8:
            ax.arrow(x+0.5,y+0.1,0,0.8,head_width=0.2, head_length=0.2,length_includes_head=True,
                    fill = True)
        elif a == 1:
            ax.arrow(x+0.85,y+0.15,-0.7,0.7,head_width=0.2, head_length=0.2,length_includes_head=True,
                    fill = True)
        elif a == 2:
            ax.arrow(x+0.9,y+0.5,-0.8,0,head_width=0.2, head_length=0.2,length_includes_head=True,
                    fill = True)
        elif a == 0:
            circ = Circle((x+0.5,y+0.5), radius=0.2, fill=True, lw=2., color='tab:blue', ec='k')
            ax.add_patch(circ)
            
    for i in range(len(actions)):
        y, x = moves[i][0]
        v0, v1 = ind_to_velocity(velocities[i]).squeeze()
        ax.arrow(x+0.5, y+0.5, 0.95*v0, -0.95*v1, head_width=0.2, head_length=0.2,length_includes_head=True, fill = True, color = 'r')
        
        

    plt.show()
    
def getEndPosition(position, velocity):
    return position + velocity[::-1] * np.array([-1,1])

class env:
    
    def __init__(self):
        self.startingGrid = np.array(startingGrid)
        self.velocity = np.array([0,1])
        self.position = self.randomStart()
        self.done = False
    
    def reset(self):
        self.position = self.randomStart()
        self.velocity = np.array([0,1])
        self.done = False
    
    def randomStart(self):
        startIndex = np.random.choice(self.startingGrid.shape[0])
        return self.startingGrid[startIndex]
    
    def setPosition(self, y, x):
        self.position = np.array([y,x])
        
    def getPosition(self):
        return self.position
    
    def getVelocity(self):
        return velocity_to_ind(self.velocity)
        
    def getAvailableActions(self, permittedActions = permittedAcceleration):
        return getAvailableActions(self.velocity, permittedActions)
    
    def step(self, action, mode = 'Bresenham', racetrack = racetrack, deviation = False):
        currentPosition = self.position
        self.velocity += coordinateActions[action]
        currentVelocity = self.velocity
        
        traveled = []

        endPosition = getEndPosition(currentPosition, currentVelocity)
        
        if mode == 'Bresenham':
            yTrajectory, xTrajectory = skimage.draw.line(*currentPosition, *endPosition)
            
        if mode == 'aa':
            yTrajectory, xTrajectory, _ = skimage.draw.line_aa(*currentPosition, *endPosition)

        trajectory = np.dstack([yTrajectory,xTrajectory]).squeeze()

        for y, x in trajectory:
            traveled.append(np.array([y,x]))
            r = racetrack[y,x]
            if r != 0.1:
                self.done = True
                break
                
        if deviation and not self.done:
            if np.random.random() < 0.5:
                x+=1
            else:
                y-=1
            r = racetrack[y,x]
            if r != 0.1:
                self.done = True
            traveled.append(np.array([y,x]))
                
            
    
        self.setPosition(y,x)
        return y, x, velocity_to_ind(currentVelocity), r, self.done, np.array(traveled)