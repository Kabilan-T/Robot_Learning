import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow, Circle


def draw_trajectory(states):
    fig, axes = plt.subplots(nrows = int((states.shape[0] - 0.1) // 4) + 1, ncols = min(states.shape[0],4), squeeze = False, \
                             figsize = (2.6 *  min(states.shape[0],4) , 2.6 * (int((states.shape[0] - 0.1) // 4) + 1)))
      
    for ind, state in enumerate(states):
        circles = np.argwhere(state == 0)
        crosses = np.argwhere(state == 2)
        patches = [Circle((y, x), radius=0.35, color='red', fill=False, lw = 3) for x, y in circles]



        axes[ind // 4, ind % 4].matshow(np.zeros((3,3)), cmap = 'Greys')
        axes[ind // 4, ind % 4].set_xticks([x-0.5 for x in range(1,3)],minor=True )
        axes[ind // 4, ind % 4].set_yticks([y-0.5 for y in range(1,3)],minor=True)
        axes[ind // 4, ind % 4].grid(color = 'k',which="minor",ls="-",lw=2.5)
        axes[ind // 4, ind % 4].set_xticks([]) 
        axes[ind // 4, ind % 4].set_yticks([]) 
        axes[ind // 4, ind % 4].patch.set_edgecolor('black')  
        axes[ind // 4, ind % 4].patch.set_linewidth('3')
        axes[ind // 4, ind % 4].set_title('Game State ' + str(ind))

        for p in patches:
            axes[ind // 4, ind % 4].add_patch(p)

        axes[ind // 4, ind % 4].plot(crosses[:,1],crosses[:,0],c = 'b', marker='x', markersize = 35, ls='')
     
    if(states.shape[0] % 4 != 0 and states.shape[0] > 4):
        for ax in axes[-1,-(4 - (states.shape[0] % 4)):]:
            ax.axis('off')
    plt.show()

class env:
    def __init__(self):
        # total number of board configurations
        numStates = 3**9
        # get all board configuration matrices explicitly
        stateSpace = self.ind_to_state(np.arange(numStates))
        # calculate all diagonal, row and col sums for each matrix, stack them
        stateWins = np.vstack([np.trace(stateSpace, axis1 = -1), np.trace(stateSpace[:,::-1], axis1 = -1), stateSpace.sum(-1).T,\
                       stateSpace.sum(-2).T]).T
        # check for 6
        self.playerWins = (np.isin(stateWins,[6]).any(-1))
        # check for 0
        self.opponentWins = (np.isin(stateWins,[0]).any(-1))
        # see if there are any blank fields left in each matrix
        self.boardFull = np.isin(stateSpace,[1],invert = True).reshape((-1,9)).all(-1)
        
        self.state_mat = np.ones((1,3,3),dtype = int)
        self.done = False
        
    def state_to_ind(self,s_mat):
        #flatten and convert to base 10 - sum over powers of 3 is handled by @ ->vector multiplication
        return s_mat.reshape(-1,9) @ 3 ** np.arange(8, -1, -1)
    
    def ind_to_state(self, s):
        #--------|--index----|-provoke broadcasting-|-array of powers of 3-|-'a'
        return ((np.array(s).reshape((-1,1)) // (3 ** np.arange(8, -1, -1))) % 3).reshape((-1,3,3))
    
    def reset(self):
        self.state_mat = np.ones((1,3,3),dtype = int)
        self.done = False
        
    def play(self, s_mat, a, p):
        s_next = s_mat.copy()
        s_next[a] = p
        ind_next = self.state_to_ind(s_next)

        #if it was X's turn
        if p == 2:
            #is there a win?
            if self.playerWins[ind_next]:
                #observation, reward, done
                return s_next, 1, True
            #is there a draw?
            elif self.boardFull[ind_next]:
                return s_next, 0, True
        #if it was O's turn
        else:
            #did O win?
            if self.opponentWins[ind_next]:
                return s_next, 0, True

        #all other cases: neither win nor draw, the game continues!
        return s_next, 0, False
    
    def get_available_actions(self, p=2):
        ###where can the player p draw a symbol?
        actions = np.argwhere(self.state_mat == 1)
        ###which states are reached this way?
        states_next = [self.play(self.state_mat, tuple(a), p)[0] for a in actions]
        states_next = np.concatenate(states_next)
        return self.state_to_ind(states_next), actions
    
    def step(self, a, player=2):
        assert not self.done, "The episode is already over!"
        # execute move for cross
        s_next_mat, r, self.done = self.play(self.state_mat, a, player)
        self.state_mat = s_next_mat
        return s_next_mat, r, self.done