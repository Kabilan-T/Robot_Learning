import numpy as np
import matplotlib.pyplot as plt

# All code taken and adapted from OpenAI Gym's Blackjack Env source code!


def cmp(a, b):
    return float(a > b) - float(a < b)


# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
deck = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


def draw_card():
    return int(np.random.choice(deck))


def draw_hand():
    return [draw_card(), draw_card()]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]


class BlackjackEnv():
    """Simple blackjack environment
    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with dealer having one face up and one face down card, while
    player having two face up cards. (Virtually for all Blackjack games today).
    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.
    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.
    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).
    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto.
    http://incompleteideas.net/book/the-book-2nd.html
    """
    def __init__(self, natural=False):
        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game
        self.reset()

    def getAvailableActions(self):
        return np.array([0,1])
    
    def step(self, action):
        if action:  # hit: add a card to players hand and return
            self.player.append(draw_card())
            if is_bust(self.player):
                done = True
                reward = -1.
            else:
                done = False
                reward = 0.
        else:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card())
            reward = cmp(score(self.player), score(self.dealer))
            if self.natural and is_natural(self.player) and reward == 1.:
                reward = 1.5
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        return (sum_hand(self.player), self.dealer[0], int(usable_ace(self.player)))

    def reset(self):
        self.dealer = draw_hand()
        self.player = draw_hand()
        return self._get_obs()
    
# helper function to pretty-print policy, black is hit, white is stick
def plotPolicy(Q):    
    fig, (ax1, ax2) = plt.subplots(1,2, constrained_layout=True)
    policy = np.argmax(Q[12:22,1:,1,:],axis = -1)
    cax1 = ax1.matshow(policy, cmap = 'Greys',origin='lower', aspect="auto")
    xticks = np.arange(0,10,1)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(['A']+["{}".format(int(x)+1) for x in np.arange(1,10,1)])
    ax1.set_xlabel("Dealer showing")
    
    yticks = np.arange(0,10,1)
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(["{}".format(int(x)+12) for x in yticks])
    ax1.set_ylabel("Player sum")
    
    ax1.set_title('Usable Ace')
    ax1.yaxis.tick_right()
    ax1.yaxis.set_label_position("right")
    ax1.xaxis.tick_bottom()
    
    
    policy = np.argmax(Q[11:22,1:,0,:],axis = -1) 
    #black is hit, white is stick
    cax2 = ax2.matshow(policy, cmap = 'Greys',origin='lower', aspect="auto")
    xticks = np.arange(0,10,1)
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(['A']+["{}".format(int(x)+1) for x in np.arange(1,10,1)])
    ax2.set_xlabel("Dealer showing")
    
    yticks = np.arange(0,11,1)
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(["{}".format(int(x)+11) for x in yticks])
    ax2.set_ylabel("Player sum")
    
    ax2.set_title('No Usable Ace')
    ax2.yaxis.tick_right()
    ax2.yaxis.set_label_position("right")
    ax2.xaxis.tick_bottom()
    plt.show()