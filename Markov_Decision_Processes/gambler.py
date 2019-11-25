
import numpy as np

from gym.envs.toy_text import discrete


class GamblerEnv(discrete.DiscreteEnv):
    """

    """

    def __init__(self, size=1000, ph=0.4):

        nS = size + 1
        nA = size // 2 + 1

        self.nrow = nS
        self.ncol = 1

        isd = np.zeros(nS)
        isd[1] = 1.

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}

        for state in range(1, nS - 1):
            for action in range(nA):

                if action == 0:
                    P[state][action].append((1.0, 0, 0.0, True))
                elif action <= min(state, nS - 1 - state):
                    new_state = state + action
                    done = new_state == nS - 1
                    reward = float(done)
                    P[state][action].append((ph, new_state, reward, done))

                    new_state = state - action
                    done = new_state == 0
                    reward = 0.0
                    P[state][action].append((1.-ph, new_state, reward, done))

        super(GamblerEnv, self).__init__(nS, nA, P, isd)

    def render(self):

        print('Last Stake = {}'.format(self.last_action))
        print('Capital = {}'.format(self.s))
