import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.newValues = util.Counter()
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        import math as m
        import copy as c
        import sys
        """

        for state in self.mdp.getStates():
            if not self.mdp.getPossibleActions(state) == ('exit',):
                self.values[state] = 0



        for i in range(iterations):
            #delta = -sys.maxint
            for state in self.mdp.getStates():

                self.newValues[state] = self.values[state]

                for direction in self.mdp.getPossibleActions(state):
                    self.newValues[state] = self.getQValue(state, direction)
                    if self.newValues[state] > self.values[state]:
                        if self.mdp.getPossibleActions(state) == ('exit',):
                            self.values[state] = self.newValues[state]
                        else:
                            self.values[state] += self.newValues[state]
                            self.values[state] /= 2.0

                #delta = max(delta, self.values[state])
                #self.values[state] = delta

        """
        self.newValues = c.deepcopy(self.values)
        for i in range(iterations - 1):
            count = 0
            for state in reversed(self.mdp.getStates()):
                if count == i*i: break
                currentHigh = -sys.maxint

                for direction in self.mdp.getPossibleActions(state):
                    temp = self.getQValue(state, direction)
                    if temp > currentHigh:
                        currentHigh = temp

                self.newValues[state] = currentHigh
                count += 1
            self.values = self.newValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #util.raiseNotDefined()
        value = 0
        for i in self.mdp.getTransitionStatesAndProbs(state, action):
            stateAndProbs = i

            nextState = stateAndProbs[0]
            probs = stateAndProbs[1]
            if self.mdp.isTerminal(nextState):
                value += self.mdp.getReward(state, 'exit', nextState)
            else:
                value += probs*(self.discount * self.values[nextState])
        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        import sys
        currentBest = -sys.maxint
        currentBestDir = None

        #if self.mdp.isTerminal(state): return 0

        for direction in self.mdp.getPossibleActions(state):
            temp = self.getQValue(state, direction)
            if temp > currentBest:
                currentBest = temp
                currentBestDir = direction

        return currentBestDir


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
