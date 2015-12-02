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
        self.mdp         = mdp
        self.discount    = discount
        self.iterations  = iterations
        self.values      = util.Counter() # A Counter is a dict with default 0

        # NOTE: The naming conventions are a bit off. This is to emphasize the
        #       connection between code and the formulas we were provided with.
        #       Enjoy!

        import sys

        values_temp = util.Counter()
        for i in range(iterations):
            for state in mdp.getStates():
                Q_max = -sys.maxint

                # A terminal state has no actions, so we must be careful to
                # reset the value to zero here.
                if mdp.isTerminal(state):
                    Q_max = 0.0

                # This is a trivial loop to find the 'best' possible action in
                # the current state, according to computed Q values.  This is
                # essentially the Pythonic way of saying the following:
                #   V_k+1(s) <- max Q(s, a)
                for action in mdp.getPossibleActions(state):
                    Q = self.getQValue(state, action)
                    if Q > Q_max:
                        Q_max = Q

                values_temp[state] = Q_max

            # Store the new values.
            self.values = values_temp
            values_temp = util.Counter()


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

        Q = 0.0

        for s_prim, T in self.mdp.getTransitionStatesAndProbs(state, action):
            s      = state
            R      = self.mdp.getReward(s, action, s_prim)
            V      = self.values[s_prim]
            gamma  = self.discount

            # It should be obvious what formula we're implementing here! :-)
            Q += T * (R + gamma * V)

        return Q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        import sys

        policy_action = None
        Q_max         = -sys.maxint

        # Below; find the highest Q value and return the action associated with
        # it, which is essentially the policy we're after.  This method will
        # automatically return None for terminal states since they have no
        # possible actions.  We break ties by returning the action we encounter
        # first which is an amazingly cheap and lousy solution, but the
        # assignments requires nothing more of us. :-)
        for action in self.mdp.getPossibleActions(state):
            Q = self.getQValue(state, action)
            if Q > Q_max:
                Q_max         = Q
                policy_action = action

        return policy_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
