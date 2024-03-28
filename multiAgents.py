# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        # Compute distance to the nearest food
        foodDist = [manhattanDistance(newPos, food) for food in newFood.asList()]
        nearestFoodDist = min(foodDist) if foodDist else 0

        # Compute distance to the ghosts
        ghostDist = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        nearestGhostDist = min(ghostDist) if ghostDist else 0

        # Check if any ghost is scared
        isScared = any(newScaredTimes)

        # Compute the number of remaining food
        remainingFood = len(newFood.asList())

        # Prioritize states where Pacman is closer to food and further from ghosts
        # If a ghost is scared, prioritize states where Pacman is closer to the ghost
        # Also, prioritize states with less remaining food
        score = successorGameState.getScore()
        score += -2 * nearestFoodDist + (nearestGhostDist if not isScared else -nearestGhostDist) - 100 * remainingFood

        return score
        return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        def minimax_decision(state, agent, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agent == 0:  # Pacman's turn (Maximizer)
                return max(minimax_decision(state.generateSuccessor(agent, action), agent + 1, depth) for action in state.getLegalActions(agent))
            elif agent == state.getNumAgents() - 1:  # Last ghost's turn (Minimizer)
                return min(minimax_decision(state.generateSuccessor(agent, action), 0, depth + 1) for action in state.getLegalActions(agent))
            else:  # Other ghosts' turn (Minimizer)
                return min(minimax_decision(state.generateSuccessor(agent, action), agent + 1, depth) for action in state.getLegalActions(agent))

        # Start with Pacman's turn at depth 0
        return max(gameState.getLegalActions(0), key=lambda action: minimax_decision(gameState.generateSuccessor(0, action), 1, 0))

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def alpha_beta_search(state, agent, depth, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agent == 0:
                v = float('-inf')
                for action in state.getLegalActions(agent):
                    v = max(v, alpha_beta_search(state.generateSuccessor(agent, action), agent + 1, depth, alpha, beta))
                    if v > beta:
                        return v
                    alpha = max(alpha, v)
                return v
            else:
                v = float('inf')
                for action in state.getLegalActions(agent):
                    if agent == state.getNumAgents() - 1:
                        v = min(v, alpha_beta_search(state.generateSuccessor(agent, action), 0, depth + 1, alpha, beta))
                    else:
                        v = min(v, alpha_beta_search(state.generateSuccessor(agent, action), agent + 1, depth, alpha, beta))
                    if v < alpha:
                        return v
                    beta = min(beta, v)
                return v
            
        alpha = float('-inf')
        beta = float('inf')
        best_action = None
        v = float('-inf')
        for action in gameState.getLegalActions(0):
            next_v = alpha_beta_search(gameState.generateSuccessor(0, action), 1, 0, alpha, beta)
            if next_v > v:
                v = next_v
                best_action = action
            alpha = max(alpha, v)
        return best_action
        util.raiseNotDefined()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        def expectimax(state, agent, depth):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            if agent == 0:  # Pacman's turn (Maximizer)
                return max(expectimax(state.generateSuccessor(agent, action), agent + 1, depth) for action in state.getLegalActions(agent))
            elif agent == state.getNumAgents() - 1:  # Last ghost's turn (Expectation)
                return sum(expectimax(state.generateSuccessor(agent, action), 0, depth + 1) for action in state.getLegalActions(agent)) / len(state.getLegalActions(agent))
            else:  # Other ghosts' turn (Expectation)
                return sum(expectimax(state.generateSuccessor(agent, action), agent + 1, depth) for action in state.getLegalActions(agent)) / len(state.getLegalActions(agent))

        # Start with Pacman's turn at depth 0
        return max(gameState.getLegalActions(0), key=lambda action: expectimax(gameState.generateSuccessor(0, action), 1, 0))
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    This function evaluates the desirability of a game state for Pacman.
    Consider the nearest food, distance to the nearest ghost, number of scared ghosts, remaining scared time, and number of remaining food positions.
    This function tries to maximize Pacman game score not being caught by a ghost.
    The function returns the score, which can be used to select the best action for Pacman.
    """
    "*** YOUR CODE HERE ***"
    # Get Pacman's position, food positions, ghost states, and scared times
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # Compute distances to nearest food and ghost
    foodDist = [manhattanDistance(newPos, food) for food in newFood.asList()]
    nearestFoodDist = min(foodDist) if foodDist else 0
    ghostDist = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
    nearestGhostDist = min(ghostDist) if ghostDist else 0

    # Count scared ghosts and get maximum scared time
    numScaredGhosts = sum(scaredTime > 0 for scaredTime in newScaredTimes)
    remainingScaredTime = max(newScaredTimes) if newScaredTimes else 0

    # Compute remaining food
    remainingFood = len(newFood.asList())

    # Calculate score
    score = currentGameState.getScore()
    score += -2 * nearestFoodDist + (nearestGhostDist if numScaredGhosts == 0 else -2 * nearestGhostDist) - 100 * remainingFood + 50 * numScaredGhosts + remainingScaredTime

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
