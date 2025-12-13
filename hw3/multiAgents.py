from util import manhattanDistance
from game import Directions
import random, util
from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and child states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        The evaluation function takes in the current and proposed child
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        """
        # Useful information you can extract from a GameState (pacman.py)
        childGameState = currentGameState.getPacmanNextState(action)
        newPos = childGameState.getPacmanPosition()
        newFood = childGameState.getFood()
        newGhostStates = childGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        minGhostDistance = min([manhattanDistance(newPos, state.getPosition()) for state in newGhostStates])

        scoreDiff = childGameState.getScore() - currentGameState.getScore()

        pos = currentGameState.getPacmanPosition()
        nearestFoodDistance = min([manhattanDistance(pos, food) for food in currentGameState.getFood().asList()])
        newFoodsDistances = [manhattanDistance(newPos, food) for food in newFood.asList()]
        newNearestFoodDistance = 0 if not newFoodsDistances else min(newFoodsDistances)
        isFoodNearer = nearestFoodDistance - newNearestFoodDistance

        direction = currentGameState.getPacmanState().getDirection()
        if minGhostDistance <= 1 or action == Directions.STOP:
            return 0
        if scoreDiff > 0:
            return 8
        elif isFoodNearer > 0:
            return 4
        elif action == direction:
            return 2
        else:
            return 1


def scoreEvaluationFunction(currentGameState):
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
    Your minimax agent (Part 1)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.getNextState(agentIndex, action):
        Returns the child game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        # Begin your code (Part 1)
        """
        The minimax function recursively explores the game tree to determine the best move for Pacman
        based on the current state and the specified depth. I first check if the depth is zero or if the current 
        state is a winning or losing state. If so, it returns the evaluation of the state. Next, I calculate the 
        next depth and the next agent to consider, cycling through the players (Pacman and Ghosts).
        The function then retrieves the legal actions that the current agent can take in the given state.
        For each legal action, it generates the next possible states by applying the action to the current 
        state. It recursively calls the minimax function on each of these next states with the updated depth
        and agent, obtaining their corresponding values. If the current agent is Pacman, it selects the 
        maximum value from next value and the corresponding index. Otherwise, if it's a Ghost, it selects 
        the minimum value. Finally, it returns the best value found and the corresponding legal action at 
        best index, representing the optimal move for Pacman at the current state.
        """
        def minimax(state, depth, agent):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            nxt_dep = depth
            nxt_agent = (agent + 1) % state.getNumAgents()
            if agent == state.getNumAgents() - 1:
                nxt_dep = depth - 1
            legal_actions = state.getLegalActions(agent)
            nxt_states = [state.getNextState(agent, action) for action in legal_actions]
            nxt_val = [minimax(nxt_state, nxt_dep, nxt_agent)[0] for nxt_state in nxt_states]
            if agent == 0:  # pacman
                best_val = max(nxt_val)
                best_index = nxt_val.index(best_val)
            else:  # ghost
                best_val = min(nxt_val)
                best_index = nxt_val.index(best_val)
            return best_val, legal_actions[best_index]
        _, best_action = minimax(gameState, self.depth, 0)
        return best_action
        #raise NotImplementedError("To be implemented")
        # End your code (Part 1)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (Part 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        # Begin your code (Part 2)
        """
        First, I check if the depth is zero or if the current state is a winning or losing state, if so, it returns
        the evaluation of the state. It then determines the next agent to move and increments the depth if 
        all agents have moved. For Pacman, I initialize best score to –infinity and loop through all legal 
        actions for Pacman, calling the function recursively for the next state. It updates best value with 
        the maximum of its current value and the result of the recursive call. If best value becomes greater 
        than beta, it returns best value as the minimizing player will never choose this branch. It also 
        updates alpha with the maximum of its current value and best value. For ghosts, I initialize best 
        value to +infinity. It also loops through all legal actions for the ghost agent, similarly updating best 
        value and beta, by just changing the use of maximum to minimum. If best value becomes less than 
        alpha, it returns best value as the maximizing player will not choose this branch. Then, I set alpha
        and beta to negative and positive infinity, respectively, initialize max value to negative infinity and 
        best action to the first legal action for Pacman. After that I write a try block which is used to 
        execute the process of evaluating Pacman's best move using the Alpha-Beta Pruning algorithm. It 
        encapsulates the loop that considers all legal actions for Pacman, calculates their utilities, and 
        updates the best action accordingly. The purpose of the try block is to handle any potential 
        exceptions that might occur during this process, ensuring the program continues to run smoothly 
        even if errors arise. Finally, the function returns the best action found, which is the action that 
        leads to the highest utility for Pacman based on the Alpha-Beta Pruning algorithm. This action 
        represents Pacman's best move in the current game state.
        """
        def alphabeta_prune(agent, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            nxt_agent = agent + 1 
            if gameState.getNumAgents() == nxt_agent:
                nxt_agent = 0
                depth += 1
            if agent == 0:  # pacman (max player)
                best_val = float("-inf")
                for action in gameState.getLegalActions(agent):
                    best_val = max(best_val, alphabeta_prune(nxt_agent, depth, gameState.getNextState(agent, action), alpha, beta))
                    if best_val > beta:
                        return best_val
                    alpha = max(alpha, best_val)
                return best_val
            else:  # ghost (min player)
                best_val = float("inf")
                for action in gameState.getLegalActions(agent):
                    best_val = min(best_val, alphabeta_prune(nxt_agent, depth, gameState.getNextState(agent, action), alpha, beta))
                    if best_val < alpha:
                        return best_val
                    beta = min(beta, best_val)
                return best_val
        alpha = float("-inf")
        beta = float("inf")
        max_val = float("-inf")
        best_action = gameState.getLegalActions(0)[0]
        try:
            for action in gameState.getLegalActions(0):
                best_val = alphabeta_prune(1, 0, gameState.getNextState(0, action), alpha, beta)
                if best_val > max_val:
                    max_val = best_val
                    best_action = action
                alpha = max(alpha, max_val)
        except Exception as e:
            print("Exception:", e)
        return best_action
        #raise NotImplementedError("To be implemented")
        # End your code (Part 2)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (Part 3)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        # Begin your code (Part 3)
        """
        First, I check if the depth is 0 or if the current state is in a winning or losing condition. If either 
        condition is met, it returns the evaluation of the state. Next, I calculate the next depth and next 
        agent for the next recursive call and handle if the current agent is the last in the list. Then the
        function retrieves the legal actions for the current agent, representing the possible actions in the 
        current state. For each legal action, it calculates the next states by applying the action to the state.
        Next, it recursively calls expectimax function for each next state to calculate the expected values 
        next value. If the current agent is Pacman, it selects the action with the maximum expected value. 
        It finds the best value as the maximum of the next value, then determines the index of the 
        maximum value in the next value. The function returns the best value and the corresponding legal 
        action. If the current agent is not Pacman, it calculates the average of the next value to represent 
        the expected value for that agent's turn. Finally, the function recursively starts with Pacman and 
        determines the best action for Pacman's turn by calling the function. This best action represents 
        Pacman's best move in the current game state, considering the Expectimax search's evaluation of 
        possible outcomes and chance events.
        """
        def expectimax(state, depth, agent):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            nxt_dep = depth
            nxt_agent = (agent + 1) % state.getNumAgents()
            if agent == state.getNumAgents() - 1:
                nxt_dep = depth - 1
            legal_actions = state.getLegalActions(agent)
            nxt_states = [state.getNextState(agent, action) for action in legal_actions]
            nxt_val = [expectimax(nxt_state, nxt_dep, nxt_agent)[0] for nxt_state in nxt_states]
            if agent == 0:
                best_val = max(nxt_val)
                best_index = nxt_val.index(best_val)
                return best_val, legal_actions[best_index]
            else:
                return sum(nxt_val)/len(nxt_val), 0
        _, best_action = expectimax(gameState, self.depth, 0)
        return best_action
        #raise NotImplementedError("To be implemented")
        # End your code (Part 3)


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (Part 4).
    """
    # Begin your code (Part 4)
    """
    I consider several factors, including the current score, Pacman's position, the positions of food 
    pellets and capsules, as well as information about ghosts and weights assigned to these factors. 
    The score is computed based on these considerations to aid Pacman in making decisions.
    For score calculation, I set the initial score to the current game score, and store Pacman's current 
    position in ‘position’, remaining food pellets in 'food', capsule positions in 'capsule' and ghost 
    states in 'ghost state'. For the new weight, I set food weight to 10, capsule weight to 50,
    ghost weight to -30 and scared ghost weight to 300. Next, I calculate the closest capsule distance
    by figuring out how close Pacman is to the closest capsule, and then adds the value to Pacman's 
    score depending on this distance. If no capsule is nearby, add the food weight to the score.
    Similarly, I use the same way to calculate closest food distance by figuring out how close Pacman is 
    to the closest food, and then adds the value to Pacman's score depending on this distance. If no 
    food is nearby, add the food weight to the score. For the ghosts, I calculate Pacman's distance to 
    each ghost. If a ghost isn't at Pacman's position and the ghost is scared, add scared ghost weight
    divided by the distance to the score. If the ghost isn't scared, add ghost weight divided by the 
    distance to the score. Finally, it returns the final score.
    """
    score = currentGameState.getScore()
    position = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    capsule = currentGameState.getCapsules()
    ghost_state = currentGameState.getGhostStates()
    food_weight = 10
    capsule_weight = 50
    ghost_weight = -30
    scared_ghost_weight = 300
    closest_capsuleDist = float("inf")
    for capsulePosition in capsule:
        capsuledist = manhattanDistance(position, capsulePosition)
        closest_capsuleDist = min(closest_capsuleDist, capsuledist)
    if closest_capsuleDist  < float("inf"):
        score += capsule_weight / closest_capsuleDist
    else:
        score += food_weight
    closest_foodDist = float("inf")
    for foodPosition in food.asList():
        fooddist = manhattanDistance(position, foodPosition)
        closest_foodDist = min(closest_foodDist, fooddist)
    if closest_foodDist < float("inf"):
        score += food_weight / closest_foodDist
    else:
        score += food_weight
    for ghost in ghost_state:
        dist = manhattanDistance(position, ghost.getPosition())
        if dist > 0:
            if ghost.scaredTimer > 0:
                score += scared_ghost_weight / dist
            else:
                score += ghost_weight / dist
    return score
    #raise NotImplementedError("To be implemented")
    # End your code (Part 4)

# Abbreviation
better = betterEvaluationFunction
