# qlearningAgents.py
# ------------------
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


from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """
    def __init__(self, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.qVals = util.Counter()#Inicilizamos

        "*** YOUR CODE HERE ***"

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        "*** YOUR CODE HERE ***"
        return self.qVals[(state, action)]#Devolver los qvalore de cada estado y accion posible


    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)#Sacamos todas las acciones posibles
        values = []#Lista con los valores
        if len(actions) == 0:#Si no tenemos acciones posibles devolvemos 0 como valor
            return 0
        else:
            for action in actions:#Recorremos cada accion
                values.append(self.getQValue(state, action))#Añadimos a nuestra lista de valores el calculo del q valor de cada accion
        return max(values)#Devolvemos ese qvalor maximo

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        "*** YOUR CODE HERE ***"
        actions = self.getLegalActions(state)#Sacamos las acciones posibles
        allActions = []
        if len(actions) == 0:#Si no tenemos acciones posibles devolvemos 0 como valor
            return None
        else:
            for action in actions:#Recorremos cada accion
                allActions.append((self.getQValue(state, action), action))#Metemos en una lista el qvalor y la accion que representa ese qvalor
            bestActions = [pair for pair in allActions if pair == max(allActions)] #Nos quedamos con la accion que mejor nos convenga
            bestActionPair = random.choice(bestActions)#De las mejore acciones elegimos una al azar para explorar
        return bestActionPair[1]#Una vez explorado nos quedamos con la mejor

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        legalActions = self.getLegalActions(state)#Sacamos las acciones
        p = self.epsilon#inicializamos epsilon
        if util.flipCoin(p):#Aleatoriamente elegimos si exploramos o si nos quedamos con la mejor accion
            return random.choice(legalActions)#Devolvemos accion aleatoria de todas las disponibles
        else:#Si no devolvemos la mejor accion 
            return self.computeActionFromQValues(state)

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
        """
        "*** YOUR CODE HERE ***"
        qSa = self.getQValue(state, action)#Sacamos el qvalor de la accion
        sample = reward + self.discount*self.computeValueFromQValues(nextState)#Calculamos la muestra
        self.qVals[(state, action)] = (1-self.alpha)*qSa + self.alpha*sample#Actualizamos la lista de qvalores con la formula 

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        return self.computeValueFromQValues(state)


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        alpha    - learning rate
        epsilon  - exploration rate
        gamma    - discount factor
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self,state)
        self.doAction(state,action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """
    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.weights = util.Counter()

    def getWeights(self):
        return self.weights#Devolvemos los pesos

    def getQValue(self, state, action):
        """
          Should return Q(state,action) = w * featureVector
          where * is the dotProduct operator
        """
        "*** YOUR CODE HERE ***"
        features = self.featExtractor.getFeatures(state,action)#Devolvemos las caracterisiticas de cada estado y accion posible
        weights = self.getWeights()#Llamamos a esos pesos para evaluar
        dotProduct = features*weights#Miramos si el peso y las caracteristicas del estado nos compensa para ir a el
        return dotProduct

    def update(self, state, action, nextState, reward):
        """
           Should update your weights based on transition
        """
        "*** YOUR CODE HERE ***"
        difference = reward + self.discount*self.computeValueFromQValues(nextState) - self.getQValue(state, action)
        weights = self.getWeights()
        if len(weights) == 0:#Si no tenemos pesos en nuestra lista
            weights[(state,action)] = 0#Metemos 0 para el peso de ese estado
        features = self.featExtractor.getFeatures(state, action)#Extraemos las caracterisiticas de el estado
        for key in features.keys():#Recorremos esas caracteristicas
            features[key] = features[key]*self.alpha*difference
        weights.__radd__(features)#Recalculamos el peso con la suma de esas caracterisiticas ajustadas
        self.weights = weights.copy()#Actualizamos los pesos globales

    def final(self, state):
      "Called at the end of each game."
      PacmanQAgent.final(self, state)#Llamamos al constructor
      if self.episodesSoFar == self.numTraining:#Revisa si ya hemos hecho todos los entrenamientos
            pass
