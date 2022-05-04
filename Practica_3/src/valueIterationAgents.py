# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

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
        self.runValueIteration()
         

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        iterations=0 #Ponemos las iteraciones a 0
        while iterations < self.iterations:
            valsPi = util.Counter()#Contador para saber los valores en funcion de la iteracion
            states=self.mdp.getStates()#Todos los estados posibles
            for state in states:#Recorremos todos los estados
                if not self.mdp.isTerminal(state):#Si el estado no es terminal 
                    valuesTer = util.Counter()#Inicializamos otro contador
                    actions = self.mdp.getPossibleActions(state)#Sacamos las posibles acciones
                    for action in actions:#Recorremos todas las acciones
                        valuesTer[action] = self.computeQValueFromValues(state, action)#Calculamos el q valor de cada accion
                    valsPi[state] = max(valuesTer.values())#Guardamos el valor maximo de cada estado
            iterations += 1#Aumentamos las iteraciones
            self.values = valsPi.copy() #Copiamos los valores para la siguiente iteracion


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
        statesAndProbs=self.mdp.getTransitionStatesAndProbs(state,action)#Sacamos tuplas con el estado y la probabilidad de transicion
        currVal=0#Inicializamos el valor a 0
        for pair in statesAndProbs:#Recorremos los pares de estado y probabilidad
            currVal+=pair[1]*(self.mdp.getReward(state,action,pair[0])+self.discount*self.values[pair[0]])#Calculamos el qValor con la formula
        return currVal


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):#Si no hay acciones a seguir porque es estado terminal devolvemos none
            return None
        actions=self.mdp.getPossibleActions(state)#Extraemos las acciones posibles
        if len(actions) == 0:#Si no hay acciones por lo que sea tambien devolvemos none
            return None
        values = util.Counter()#Contador con los valores que vamos a tener
        for action in actions:#Parca cada accion que tengamos
            values[action] = self.computeQValueFromValues(state, action)#Calculamos su q valor
        return values.argMax()#Devolvemos la accion que hace que el valor sea maximo

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        value = 0
        maxValue = 99999999999999999
        states = self.mdp.getStates()
        iterations = self.iterations
        for i in range(iterations):
            current = util.Counter()
            length = len(states)
            state = states[i%length]
            if not self.mdp.isTerminal(state):
                maxValues = []
            else:
                current[state] = 0
            actions = self.mdp.getPossibleActions(state)
            for action in actions:
                value = 0
                nextAction = next
                statesAndProbabilities = self.mdp.getTransitionStatesAndProbs(state,action)
                for nextAction, prob in statesAndProbabilities:
                    reward = self.mdp.getReward(state,action,nextAction)
                    discount = self.discount
                    value = value + prob * (reward + discount*self.values[nextAction])
                maxValues = maxValues + [value]
                if value > maxValue:
                    maxValue = value
                length = len(maxValues)
                if length!=0:
                    current[state] = max(maxValues)
            self.values[state] = current[state]

class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        fringe = util.PriorityQueue()
        states = self.mdp.getStates()#Obtener todos los estados
        predecessors = {} # Crear un nuevo diccionario vacío para los predecesores
        for tState in states:#Recorremos todos los estados
            previous = set()# Inicializar el conjunto, sin elementos duplicados en el conjunto
            for state in states:
                actions = self.mdp.getPossibleActions(state)#Sacamos las acciones posibles para cada estado
                for action in actions:#Recorremos todas las acciones
                    transitions = self.mdp.getTransitionStatesAndProbs(state, action)#Tuplas con las transiciones y sus probabliidades
                    for next, probability in transitions:#Recorremos cada elemento de ese conjunto de tuplas
                        if probability != 0:#Si no hay probabilidad de ir a ese estado
                            if tState == next:#Si ya hemos recorrido ese estado
                                previous.add(state)#Añadimos a recorridos
            predecessors[tState] = previous#Añadimos al array de predecesores el estado que acabamos de recorrer
        for state in states:#Volvemos a iterar sobre los estados
            if self.mdp.isTerminal(state) == False:#Si el estado actual no es terminal
                current = self.values[state]#Extraemos el valor del estado actual
                qValues = []#Inicilizamos el array de qvalores vacio
                actions = self.mdp.getPossibleActions(state)#Extraemos las accione sposibles
                for action in actions:#Recorremos esas acciones
                    tempValue = self.computeQValueFromValues(state, action)#Calculamos los qvalores para esas acciones
                    qValues = qValues + [tempValue]#Añadimos los qvalores a la lista
                maxQvalue = max(qValues)#Extraemos la accion que maximiza ese qvalor
                diff = current - maxQvalue#Calculamos el valor real restandole el mayor qvalor al valor actual
                if diff > 0:#Si el valor real es mayor que 0
                    diff = diff * -1#Lo multiplicamos por -1 para convertirlo a negativo
                fringe.push(state, diff)#Añadimos el estado actual a la cola de prioridad que es nuestra franja de trabajo
        for i in range(0, self.iterations):
            if fringe.isEmpty():#sI NO TENEMOS ESTADOS DONDE TRABAJAR
                break#Paramos las iteraciones
            s = fringe.pop()#Extraemos el primero elemento de la cola
            if not self.mdp.isTerminal(s):#Si no es un estado terminal
                values = []#Inicializamos lista de valores 
                for action in self.mdp.getPossibleActions(s):#Recorremos las acciones posibles
                    value = 0
                    for next, prob in self.mdp.getTransitionStatesAndProbs(s, action):#Recorremos cada estado siguiente y la probabilidad que este
                        reward = self.mdp.getReward(s, action, next)
                        value = value + (prob * (reward + (self.discount * self.values[next])))#Calculamos el valor del estado con la formula
                    values.append(value)#Añadimos a la lista de valores
                self.values[s] = max(values)#Nos quedamos con el valor mas grande
            for previous in predecessors[s]:#Recorremos los estados anteriores al actual
                current = self.values[previous]#Extraemos los valores de ese estado anterior
                qValues = []
                for action in self.mdp.getPossibleActions(previous):#Vemos las acciones del estado anterior
                    qValues += [self.computeQValueFromValues(previous, action)]#Calculamos los qvalores de el predecesor
                maxQ = max(qValues)#Nos quedamos con el valor maximo
                diff = abs((current - maxQ))#Obtenemos la diferencia de lo extraido y lo calculado
                if (diff > self.theta):
                    fringe.update(previous, -diff)#Actualizamos la franja de trabajo con el estado anterior dandole como prioridad la diferencia calculada previamente

       
   
       
