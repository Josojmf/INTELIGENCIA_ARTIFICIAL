# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's succ:", problem.getsucc(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"
    actual = problem.getStartState()#Definimos el primer nodo raiz
    visited = []#Array que contendra los nodos ya visitados
    stripe = util.Stack()#Defino la pila que contendrá todos los nodos de la franja actual, estos nodos son los que iremos desplegando en cada iteracion 
    stripe.push((actual, []))#Incluyo el nodo raiz en la pila

    while not stripe.isEmpty(): #Mientras que la franja de trabajo no este vacia
        actual, path = stripe.pop()
        visited.append(actual)#Marco como visitado el nodo actual 

        if problem.isGoalState(actual):#Si el nodo actual es el estado meta 
            return path#Terminamos y devuelvo el camino que hemos obtenido

        for succ in problem.getSuccessors(actual):#Desplegamos el nodo anterior y miramos sus sucesores
            if succ[0] not in visited:#Comprobamos que el nodo mas a la izquierda no este visitado  
                stripe.push((succ[0], path + [succ[1]]))#Si no está visitado lo añadimos a la franja de trabajo e incluimos el siguiente en el camino

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    actual=problem.getStartState()#Definimos el primer nodo raiz
    stripe = util.Queue()#Definimos una pila que contendrá los nodos de la franaj que se irán desplegando
    visited = []#Array con estados ya visitados
    path = []#Array con el camino que se ha ido siguiendo
    stripe.push((actual, []))#Encolamos el estado actual(inicial)
    visited.append(actual)#Marcamos como visitado el nodo raiz

    while not stripe.isEmpty():#Mientras que franja de trabajo no este vacia
        actual, path = stripe.pop()

        if problem.isGoalState(actual):#Si el nodo actual es el nodo meta
            return path #Terminamos y devolvemos el camino hasta l nodo actual

        else:

            for succ, direction,cost in problem.getSuccessors(actual):#Desplegamos los sucesores de mi nodo acatual

                if not succ in visited:#Para sucesor no visitado(En este caso miramos todos los sucesores y nos solo el de mas a alq izquierda)
                    visited.append(succ)#Marcamos cada sucesor como visitado
                    stripe.push((succ, path + [direction]))#Lo incluimos en la franja junto al camino previo y la direccion que hemos tomado

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    stripe = util.PriorityQueue() #DEfinimos una cola con prioridad como nuestra estructura de datos
    stripe.push((problem.getStartState(),[],0),0) #Introducimos en la cola el nodo inicial, las direcciones a seguir y el coste
                                                  #inicial como primer argumento, y la prioridad 0 como segundo
    visited = [] #Declaramos un array que contendrá los nodos visitados

    while not stripe.isEmpty(): # Mientras que la franja de trabajo no esté vacia 
        node, actions, accCost = stripe.pop() # Asignaremos el primer elemento de la fila a las variables node actions y coste 

        if(not node in visited): #Si el primer nodo de la cola no esta visitadp
            visited.append(node) #Lo marcamos como visitado

            if problem.isGoalState(node): #Comprobamos si hemos encontrado el nodo destino
                return actions # Si es asi devolvemos las acciones que hemos tomado hasta encontrar ese nodo

            for child, direction, cost in problem.getSuccessors(node): #Si no es el destino expandimos el nodo y sus sucesores
                stripe.push((child, actions+[direction],accCost + cost),accCost + cost) #Metemos en la franja de trabajo cada sucesor, las acciones
                                                                                          #previas y la nueva que hay que tomar,el coste acumulado, 
                                                                                          #y por ultimo le indicamos la prioridad con el coste acumulado,
                                                                                          # asi siempre tomara el camino con menos coste

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    
    "*** YOUR CODE HERE ***"
    stripe = util.PriorityQueue() #Definimos una cola con prioridad como estructura de datos
    stripe.push((problem.getStartState(),[],0),heuristic(problem.getStartState(), problem)) #Metemos el nodo raiz con el conjunto vacio de acciones tomadas y 
                                                                                            #coste 0 por ser el primero en la cola
    visited = [] #Declaramos un array vacio que sera el conjunto de nodos visitados
    while not stripe.isEmpty(): #Mientras que la franja de trabajo no este vacia
        node, actions, accCost = stripe.pop() #Sacamos el nodo actual,las acciones tomadas hasta el momento y el coste accumulado total

        if(not node in visited): #Si no esta visitado ya el nodo
            visited.append(node) #LO marcamos como visitado

            if problem.isGoalState(node): #Si es el nodo destino
                return actions #Devolvemos las acciones tomadas hasta ese punto

            for child, direction, cost in problem.getSuccessors(node): #Expandimos el nodo actual y sus sucesores
                g = accCost + cost #Calculamos el coste real hasta ese punto
                stripe.push((child, actions+[direction], accCost + cost), g + heuristic(child, problem)) #Metemos lo mismo que en el caso del ucs pero sumandole la 
                                                                                                         #heurisitca para ver que camino es mejor
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
