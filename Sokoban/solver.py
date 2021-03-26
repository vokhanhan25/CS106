import sys
import collections
import numpy as np
import heapq
import time
import numpy as np
global posWalls, posGoals


"""Load puzzles and define the rules of sokoban"""
class PriorityQueue:
    """Define a PriorityQueue data structure that will be used"""
    def  __init__(self):
        self.Heap = []
        self.Count = 0

    def push(self, item, priority):
        entry = (priority, self.Count, item)
        heapq.heappush(self.Heap, entry)
        self.Count += 1

    def pop(self):
        (_, _, item) = heapq.heappop(self.Heap)
        return item

    def isEmpty(self):
        return len(self.Heap) == 0

def transferToGameState(layout):
    """Transfer the layout of initial puzzle"""
    layout = [x.replace('\n','') for x in layout]
    layout = [','.join(layout[i]) for i in range(len(layout))]
    layout = [x.split(',') for x in layout]
    maxColsNum = max([len(x) for x in layout])
    for irow in range(len(layout)):
        for icol in range(len(layout[irow])):
            if layout[irow][icol] == ' ': layout[irow][icol] = 0   # free space
            elif layout[irow][icol] == '#': layout[irow][icol] = 1 # wall
            elif layout[irow][icol] == '&': layout[irow][icol] = 2 # player
            elif layout[irow][icol] == 'B': layout[irow][icol] = 3 # box
            elif layout[irow][icol] == '.': layout[irow][icol] = 4 # goal
            elif layout[irow][icol] == 'X': layout[irow][icol] = 5 # box on goal
        colsNum = len(layout[irow])
        if colsNum < maxColsNum:
            layout[irow].extend([1 for _ in range(maxColsNum-colsNum)]) 

    # print(layout)
    return np.array(layout)
def transferToGameState2(layout, player_pos):
    """Transfer the layout of initial puzzle"""
    maxColsNum = max([len(x) for x in layout])
    temp = np.ones((len(layout), maxColsNum))
    for i, row in enumerate(layout):
        for j, val in enumerate(row):
            temp[i][j] = layout[i][j]

    temp[player_pos[1]][player_pos[0]] = 2
    return temp

def PosOfPlayer(gameState):
    """Return the position of agent"""
    return tuple(np.argwhere(gameState == 2)[0]) # e.g. (2, 2)

def PosOfBoxes(gameState):
    """Return the positions of boxes"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 3) | (gameState == 5))) # e.g. ((2, 3), (3, 4), (4, 4), (6, 1), (6, 4), (6, 5))

def PosOfWalls(gameState):
    """Return the positions of walls"""
    return tuple(tuple(x) for x in np.argwhere(gameState == 1)) # e.g. like those above

def PosOfGoals(gameState):
    """Return the positions of goals"""
    return tuple(tuple(x) for x in np.argwhere((gameState == 4) | (gameState == 5))) # e.g. like those above

def isEndState(posBox):
    """Check if all boxes are on the goals (i.e. pass the game)"""
    return sorted(posBox) == sorted(posGoals)

def isLegalAction(action, posPlayer, posBox):
    """Check if the given action is legal"""
    xPlayer, yPlayer = posPlayer
    if action[-1].isupper(): # the move was a push
        x1, y1 = xPlayer + 2 * action[0], yPlayer + 2 * action[1]
    else:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
    return (x1, y1) not in posBox + posWalls

def legalActions(posPlayer, posBox):
    """Return all legal actions for the agent in the current game state"""
    allActions = [[-1,0,'u','U'],[1,0,'d','D'],[0,-1,'l','L'],[0,1,'r','R']]
    xPlayer, yPlayer = posPlayer
    legalActions = []
    for action in allActions:
        x1, y1 = xPlayer + action[0], yPlayer + action[1]
        if (x1, y1) in posBox: # the move was a push
            action.pop(2) # drop the little letter
        else:
            action.pop(3) # drop the upper letter
        if isLegalAction(action, posPlayer, posBox):
            legalActions.append(action)
        else: 
            continue     
    return tuple(tuple(x) for x in legalActions) # e.g. ((0, -1, 'l'), (0, 1, 'R'))

def updateState(posPlayer, posBox, action):
    """Return updated game state after an action is taken"""
    xPlayer, yPlayer = posPlayer # the previous position of player
    newPosPlayer = [xPlayer + action[0], yPlayer + action[1]] # the current position of player
    posBox = [list(x) for x in posBox]
    if action[-1].isupper(): # if pushing, update the position of box
        posBox.remove(newPosPlayer)
        posBox.append([xPlayer + 2 * action[0], yPlayer + 2 * action[1]])
    posBox = tuple(tuple(x) for x in posBox)
    newPosPlayer = tuple(newPosPlayer)
    return newPosPlayer, posBox

def isFailed(posBox):
    """This function used to observe if the state is potentially failed, then prune the search"""
    rotatePattern = [[0,1,2,3,4,5,6,7,8],
                    [2,5,8,1,4,7,0,3,6],
                    [0,1,2,3,4,5,6,7,8][::-1],
                    [2,5,8,1,4,7,0,3,6][::-1]]
    flipPattern = [[2,1,0,5,4,3,8,7,6],
                    [0,3,6,1,4,7,2,5,8],
                    [2,1,0,5,4,3,8,7,6][::-1],
                    [0,3,6,1,4,7,2,5,8][::-1]]
    allPattern = rotatePattern + flipPattern

    for box in posBox:
        if box not in posGoals:
            board = [(box[0] - 1, box[1] - 1), (box[0] - 1, box[1]), (box[0] - 1, box[1] + 1), 
                    (box[0], box[1] - 1), (box[0], box[1]), (box[0], box[1] + 1), 
                    (box[0] + 1, box[1] - 1), (box[0] + 1, box[1]), (box[0] + 1, box[1] + 1)]
            for pattern in allPattern:
                newBoard = [board[i] for i in pattern]
                if newBoard[1] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posWalls: return True
                elif newBoard[1] in posBox and newBoard[2] in posWalls and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[2] in posBox and newBoard[5] in posBox: return True
                elif newBoard[1] in posBox and newBoard[6] in posBox and newBoard[2] in posWalls and newBoard[3] in posWalls and newBoard[8] in posWalls: return True
    return False

"""Implement all approcahes"""

def depthFirstSearch(gameState):
    """Implement depthFirstSearch approach"""
    beginBox = PosOfBoxes(gameState) #get the location of boxes
    beginPlayer = PosOfPlayer(gameState) #get the location of player

    startState = (beginPlayer, beginBox) # get the start state from  the location of boxes and player
    # declare queue, push start state into the queue, this queue stores the states
    frontier = collections.deque([[startState]]) 
    exploredSet = set() #Declare the set of states have explore
    actions = [[0]] # declare actions array, this queue stores the actions of the corresponding states
    temp = [] #declare array temp
    while frontier: # while the frontier node is not empty
        node = frontier.pop()  # get the node at the right of the queue
        node_action = actions.pop() #get the action at the right of the action array
        if isEndState(node[-1][-1]): # check if all boxes are on the goals
            temp += node_action[1:]  #if not yet explored, add this node into exploredSet
            break #stop DFS
        if node[-1] not in exploredSet:#check if the node is explopred
            exploredSet.add(node[-1])  #if not yet explored, add this node into exploredSet
            for action in legalActions(node[-1][0], node[-1][1]): #loop through valid actions

                #store the new state after taking valid action on newPosPlayer, newPosBox
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) 
                 # if the state is potentially failed
                if isFailed(newPosBox):
                    continue #if failed, skip and browse next action
                frontier.append(node + [(newPosPlayer, newPosBox)]) # push the new state to the right of the frontier queue
                actions.append(node_action + [action[-1]]) # push the new action to the right of the action queue
    return temp #return array temp
 

def breadthFirstSearch(gameState):
    beginBox = PosOfBoxes(gameState) #get the location of boxes
    beginPlayer = PosOfPlayer(gameState) #get the location of player

    startState = (beginPlayer, beginBox) # get the start state from  the location of boxes and player

    # declare frontier queue, push start state into the queue, this queue stores the states
    frontier = collections.deque([[startState]]) 

    # declare actions queue, this queue stores the actions of the corresponding states
    actions = collections.deque([[0]]) 
    exploredSet = set() #Declare the set of states have explore
    temp = [] #declare array temp

    ### Implement breadthFirstSearch here
    while frontier: # while the frontier node is not empty
        node = frontier.popleft() # get the node at the left of the fronier queue
        node_action = actions.popleft() #get the action at the left of the actions queue
        if isEndState(node[-1][-1]):  # check if all boxes are on the goals
            temp += node_action[1:] #if true, store action to temp array
            break # stop BFS
        if node[-1] not in exploredSet: #check if the node is explopred
            exploredSet.add(node[-1]) #if not yet explored, add this node into exploredSet
            for action in legalActions(node[-1][0], node[-1][1]): #loop through valid actions
                #store the new state after taking valid action on newPosPlayer, newPosBox
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) 
                if isFailed(newPosBox): # if the state is potentially failed
                    continue #if failed, skip and browse next action
                # push the new state to the right of the frontier queue
                frontier.append(node + [(newPosPlayer, newPosBox)]) 

                # push the new action to the right of the action queue
                actions.append(node_action + [action[-1]]) 
    return temp #return array temp

def cost(actions):
    return actions.count('l') + actions.count('r') + actions.count('u') + actions.count('d')

def uniformCostSearch(gameState):
    """Implement uniformCostSearch approach"""
    beginBox = PosOfBoxes(gameState) #get the location of boxes
    beginPlayer = PosOfPlayer(gameState) #get the location of player
 
    startState = (beginPlayer, beginBox) # get the start state from  the location of boxes and player

    # declare frontier priority queue, stores the states and the cost to them, prioritizing the states with the smallest cost to it
    frontier = PriorityQueue() 

    #push start state into the queue, the start state has the cost to it = 0
    frontier.push([startState], 0)
    
    exploredSet = set() #Declare the set of states have explore

    # declare priority queue, stores the actions and the cost, prioritizing the actions with the smallest cost

    # declare actions priority queue, this queue stores the actions of the corresponding states
    # prioritizing the states with the smallest cost to corresponding state
    actions = PriorityQueue() 
    actions.push([0], 0) #initiates the actions of the first state
    temp = [] #declare array temp
    
    ### Implement uniform cost search here
    while frontier: # while the frontier node is not empty
        node = frontier.pop()  # get the node has smallest cost
        node_action = actions.pop() #get the action has smalles cost

        if isEndState(node[-1][-1]): # check if all boxes are on the goals
            temp += node_action[1:]  #if true, store action to temp array
            break #stop UCSs

        if node[-1] not in exploredSet: #if true, store action to temp array
            exploredSet.add(node[-1]) #if not yet explored, add this node into exploredSet
            for action in legalActions(node[-1][0], node[-1][1]): #loop through valid actions

                #store the new state after taking valid action on newPosPlayer, newPosBox
                newPosPlayer, newPosBox = updateState(node[-1][0], node[-1][1], action) 

                if isFailed(newPosBox):  # if the state is potentially failed
                    continue #if failed, skip and browse next action
                
                # push the new state to priority queue frontier, priority small cost
                frontier.push(node + [(newPosPlayer, newPosBox)], cost(action))

                # push the actions corresponding the new state to priority queue actions, priority small cost
                actions.push(node_action + [action[-1]], cost(action))
    return temp #return array temp

"""Read command"""
def readCommand(argv):
    from optparse import OptionParser
    
    parser = OptionParser()
    parser.add_option('-l', '--level', dest='sokobanLevels',
                      help='level of game to play', default='level1.txt')
    parser.add_option('-m', '--method', dest='agentMethod',
                      help='research method', default='bfs')
    args = dict()
    options, _ = parser.parse_args(argv)
    with open('assets/levels/' + options.sokobanLevels,"r") as f: 
        layout = f.readlines()
    args['layout'] = layout
    args['method'] = options.agentMethod
    return args

def get_move(layout, player_pos, method):
    time_start = time.time()
    global posWalls, posGoals
    # layout, method = readCommand(sys.argv[1:]).values()
    gameState = transferToGameState2(layout, player_pos)
    posWalls = PosOfWalls(gameState)
    posGoals = PosOfGoals(gameState)
    if method == 'dfs':
        result = depthFirstSearch(gameState)
    elif method == 'bfs':
        result = breadthFirstSearch(gameState)    
    elif method == 'ucs':
        result = uniformCostSearch(gameState)
    else:
        raise ValueError('Invalid method.')
    time_end=time.time()
    print('Runtime of %s: %.2f second.' %(method, time_end-time_start))
    print(result)
    number_of_steps = len(result)
    print('Number of Steps:', number_of_steps)
    return result
