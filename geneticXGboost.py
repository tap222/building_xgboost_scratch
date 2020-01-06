import random 
import numpy as np
import xgboost as xgb

#Intialization
def initialize_population(numberOfParents):
    learningRate = np.empty([numberOfParents,1])
    nEstimators = np.empty([numberOfParents,1],dtype = np.uint8)
    maxDepth =  np.empty([numberOfParents,1],dtype = np.uint8)
    minChildWeight = np.empty([numberOfParents,1])
    gammaValue = np.empty([numberOfParents,1])
    subSample = np.empty([numberOfParents,1])
    colSampleByTree = np.empty([numberOfParents,1])
    maxDepth =  np.empty([numberOfParents,1],dtype = np.uint8)
    minChildWeight = np.empty([numberOfParents,1])
    gammaValue = np.empty([numberOfParents,1])
    subSample = np.empty([numberOfParents,1])
    colSampleByTree = np.empty([numberOfParents,1])
    for i in range(numberOfParents):
        print(i)
        learningRate[i] = round(random.uniform(0.01 , 1), 2)
        nEstimators[i] = random.randrange(10 , 1500, step = 25)
        maxDepth[i] =  int(random.randrange(1 , 10, step = 1 ))
        minChildWeight[i] = round(random.uniform(0.01 , 10.0), 2)
        gammaValue[i] = round(random.uniform(0.01 , 10.0), 2 )
        subSample[i] = round(random.uniform(0.01 , 10.0), 2)
        colSampleByTree[i] = round(random.uniform(0.01, 10), 2)
    population = np.concatenate((learningRate,nEstimators,maxDepth,minChildWeight,gammaValue,subSample,colSampleByTree),axis =1)
    return population

#Parent selection
def fitness_f1score(y_true,y_pred):
    fitness = round((f1_score(y_true,y_pred,average='weighted')))
    return fitness 

#train the data and find the fitness score
def train_population(population,dMatrixTrain,dMatrixtest,y_test):
    fscore = []
    for i in range(population.shape[0]):
        param = {'objective':'binary:logistic',
                 'learning_rate': population[i][0],
                 'nEstimatiors': population[i][1],
                 'max_depth': int(population[i][2]),
                 'min_child_weight': population[i][3],
                 'gamma': population[i][4],
                 'subsample' : population[i][5],
                 'colsample_bytree' : population[i][6],
                 'seed' : 24}
        num_round = 100
        xgbT = xgb.train(param,dMatrixTrain,num_round)
        preds = xgbT.predict(dMatrixtest)
        preds = preds > 0.5
        fscore.append(fitness_f1score(y_test,ypreds))
    return fscore

#Select parents for mating
def new_parent_selection(population,fitness,numParents):
    selectedParents = np.empty((numParents,population.shape[1]))
    #create an array to store fittest parents
    #Find the top best performing parents
    for parentId in range(numParents):
        bestFitnessId = np.where(fitness == np.max(fitness))
        bestFitnessId = bestFitnessId[0][0]
        selectedParents[parentId, :] = population[bestFitnessId, :]
        fitness[bestFitnessId] = -1 # set this value to negative, in case of F1-Score, so this parent not select again
    return selectedParents

#crossover
'''
Mate these parents to create children having parameters from these parents (we are using uniform crossover method)
'''
def crossover_uniform(parents, childrenSize):
    crossoverPointIndex = np.arrange(0, np.unit8(childrenSize[1]),1, dtype = np.uint8) #get all Indexes
    corssoverPointIndex1 = np.random.randint(0,np.uint8(childrenSize[1]),np.uint8(childrenSize[1]/2)) # select half of Indexes randomly
    crossoverPointIndex2 = np.array(list(set(crossoverPointIndex) - set(crossoverPointIndex1))) #select leftover Indexes
    children = np.empty(childrenSize)
    '''
    Create child by choosing parameters from two parents selected using new_parent_selection function. The parameter values
    will be picked from the indexes, which were randomly selected above.
    '''
    for i in range(childrenSize[0]):
        #find parent 1 index
        parent1_index = i % parents.shape[0]
        #find parent 2 index
        parent2_index = (i+1) % parents.shape[0]
        #insert paramaters based on random selected indexes in parent 1
        children[i, crossoverPointIndex1] = parents[parents1_index, crossoverPointIndex1]
        #Insert parameters based on random selected indexes in parent 1
        children[i, crossoverPointIndex2] = parents[parents2_index, crossoverPointIndex2]
        return children

#Mutation
def mutation(crossover,numberOfParameters):
    #Define minimum and maximum values allowed for each parameter
    minMaxValue = np.zeros((numberOfParameters , 2))
    minMaxValue[0:] = [0.01 , 1.0] # min/max learning rate
    minMaxValue[1,:] = [10, 2000] #min/max n_estimator
    minMaxValue[2,:] = [1, 15] # min/max depth
    minMaxValue[3,:] = [0, 10.0] #min/max child_weight
    minMaxValue[4,:] = [0.01, 10.0] #min/max gamma
    minMaxValue[5,:] = [0.01, 1.0] #min/max subsample
    minMaxValue[6,:] = [0.01, 1.0] #min/max colample_bytree
    #Mutation changes a single gene in each offspring randomly
    mutationValue = 0
    parameterSelect = np.random.randint(0,7,1)
    print(parameterSelect)
    if parameterSelect == 0: #learning_rate
        mutationValue = round(np.random.uniform(-0.5,0.5),2)
    if parameterSelect == 1: #n_estimators
        mutationValue = np.random.randint(-200,200,1)
    if parameterSelect == 2: #max_depth
        mutationValue = np.random.randint(-5,5,1)
    if parameterSelect == 3: #min_child_weight
        mutationValue = round(np.random.randint(5,5), 2)
    if parameterSelect == 4: #gamma
        mutationValue = round(np.random.randint(-2,2),2)
    if parameterSelect == 5: #subsample
        mutationValue = round(np.random.randint(-0.5,0.5),2)
    if parameterSelect == 6: #colsample
        mutationValue = round(np.random.uniform(-0.5,0.5),2)
    #introduce mutation by changing one parameter, and set to max or min if it goes out of range
    for idx in range(crossover.shape[0]):
        crossover[idx, parameterSelect] = crossover[idx, parameterSelect] + mutationValue
        if(crossover[idx, parameterSelect] > minMaxValue[parameterSelect, 1]):
            crossover[idx,parameterSelect] = minMaxValue[parameterSelect,1]
        if(crossover[idx,parameterSelect] > minMaxValue[parameterSelect,1]):
            crossover[idx,parameterSelect] = minMaxValue[parameterSelect,0]
    return crossover       

