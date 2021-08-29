# Functions for the Genetic Algorithm implementation
from statistics import mean

from deap import creator
from deap import base
from deap import tools
from deap import algorithms

# Generic functions for plots and data manipulation
import pandas as pd
import numpy as np
import array
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from keras.models import load_model

# the same preprocessing function we used for the first assignment
def preprocessAndRun():
    train = pd.read_csv('mnist_train.csv')
    test = pd.read_csv('mnist_test.csv')

    X_train = train.drop("label", axis=1, inplace=False)
    X_test = test.drop("label", axis=1, inplace=False)

    # Unsigned int to float
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    y_train = train.label
    y_test = test.label

    # Make categorical
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    # 2. Normalization
    # Pixel values between 0 and 255, so we will use MinMaxScaler to scale it to [0,1].
    norm = MinMaxScaler()
    X_train = norm.fit_transform(X_train)
    y_train = norm.fit_transform(y_train)

    # for later use
    X_test = norm.fit_transform(X_test)
    y_test = norm.fit_transform(y_test)
    return X_train, X_test, y_train, y_test


Xtrain, Xtest, ytrain, ytest = preprocessAndRun()

# The best model from the previous assignment
model = load_model("my_model")

# Genetic Algorithm parameters
# Population size
popsize = 200

# Crossover term
cxpb = 0.9

# Mutation probability
mutpb = 0.01

# Number of generations
ngen = 1000

# Create and register functions ( code taken from the DEAP documentation)
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox = base.Toolbox()
# Attribute generator
toolbox.register("attr_bool", random.randint, 0, 1)
# Structure initializers. 784 because every individual is a string of 784 bits.
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, 784)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)


# fitness function
def fitness(ind):
    loss, accuracy = model.evaluate(Xtest, ytest, verbose=0)
    # We want to penalty the individual depending on how many inputs he has.
    # Less inputs means more fit individual.
    # We also want the higher accuracy value, so the fitness value is a combination of both.
    # Higher fitness_ind will mean more fit individual, because we are based on accurcy.
    fitness_ind = accuracy - (np.count_nonzero(ind) / 784) * 0.6
    return fitness_ind,


toolbox.register("evaluate", fitness)

# Operator registering

# We will user two point crossover
toolbox.register("mate", tools.cxTwoPoint)
# We will use bit-flip mutation
toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
# We will use Tournament selection. We use tournament size 3, because it is used at the manual of DEAP too.
toolbox.register("select", tools.selTournament, tournsize=3)

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("max", max)
bestInd = list()
howManyGens = list()
bestFitnessPerGen = list()
bestFitnessPerGenForAllRuns = list()
for runs in range(10):

    # Initializing population
    pop = toolbox.population(n=popsize)
    print("Start of evolution")

    # Firstly, we need to evaluate the entire population.
    fitness_results = list(map(toolbox.evaluate, pop))
    for ind, fitness_eval in zip(pop, fitness_results):
        ind.fitness.values = fitness_eval
    fitness_list = [ind.fitness.values[0] for ind in pop]

    # Holding evolution records for every gen
    record = stats.compile(pop)
    log = tools.Logbook()
    log.record(gen=0, **record)

    generation = 0
    end = False
    bestFitness = 0

    while generation < 1000 and not end:
        generation += 1
        # ------- Evaluate current generation --------
        print("-- Generation %i --" % generation)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring (Code taken from the documentation of DEAP)
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # We need to find the fitness valeus for the new generation, ,the offspring of the previous gen
        new_gen_ind = [ind for ind in offspring if not ind.fitness.valid]
        new_gen_fitness = (map(toolbox.evaluate, new_gen_ind))
        for ind, fitness_eval in zip(new_gen_ind, new_gen_fitness):
            ind.fitness.values = fitness_eval
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        # Fitnesses of the offsprings
        fitness_list = [ind.fitness.values[0] for ind in pop]

        record = stats.compile(pop)
        log.record(gen=generation, **record)

        # Returning a list of the best values for checking
        best = log.select("max")
        # If the next best individual is less than 1% better thant the best individual of
        # the previous gen, we stop the evolution
        if (generation > 25) and (best[-1][0]-best[-25][0]) < (0.0001*best[-25][0]):
            print("Evolution stopped because there is no significant improvement between best individuals")
            end = True
        # Check if they have the same fitness values
        elif (generation > 25) and (best[-1][0] == best[-25][0]):
            print("Evolution stopped because the best individuals are the same")
            end = True
        else:

            end = False
        # Finding best individual if needed
        best_fitness = max(fitness_list)
        if best_fitness > bestFitness:
            bestFitness = best_fitness
            bestInd = pop[fitness_list.index(best_fitness)]
        bestFitnessPerGen.append(best_fitness)

    howManyGens.append(generation)
    bestFitnessPerGenForAllRuns.append(max(bestFitnessPerGen))
    # Evolution curve staff
    plt.plot(bestFitnessPerGen)
    plt.title('Evolution of the best individual')
    plt.ylabel('Fitness value')
    plt.xlabel('Generation')
    plt.show()
    print("-- End of (successful) evolution --")

print("-- End of all runs --")
print("Average number of gens", mean(howManyGens))
print("Average fitness of best sol", mean(bestFitnessPerGenForAllRuns))

# Now we have to compare the best ANN created from the GA algorithm
# to the best ANN from the first assignment.
# Every feature that isnt activated at the best individual
# of the GA, must be deactivated (0) at the test and train datasets too.
# That the reason we are searching for feature significance.

for i in range(len(bestInd)):
    if bestInd[i] == 0:
        Xtrain[:,i] = 0
        Xtest[:,i] = 0

# Firstly, we dont retrain the ANN.
loss, accuracy = model.evaluate(Xtest, ytest, verbose=0)
print("Accuracy for non retrained model is", accuracy)

# Then, we try with retraining.
model.fit(Xtrain, ytrain, epochs=80, verbose=0)
loss2, accuracy2 = model.evaluate(Xtest, ytest, verbose=0)
print("Accuracy for  retrained model is", accuracy2)

