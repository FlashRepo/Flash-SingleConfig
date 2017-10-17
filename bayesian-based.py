from __future__ import division
import pandas as pd
import sys
from os import listdir
from random import shuffle
from sklearn.tree import DecisionTreeRegressor
from policies import policy1, policy2
import numpy as np
import pickle


class solution_holder:
    def __init__(self, id, decisions, objective, rank):
        self.id = id
        self.decision = decisions
        self.objective = objective
        self.rank = rank


def get_data(filename, initial_size):
    """
    :param filename:
    :param Initial training size
    :return: Training and Testing
    """
    pdcontent = pd.read_csv(filename)
    indepcolumns = [col for col in pdcontent.columns if "<$" not in col]
    depcolumns = [col for col in pdcontent.columns if "<$" in col]
    sortpdcontent = pdcontent.sort(depcolumns[-1])
    ranks = {}
    for i, item in enumerate(sorted(set(sortpdcontent[depcolumns[-1]].tolist()))):
        ranks[item] = i

    content = list()
    for c in xrange(len(sortpdcontent)):
        content.append(solution_holder(
                                       c,
                                       sortpdcontent.iloc[c][indepcolumns].tolist(),
                                       sortpdcontent.iloc[c][depcolumns].tolist(),
                                       ranks[sortpdcontent.iloc[c][depcolumns].tolist()[-1]]
                                       )
                       )

    shuffle(content)
    indexes = range(int(len(content) * 0.8))  # for experiments described in the paper
    train_indexes, test_indexes = indexes[:initial_size],  indexes[initial_size:]
    assert(len(train_indexes) + len(test_indexes) == len(indexes)), "Something is wrong"
    train_set = [content[i] for i in train_indexes]
    test_set = [content[i] for i in test_indexes]

    return [train_set, test_set]


def get_best_configuration_id(train, test):
    train_independent = [t.decision for t in train]
    train_dependent = [t.objective[-1] for t in train]

    test_independent = [t.decision for t in test]

    model = DecisionTreeRegressor()
    model.fit(train_independent, train_dependent)
    predicted = model.predict(test_independent)
    predicted_id = [[t.id,p] for t,p in zip(test, predicted)]
    predicted_sorted = sorted(predicted_id, key=lambda x: x[-1])
    # Find index of the best predicted configuration
    best_index = predicted_sorted[0][0]
    return best_index


def run_active_learning(filename, initial_size, max_lives=10):
    steps = 0
    lives = max_lives
    training_set, testing_set = get_data(filename, initial_size)
    dataset_size = len(training_set) + len(testing_set)
    while (initial_size+steps) < dataset_size - 1:
        best_id = get_best_configuration_id(training_set, testing_set)
        # print best_index, len(testing_set)
        best_solution = [t for t in testing_set if t.id == best_id][-1]

        list_of_all_solutions = [t.objective[-1] for t in training_set]
        if best_solution.objective[-1] < min(list_of_all_solutions):
            lives = max_lives
        else:
            lives -= 1
        training_set.append(best_solution)
        # find index of the best_index
        best_index = [i for i in xrange(len(testing_set)) if testing_set[i].id == best_id]
        assert(len(best_index) == 1), "Something is wrong"
        best_index = best_index[-1]
        del testing_set[best_index]
        assert(len(training_set) + len(testing_set) == dataset_size), "Something is wrong"
        if lives == 0:
            break
        steps += 1

    return training_set, testing_set


def wrapper_run_active_learning(filename, initial_size):
    training_set, testing_set= run_active_learning(filename, initial_size)
    global_min = min([t.objective[-1] for t in training_set + testing_set])
    best_training_solution = [ tt.rank for tt in training_set if min([t.objective[-1] for t in training_set]) == tt.objective[-1]]
    best_solution = [tt.rank for tt in training_set + testing_set if tt.objective[-1] == global_min]
    return [(min(best_training_solution) - min(best_solution)), len(training_set), len(training_set) + len(testing_set)]

if __name__ == "__main__":
    import time
    # filenames = ["./Data/"+f for f in listdir("./Data") if '.csv' in f]
    filenames = [
        # './Data/SS-A1.csv', './Data/SS-A2.csv', './Data/SS-B1.csv', './Data/SS-B2.csv', './Data/SS-C1.csv',
        #  './Data/SS-C2.csv', './Data/SS-D1.csv', './Data/SS-D2.csv', './Data/SS-E1.csv', './Data/SS-E2.csv',
        #  './Data/SS-F1.csv', './Data/SS-F2.csv', './Data/SS-G1.csv', './Data/SS-G2.csv', './Data/SS-H1.csv',
        #  './Data/SS-H2.csv', './Data/SS-I1.csv', './Data/SS-I2.csv', './Data/SS-J1.csv', './Data/SS-J2.csv',
        #  './Data/SS-K1.csv', './Data/SS-K2.csv', './Data/SS-L1.csv', './Data/SS-L2.csv',
        './Data/SS-M1.csv',
        './Data/SS-M2.csv', './Data/SS-N1.csv', './Data/SS-N2.csv', './Data/SS-O1.csv', './Data/SS-O2.csv']
    initial_size = 20
    stats_dict = {}
    for filename in filenames:
        initial_time = time.time()
        stats_dict[filename] = {}
        rank_diffs = []
        evals = []
        size_of_test_set = []
        print filename + " | ",
        for _ in xrange(30):
            print "+ ",
            return_vals = wrapper_run_active_learning(filename, initial_size)
            rank_diffs.append(return_vals[0])
            evals.append(return_vals[1])
            size_of_test_set.append(return_vals[2])
        stats_dict[filename]["rank_diff"] = rank_diffs
        stats_dict[filename]["evals"] = evals
        stats_dict[filename]["testsize"] = size_of_test_set

        print " | Total Time: ", time.time() - initial_time

        pickle.dump(stats_dict, open("./PickleLocker/Flash_Stats_" + filename.split('/')[-1] + ".p", "w"))

