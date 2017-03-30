from __future__ import division
import pandas as pd
import sys
from os import listdir
from random import shuffle
from sklearn.tree import DecisionTreeRegressor
from policies import policy1, policy2
import numpy as np


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
    indepcolumns = [col for col in pdcontent.columns if "$<" not in col]
    depcolumns = [col for col in pdcontent.columns if "$<" in col]
    sortpdcontent = pdcontent.sort_values(by=depcolumns[-1])
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
    indexes = range(len(content))
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
    print (min(best_training_solution) - min(best_solution)), len(training_set), " | ",
    return (min(best_training_solution) - min(best_solution)), len(training_set)

if __name__ == "__main__":
    filenames = ["./Data/"+f for f in listdir("./Data")]
    initial_size = 20
    evals_dict = {}
    rank_diffs_dict = {}
    stats_dict = {}
    for filename in filenames:
        evals_dict[filename] = []
        rank_diffs_dict[filename] = []
        stats_dict[filename] = {}
        rank_diffs = []
        evals = []
        print filename
        for _ in xrange(20):
            temp1, temp2 = wrapper_run_active_learning(filename, initial_size)
            rank_diffs.append(temp1)
            evals.append(temp2)
        print
        evals_dict[filename] = evals
        rank_diffs_dict[filename] = rank_diffs
        stats_dict[filename]["mean_rank_diff"] = np.mean(rank_diffs)
        stats_dict[filename]["std_rank_diff"] = np.std(rank_diffs)
        stats_dict[filename]["mean_evals"] = np.mean(evals)
        stats_dict[filename]["std_evals"] = np.std(evals)

    import pickle
    pickle.dump(evals_dict, open("./PickleLocker/ActiveLearning_Evals.p", "w"))
    pickle.dump(rank_diffs_dict, open("./PickleLocker/ActiveLearning_Rank_Diff.p", "w"))
    pickle.dump(stats_dict, open("./PickleLocker/ActiveLearning_Stats.p", "w"))

