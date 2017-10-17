from __future__ import division
import pandas as pd
import sys
from os import listdir
from random import shuffle
from sklearn.tree import DecisionTreeRegressor
from policies import policy1, policy2
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle


class solution_holder:
    def __init__(self, id, decisions, objective, rank):
        self.id = id
        self.decision = decisions
        self.objective = objective
        self.rank = rank


def split_data(filename):
    pdcontent = pd.read_csv(filename)
    indepcolumns = [col for col in pdcontent.columns if "<$" not in col]
    depcolumns = [col for col in pdcontent.columns if "<$" in col]
    sortpdcontent = pdcontent.sort(depcolumns[-1])
    content = list()
    ranks = {}
    for i, item in enumerate(sorted(set(sortpdcontent[depcolumns[-1]].tolist()))):
        ranks[item] = i
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
    train_indexes, validation_indexes, test_indexes = indexes[:int(0.4*len(indexes))], indexes[int(.4*len(indexes)):int(.6*len(indexes))],  indexes[int(.6*len(indexes)):]
    assert(len(train_indexes) + len(validation_indexes) + len(test_indexes) == len(indexes)), "Something is wrong"
    train_set = [content[i] for i in train_indexes]
    validation_set = [content[i] for i in validation_indexes]
    test_set = [content[i] for i in test_indexes]

    return [train_set, validation_set, test_set]


def mre_progressive(train, test, threshold=0.1):
    train_independent = [t.decision for t in train]
    train_dependent = [t.objective[-1] for t in train]

    test_independent = [t.decision for t in test]
    test_dependent = [t.objective[-1] for t in test]

    model = DecisionTreeRegressor()
    model.fit(train_independent, train_dependent)
    predicted = model.predict(test_independent)

    mre = []
    for org, pred in zip(test_dependent, predicted):
        mre.append(abs(org - pred)/ abs(org))
    return np.mean(mre)


def wrapper_mre_progressive(train_set, validation_set, threshold=0.1):
    initial_size = 10
    training_indexes = range(len(train_set))
    shuffle(training_indexes)
    sub_train_set = [train_set[i] for i in training_indexes[:initial_size]]
    steps = 0
    while (initial_size+steps) < len(train_set) - 1:
        print "size of sub_train: ", len(sub_train_set)
        mre_returned = mre_progressive(sub_train_set, validation_set)
        if mre_returned < threshold: break
        steps += 200
        for i in xrange(steps-200, steps):
            sub_train_set.append(train_set[initial_size+i])

    return sub_train_set


def find_lowest_rank(train, test, bracket=10):
    # Test data
    train_independent = [t.decision for t in train]
    train_dependent = [t.objective[-1] for t in train]

    sorted_test = sorted(test, key=lambda x: x.objective[-1])
    for r, st in enumerate(sorted_test): st.rank = r
    test_independent = [t.decision for t in sorted_test]
    test_dependent = [t.objective[-1] for t in sorted_test]

    model = DecisionTreeRegressor()
    model.fit(train_independent, train_dependent)
    predicted = model.predict(test_independent)

    predicted_id = [[i, p] for i, p in enumerate(predicted)]
    predicted_sorted = sorted(predicted_id, key=lambda x: x[-1])
    # assigning predicted ranks
    predicted_rank_sorted = [[p[0], p[-1], i] for i,p in enumerate(predicted_sorted)]
    select_few = predicted_rank_sorted[:10]
    return [sf[0] for sf in select_few]

if __name__ == "__main__":
    import time
    datafolder = "./Data/"
    evals_dict = {}
    rank_diffs_dict = {}
    stats_dict = {}
    # files = [datafolder + f for f in listdir(datafolder) if '.csv' in f]
    files = ['./Data/SS-N1.csv']

    for file in files:
        initial_time = time.time()
        evals_dict[file] = []
        rank_diffs_dict[file] = []
        stats_dict[file] = {}
        mmre_rank_diffs = []
        mmre_evals = []
        size_of_test_set = []
        print file + " | ",
        for _ in xrange(1):
            print "+ ",
            datasets = split_data(file)
            train_set = datasets[0]
            validation_set = datasets[1]
            test_set = datasets[2]

            sub_train_set_rank = wrapper_mre_progressive(train_set, validation_set)
            lowest_rank = find_lowest_rank(sub_train_set_rank, train_set+test_set)
            len_mre_train_set = len(sub_train_set_rank)
            min_mre_prog = min(lowest_rank)
            mmre_rank_diffs.append(min_mre_prog)
            mmre_evals.append(len_mre_train_set + len(validation_set) + 10)
            size_of_test_set.append(len(train_set) + len(test_set))

        stats_dict[file]["rank_diff"] = mmre_rank_diffs
        stats_dict[file]["evals"] = mmre_evals
        stats_dict[file]["testsize"] = size_of_test_set

        print " | Total Time: ", time.time() - initial_time

        pickle.dump(stats_dict, open('./PickleLocker/Residual_Stats_' + file.split('/')[-1] + '24.p', 'w'))





