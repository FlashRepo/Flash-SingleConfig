from __future__ import division

import pandas as pd
from random import shuffle
import numpy as np
import sys

def test():
    sys.stdout.flush()
    data_file = "./Data/rs-6d-c3_obj1.csv"

    content = pd.read_csv(data_file)
    columns = content.columns

    indep_columns = [c for c in columns if '$<' not in c]
    dep_columns = [c for c in columns if '$<' in c][-1]

    indep = content[indep_columns]
    dep = content[dep_columns]

    indexes = range(len(content))
    shuffle(indexes)

    train_indexes = indexes[:int(0.1 * len(content))]
    test_indexes = indexes[int(0.1 * len(content)):]

    train_indep = [indep.iloc[i] for i in train_indexes]
    train_dep = [dep.iloc[i] for i in train_indexes]

    test_indep = [indep.iloc[i] for i in test_indexes]
    test_dep = [dep.iloc[i] for i in test_indexes]

    points = []
    for i in xrange(1, len(train_indexes), 1):
        print '. ',
        sys.stdout.flush()
        t_train_indep = train_indep[:i+1]
        t_train_dep = train_dep[:i+1]

        from sklearn.tree import DecisionTreeRegressor
        model = DecisionTreeRegressor()
        model.fit(t_train_indep, t_train_dep)

        predict = model.predict(test_indep)

        t_mmre = []
        for act,pred in zip(test_dep, predict):
            t_mmre.append(abs(act-pred)/act)

        points.append([i+1, (1-np.median(t_mmre))*100])

    import matplotlib.pyplot as plt
    plt.plot([p[0] for p in points], [p[1] for p in points], color='black')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Size of Training Set')
    plt.savefig('figure1.eps')

if __name__ == "__main__":
    test()