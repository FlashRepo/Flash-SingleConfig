import pickle
import os

locker = './'
files = [locker+f for f in os.listdir(locker) if '.py' not in f]

names = {}
for f in files:
    if f.split('.csv')[0] not in names.keys():
        names[f.split('.csv')[0]] = 1
    else:
        names[f.split('.csv')[0]] += 1

# names = set(names)
for key in names.keys():
    pfiles = []
    for f in files:
        if key in f:
            pfiles.append(f)
    store = {}
    for p in pfiles:
        print p
        t = pickle.load(open(p))
        k = t.keys()[0]
        if len(store.keys()) == 0:
            store[k] = {}
            store[k]['evals'] = t[k]['evals']
            store[k]['rank_diff'] = t[k]['rank_diff']
            store[k]['testsize'] = t[k]['testsize']
        else:
            store[k]['evals'].extend(t[k]['evals'])
            store[k]['rank_diff'].extend(t[k]['rank_diff'])
            store[k]['testsize'].extend(t[k]['testsize'])
    pickle.dump(store, open(key+'.csv.p', 'w'))
