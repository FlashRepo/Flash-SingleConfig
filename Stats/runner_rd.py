from sk import rdivDemo
import pickle
import os

locker = '../PickleLocker/'
files = [locker + f for f in os.listdir(locker) if '.p' in f]

residual = {}
rank = {}
bayesian = {}

for f in files:
    name = f.split('/')[-1].split('_')[-1].replace('.csv.p', '')
    if 'Flash' in f:
        bayesian[name] = f
    elif 'Rank' in f:
        rank[name] = f
    elif 'Residual' in f:
        residual[name] = f
    print name

names = sorted(residual.keys())
data = []
data1 = []
names = ['SS-A2', 'SS-C1', 'SS-C2', 'SS-E2', 'SS-J1', 'SS-J2', 'SS-K1',]

for name in names:
    # print name
    key = './Data/' + name + '.csv'
    res = ['Residual-based'] + pickle.load( open(residual[name], "rb" ) )[key]['rank_diff']
    rnk = ['Rank-based'] + pickle.load( open( rank[name], "rb" ) )[key]['rank_diff']
    flsh = ['Flash'] + pickle.load( open( bayesian[name], "rb" ) )[key]['rank_diff']

    # data.append([name, mean(res['rank_diff']), mean(rnk['rank_diff']), mean(flsh['rank_diff'])])
    # data.append([name, mean(res['evals']), mean(rnk['evals']), mean(flsh['evals']), sizes[name]*0.2])
    rdivDemo(name, [res, rnk, flsh], globalMinMax=False, isLatex=True)


# for i,file in enumerate(files):
#     lists = list()
#     lists.append(["AL-Rank"] + al_evals[file])
#     lists.append(["P-Rank"] + rank_evals[file])
#     lists.append(["P-MMRE"] + mmre_evals[file])
#     rdivDemo("SS" + str(i+1), lists, globalMinMax=False, isLatex=True)
