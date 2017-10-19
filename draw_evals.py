import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

sizes = {
    'SS-A1'  : 1343,
    'SS-A2'  : 1343,
    'SS-B1'  : 206,
    'SS-B2'  : 206,
    'SS-C1'  : 1512,
    'SS-C2'  : 1512,
    'SS-D1'  : 196,
    'SS-D2'  : 196,
    'SS-E1'  : 756,
    'SS-E2'  : 756,
    'SS-F1'  : 196,
    'SS-F2'  : 196,
    'SS-G1'  : 196,
    'SS-G2'  : 196,
    'SS-H1'  : 259,
    'SS-H2'  : 259,
    'SS-I1'  : 1080,
    'SS-I2'  : 1080,
    'SS-J1'  : 3840,
    'SS-J2'  : 3840,
    'SS-K1'  : 2880,
    'SS-K2'  : 2880,
    'SS-L1'  : 1023,
    'SS-L2'  : 1023,
    'SS-M1'  : 239360,
    'SS-M2'  : 239360,
    'SS-N1'  : 53662,
    'SS-N2'  : 53662,
    'SS-O1'  : 65424,
    'SS-O2'  : 65424,
}

def mean(lst):
    return round(np.mean(lst), 3)

locker = './PickleLocker/'
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
for name in names:
    key = './Data/' + name + '.csv'
    res = pickle.load( open(residual[name], "rb" ) )[key]
    rnk = pickle.load( open( rank[name], "rb" ) )[key]
    flsh = pickle.load( open( bayesian[name], "rb" ) )[key]

    # data.append([name, mean(res['rank_diff']), mean(rnk['rank_diff']), mean(flsh['rank_diff'])])
    data.append([name, mean(res['evals']), mean(rnk['evals']), mean(flsh['evals']), sizes[name]*0.2])

# for d in data: print d
# for d in data1: print d

import numpy as np
import matplotlib.pyplot as plt
data = sorted(data, key=lambda x: x[0])
N = len(data)
residual_evals = [d[1] for d in data]
rank_evals = [d[2] for d in data]
flash_evals = [d[3] for d in data]
size_evals = [d[4] for d in data]

# converting to ratios
size_evals = [(d / proj) * 100 for d, proj in zip(size_evals, residual_evals)]
flash_evals = [(d / proj) * 100 for d, proj in zip(flash_evals, residual_evals)]
rank_evals = [(d / proj) * 100 for d, proj in zip(rank_evals, residual_evals)]
residual_evals = [100 for _ in flash_evals]

# print sorted([[i, (d/proj)*100] for i, (d, proj) in enumerate(zip(residual_evals, flash_evals))], key=lambda x:x[-1])


space = 5
ind = np.arange(space, space*(len(data)+1), space)  # the x locations for the groups
width = 1.5        # the width of the bars

fig, ax = plt.subplots()
ax.plot([i for i in xrange(3, 140)], [100 for _ in xrange(3, 140)], linestyle='--', color='black', label='Residual-based')
rects1 = ax.bar(ind, rank_evals, width, color='#f0f0f0', label='Rank-based')
# r1 = ax.bar(ind, size_evals, width, color='y', bottom=residual_evals)

rects2 =  ax.bar(ind + 1 * width, flash_evals, width, color='#636363', label='Flash')
# r2 =  ax.bar(ind + 1 * width, size_evals, width, color='y', bottom=rank_evals)


# rects3 = ax.bar(ind + 2 * width, flash_evals, width, color='#bdbdbd', label='Flash')

ax.set_ylabel('Evaluations as % of Projective Sampling')
# ax.set_title('Scores by group and gender')

ax.set_xticks(ind + 2*width / 2)
ax.set_xticklabels([x[0] for x in data], rotation='vertical')

ax.set_xlim(3, 140)
# ax.set_ylim(1, 1000)

plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3, fancybox=True, frameon=False)

handles, labels = ax.get_legend_handles_labels()
print handles
print labels
# plt.legend([handles[1], handles[2], handles[0]], [labels[1], labels[2], labels[0]], bbox_to_anchor=(0.85, 1.1), ncol=3, fancybox=True, frameon=False)

fig.set_size_inches(14, 5)
# plt.show()
plt.savefig('one_obj_evals.eps', bbox_inches='tight')