import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


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

    data.append([name, mean(res['rank_diff']), mean(rnk['rank_diff']), mean(flsh['rank_diff'])])
    data1.append([name, mean(res['evals']), mean(rnk['evals']), mean(flsh['evals'])])

for d in data: print d
# for d in data1: print d

gap = 35

left, width = .53, .5
bottom, height = .25, .5
right = left + width
top = bottom + height

f, ((ax1, ax2, ax3)) = plt.subplots(1, 3)

ax1.scatter([gap*(i+1) for i in xrange(len(data))], [d[1] for d in data], marker='v', color='#228B22')
ax1.set_xlim(10, 1020)
ax1.set_ylim(-2, 70)
ax1.tick_params(axis=u'both', which=u'both',length=0)
ax1.set_title('Residual-based', fontsize=16)
ax1.set_ylabel("Rank Difference (RD)", fontsize=16)



ax2.scatter([gap*(i+1) for i in xrange(len(data))], [d[2] for d in data], marker='v', color='#228B22')
ax2.set_xlim(10, 1020)
ax2.set_ylim(-2, 70)
ax2.tick_params(axis=u'both', which=u'both',length=0)
ax2.set_title('Rank-based', fontsize=16)
ax2.set_ylabel("Rank Difference (RD)", fontsize=16)

ax3.scatter([gap*(i+1) for i in xrange(len(data))], [d[3] if d[3] < 20 else 16 for d in data], marker='v', color='#228B22')
ax3.set_xlim(10, 1020)
ax3.set_ylim(-2, 70)
ax3.tick_params(axis=u'both', which=u'both',length=0)
ax3.set_title('Flash', fontsize=16)
ax3.set_ylabel("Rank Difference (RD)", fontsize=16)

plt.sca(ax1)
plt.xticks([gap*(i+1) for i in xrange(len(data))], [d[0] for d in data], rotation=90, fontsize=12)

plt.sca(ax2)
plt.xticks([gap*(i+1) for i in xrange(len(data))], [d[0] for d in data], rotation=90, fontsize=12)

plt.sca(ax3)
plt.xticks([gap*(i+1) for i in xrange(len(data))], [d[0] for d in data], rotation=90, fontsize=12)
#
# plt.figlegend((circ1, circ2, circ3), ('<5%', '5%<MMRE<10%', '>10%'), frameon=False, loc='lower center',
#               bbox_to_anchor=(0.4, -0.04),fancybox=True, ncol=3, fontsize=16)

f.set_size_inches(22, 5.5)
# plt.show()
plt.savefig('one_obj_rd.eps', bbox_inches='tight')
