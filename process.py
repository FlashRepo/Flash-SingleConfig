content = open('residual-based.py').readlines()


files = [
    # './Data/SS-M1.csv', './Data/SS-M2.csv', './Data/SS-N1.csv',
    './Data/SS-N2.csv', './Data/SS-O1.csv',
    # './Data/SS-O2.csv'
]
for file in files:
    for r in xrange(30):
        modified = []
        for i,c in enumerate(content):
            if i == 144:
                modified += "        pickle.dump(stats_dict, open('./PickleLocker/Residual_Stats_' + file.split('/')[-1] + '" + str(r) + ".p', 'w'))\n"
            elif i == 112:
                modified += "    files = ['" + file +"']\n"
            else:
                modified += c

        mfile = file.split('-')[-1].replace('.csv', '')
        nfile = 'auto-residual-based-' + mfile + '-' + str(r) + '.py'
        f = open(nfile, 'w')
        for m in modified:
            f.write("%s" % m)
        f.close()

#
# content = open('rank-based.py').readlines()
# #
# #
# files = ['./Data/SS-M1.csv', './Data/SS-M2.csv', './Data/SS-N1.csv', './Data/SS-N2.csv', './Data/SS-O1.csv', './Data/SS-O2.csv']
# for file in files:
#     for r in xrange(30):
#         modified = []
#         for i,c in enumerate(content):
#             if i == 149:
#                 modified += "        pickle.dump(stats_dict, open('./PickleLocker/Rank_Stats_' + file.split('/')[-1] + '" + str(r) + ".p', 'w'))\n"
#             elif i == 118:
#                 modified += "    files = ['" + file +"']\n"
#             else:
#                 modified += c
#
#         mfile = file.split('-')[-1].replace('.csv', '')
#         nfile = 'auto-rank-based-' + mfile + '-' + str(r) + '.py'
#         f = open(nfile, 'w')
#         for m in modified:
#             f.write("%s" % m)
#         f.close()