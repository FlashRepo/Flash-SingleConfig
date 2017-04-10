from sk import rdivDemo
import pickle

al_eval_p = "../PickleLocker/ActiveLearning_Rank_Diff.p"
mmre_eval_p = "../PickleLocker/Progressive_MMRE_Rank_Diff.p"
rank_eval_p = "../PickleLocker/Progressive_Rank_Rank_Diff.p"


al_evals = pickle.load(open(al_eval_p, "r"))
mmre_evals = pickle.load(open(mmre_eval_p, "r"))
rank_evals = pickle.load(open(rank_eval_p, "r"))

files = sorted(al_evals.keys())


for i,file in enumerate(files):
    lists = list()
    lists.append(["AL-Rank"] + al_evals[file])
    lists.append(["P-Rank"] + rank_evals[file])
    lists.append(["P-MMRE"] + mmre_evals[file])
    rdivDemo("SS" + str(i+1), lists, globalMinMax=False, isLatex=True)
