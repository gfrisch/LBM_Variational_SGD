"""
    Experiment comparing V-EM inference and V-SGD on the Latent Block Model.

    Dependency on SparseBM module for V-EM inference.
"""

from sparsebm import generate_LBM_dataset
from sparsebm import LBM
from sparsebm.utils import CARI
from lbm_binary import LbmBernoulli
import numpy as np
import time
import pickle
import os
import torch
import multiprocessing as mp

torch.set_num_threads(1)
device = torch.device("cpu")

directory = "./res_init_lbm_binary_stochastic"
if not os.path.exists(directory):
    try:
        os.makedirs(directory)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise

n1, n2, nq, nl = 1000, 1000, 4, 4


def process_exp(eps, repeat):
    pi_sim = np.array(
        [
            [eps, eps, eps, 1 - eps],
            [eps, eps, 1 - eps, eps],
            [eps, 1 - eps, 1 - eps, eps],
            [eps, eps, eps, eps],
        ]
    )
    results = {"eps": eps, "repeat": repeat}
    print(
        f"---------------- REPEAT {repeat}/100 eps {eps}------------------------"
    )
    dataset = generate_LBM_dataset(
        number_of_rows=n1,
        number_of_columns=n1,
        nb_row_clusters=nq,
        nb_column_clusters=nl,
        row_cluster_proportions=np.array([0.25] * 4),
        column_cluster_proportions=np.array([0.25] * 4),
        connection_probabilities=pi_sim,
        sparse=False,
        verbosity=0,
    )
    graph = dataset["data"]
    X = np.array(graph.todense())
    row_cluster_indicator = dataset["row_cluster_indicator"]
    column_cluster_indicator = dataset["column_cluster_indicator"]
    number_of_row_clusters = row_cluster_indicator.shape[1]
    number_of_columns_clusters = column_cluster_indicator.shape[1]

    model = LBM(
        nq,
        nl,
        n_init_total_run=10,
        n_iter_early_stop=10,
        n_init=100,
        verbosity=0,
        use_gpu=False,
    )
    start_time = time.time()
    model.fit(graph)
    sparsebm_time = time.time() - start_time

    sparsebm_cari = CARI(
        row_cluster_indicator.argmax(1),
        column_cluster_indicator.argmax(1),
        model.row_labels,
        model.column_labels,
    )
    print("Co-Adjusted Rand index is {:.2f}".format(sparsebm_cari))

    def callback(model, epoch):
        cari = CARI(
            row_cluster_indicator.argmax(1),
            column_cluster_indicator.argmax(1),
            model.tau_1.argmax(1).cpu().detach().numpy(),
            model.tau_2.argmax(1).cpu().detach().numpy(),
        )
        print("Co-Adjusted Rand index is {:.2f}".format(cari))
        return cari

    model_sto = LbmBernoulli(device=device)
    start_time = time.time()
    model_sto.fit(X, nq, nl, lr=5e-2)
    sto_time = time.time() - start_time
    sto_cari = callback(model_sto, 1)

    results.update(
        {
            "sparsebm_time": sparsebm_time,
            "sparsebm_cari": sparsebm_cari,
            "sto_cari": sto_cari,
            "sto_time": sto_time,
        }
    )

    pickle.dump(
        results,
        open(
            directory + "/eps_" + str(eps) + "_repeat_" + str(repeat) + ".pkl",
            "wb",
        ),
    )


eps = np.linspace(0.35, 0.49, 10)
iterable = [(ep, i) for ep in eps for i in range(0, 50)]
print("Number of processors: ", mp.cpu_count())
pool = mp.Pool(mp.cpu_count())
results = pool.starmap_async(process_exp, iterable).get()
pool.close()
print("finished")

###########################################################################
######################   Plotting results         #########################
###########################################################################

directory = "./res_init_lbm_binary_stochastic"

import pickle
import glob
import numpy as np
from collections import defaultdict

files = sorted(glob.glob(directory + "/*.pkl"))
results = defaultdict(list)
result_array = []
for f in sorted(files):
    data = pickle.load(open(f, "rb"))
    results[data["eps"]].append(
        {
            data["eps"],
            data["repeat"],
            data["sparsebm_time"],
            data["sparsebm_cari"],
            data["sto_cari"],
            data["sto_time"],
        }
    )
    result_array.append(
        [data["eps"], data["sto_cari"], "Stochastic", data["sto_time"]]
    )
    result_array.append(
        [data["eps"], data["sparsebm_cari"], "Standard", data["sparsebm_time"]]
    )
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

result_array = pd.DataFrame(
    result_array, columns=["epsilon", "CARI", "Inference", "Time"]
)
fig, ax = plt.subplots(figsize=(4, 3))
g = sns.pointplot(
    y="CARI",
    x="epsilon",
    hue="Inference",
    data=result_array,
    ax=ax,
    palette="Paired",
    capsize=0.2,
    scale=0.6,
    errwidth=0.5,
    markers=["*", "+", "x"],
    linestyles=["solid", "dotted"],
)
for l in g.get_lines():
    plt.setp(l, linewidth=0.5)
labels = ax.get_xticklabels()  # get x labels
for i, l in enumerate(labels):
    labels[i] = round(sorted(result_array.epsilon.unique())[i], 2)
    if i % 3 != 0:
        labels[i] = ""  # skip even labels
ax.set_xticklabels(labels)  # set new labels

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_ylabel("Coclustering Adjusted Rand Index")
ax.set_xlabel("$\epsilon$")
ax.set_ylim(0, 1.1)
fig.tight_layout()
plt.show()
