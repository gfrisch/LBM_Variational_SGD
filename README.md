# Latent Block Model with varitional SGD

Inference of the Latent block model with a stochastic gradient descent on the variational criterion

### Requires pytorch module !

## Example of use:
- with the following example, the module SparseBM is necessary (to be installed with pip)
```python
import sparsebm
import numpy as np
dataset = sparsebm.generate_LBM_dataset()
X = np.array(dataset["data"].todense(), dtype=int) # Matrix to co-cluster.
nb_row_clusters = dataset['row_cluster_indicator'].shape[1]
nb_row_clusters = dataset['column_cluster_indicator'].shape[1]
# Initiate model
from lbm_binary import LbmBernoulli
import torch
model = LbmBernoulli(device=torch.device('cpu'))
# Fit model to the X matrix with a given nb of clusters.
model.fit(X, nb_row_clusters, nb_row_clusters, lr=5e-2)
# Get original partitions
row_cluster_indicator = dataset["row_cluster_indicator"]
column_cluster_indicator = dataset["column_cluster_indicator"]
from torch_utils import CARI
cari = CARI(
    row_cluster_indicator.argmax(1),
    column_cluster_indicator.argmax(1),
    model.tau_1.argmax(1).cpu().detach().numpy(),
    model.tau_2.argmax(1).cpu().detach().numpy(),
)
print(f"CoARI : {cari}")
```


## Compare V-EM with V-SGD:
- execute file *experiment_V-EM-V-SGD_LBM.py*
