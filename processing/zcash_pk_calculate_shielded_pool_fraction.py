# %%
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4 
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6

import struct
import numpy as np
import pandas as pd
import networkx as nx
import scipy
from scipy.stats import entropy
from scipy.sparse import csr_matrix 
from tqdm import tqdm
from matplotlib import pyplot as plt
from collections import defaultdict
import pickle
import argparse
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

from functions import absorbing_markov, reduce_Q

parser = argparse.ArgumentParser(description='Run demo')
parser.add_argument('--nx_gpickle_file', type=str, default="", help='networkx pickle file to process')

args = parser.parse_args('--nx_gpickle_file graph/graph_pk_zcash_spb.gpickle'.split(' '))

np.random.seed(123)

G = nx.read_gpickle(args.nx_gpickle_file)

# %%
print("Deleting zero sat edges")
zeroSats=0
toBeDeletedEdges = []
for e in tqdm(G.edges.data("weight")):
    if e[2]==0:
        zeroSats+=1
        toBeDeletedEdges.append(e)
G.remove_edges_from(toBeDeletedEdges)
print("Number of transactions with 0 Satoshi outputs", zeroSats)
print("Number of remaining edges:",G.size())

#print("finding connected components")
#connected_components = sorted(nx.weakly_connected_components(G), key=len)

from scipy.sparse import identity
from scipy.sparse.linalg import inv

print("Transforming to markov chain")

I, Q, R, node_recode, absorber_recode = absorbing_markov(G)

# %%
i=absorber_recode['shielded_pool']-Q.shape[1]

# %%
Q_slice, R_slice, slice_size, conv_lim, verbose = Q, R[:, i:i+1], 1, 1e-6, True

slices = range(0, R_slice.shape[1], slice_size)
if verbose:
    slices = tqdm(slices)
for s in slices:
    qkr = R_slice[:, s:s+slice_size].todense()
    coll = qkr.copy()
    while True:
        qkr = Q_slice*qkr
        coll += qkr
        if(qkr.max() <= conv_lim):
            break

# %%
shielded_pool_absorbtion_ratio = coll
shielded_pool_absorbtion_ratio = np.array(shielded_pool_absorbtion_ratio).squeeze()

# %%
entropies = np.load('zcash_pk/entropies.npy')
unspent = np.load('zcash_pk/unspent.npy')

# %%
entropies_utxo = entropies[unspent]

# %%
entropies_utxo.mean()

# %%
shielded_pool_absorbtion_ratio[unspent].mean()

# %%
import pickle
with open('zcash_pk/recode', 'rb') as handle:
    recode = pickle.load(handle)

# %%
shielded_pool_id = recode['shielded_pool']

# %% [markdown]
# ### To calculate corrected entropy:
# 
# Paths absorbed in the shielded pool all continue outside of the timeframe. Let's assume the nodes reached this way are disjunct from the nodes reached throught other paths.
# 
# Original value:
# $$
# \textrm{entropy} = -\sum_{i!=k} log(p_i) p_i - p_k log(p_k)
# $$
# 
# We want to modify this entropy so as if the absorbed paths continued according to distribution $r_{k}$:
# $$
# \textrm{entropy'} = -\sum_{i!=k} log(p_i) p_i - \sum_{j} log(p_k r_{k_j}) p_k r_{k_j}
# $$
# where
# $$
# \sum_{j} log(p_k r_{k_j}) p_k r_{k_j} = p_k\sum_j (\log(p_k)+\log(r_{k_j}))r_{k_j} = p_k \log(p_k)\sum_j r_{k_j} + p_k\sum_j \log(r_{k_j})r_{k_j} = \\
# = p_k log(p_k) - p_k \left(-\sum_j log(r_{k_j})r_{k_j}\right) = p_k log(p_k) - p_k \textrm{pool_entropy}
# $$
# 
# This yields
# $$
# \textrm{entropy'} = \textrm{entropy} + p_k log(p_k) - (p_k log(p_k) - p_k \textrm{pool_entropy}) = \textrm{entropy} + p_k \textrm{pool_entropy}
# $$

# %%
pool_out_values = pd.read_csv('../data/sqlite_outputs/zcash_shielded_input_distribution.tsv.gz', sep="\t", header=None)

# %%
pool_entropy = scipy.stats.entropy(pool_out_values)[0]
pool_entropy

# %%
entropies_modified = entropies + shielded_pool_absorbtion_ratio*pool_entropy

# %%
entropies[unspent].mean()

# %%
entropies_modified[unspent].mean()

# %%
np.save('zcash_pk/entropies_modified', entropies_modified)

# %%



