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

from functions import absorbing_markov, reduce_Q

parser = argparse.ArgumentParser(description='Run demo')
parser.add_argument('--nx_gpickle_file', type=str, default="", help='networkx pickle file to process')
parser.add_argument('--workers', type=int, default=50, help='number of threads to use')
parser.add_argument('--precision', type=float, default=0.001, help='precision of convergence')
parser.add_argument('--save_entropies', type=str, default="entropies", help='file to save entropies to')
parser.add_argument('--save_recode', type=str, default="recode", help='file to save node recode dict to')
parser.add_argument('--save_absorber_recode', type=str, default="absorber_recode", help='file to save absorber recode dict to')
parser.add_argument('--save_unspent_list', type=str, default="unspent", help='file to save unspent recoded nodes')


args = parser.parse_args()

G = nx.read_gpickle(args.nx_gpickle_file)

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

print("Reducing graph")
Q_reduced, R_reduced, I_reduced, reduction_mapping = reduce_Q(Q, R)

from scipy import special
import pathos.multiprocessing as multiprocessing
from pathos.pools import ProcessPool
from time import sleep

workers = args.workers

pool = ProcessPool(workers)
step = 16
worker_step = R_reduced.shape[1]//workers + 1

def do_approximate_partial_entropy(args):
    Q, R, slice_size, conv_lim, verbose = args
    entropies = np.zeros(R_reduced.shape[0])
    slices = range(0, R.shape[1], slice_size)
    if verbose:
        slices = tqdm(slices)
    for i in slices:
        qkr = R[:, i:i+slice_size].todense()
        coll = qkr.copy()
        while True:
            qkr = Q*qkr
            coll += qkr
            if(qkr.max() <= conv_lim):
                break
        entropies += special.entr(np.asarray(coll)).sum(axis=1)
    return entropies

precision = args.precision

regular_task_list = [
    (Q_reduced, R_reduced[:, i:i+worker_step], step, precision, i==0) for i in range(0, R_reduced.shape[0], worker_step)
]
print("Starting calculations")
partial_entropies = list(pool.map(do_approximate_partial_entropy, regular_task_list))
pool.close()
pool.join()

print("Done, saving results")

entropies = np.vstack(partial_entropies).sum(axis=0)
entropies_unreduced = entropies[reduction_mapping]
np.save(args.save_entropies, entropies_unreduced)
with open(args.save_recode, 'wb') as handle:
    pickle.dump(node_recode, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open(args.save_absorber_recode, 'wb') as handle:
    pickle.dump(absorber_recode, handle, protocol=pickle.HIGHEST_PROTOCOL)

weights = list(G.edges.data('weight'))
weights_out_minus_in = defaultdict(lambda:0)
for n1, n2, w in weights:
    weights_out_minus_in[n1] += w
    weights_out_minus_in[n2] -= w
has_unspent = [n for n, w in weights_out_minus_in.items() if w < 0]
has_unspent_recode = [node_recode[n] for n in has_unspent]
np.save(args.save_unspent_list, np.array(has_unspent_recode))


