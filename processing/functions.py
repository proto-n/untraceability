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
from scipy.sparse import identity
from scipy.sparse.linalg import inv
import pickle
import argparse
import pandas as pd

def absorbing_markov(G, exclude_source=[], verbose=False):
    """ Returns `I`, `Q`, `R`, `nodes_recode` and `absorber_recode` 
    for calculating the asorbing probabilities using compact formula.
    The value `nodes_recode` contains the mapping of nodes to indices
    in `Q`, and the value `absorber_recode`contains the mapping of
    absorbing nodes to indices in `R`.
    """

    edgelist = list(G.edges)
    weights = list(G.edges.data('weight'))

    print("Graph loaded") if verbose else None

    # edge -> weight mapping
    weights_dict = dict([((i,j), k) for i,j,k in weights])

    # recode id-s using integers [0, n]
    nodes = list(set([i for j in edgelist for i in j]))
    nodes = pd.Series([i for j in edgelist for i in j]).drop_duplicates().values.tolist()

    # finding sources (absorbers in markov terminology)
    weights_out_minus_in = defaultdict(lambda:0)
    for n1, n2, w in weights:
        weights_out_minus_in[n1] += w
        weights_out_minus_in[n2] -= w

    # create numpy mapping
    weights_out_minus_in_nodes = np.empty(len(weights_out_minus_in), dtype=object)
    weights_out_minus_in_nodes[:], weights_out_minus_in_values = zip(*weights_out_minus_in.items())
    weights_out_minus_in_values = np.array(weights_out_minus_in_values)

    weights_extra_in = np.zeros_like(weights_out_minus_in_values)
    for j,i in enumerate(weights_out_minus_in_nodes):
        weights_extra_in[j] = G.nodes[i].get('balance', 0)
        
    deficited = weights_out_minus_in_values > weights_extra_in
    weights_extra_in += np.where(deficited, (weights_out_minus_in_values-weights_extra_in), np.zeros_like(weights_extra_in))
    weights_extra_in_dict = dict(zip(weights_out_minus_in_nodes, weights_extra_in))

    weights_out_minus_in_values -= weights_extra_in
   
    #(weights_out_minus_in_values>0).sum(), (weights_out_minus_in_values==0).sum(), (weights_out_minus_in_values<0).sum()

    # list of sources
    source_nodes = weights_out_minus_in_nodes[weights_extra_in>0]
    if(exclude_source != []):
        source_nodes = np.setdiff1d(source_nodes, exclude_source)

    # simply recoding the original graph
    nodes_recode_dict = dict(zip(nodes, range(len(nodes))))
    nodes_recode_reverse_dict = dict([(b,a) for a,b in nodes_recode_dict.items()])
    edgelist_recoded = [(nodes_recode_dict[a], nodes_recode_dict[b]) for a,b in edgelist]
    weights_dict_recoded = dict([((nodes_recode_dict[a], nodes_recode_dict[b]), c) for ((a,b), c) in weights_dict.items()])

    # dummy nodes for sources, self edges for dummy nodes
    dummy_nodes = np.arange(len(source_nodes))+len(nodes)
    dummy_node_coresponding = dict(zip(source_nodes, dummy_nodes))
    dummy_node_corresponding_reverse = dict([(b,a) for a,b in dummy_node_coresponding.items()])
    dummy_edges = [(dummy_node_coresponding[s], nodes_recode_dict[s]) for s in source_nodes]
    dummy_self_edges = [(dummy_node_coresponding[s], dummy_node_coresponding[s])  for s in source_nodes]

    dummy_weights_dict = dict([((dummy_node_coresponding[n], nodes_recode_dict[n]), weights_extra_in_dict[n]) for n in source_nodes])
    dummy_self_weights_dict = dict([((dummy_node_coresponding[n], dummy_node_coresponding[n]), 1) for n in source_nodes])

    # merge real and dummy nodes, endges
    edgelist_recoded_full = edgelist_recoded + dummy_edges + dummy_self_edges
    weights_dict_recoded_full = weights_dict_recoded.copy()
    weights_dict_recoded_full.update(dummy_weights_dict)
    weights_dict_recoded_full.update(dummy_self_weights_dict)

    print("Sparse matrix") if verbose else None
    
    # original weighted adjacency matrix
    spmat = csr_matrix(([weights_dict_recoded_full[i] for i in edgelist_recoded_full], list(zip(*edgelist_recoded_full))))

    # raw sparse data and normalization
    spmat_T = spmat.T.tocsr()
    spmat_T_indices = spmat_T.nonzero() #[rows, cols]
    spmat_T_data = np.array(spmat_T[spmat_T_indices]).squeeze()

    spmat_T_rowsums = np.array(spmat_T.sum(axis=1)).squeeze()
    spmat_T_normalized = csr_matrix((spmat_T_data / spmat_T_rowsums[spmat_T_indices[0]], spmat_T_indices))

    assert(((np.array(spmat_T_normalized.sum(axis=1))-1) < 1e-6).all())
    
    # create transition matrix
    trMat = spmat_T_normalized

    abs_length = len(nodes)

    Q = trMat[0:abs_length, 0:abs_length].tocsc()
    R = trMat[0:abs_length, abs_length:].tocsc()
    O = trMat[abs_length:,0:abs_length].tocsc()
    I = identity(abs_length).tocsc()
    
    return I, Q, R, nodes_recode_dict, dummy_node_coresponding

def get_chains(QQ, RR):
    """Finding and extracting chains where there is a single outlink with
    probability 1."""
    # finding such nodes
    QQ_nnz = QQ.getnnz(axis=1)
    to_eliminate_indices = np.nonzero((QQ_nnz==1) & (RR.getnnz(axis=1)==0))[0]
    
    ## utility variables for extraction loop
    # target of single outlink
    nextnodes = np.zeros(QQ.shape[0], dtype=int)
    nextnodes[to_eliminate_indices] = QQ[to_eliminate_indices].nonzero()[1]
    # whether it is the target of such a single outlink
    nextnode_set = np.zeros(QQ.shape[0]+1, dtype=np.uint8)
    nextnode_set[QQ[to_eliminate_indices].nonzero()[1]] = 1
    # is it the end of a chain? we set this as we go
    endnode = np.zeros(QQ.shape[0]+1, dtype=np.uint8)
    # which node belongs in which chain
    chain_indices = np.zeros(len(QQ_nnz))-1

    current_chain = 0 # we increase this as we find chains
    to_connect = [] # we always connect the inputs of the first node in the chain to the last node in the chain
    to_eliminate = [] # we eliminate all but the last node in the chain
    refer_to = [] # all the nodes in the chain refer to the last
    for i in tqdm(to_eliminate_indices):
        if nextnode_set[i]: # we only start chasing the chain when starting from its first node
            continue
        chain = []
        next_node = i
        while True: # following the chain through
            # sometimes there is a merge of two chains into a single continuation
            # in these cases, we keep the original chain ids of the continuation
            # A (0) -> B (0) -> C (0) -> F (0) -> G -> A
            # D (1)-> E (1) -> C (0)
            # (chain_indices in parenthesis)
            # G here is the ending node, before it loops back to A
            
            # if we are here first, we set chain id
            if chain_indices[next_node] == -1:
                # else set to current chain
                chain_indices[next_node] = current_chain
        
            # can we move forward one node in the chain?
            next_node_candidate = nextnodes[next_node]
            if next_node_candidate == 0 or chain_indices[next_node_candidate] == current_chain or endnode[next_node] or next_node_candidate == next_node:
                # if (chain is ending) or (chain loops back onto itself) or (we have previously stopped at this point) or (next point contains loop edge)
                current_chain += 1
                endnode[next_node] = 1
                # stop chasing the chain
                break
            chain.append(next_node)
            next_node = next_node_candidate

        # we eliminate all but the last
        to_eliminate.append(chain)
        # we connect the inputs of the start of the chain to the end
        for j in chain:
            to_connect.append((j, next_node))
        # all nodes in the chain refer to the end
        refer_to.append((chain, next_node))

    # needs deduplication beacuse of the merging of chains
    to_eliminate_deduplicated = list(set([i for c in to_eliminate for i in c]))
    return to_connect, to_eliminate_deduplicated, refer_to

def reduce_Q(Q, R, copy=True, verbose=True):
    """Reduces the markov chain by eliminating nodes where there is a single
    out-edge with probability 1. This doesn't change limes the for the nodes
    which are kept. Returns the reduced `Q`, `R`, and `I` values, as well as
    the `reduction_mapping` variable. The original limes can be fond by taking
    indices from the reduced limes according to the reduction mapping."""
    if copy:
        QQ = Q.copy()
        RR = R.copy()
    else:
        QQ = Q
        RR = R
    print("Getting chains") if verbose else None
    to_connect, to_eliminate, refer_to = get_chains(Q, R)
    
    print("Reducing chains") if verbose else None

    
    # reducing the columns of Q to the ones being kept
    to_keep = np.setdiff1d(np.arange(QQ.shape[0]), to_eliminate)
    QQ_keep_cols = QQ[:, to_keep]

    # calculating the reducion mapping: which original node refers to
    # which resulting (reduced) node id. Multiple original ids refer to
    # the same reduced id (all of the chain refers to the end)
    refer_to_expanded = [(a,b) for (a_s, b) in refer_to for a in a_s]
    refer_to_ix, refer_to_val = [list(i) for i in zip(*refer_to_expanded)] 
    
    reduced_mapping = np.zeros(QQ.shape[0], dtype=int)-1
    reduced_mapping[to_keep] = np.arange(to_keep.shape[0])
    reduced_mapping[refer_to_ix] = reduced_mapping[refer_to_val]
    
    assert np.bincount(np.isin(refer_to_val, to_keep))[0]==0
    assert (reduced_mapping!=-1).all()
    

    # collecting inputs of eliminated nodes 
    connect_from, connect_to = [np.array(list(i)) for i in zip(*to_connect)]
    connect_from_unique, uniq_idx = np.unique(connect_from, return_index=True)
    connect_to_unique = np.array(connect_to)

    connect_from_Q = QQ[:, connect_from_unique]
    connect_from_nnz = connect_from_Q.nonzero()
    if getattr(connect_from_Q[connect_from_nnz], 'toarray', False):
        cfq = connect_from_Q[connect_from_nnz].toarray()
    else:
        cfq = connect_from_Q[connect_from_nnz]
    connect_from_values = np.array(cfq).squeeze(0)
    
    #assert np.setdiff1d(connect_from_nnz[0], to_keep).shape[0] == 0
    connect_from_nnz_rows, connect_from_nnz_cols = connect_from_nnz
    connect_to_cols = connect_to_unique[connect_from_nnz_cols]
    # connect_to_cols results in the col ids of chain ends according to the
    # original (non reduced) mapping
    

    QQ_keep_cols_nnz = QQ_keep_cols.nonzero()
    QQ_keep_cols_values = np.array(QQ_keep_cols[QQ_keep_cols_nnz]).squeeze(0)
    
    # adding the original kept columns and the connected inputs of eliminated nodes
    Q_reduced_cols = csr_matrix((
        np.concatenate([QQ_keep_cols_values, connect_from_values]),
        [
            np.concatenate([QQ_keep_cols_nnz[0], connect_from_nnz_rows]),
            np.concatenate([QQ_keep_cols_nnz[1], reduced_mapping[connect_to_cols]]),
        ]
    ), shape=(QQ.shape[0], to_keep.shape[0]))
    Q_reduced = Q_reduced_cols[to_keep, :]
    R_reduced = RR.tocsr()[to_keep, :]
    I = identity(Q_reduced.shape[0]).tocsc()
    
    assert (Q_reduced.sum(axis=1)>1+1e-6).sum() == 0
    
    print("Reduced size from", QQ.shape[0], "nodes to", Q_reduced.shape[0], "nodes") if verbose else None
    print("Reduction ratio: %0.2f%%" % (Q_reduced.shape[0]*100/QQ.shape[0])) if verbose else None
    return Q_reduced, R_reduced, I, reduced_mapping