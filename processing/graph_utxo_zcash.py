# %%
import struct
import numpy as np
import networkx as nx
import scipy
import csv
import gzip
import pandas as pd
from scipy.stats import entropy
from scipy.sparse import csr_matrix
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt

# %%
all_in = []
all_out = []
for i in range(1,8):
    day1Inputs = "../data/zcash/inputs/blockchair_zcash_inputs_2021020"+str(i)+".tsv.gz"
    day1Outputs = "../data/zcash/outputs/blockchair_zcash_outputs_2021020"+str(i)+".tsv.gz"

    inputs = pd.read_csv(day1Inputs, sep = '\t')
    inputs = inputs.drop_duplicates(subset=['transaction_hash', 'index'])
    all_in.append(inputs)
    outputs = pd.read_csv(day1Outputs, sep = '\t')
    outputs = outputs.drop_duplicates(subset=['transaction_hash', 'index'])
    all_out.append(outputs)

# %%
all_txs = []
for i in range(1,8):
    day1tx = "../data/zcash/transactions/blockchair_zcash_transactions_2021020"+str(i)+".tsv.gz"

    txs_ = pd.read_csv(day1tx, sep = '\t')
    txs_ = txs_.drop_duplicates(subset=['hash'])
    all_txs.append(txs_)

# %%
inputs = pd.concat(all_in, axis=0)
outputs = pd.concat(all_out, axis=0)

txs = pd.concat(all_txs, axis=0)

# %%
## Address-based BTC Tx Graph
G = nx.DiGraph()

for value, txHash, index, spendingTxHash in tqdm(inputs[['value', 'transaction_hash', 'index', 'spending_transaction_hash']].values):
    if spendingTxHash not in G:
        G.add_node(spendingTxHash)
    if (txHash, index) not in G:
        G.add_node((txHash, index))
    G.add_edge((txHash, index), spendingTxHash, weight = value)

for value, txHash, index in tqdm(outputs[['value', 'transaction_hash', 'index']].values):
    if txHash not in G:
        G.add_node(txHash)
    if (txHash, index) not in G:
        G.add_node((txHash, index))
    G.add_edge(txHash, (txHash, index), weight = value)

# %%
G.add_node(("shielded_pool",0))

# %%
shielded_balance = pd.read_csv('../data/sqlite_outputs/zcash_shielded_sum.tsv.gz', sep="\t", header=None)
shielded_balance_original = shielded_balance
shielded_balance = shielded_balance.iloc[0][0]

# %%
sp_index = 0

# %%
for txHash, shielded_delta in txs[txs['shielded_value_delta']!=0][['hash','shielded_value_delta']].values:
    if shielded_delta > 0:
        G.add_node(("shielded_pool",sp_index+1))
        G.add_edge(txHash, ("shielded_pool", sp_index+1), weight=shielded_delta)
        G.add_edge(("shielded_pool", sp_index), ("shielded_pool", sp_index+1), weight=shielded_balance)
        sp_index += 1
    elif shielded_delta < 0:
        G.add_edge(("shielded_pool", sp_index), txHash, weight=-shielded_delta)
    shielded_balance += shielded_delta

# %%
print("Writing the graph to file")
nx.write_gpickle(G, "graphs/graph_utxo_zcash.gpickle")
print("Finished")

# %%
G.nodes[('shielded_pool',0)]['balance'] = shielded_balance_original.iloc[0][0]
print("Writing the graph to file")
nx.write_gpickle(G, "graph/graph_utxo_zcash_spb.gpickle")
print("Finished")

# %%



