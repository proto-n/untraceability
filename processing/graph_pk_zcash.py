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
inputs_agg = inputs.groupby(['spending_transaction_hash', 'recipient'])['value'].sum().reset_index()
outputs_agg = outputs.groupby(['transaction_hash', 'recipient'])['value'].sum().reset_index()

# %%
G = nx.DiGraph()

for recipient, value, txHash in tqdm(inputs_agg[['recipient', 'value', 'spending_transaction_hash']].values):
    if txHash not in G:
        G.add_node(txHash)
    if recipient not in G:
        G.add_node(recipient)
    G.add_edge(recipient, txHash, weight = value)

for recipient, value, txHash in tqdm(outputs_agg[['recipient', 'value', 'transaction_hash']].values):
    if txHash not in G:
        G.add_node(txHash)
    if recipient not in G:
        G.add_node(recipient)
    G.add_edge(txHash, recipient, weight = value)

# %%
G.add_node("shielded_pool")

# %%
for txHash, shielded_delta in txs[txs['shielded_value_delta']!=0][['hash','shielded_value_delta']].values:
    if shielded_delta > 0:
        G.add_edge(txHash, "shielded_pool", weight=shielded_delta)
    elif shielded_delta < 0:
        G.add_edge("shielded_pool", txHash, weight=-shielded_delta)

# %%
balance_inputs = pd.read_csv('../data/sqlite_outputs/zcash_inputs_sum.tsv.gz', sep="\t", header=None)
balance_outputs = pd.read_csv('../data/sqlite_outputs/zcash_outputs_sum.tsv.gz', sep="\t", header=None)

# %%
balance_inputs = balance_inputs.set_index(0).rename(columns={1: 'out_amount'})
balance_outputs = balance_outputs.set_index(0).rename(columns={1: 'in_amount'})
balance_joined = balance_inputs.join(balance_outputs, how="outer")
balance_joined.fillna(0, inplace=True)
balance_joined['balance'] = balance_joined['in_amount']-balance_joined['out_amount']

# %%
balance_nonzero = balance_joined[balance_joined['balance']!=0]

# %%
outputs[outputs['recipient'].str.startswith('d-')].iloc[0]['transaction_hash']

# %%
for node, balance in tqdm(balance_nonzero.reset_index()[[0, 'balance']].values):
    if node in G.nodes:
        G.nodes[node]['balance'] = balance

# %%
shielded_balance = pd.read_csv('../data/sqlite_outputs/zcash_shielded_sum.tsv.gz', sep="\t", header=None)

# %%
shielded_balance

# %%
# lets not add it for now

# %%
print("Writing the graph to file")
nx.write_gpickle(G, "graph/graph_pk_zcash.gpickle")
print("Finished")

# %%
G.nodes['shielded_pool']['balance'] = shielded_balance[0].values[0]
print("Writing the graph to file")
nx.write_gpickle(G, "graph/graph_pk_zcash_spb.gpickle")
print("Finished")

# %%



