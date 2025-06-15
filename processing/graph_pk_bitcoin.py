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
    day1Inputs = "../data/bitcoin/inputs/blockchair_bitcoin_inputs_2021020"+str(i)+".tsv.gz"
    day1Outputs = "../data/bitcoin/outputs/blockchair_bitcoin_outputs_2021020"+str(i)+".tsv.gz"

    inputs = pd.read_csv(day1Inputs, sep = '\t')
    inputs = inputs.drop_duplicates(subset=['transaction_hash', 'index'])
    all_in.append(inputs)
    outputs = pd.read_csv(day1Outputs, sep = '\t')
    outputs = outputs.drop_duplicates(subset=['transaction_hash', 'index'])
    all_out.append(outputs)

# %%
inputs = pd.concat(all_in, axis=0)
outputs = pd.concat(all_out, axis=0)

# %%
inputs_agg = inputs.groupby(['spending_transaction_hash', 'recipient'])['value'].sum().reset_index()
outputs_agg = outputs.groupby(['transaction_hash', 'recipient'])['value'].sum().reset_index()

# %%
## Address-based BTC Tx Graph
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
balance_inputs = pd.read_csv('../data/sqlite_outputs/bitcoin_inputs_sum.tsv.gz', sep="\t", header=None)
balance_outputs = pd.read_csv('../data/sqlite_outputs/bitcoin_outputs_sum.tsv.gz', sep="\t", header=None)

# %%
balance_inputs = balance_inputs.set_index(0).rename(columns={1: 'out_amount'})
balance_outputs = balance_outputs.set_index(0).rename(columns={1: 'in_amount'})
balance_joined = balance_inputs.join(balance_outputs, how="outer")
balance_joined.fillna(0, inplace=True)
balance_joined['balance'] = balance_joined['in_amount']-balance_joined['out_amount']

# %%
for node, balance in tqdm(balance_joined.reset_index()[[0, 'balance']].values):
    G.nodes[node]['balance'] = balance

# %%
print("Writing the graph to file")
nx.write_gpickle(G, "graph/graph_pk_bitcoin.gpickle")
print("Finished")

# %%



