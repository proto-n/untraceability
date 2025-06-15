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
print("Writing the graph to file")
nx.write_gpickle(G, "graph/graph_utxo_bitcoin.gpickle")
print("Finished")

# %%



