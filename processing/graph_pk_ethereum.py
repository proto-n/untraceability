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
import math

# %%
all_call = []
for i in range(1,8):
    day1calls = "../data/ethereum/calls/blockchair_ethereum_calls_2021020"+str(i)+".tsv.gz"
    calls = pd.read_csv(day1calls, sep = '\t')
    calls = calls[(calls['failed']==0) & (calls['value']!=0)]
    
    all_call.append(calls)

calls = pd.concat(all_call, axis=0)
calls['value_gwei'] = [int(i)/10**9 for i in calls['value']]

# %%
eth = nx.DiGraph()

# %%
for value, sender, recipient in tqdm(calls[['value_gwei', 'sender', 'recipient']].values):
    if sender not in eth:
        eth.add_node(sender)
    if recipient not in eth:
        eth.add_node(recipient)
    if eth.has_edge(sender, recipient):
        eth[sender][recipient]['weight'] += value
    else:
        eth.add_edge(sender, recipient, weight = value)

# %%
call_inputs = pd.read_csv('../data/sqlite_outputs/eth_call_recipient.tsv.gz', sep="\t", header=None)
call_outputs = pd.read_csv('../data/sqlite_outputs/eth_call_sender.tsv.gz', sep="\t", header=None)

call_inputs[2] = [int(i)/10**9 for i in call_inputs[1]]
call_outputs[2] = [int(i)/10**9 for i in call_outputs[1]]

# %%
call_outputs = call_outputs.set_index(0).rename(columns={2: 'out_amount'})
call_inputs = call_inputs.set_index(0).rename(columns={2: 'in_amount'})

# %%
balance_joined = call_outputs[['out_amount']].join(call_inputs[['in_amount']], how="outer")
balance_joined.fillna(0, inplace=True)
balance_joined['balance'] = balance_joined['in_amount']-balance_joined['out_amount']

# %%
fees = pd.read_csv('../data/sqlite_outputs/eth_fee_sum.tsv.gz', sep="\t", header=None)
fees[2] = [int(i)/10**9 for i in fees[1]]

# %%
fees = fees.set_index(0).rename(columns={2: 'fee_amount'})

# %%
balance_joined = balance_joined.join(fees['fee_amount'], how="left")
balance_joined['fee_amount'].fillna(0, inplace=True)
balance_joined['balance'] -= balance_joined['fee_amount']

# %%
for node, balance in tqdm(balance_joined.reset_index()[[0, 'balance']].values):
    if(type(node)==str and node in eth.nodes and balance > 0):
        eth.nodes[node]['balance'] = balance

# %%
print("Writing the graph to file")
nx.write_gpickle(eth, "graph/graph_pk_ethereum.gpickle")
print("Finished")

# %%
balance_joined['balance'].isna().any()

# %%
balance_joined['balance'].replace([np.inf, -np.inf], np.nan).isna().any()

# %%



