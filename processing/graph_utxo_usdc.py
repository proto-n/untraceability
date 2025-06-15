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
from collections import defaultdict

# %%
import math

# %%
input_sum = pd.read_csv(
    '../data/sqlite_outputs/erc20_inputs_sum.tsv.gz',
    sep="\t",
    header=None,
    names=['sender', 'token_address', 'value']
)

# %%
input_sum_usdc = input_sum[input_sum['token_address'] == 'A0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48'.lower()].drop('token_address', axis=1)

# %%
output_sum = pd.read_csv(
    '../data/sqlite_outputs/erc20_outputs_sum.tsv.gz',
    sep="\t",
    header=None,
    names=['recipient', 'token_address', 'value']
)

# %%
output_sum_usdc = output_sum[output_sum['token_address'] == 'A0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48'.lower()].drop('token_address', axis=1)

# %%
input_sum_usdc['value'] = [int(i)/10**6 for i in input_sum_usdc['value']]
output_sum_usdc['value'] = [int(i)/10**6 for i in output_sum_usdc['value']]

# %%
output_sum_usdc = output_sum_usdc.rename(columns={'value':'in_amount'})
input_sum_usdc = input_sum_usdc.rename(columns={'value':'out_amount'})

# %%
output_sum_usdc = output_sum_usdc.set_index('recipient')
input_sum_usdc = input_sum_usdc.set_index('sender')

# %%
balance_joined = input_sum_usdc[['out_amount']].join(output_sum_usdc[['in_amount']], how="outer")
balance_joined.fillna(0, inplace=True)
balance_joined['balance'] = balance_joined['in_amount']-balance_joined['out_amount']

# %%
all_tx = []
for i in range(1,8):
    daytx = "../data/ethereum/erc-20/transactions/blockchair_erc-20_transactions_2021020"+str(i)+".tsv.gz"
    tx = pd.read_csv(daytx, sep = '\t')
    tx = tx[tx['token_address']=='A0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48'.lower()]
    
    all_tx.append(tx)

txs = pd.concat(all_tx, axis=0)
txs['value'] = [int(i)/10**6 for i in txs['value']]

# %%
erc20 = nx.DiGraph()

# %%
balance_joined_nonzero = balance_joined.copy()
balance_joined_nonzero.loc[balance_joined_nonzero['balance']<0, 'balance'] = 0

# %%
node_last_index = defaultdict(lambda: 0)
balances = dict(balance_joined_nonzero.reset_index()[['index','balance']].values)

# %%
for value, sender, recipient in tqdm(txs[['value', 'sender', 'recipient']].values):
    if(value == 0):
        continue

    sender_last = node_last_index[sender]
    recipient_last = node_last_index[recipient]
    if (sender, sender_last) not in erc20:
        erc20.add_node((sender, sender_last))
    if (recipient, recipient_last) not in erc20:
        erc20.add_node((recipient, recipient_last))
    if (recipient, recipient_last+1) not in erc20:
        erc20.add_node((recipient, recipient_last+1))

    node_last_index[recipient] += 1
    
    bal = balances.get(recipient, 0)
    erc20.add_edge((recipient, recipient_last), (recipient, recipient_last+1), weight = bal)
    erc20.add_edge((sender, sender_last), (recipient, recipient_last+1), weight = value)
    balances[recipient] = balances.get(recipient, 0) + value
    balances[sender] = balances.get(sender, 0) + value

# %%
for node, balance in tqdm(balance_joined.reset_index()[['index', 'balance']].values):
    if(type(node)==str and (node, 0) in erc20.nodes and balance > 0):
        erc20.nodes[(node,0)]['balance'] = balance

# %%
print("Writing the graph to file")
nx.write_gpickle(erc20, "graph/graph_utxo_usdc.gpickle")
print("Finished")

