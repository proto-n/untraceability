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
import math

# %%
input_sum = pd.read_csv(
    '../data/sqlite_outputs/erc20_inputs_sum.tsv.gz',
    sep="\t",
    header=None,
    names=['sender', 'token_address', 'value']
)

# %%
input_sum_dai = input_sum[input_sum['token_address'] == '6B175474E89094C44Da98b954EedeAC495271d0F'.lower()].drop('token_address', axis=1)

# %%
output_sum = pd.read_csv(
    '../data/sqlite_outputs/erc20_outputs_sum.tsv.gz',
    sep="\t",
    header=None,
    names=['recipient', 'token_address', 'value']
)

# %%
output_sum_dai = output_sum[output_sum['token_address'] == '6B175474E89094C44Da98b954EedeAC495271d0F'.lower()].drop('token_address', axis=1)

# %%
input_sum_dai['value'] = [int(i)/10**6 for i in input_sum_dai['value']]
output_sum_dai['value'] = [int(i)/10**6 for i in output_sum_dai['value']]

# %%
all_tx = []
for i in range(1,8):
    daytx = "../data/ethereum/erc-20/transactions/blockchair_erc-20_transactions_2021020"+str(i)+".tsv.gz"
    tx = pd.read_csv(daytx, sep = '\t')
    tx = tx[tx['token_address']=='6B175474E89094C44Da98b954EedeAC495271d0F'.lower()]
    
    all_tx.append(tx)

txs = pd.concat(all_tx, axis=0)
txs['value'] = [int(i)/10**6 for i in txs['value']]

# %%
erc20 = nx.DiGraph()

for value, sender, recipient in tqdm(txs[['value', 'sender', 'recipient']].values):
    if sender not in erc20:
        erc20.add_node(sender)
    if recipient not in erc20:
        erc20.add_node(recipient)
    if erc20.has_edge(sender, recipient):
        erc20[sender][recipient]['weight'] += value
    else:
        erc20.add_edge(sender, recipient, weight = value)

# %%
output_sum_dai = output_sum_dai.rename(columns={'value':'in_amount'})
input_sum_dai = input_sum_dai.rename(columns={'value':'out_amount'})

# %%
output_sum_dai = output_sum_dai.set_index('recipient')
input_sum_dai = input_sum_dai.set_index('sender')

# %%
balance_joined = input_sum_dai[['out_amount']].join(output_sum_dai[['in_amount']], how="outer")
balance_joined.fillna(0, inplace=True)
balance_joined['balance'] = balance_joined['in_amount']-balance_joined['out_amount']

# %%
(balance_joined['balance']>=0).value_counts()

# %%
# corrupted files
balance_joined[balance_joined['balance']<0]

# %%
for node, balance in tqdm(balance_joined.reset_index()[['index', 'balance']].values):
    if(type(node)==str and node in erc20.nodes and balance > 0):
        erc20.nodes[node]['balance'] = balance

# %%
print("Writing the graph to file")
nx.write_gpickle(erc20, "graph/graph_pk_dai.gpickle")
print("Finished")

# %%



