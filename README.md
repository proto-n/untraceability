# Towards Measuring the Traceability of Cryptocurrencies

Implementation of the code for the paper [Towards Measuring the Traceability of Cryptocurrencies](https://arxiv.org/abs/2211.04259) published at ICBC'25.

## Steps to reproduce:

1. Run the scripts in `data/` in the given order. These scripts run long running downloads and long running sqlite aggregations. See more instructions in some of the scripts themselves.
2. Run any graph_* script you need in `processing/` to create the networkx graphs.
3. Run `processing/experiments/*/run.sh` from its directory to calculate untraceabilities (absorbing entropies).
4. Run `processing/zcash_*` to readjust zcash results by considering the shielded pool.
5. The notebook `processing/expected_steps_to_absorption.ipynb` creates the figures in the paper.