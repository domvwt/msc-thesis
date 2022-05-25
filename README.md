# UK Companies Graph Study

## TODO

- create graph structure of company ownership
- use networkx / igraph to start with?
- may need to sample data for persons with > 1 interest
- use embeddings for names and addresses
- get lat long for postcode

## Data

Put this into a table

- <http://download.companieshouse.gov.uk/BasicCompanyDataAsOneFile-2022-05-01.zip> - retrieved 2022-05-24
- <http://download.companieshouse.gov.uk/persons-with-significant-control-snapshot-2022-05-24.zip> - retrieved 2022-05-24
- [Open Ownership Data](https://oo-register-production.s3-eu-west-1.amazonaws.com/public/exports/statements.latest.jsonl.gz) - retrieved 2022-05-24

## Libraries

- PyTorch
  - [PT-Geometric](https://github.com/pyg-team/pytorch_geometric) - stars: 14.7k
- Tensorflow
  - [Graph-Nets](https://github.com/deepmind/graph_nets) - stars: 5.1k
  - [Spektral](https://keras.io/examples/graph/gnn_citations/) - stars: 2.1k
  - [Stellar](https://keras.io/examples/graph/gnn_citations/) - stars: 2.4k
  - [TF-GNN](https://github.com/tensorflow/gnn) - stars: under 1k
- Backend Agnostic
  - [DGL](https://github.com/dmlc/dgl) - stars: 9.6k

## Other Dependencies

```bash
# SciPy dependencies
sudo apt install gcc gfortran python3-dev libopenblas-dev liblapack-dev cython
```

```bash
# Graph plotting
sudo apt install graphviz
```
