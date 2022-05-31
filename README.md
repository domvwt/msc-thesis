# UK Companies Graph Study

## TODO

- create graph structure of company ownership
- use networkx / igraph to start with?
- may need to sample data for persons with > 1 interest
- use embeddings for names and addresses
- get lat long for postcode(?)
- join companies house data
- join open corporates data(?)

## Data

|Data                           |Provider       |Date Retrieved|
|:------------------------------|:--------------|:-------------|
|[Company Data][CompanyData]    |Companies House|2022-05-24    |
|[PSC Data][PSCData]            |Companies House|2022-05-24    |
|[Ownership Data][OwnershipData]|Open Ownership |2022-05-24    |

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
 <!-- Links -->
[CompanyData]: http://download.companieshouse.gov.uk/BasicCompanyDataAsOneFile-2022-05-01.zip
[PSCData]: http://download.companieshouse.gov.uk/persons-with-significant-control-snapshot-2022-05-24.zip
[OwnershipData]: https://oo-register-production.s3-eu-west-1.amazonaws.com/public/exports/statements.2022-05-27T19:23:50Z.jsonl.gz