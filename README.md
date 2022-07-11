# UK Companies Graph Study

## TODO

- use embeddings for names and addresses (?)
- get lat long for postcode (?)
  - <https://github.com/symerio/pgeocode>
  - need to encode neighbour location for nodes
- connected components
  - label with minimum entity id
  - split into train and test
- perturbation
  - rewire edge within component
  - create edge within component
  - create edge to external component
  - make sure ownership is not above 100% - we can do this by rewiring an existing owner's edge?
  - edge rewire vs edge creation
- message passing - permutation invariant functions
  - <https://openreview.net/pdf?id=tf8a4jDRFCv>
  - <https://antoniolonga.github.io/Pytorch_geometric_tutorials/posts/post5.html>
  - std
  - var
  - skewness
  - kurtosis
  - gini?
  - average difference?
- evaluation
  - check nodes with highest anomaly score
  - check greatest errors
  - use SHAP to determine why predictions were made (if possible?)
  - could explainability be an avenue for further research?

## Remember

[Litmaps](https://app.litmaps.com/shared/workspace/222F8A9E-DF91-4B50-BB96-1121DA98BEC4/map/5DB136D3-A5F5-49DC-AA5C-F02E42DD1A94)

## Data

|Data                           |Provider                         |Date Retrieved|
|:------------------------------|:--------------------------------|:-------------|
|[Company Data][CompanyData]    |[Companies House][CompaniesHouse]|2022-05-24    |
|[PSC Data][PSCData]            |[Companies House][CompaniesHouse]|2022-05-24    |
|[Ownership Data][OwnershipData]|[Open Ownership][OpenOwnership]  |2022-05-24    |
|[Offshore Leaks][OffshoreLeaks]|[ICIJ][ICIJ]                     |2022-06-02    |

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

### Scipy Build

```bash
# SciPy dependencies
sudo apt install gcc gfortran python3-dev libopenblas-dev liblapack-dev cython
```

### Graph Plotting

```bash
# Graph plotting
sudo apt install graphviz
```

### Graph Database - Neo4j

```bash
curl -fsSL https://debian.neo4j.com/neotechnology.gpg.key |sudo gpg --dearmor -o /usr/share/keyrings/neo4j.gpg
echo "deb [signed-by=/usr/share/keyrings/neo4j.gpg] https://debian.neo4j.com stable 4.1" | sudo tee -a /etc/apt/sources.list.d/neo4j.list
```

```bash
sudo apt update
sudo apt install neo4j
```

 <!-- Links -->

[CompaniesHouse]: https://www.gov.uk/government/organisations/companies-house
[OpenOwnership]: https://www.openownership.org/
[ICIJ]: https://offshoreleaks.icij.org/
[CompanyData]: http://download.companieshouse.gov.uk/BasicCompanyDataAsOneFile-2022-05-01.zip
[PSCData]: http://download.companieshouse.gov.uk/persons-with-significant-control-snapshot-2022-05-24.zip
[OwnershipData]: https://oo-register-production.s3-eu-west-1.amazonaws.com/public/exports/statements.2022-05-27T19:23:50Z.jsonl.gz
[OffshoreLeaks]: https://offshoreleaks-data.icij.org/offshoreleaks/csv/full-oldb.20220503.zip#_ga=2.29179839.263289515.1654130741-1571958430.1654130741
