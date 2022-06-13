## Literature Review Notes

### Keep

#### A Comprehensive Survey on Graph Anomaly Detection with Deep Learning (2021)

##### Contributions

- First survey in graph anomaly detection with deep learning
- Offers a taxonomy for graph anomaly detection tasks
    - Anomalous Nodes
    - Anomalous Edges
    - Anomalous Subgraphs
    - Anomalous Graphs
- Taxonomy of graph types
    - plain / attributed
    - static / dynamic
    
> "...because the copious types of graph anomalies cannot be directly represented in Euclidean feature space, it is not feasible to directly apply traditional anomaly detection techniques to graph anomaly detection, and researchers have intensified their efforts to GAD recently."

#### Deep Learning for Anomaly Detection: A Review (2020)
https://dl.acm.org/doi/10.1145/3439950

- Covers **node detection** with GADL (few references)
- Covers **edge detection** with GADL (few references)

> Non-deep learning based techniques also lack the capability to capture the non-linear properties of real objects [24]. Hence, the representations of objects learned by them are not expressive enough to fully support graph anomaly detection.
Describes a taxonomy of graph anomaly detection tasks:

#### Fraud detection: A systematic literature review of graph-based anomaly detection approaches (2020)

- Covers **node detection** with GADL (many references)

#### A Unifying Review of Deep and Shallow Anomaly Detection (2022)

Divides methods into 

- Classification
- Probabilistic
- Reconstruction
- Distance

> A conditional or contextual anomaly is a data instance that is anomalous in a specific context, such as time, space, or the connections in a graph.

**Note that most of these methods are not suitable for the anomalous node detection task**


#### Anomaly Detection in Graphs of Bank Transactions for Anti Money Laundering Applications

- Labelled dataset of genuine bank transaction
- Method for injecting typical anomaly patterns (simulation)
- Uses traditional methods only: isolation forest
- Generated egonet features

#### Deep Learning for Anomaly Detection: A Survey (2019)

#### Financial Crime

## Other

### Towards Federated Graph Learning for Collaborative Financial Crimes Detection

- We demonstrated that our federated model outperforms local model by 20% with the UK FCA TechSprint data set. This new platform opens up a door to efficiently detecting global money laundering activity.

