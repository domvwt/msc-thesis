# Further Work

- Explainability of Graph Neural Networks
  - [GNNExplainer: Generating Explanations for Graph Neural Networks](https://arxiv.org/abs/1903.03894)
  - [Graph Neural Networks: A Review of Methods and Applications](https://arxiv.org/abs/1812.08434)
  - Important for the application of GNNs in industry
- Skip connections and pooling operators
- Additional linear layers post message passing
- Stacking / Ensemble of GNNs and other models
- Dynamic graphs (e.g. temporal graphs)
  - Capturing changes in company ownership structure over time

- Additional features
  - Geographical

## Notes on the dataset

- Anomalies designed to simulate the occurrence of real entities (companies and natural persons) that are not anomalous in their own right but are anomalous in the context of the businesses that they have an interest in.
- Could be possible to use the Paradise Papers and other leaked sources to construct a dataset of genuine anomalies.
- Improve the anomaly simulation algorithm to generate more realistic anomalies and preserve the original distribution of the dataset.
- Frequency of anomalies is a parameter of the simulation algorithm. However alternative values were not tested in this work.

## Notes on performance

- Possible that some form of target leakage occurs during the simulation of the anomalies. Difficult to know without being able to explain the predictions of the model and investigating the feature importances.
