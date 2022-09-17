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

## Notes on models

- Highly sensitive to initialisation weights. Could be possible to improve performance by using a more sophisticated initialisation scheme. For example, [this paper](https://arxiv.org/abs/1905.12265) proposes a method for initialising GNNs that is based on the spectral properties of the graph Laplacian. This method is also used in [this paper](https://arxiv.org/abs/2003.00982) which achieves state-of-the-art performance on the Cora dataset. IS THIS APPLICABLE TO OUR DATASET?
- More sophisticated architectures exist, the Graph Attention Network (GAT), Hetereogeneous Graph Attention Network (HGT) and Heterogeneous Graph Transformer (HGT) are all state-of-the-art on the Cora dataset. Could be possible to improve performance by using one of these architectures. Caveat
is that the current architecture is already very complex and it is not clear that the additional complexity of these architectures would be justified by the performance gains. Also at the limits of what is possible with the current dataset and the current hardware available to me.
- Validation data more valuable as an indicator that the model is in a global minimum than as additional training data. Could be possible to improve performance by using a larger proportion of the data for training and a smaller proportion for validation.
- Large number of epochs required to train the model
- Non linear progress in performance. Reducing learning rate did not appear to help in this regard.
- Models did not show signs of overfitting regardless of the number of epochs trained for. This is surprising given the large number of parameters in the model.
- Regularisation did not improve performance. This is surprising given the large number of parameters in the model but could be explained by the fact that the model is not overfitting.
- Could use a learning rate scheduler to improve performance.
