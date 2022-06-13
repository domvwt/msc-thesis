## Literature Review Notes

There are few published studies focussed specifically on the detection of anomalous business ownership structures. We offer a review of the most relevant literature below.

### Detecting Financial Fraud

@luna_finding_2018 describe a procedure for identifying suspected shell company accounts using distance and density based anomaly detection techniques. The authors were successful in detecting shell companies through observing differences in transactional behaviour. A notable caveat is that the data was simulated for the purposes of the study, which leaves questions around its applicability to real world scenarios.

Recent work by @dumitrescu_anomaly_2022-1 demonstrates how local neighbourhood features and statistical scores can be used in Anti-Money Laundering (AML) models. Relevant features included unsupervised anomalous node detection techniques [@hutchison_oddball_2010] and local neighbourhood connectivity features [@molloy_graph_2016] calculated on *reduced egonets*. A strength of the study is that it was conducted on genuine labelled transactional data with positive results. The authors did not implement any GCN or other deep learning approaches for comparison.

@fronzetti_colladon_using_2017 explore a range of social network analysis techniques for the identification of money laundering using data kept by an Italian factoring company. The authors found that constructing many networks from different projections of the graph entities improved the power of individual risk metrics. Degree centrality was determined to be a significant risk predictor in all cases, while in certain scenarios network constraint proved to be informative. It should be acknowledged that the results obtained are for the clients of a single business and that additional work is required to demonstrate wider validity.

### Anomaly Detection for Graphs

<!-- Graph-based Anomaly Detection and Description: A Survey (2014) -->

@akoglu_graph_2015 highlight four main reasons for the suitability of graph structures in anomaly detection:

**Inter-dependent nature of the data** -- "Data objects are often related to each other and share dependencies." This can be observed in business ownership data through the relationships that connect individuals and companies in legal hierarchies and communities. 

**Powerful representation** -- Graphs offer a powerful way of representing inter-dependencies and long range correlations between related entities. By using different node and edge types, as well as additional attributes, it is possible to represent rich datasets. These properties are valuable in capturing the different types of entities present in a business ownership graph. A business, for example, will have attributes that are not shared by individuals, such as an industry classification code.

**Relational nature of problem domains** -- "The nature of anomalies could exhibit themselves as relational". In context of detecting anomalous business ownership, it is evident that individuals and businesses may be anomalous predominantly through their unusual relationships to other entities.

**Robust machinery** -- "Graphs serve as more adversarially robust tools." It is suggested that graph based systems are ideally suited for fraud detection, as bad actors will find it difficult to alter or fake their position in the global structure.

<!-- A Comprehensive Survey on Graph Anomaly Detection with Deep Learning (2021) -->

A thorough description of graph anomaly detection tasks and approaches is offered by @ma_comprehensive_2021. Their taxonomy categorises tasks based on the graph component being targeted: nodes, edges, sub-graphs, or full graphs. In addition, the authors state their belief that "because the copious types of graph anomalies cannot be directly represented in Euclidean feature space, it is not feasible to directly apply traditional anomaly detection techniques to graph anomaly detection".

### Anomalous Node Detection with GCNs

@kipf_semi-supervised_2017 propose a scalable GCN architecture for classifying nodes in a partially labelled dataset. Early attempts to apply deep learning to graph structures utilised RNN architectures which prove difficult to scale [@gori_new_2005, @scarselli_graph_2009, @li_gated_2017]. Kipf et al. extend prior work on spectral GCNs [@bruna_spectral_2014, @defferrard_convolutional_2017] to produce a flexible model that scales in linear time with respect to the number of graph edges.

@ding_deep_2019 combine a GCN architecture with an autoencoder in a method that identifies anomalous nodes by reconstruction error. The proposed method, DOMINANT, uses a GCN to generate node embeddings and separately reconstructs both the graph topology and the node attributes. This strategy is further developed and applied to multi-view data through the combination of multiple graph encoders [@peng_deep_2022].

An alternative method is proposed in @li_specae_2019, in which a spectral convolution and deconvolution framework is used to identify anomalous nodes in conjunction with a density estimation model. The approach continues to demonstrate the importance of combining multiple perspectives of the network data, with the innovation being the use of a Gaussian Mixture Model to combine representations in a single view. 
