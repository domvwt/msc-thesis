---
bibliography: ./thesis/thesis.bib
panhan:
- output_file: ./thesis/thesis.pdf
  use_preset: journal
- output_file: ./thesis/thesis-alt.pdf
  pandoc_args:
    template: false
  use_preset: journal
- output_file: ./thesis/thesis-processed.md
  output_format: markdown
  use_preset: default
reference-section-title: References
subtitle: Final Project
title: Detecting Anomalous Business Ownership Structures with Graph
  Neural Networks
---

```{=html}
<!-- * NOTE: Export bibliography from Zotero in Better BibLaTeX format. -->
```
```{=html}
<!--
Configuration by Panhan - config handler for Pandoc
https://github.com/domvwt/panhan
https://pandoc.org/
-->
```
```{=html}
<!-- Doing Your Research Project:
      file:///home/domvwt/Downloads/Judith_Bell_Doing_Your_Research_Project.pdf -->
```
```{=tex}
\newpage
```
# Introduction

## Background

In October of 2021, The International Consortium of Investigative
Journalists (ICIJ) revealed the findings of their Pandora Papers
investigation. Through examination of nearly 12 million confidential
business records, the ICIJ found evidence implicating thousands of
individuals and businesses in efforts to conceal the ownership of
companies and assets around the world [@icijOffshoreHavensHidden2021].
The intentions behind this secrecy varied from legitimate privacy
concerns to criminal activities, including money laundering, tax
evasion, and fraud
[@europeanunionagencyforlawenforcementcooperationShadowMoneyInternational2021].

To put these numbers in perspective, a 2019 study by the European
Commission estimated that a total of USD 7.8 trillion was held offshore
as of 2016. The share of this attributed to the European Union (EU) was
USD 1.6 trillion, which corresponds to an estimated tax revenue loss to
the EU of EUR 46 billion
[@europeancommission.directorategeneralfortaxationandcustomsunion.EstimatingInternationalTax2019].

Identifying the beneficiaries of a company is challenging due to the
ease with which information can be concealed or simply not declared.
Further complication is introduced by the interconnected nature of
businesses and individuals, as well as the ingenuity of criminals in
masking illicit activity. These difficulties place significant strain on
the resources of law enforcement agencies and financial institutions
[@stevenm.CombatingIllicitFinancing2019].

In April 2016, the United Kingdom made it mandatory for businesses to
keep a register of People with Significant Control. This includes people
who own more than 25% of the company's shares [@KeepingYourPeople2016].
Ownership data is curated and processed by the Open Ownership
organisation for the purposes of public scrutiny and research
[@OpenOwnership2022]. It is the data provided by Open Ownership that
forms the basis of this study. Details of suspicious or illegitimate
business owners are not readily available due to the sensitive nature of
such records. We propose a method for simulating anomalous ownership
structures as part of our experimental methods.

To model the complex network of global business ownership, it is
necessary to represent companies, people, and their relationships in a
graph structure. With data in this format, it is possible to consider
the features of an entity's local neighbourhood when making a decision,
in addition to the entity's own characteristics. Anomaly detection
algorithms that operate on graph structures remain at the frontier of
machine learning research.

To the best of the author's knowledge, there is indeed no published
research studying the effectiveness of graph anomaly detection
techniques on business ownership networks. The following proposal is a
study into the application of state of the art anomaly detection
techniques to business ownership graphs.

## Aims, Objectives and Research Questions

### Aims

The primary aim of this project is to develop an effective approach for
detecting anomalous entities in a business ownership network. Second,
the project will offer a comparison of Graph Neural Network (GNN) models
to traditional anomaly detection approaches on a business ownership
graph. Finally, we contribute a dataset describing a real business
ownership network that is suitable for graph learning.

### Objectives

-   Compose a business ownership graph from open data sources.
-   Train and evaluate GNN models for anomaly detection.
-   Perform the anomaly detection task with traditional machine learning
    methods.
-   Compare approaches in terms of effectiveness and applicability.

### Research Questions

```{=html}
<!-- https://writingcenter.gmu.edu/guides/how-to-write-a-research-question -->
```
The questions driving the research are as follows:

-   What is the most effective strategy for detecting anomalous entities
    in business ownership networks?
-   How do GNN models compare to traditional approaches in terms of
    classification performance?
-   What are the challenges that arise in building and training a GNN
    model and what recommendations can be made to mitigate these?

## Prior and Related Work

There are few published studies focussed specifically on the detection
of anomalous business ownership structures. We offer a review of the
most relevant literature below.

### Detecting Financial Fraud

@lunaFindingShellCompany2018 describe a procedure for identifying
suspected shell company accounts using distance and density based
anomaly detection techniques. The authors were successful in detecting
shell companies through observing differences in transactional
behaviour. A notable caveat is that the data was simulated for the
purposes of the study, which leaves questions around its applicability
to real world scenarios.

Recent work by @dumitrescuAnomalyDetectionGraphs2022 demonstrates how
local neighbourhood features and statistical scores can be used in
Anti-Money Laundering (AML) models. Relevant features included
unsupervised anomalous node detection techniques [@
@akogluOddballSpottingAnomalies2010] and local neighbourhood
connectivity features [@molloyGraphAnalyticsRealtime2016] calculated on
*reduced egonets*. A strength of the study is that it was conducted on
genuine labelled transactional data with positive results. The authors
did not implement any GNN or other deep learning approaches for
comparison.

@fronzetticolladonUsingSocialNetwork2017 explore a range of social
network analysis techniques for the identification of money laundering
using data kept by an Italian factoring company. The authors found that
constructing many networks from different projections of the graph
entities improved the power of individual risk metrics. Degree
centrality was determined to be a significant risk predictor in all
cases, while in certain scenarios network constraint proved to be
informative. It should be acknowledged that the results obtained are for
the clients of a single business and that additional work is required to
demonstrate wider validity.

### Anomaly Detection for Graphs

```{=html}
<!-- Graph-based Anomaly Detection and Description: A Survey (2014) -->
```
@akogluGraphbasedAnomalyDetection2014 highlight four main reasons for
the suitability of graph structures in anomaly detection:

**Inter-dependent nature of the data** -- "Data objects are often
related to each other and share dependencies." This can be observed in
business ownership data through the relationships that connect
individuals and companies in legal hierarchies and communities.

**Powerful representation** -- Graphs offer a powerful way of
representing inter-dependencies and long range correlations between
related entities. By using different node and edge types, as well as
additional attributes, it is possible to represent rich datasets. These
properties are valuable in capturing the different types of entities
present in a business ownership graph. A business, for example, will
have attributes that are not shared by individuals, such as an industry
classification code.

**Relational nature of problem domains** -- "The nature of anomalies
could exhibit themselves as relational". In context of detecting
anomalous business ownership, it is evident that individuals and
businesses may be anomalous predominantly through their unusual
relationships to other entities.

**Robust machinery** -- "Graphs serve as more adversarially robust
tools." It is suggested that graph based systems are ideally suited for
fraud detection, as bad actors will find it difficult to alter or fake
their position in the global structure.

```{=html}
<!-- A Comprehensive Survey on Graph Anomaly Detection with Deep Learning (2021) -->
```
A thorough description of graph anomaly detection tasks and approaches
is offered by @maComprehensiveSurveyGraph2021. Their taxonomy
categorises tasks based on the graph component being targeted: nodes,
edges, sub-graphs, or full graphs. The authors state their belief that
"because the copious types of graph anomalies cannot be directly
represented in Euclidean feature space, it is not feasible to directly
apply traditional anomaly detection techniques to graph anomaly
detection".

### Anomalous Node Detection with GNNs

@kipfSemiSupervisedClassificationGraph2017 propose a scalable GNN
architecture for classifying nodes in a partially labelled dataset.
Early attempts to apply deep learning to graph structures utilised RNN
architectures which prove difficult to scale
[@goriNewModelEarning2005; @liGatedGraphSequence2017; @scarselliGraphNeuralNetwork2009].
Kipf et al. extend prior work on spectral GNNs
[@brunaSpectralNetworksLocally2014; @defferrardConvolutionalNeuralNetworks2017]
to produce a flexible model that scales in linear time with respect to
the number of graph edges.

@dingDeepAnomalyDetection2019 combine a GNN architecture with an
autoencoder in a method that identifies anomalous nodes by
reconstruction error. The proposed method, DOMINANT, uses a GNN to
generate node embeddings and separately reconstructs both the graph
topology and the node attributes. This strategy is further developed and
applied to multi-view data through the combination of multiple graph
encoders [@pengDeepMultiViewFramework2022].

An alternative method is offered by @liSpecAESpectralAutoEncoder2019, in
which a spectral convolution and deconvolution framework is used to
identify anomalous nodes in conjunction with a density estimation model.
The approach continues to demonstrate the importance of combining
multiple perspectives of the network data, with the innovation being the
use of a Gaussian Mixture Model to combine representations in a single
view.

### Recent Developments

@velickovicGraphAttentionNetworks2018 demonstrate the use of
self-attention layers to address shortcomings in the representations
captured by GNN architectures. However, a comparison of Relational Graph
Attention (GAT) models to GNNs showed that relative performance was task
dependent and that current GAT models could not be shown to consistently
outperform GNNs on benchmark exercises
[@busbridgeRelationalGraphAttention2019].

In an application of graph attention based models to financial fraud
detection, @wangSemisupervisedGraphAttentive2019 show that their SemiGNN
model outperforms established approaches when predicting risk of default
and in attribute prediction. Baseline methods used for comparison
included a XGBoost [@chenXGBoostScalableTree2016], GNN, GAT, and LINE
[@tangLINELargescaleInformation2015].

The GraphSAGE [@hamiltonInductiveRepresentationLearning2017] node
embedding framework is among the most popular

# Dataset

Training a supervised classification model requires access to a dataset
of labelled examples. Given the typical infrequency of fraudulent events
to legitimate cases, it is common for fraud classification projects
require large amounts of data in order to provide a modest number of
fraudulent examples from which the model can learn. However, considering
the sensitive nature of fraud investigations, it is not surprising to
find that no such data is available in the public domain.

In the following section we detail the public data sources used in this
study and the steps necessary for processing them. We further propose a
method for simulating anomalous business ownership structures using this
publically available data.

## External Sources

### People with Significant Control Register

Since 2016, it has been a requirement for all businesses in the United
Kingdom to declare People of Significant Control (PSC)
[@KeepingYourPeople2016]. This includes all shareholders with ownership
greater than 25%, any persons with more than 25% of voting rights, and
any person with the right to appoint or remove the majority of the board
of directors [@PeopleSignificantControl2022]. The register is available
as a daily snapshot, available to download from the Companies House
webpage [@CompaniesHouse2022a].

Included in the register are the name, address, and identification
number of each company for which a declaration has been received. Also
listed are the name, address, country of origin, date of birth, and
nature of control for each person listed as a PSC.

Rather than consume this data directly, we obtain this information from
a data feed curated by the Open Ownership organisation.

### Open Ownership Register

The Open Ownership organisation maintains a database of over 16 million
beneficial ownership records, including those collated in the UK PSC
register and from additional sources made available by other countries
[@OpenOwnership2022]. This data is provided in a standardised format
that is conducive to machine processing and graph generation. Additional
data quality processing is undertaken by Open Ownership, such as the
merging of duplicated records for persons that have more than one PSC
declaration [@BeneficialOwnershipData2022].

The Open Ownership Register is used as a canonical source of company,
person, and ownership relationships for the generation of our UK
business ownership graph.

![Open Ownership data schema
[@BeneficialOwnershipData2022]](figures/data-schema-model-2.svg){#fig:data-schema-model-2
width="3in"}

### The Free Company Data Product

The Free Company Data Product is a monthly snapshot of data for all live
companies on the UK public register. Taken from the Companies House data
products website, this data includes:

-   basic information including company type and registered office
    address
-   the nature of business or standard industrial classification (SIC)
-   company status, such as 'live' or 'dissolved'
-   date of last accounts or confirmation statement filed
-   date of next accounts or confirmation statement due
-   previous company names

[@CompaniesHouseData]

This data serves as an additional source of node features for company
entities in the generated dataset.

## Initial Data Preparation

The Open Ownership data file consists of over 20 million records stored
as nested JSON objects. This includes data for businesses and relevant
persons registered outside of the UK. The Apache Spark framework
[@ApacheSparkUnified2022] is used for bulk data preparation due to its
parallel and out-of-core processing capabilties, as well as its support
for nested data structures.

As the scope of this study covers only UK registered companies and their
shareholders, records for entities that do not have a shareholding
interest in a UK company are discarded. Non-UK companies that are
registered as a shareholder of a UK company are also discarded, as we
are unable to obtain information for these entities via Companies House.
Computational resource constraints also prevent handling of a larger
dataset. To further limit dataset size, and in the interests of only
considering accurate and up to date information, we also filter out
companies that are listed as dissolved.

```{=html}
<!-- TODO: INSERT NUMBER OF RECORDS HERE, PERSONS, COMPANIES, RELATIONSHIPS -->
```
While the initial nested data schema is desirable for clean
representation and compact storage, we require a flat relational table
structure for analytical and model training purposes. Relevant data
items are extracted into a flat table for each entity type. This results
in three tables of interim output: company information, person
information, and statements of control that link entities to their
ownership interests.

The Companies House data is joined to the company entity table via the
UK company registration number.

```{=html}
<!-- TODO: INSERT FINAL TABLE SCHEMA -->
```
## Graph Generation

### Nodes and Edges

The company, person, and ownership relationship tables prepared in the
previous steps are used to create a graph data structure. Companies and
persons are represented as nodes in the graph, and the ownership
relationships represented as directed edges (from owner to owned entity)
between them. The resulting graph is attributed with node features and
edge weights. Since the graph consists of two types of nodes with
disitinct feature sets, it can be described as an attributed
heterogeneous graph [@maComprehensiveSurveyGraph2021].

### Connected Components

Connected components are labelled in the graph to facilitate additional
filtering and to allow for parallel computation of topological features.

Two nodes, $u$ and $v$, are considered to be connected if a path exists
between them in the graph $G$. If no path exists between $u$ and $v$
then they are said to be disconnected. Not all entities in the ownership
graph are connected by any path, and so the ownership graph $G$ can be
viewed as a set of disjoint sets of nodes, or components. The graph
$G = (V, E)$ is described by its components thus:
$G[V_1], G[V_2], ... G[V_{\omega}]$. If the number of nodes in a graph
is denoted by $|V|$, the number of nodes in a component $\omega$ can be
described by $|V_{\omega}|$[@bondyGraphTheory1982, p.13].

![Initial distribution of component sizes. A single super-component
consisting of 27,008 nodes is excluded from the
plot.](figures/component-size-distribution-initial.png){#fig:init-comp-dist
short-caption="Initial distribution of component sizes"}

The vast majority of businesses in the dataset have only one or two
declared PSC shareholders. The initial distribution of component sizes
is shown in figure {@fig:init-comp-dist}. These small subgraphs are not
of interest in this study and are excluded, along with any component
that contains fewer than ten nodes or with a ratio of less than 1 in 10
natural persons to companies.

The rationale for excluding these components is threefold: first, the
small number of nodes and relationships presents little information for
a model to learn from; second, these small networks are less likely to
be of interest in real world fraud detection, as illegitimate ownership
is usually concealed behind multiple layers of ownership; and from a
practical standpoint, the volume of data is brought down to a manageable
size. Networks with fewer than 1 in 10 natural persons to companies are
excluded, as a focus of this study is on the performance of anomaly
detection methods on heterogeneous graph data, as opposed to homogeneous
networks.

Finally, we address an observed data quality issue in which multiple
nodes in the same component may share the same name. These nodes are
merged into a single node, with the resulting node taking on all the
ownership relationships of the merged nodes.

## Structural Anomaly Simulation

To simulate structural anomalies, 10% of the nodes are selected at
random and marked as anomalous ($y = 1$) and the remaining 90% marked as
normal ($y = 0$). A single outgoing edge is chosen from each anomalous
node, $\epsilon_{norm} = \epsilon(u_i, v_i)$, and its source ID replaced
with that of another node marked as anomalous, $u_j$. This produces the
new anomalous edge $\epsilon_{anom} = \epsilon(u_j, v_i)$ and eliminates
the original edge $\epsilon_{norm}$. The result is an exchanging of
ownership interests between anomalous nodes, while preserving the
overall structure of the graph. The procedure is illustrated in figure
{@fig:anomaly-sim-process}. This process is repeated until all anomalous
nodes have one edge replaced with a different edge so that for all
anomalous nodes $\epsilon_{anom} \neq \epsilon_{norm}$. The outgoing
edges of normal nodes are not altered, though their incoming edges may
be affected by the anomaly simulation process.

The final step of the simulation is to discard any small components ($n$
\< 9) that have been created as a result of the anomaly simulation
process. This is done to ensure that anomalous nodes belong to
components with a similar size distribution to the original graph, as
demonstrated in figures @fig:comp-dist-anomaly-sim and
@fig:anom-dist-anomaly-sim. The final proportion of anomalous nodes in
the simulated graph is 7.3%.

![Process for simulating structural anomalies. The initial graph is
shown in (1). A node is selected for anomalisation (orange circle) in
(2). The outgoing edge (red arrow) is swapped for that of another
anomalised node (green arrow), resulting in the exchange of pink nodes
for green
(3).](figures/anomaly-simulation-process.png){#fig:anomaly-sim-process
short-caption="Anomaly Simulation Process"}

The anomaly simulation process introduces unusual ownership
relationships into the graph. The true freuqency of these occurrences is
unknown, but is is expected to be far lower than 7.3% of nodes. This
anomaly rate is chosen to ensure that model training is not impossible
due to extreme class imbalance, and the same effect can be achieved by
oversampling the minority class [@chawlaSMOTESyntheticMinority2002] or
undersampling the majority class [@fernandezLearningImbalancedData2018,
p.82].

When labelling the unaltered nodes as normal, it is assumed that the
data originally provided by Open Ownership is accurate and that if any
anomalies are present they will have a negligible impact on experimental
results. Nevertheless, anomalies in the initial dataset will be seen by
all candidate models, and so should not bias the results.

## Node Features

In order to train a baseline model for comparison, a set of node level
features is generated to represent each node in a tabular format
[@leskovecTraditionalMethodsMachine2021]. The following topological
features are extracted for each node:

-   Indegree: The number of incoming edges to the node.
-   Outdegree: The number of outgoing edges from the node.
-   Closeness Centrality: Inverse of the sum of shortest path distances
    to all other nodes.
-   Clustering Coefficient: Connectedness of neighbouring nodes.
-   PageRank: A measure of node importance, based on the importance of
    neighbouring nodes [@pagePageRankCitationRanking1998].

To capture information about the node's position in the graph, aggregate
statistics are calculated for the aforementioned topological features
for the node's neighbours and added as features:

-   Minimum
-   Maximum
-   Sum
-   Mean
-   Standard Deviation

The count of immediate neighbours is also included as a feature.

# Methods

## Traditional Machine Learning Approaches to Binary Classification

In order to assess the relative improvement that can be achieved by
using Graph Neural Networks, we compare their performance to a baseline
model trained on a set of node level features.

### Gradient Boosted Trees

Gradient boosted tree ensembles achieve robust performance on a wide
range of machine learning tasks
[@friedmanGreedyFunctionApproximation2001]. Modern applications are
favoured for their strong performance in capturing complex dependencies,
native handling of heterogeneous and missing data, and numerical scale
invariance. We use the CatBoost library to train our baseline model, as
the current state-of-the-art implementation
[@prokhorenkovaCatBoostUnbiasedBoosting2019].

## Graph Neural Networks

The PyTorch Geometric library offers a comprehensive set of methods for
deep learning on graph data structures
[@feyFastGraphRepresentation2019]. We use the implementations provided
in this library to train our Graph Neural Network models.

We select two architectures, GraphSAGE
[@hamiltonInductiveRepresentationLearning2018] and kGNN
[@morrisWeisfeilerLemanGo2021], for our experiment. These models are
chosen for their demonstrated performance at node classification tasks
and ability to handle heterogeneous graph data with slight modification
(detailed below) [@hamiltonInductiveRepresentationLearning2018,
@morrisWeisfeilerLemanGo2021]. A further consideration in this choice of
models is the size of our dataset and the available computational
resources. Both architectures are relatively lightweight compared to
other models, such as Graph Attention Networks
[@velivckovicGraphAttentionNetworks2018], and can be trained on a single
GPU in a reasonable amount of time.

Since both GNN architectures selected for this study were originally
implemented for learning on homogeneous graphs, we use a method provided
by PyTorch Geometric for adapting these homogeneous architectures for
learning on our heterogeneous dataset. A homogeneous model is adapted
for heterogeneous learning by duplicating message passing functions to
operate on each edge type individually. Node representations are learned
for each node type and aggregated using a user provided function that we
choose via neural architecture search. This technique is described by
@schlichtkrullModelingRelationalData2017 and illustrated in figure
{@fig:to-hetero}.

![Converting homogeneous GNN architectures for heterogeneous learning
[@HeterogeneousGraphLearning]](figures/to_hetero.svg){#fig:to-hetero
short-caption="Converting homogeneous GNN architectures for heterogeneous learning"
width="5in"}

### GraphSAGE

The GraphSAGE model proposed by
@hamiltonInductiveRepresentationLearning2018 is a node embedding
framework that uses both node attributes and the attributes of
neighbouring nodes to generate node representations. During training,
the model learns how to effectively aggregate information from the
neighbourhood of each node and combine this with the node's own features
to produce a learned representation.

The aggregation architecture can be any symmetric function, such as the
mean or sum, or a more complex function such as a neural network that
can operate on an ordered set of node features. For supervised learning
tasks, the parameters of the aggregation function are learned via
backpropagation.

Edge attributes are not used in the GraphSAGE model and are therefore
ignored during training.

![GraphSAGE neighbourhood sampling and aggregation
[@hamiltonInductiveRepresentationLearning2018]](figures/graphSAGE.png){#fig:graphsage
short-caption="GraphSAGE architecture" width="5in"}

### Higher Order GNN (kGNN)

The Higher-Order GNN (or kGNN) proposed by @morrisWeisfeilerLemanGo2021
is an extension of the prototypical GNN described by
@kipfSemiSupervisedClassificationGraph2017 to higher order graph
representations. These higher order representations are learned by
associating all unordered $k$-tuples of nodes and learning a
representation for each tuple, where $k$ is a hyperparameter of the
model. These tuple representations are hierarchically combined in a
final dense layer to produce a representation for each node that takes
into account its local neighbourhood and higher order topological
features. This is illustrated in figure {@fig:higher-order-gnn}.

As with the GraphSAGE model, an aggregation architecture is used to
learn the node tuple representations. The weights for each layer are
also trained via backpropagation for supervised learning tasks. Unlike
the GraphSAGE model, the kGNN model does incorporate edge weights in its
learning process.

![Hierarchical 1-2-3 GNN network architecture
[@morrisWeisfeilerLemanGo2021]](figures/learning-higher-order-graph-properties.png){#fig:higher-order-gnn
short-caption="Hierarchical 1-2-3 GNN network architecture"}

## Experimental Setup

### Data Splitting

The dataset is split into training, validation, and test sets. The
training set is used to train the model, the validation set is used to
tune hyperparameters, and the test set is used to evaluate the model's
performance. The split is performed using a random 80/10/10 split, with
random selection applied to each component in the graph to ensures that
nodes in the same component are not split across multiple sets.

### Class Weighting

Class imbalance in the dataset is addressed by assigning a weight to
each class during model training. A weight of 10 is applied to anomalous
nodes and a weight of 1 for normal nodes. These weights are used as
multipliers for the errors of their respective classes, leading to
increased penalty and greater emphasis for anomalous nodes. This is a
cost sensitive approach to class imbalance that conveniently does not
require the use of oversampling or undersampling
[@heLearningImbalancedData2009].

### Evaluation Metrics

We use two threshold-free metrics to evaluate model performance: the
area under the precision-recall curve (AUC-PR) and the area under the
receiver operating characteristic curve (AUC-ROC). These metrics are
chosen as they are insensitive to the choice of threshold, and are more
robust to class imbalance than threshold dependent metrics such as
accuracy. [@maImbalancedLearningFoundations2013, p.72]

#### Area under the Receiver Operating Characteristic Curve (AUC-ROC)

The Receiver Operating Characteristic Curve (ROC) is a plot of the true
positive rate against the false positive rate along a range of threshold
settings. A perfect model will achieve a 100% true positive rate with a
0% false positive rate, while a zero-skill classifier will achieve a 50%
true positive rate with a 50% false positive rate. The area under the
ROC curve (AUC-ROC) provides a way to summarise model performance over
the range of threshold values. A perfect model will have an AUC-ROC of
1, while the zero-skill model will have an AUC-ROC of 0.5.
[@fawcettIntroductionROCAnalysis2006]

#### Area under the Precision-Recall Curve (AUC-PR)

The Precision-Recall Curve (PR) is a plot of precision against recall
along the range of recall values. A perfect model will achieve a 100%
precision and 100% recall, while a zero-skill classifier will achieve a
rate of precision equal to the proportion of positive cases in the
dataset along for all recall values. The area under the PR curve
(AUC-PR) provides a way to summarise model performance over the range of
recall values. A perfect model will have an AUC-PR of 1, while the
zero-skill model will have an AUC-PR equal to the proportion of positive
cases in the dataset. The PR curve can provide a more accurate view of
model peformance on imbalanced datasets.
[@saitoPrecisionRecallPlotMore2015]

### Graph Neural Network Training

#### Optimiser

The Adam optimiser [@kingmaAdamMethodStochastic2017] is used to train
the GNNs with an initial learning rate of 0.01.

#### Neural Architecture Search

For each of the GNN models (GraphSAGE and kGNN), we learn an optimal
neural architecture for the anomalous node classification task. We use
the Optuna library [@akibaOptunaNextgenerationHyperparameter2019] to
explore the search space, training candidate models on the training
dataset and selecting the architecture that achieves the highest AUC-PR
on the validation data.

Each model is trained for a maximum of 2000 epochs, with an early
stopping callback used to terminate trials that do not improve for 200
consecutive epochs. Thirty trials are performed for each model. Both GNN
models share the same search spaces, the parameters of the space are
provided in the appendix tables {@tbl:gnn-search-space-cont} and
{@tbl:gnn-search-space-cat}.

#### Hyperparameter Tuning

Parameters for dropout and weight decay are also tuned using Optuna.
These trials take place after the architecture search, in order to limit
the dimensionality of the search space in each experiment. A total of 20
trials are performed for each pair of candidate values, with the best
hyperparameters selected based on the AUC-PR score on the validation
set.

### CatBoost Training

The CatBoost classifier is trained in a similar manner to the GNNs. Each
candidate model is trained and evaluated on the same training and
validation sets as the GNN models, and evaluated using the AUC-PR
metric. The hyperparameter search space is provided in the appendix
table {@tbl:catboost-search-space}.

# Results and Discusson

## Neural Architecture Search and Hyperparameter Tuning

It was observed during the model tuning process that both GNN models
displayed a great deal of variance in performance, even between trials
when all hyperparameters were held constant. This suggests a high deal
of sensitivity to the initialisation of the model weights, and that the
model architecture is not the only factor in determining model
performance.

In order to ensure that the final model is among the best possible
candidates, we perform several round of training with the best
performing architecture, checking validation performance at each epoch
and saving the weights whenever the validation AUC-PR improves beyond
the previous best. This process is repeated for each of the GNN models,
and the final model is selected based on the best validation AUC-PR
score. This is in contrast to the commonly adopted strategy of training
the final model for a predetermined number of epochs on both the
training and validation datasets.

Further, it was noted that both models exhibited a non-linear
progression in performance over the course of training and showed no
signs of overfitting. While this could suggest that the models have not
converged, allowing the models to train over thousands of epochs showed
that the validation AUC-PR continues to increase and decrease in a
non-linear fashion. It may be worth studying this behaviour in future
work to determine whether it is a result of the model architecture, the
dataset, or some other factor in the experimental design.

### GraphSAGE

The best performing architecture for the GraphSAGE models was found to
be a 2-layer model with 256 hidden channels, a bias term, and an
additional linear layer after the final message passing layer. A sum
aggregation function was selected for both the message passing layers
and for the aggregation of heterogeneous node representations. A leaky
relu activation function was selcted for all layers. Neither dropout nor
weight decay were found to be beneficial to model performance.

### kGNN

The best performing architecture for the kGNN model was found to be a
4-layer model with 128 hidden channels, no bias term, and an additional
linear layer after the final message passing layer. A minimum pooling
aggregation function was selected for both the message passing layers
and for the aggregation of heterogeneous node representations. A gelu
activation function was selcted for all layers. Neither dropout nor
weight decay were found to be beneficial to model performance.

### CatBoost

The most successful CatBoost model was found to have the following
parameters:

-   learning rate: 0.09
-   depth: 9
-   boosting_type: Plain
-   bootstrap_type: MVS
-   colsample_bylevel: 0.08

## Model Performance

Both of the GNN models achieved significantly higher AUC-ROC and AUC-PR
scores than the CatBoost model. The kGNN model achieved the highest
AUC-ROC and AUC-PR scores, with an AUC-ROC of 0.982 and an AUC-PR of
0.904. The GraphSAGE model achieved an AUC-ROC of 0.953 and an AUC-PR of
0.767. The CatBoost model achieved an AUC-ROC of 0.639 and an AUC-PR of
0.104.

95% confidence intervals are provided for the AUC-ROC and AUC-PR scores.
These are calculated using the bootstrap method
[@DavisonBootstrapMethods] for 10,000 iterations.

![ROC and PR curves on the test
set.](figures/roc-pr-curve.png){#fig:roc-pr-curve
short-caption="ROC and PR curves on the test set."}

  ------------------------------------------------------------------------
  Model              AUC-ROC 95% CI                 AUC-PR 95% CI
  ----------- -------------- --------------- ------------- ---------------
  CatBoost             0.639 \[0.619,                0.104 \[0.092,
                             0.659\]                       0.116\]

  GraphSAGE            0.943 \[0.929,                0.735 \[0.695,
                             0.957\]                       0.775\]

  kGNN             **0.982** \[0.974,            **0.904** \[0.876,
                             0.990\]                       0.932\]
  ------------------------------------------------------------------------

  : Model performance on the test set. {#tbl:results}

The stand out performance of the kGNN model may be explained by its
capacity to learn and combine higher order features of the graph. A
potential avenue for further study would be to investigate what features
the model is learning and how they influence the model's performance.
Work by @yingGNNExplainerGeneratingExplanations2019 on the GNNExplainer
tool is likely worth exploring.

It should also be noted that the kGNN model has two additional message
passing layers compared to the GraphSAGE model. This may be a
contributing factor to the model's superior performance, however the
kGNN consists of half as many hidden channels. This model depth may be
another contributing factor to the model's superior performance.

```{=tex}
\newpage
```
# References

::: {#refs}
:::

```{=tex}
\newpage
```
# Appendix

![Distribution of component sizes before and after anomaly
simulation.](figures/component-sizes.png){#fig:comp-dist-anomaly-sim
short-caption="Distribution of component sizes."}

![Distribution of anomalous nodes by component size before and after
anomaly
simulation.](figures/anomaly-distribution-component-size.png){#fig:anom-dist-anomaly-sim
short-caption="Distribution of anomalous nodes by component size."}

![Distribution of node degrees before and after anomaly
simulation.](figures/degree-distributions-post.png){#fig:node-degrees-anomaly-sim
short-caption="Distribution of node degrees."}

  Property          Min   Max
  ----------------- ----- ------
  hidden layers     1     8
  hidden channels   2     256
  weight decay      0     0.01
  dropout           0     0.5

  : Neural architecture search space for GNN models - continuous.
  {#tbl:gnn-search-space-cont}

  Property                              Choices
  ------------------------------------- ------------------------
  activation function                   relu, gelu, leaky_relu
  linear layer post-message passing     true, false
  message passing aggregation           min, max, sum, mean
  heterogeneous embedding aggregation   min, max, sum, mean

  : Neural architecture search space for GNN models - categorical.
  {#tbl:gnn-search-space-cat}