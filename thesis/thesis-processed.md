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
# Introduction

# Prior and Related Work

# Dataset

Training a supervised classification model requires access to a dataset
of labelled examples. Given the typical infrequency of fraudulent events
to legitimate cases, it is common for fraud classification projects
require large amounts of data in order to provide a modest number of
fraudulent examples from which the model can learn.

Considering the sensitive nature of fraud investigations, it is not
surprising to find that no such data is available in the public domain.

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

It is assumed that the data originally provided by Open Ownership is
accurate and that if any anomalies are present they will have a
negligible impact on experimental results. Where anomalies are present,
they should have an equal effect on all models, and so should not bias
the results.

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

Since the GNN architectures selected for this study were originally
implemented for learning on homogeneous graphs, we use a method provided
by PyTorch Geometric for adapting these homogeneous architectures for
learning on our heterogeneous dataset. A homogeneous model is adapted
for heterogeneous learning by duplicating message passing functions to
operate on each edge type individually. Node representations are learned
for each node type and aggregated using a user provided function that we
choose via neural architecture search. This is technique is described by
@schlichtkrullModelingRelationalData2017 and illustrated in figure
{@fig:to-hetero}.

![Converting homogeneous GNN architectures for heterogeneous learning
[@HeterogeneousGraphLearning]](figures/to_hetero.svg){#fig:to-hetero
short-caption="Converting homogeneous GNN architectures for heterogeneous learning"
width="5in"}

### Neural Architecture Search and Hyperparameter Tuning

For each type of GNNs (GraphSAGE and GCN), we learn an optimal neural
architecture for the anomalous node classification task. We use the
Optuna library [@akibaOptunaNextgenerationHyperparameter2019] to explore
the relevant search space and select the best architecture by evaluating
performance on the validation dataset. The search spaces used for each
model are provided in the appendix tables {@tbl:graphsage-search-space}
and {@tbl:graph-conv-search-space}.

Regularisation hyperparameters are tuned separately, with the best
performing values for dropout and weight decay also selected by
evaluating performance on the validation dataset.

### GraphSAGE

The GraphSAGE model proposed by
@hamiltonInductiveRepresentationLearning2018 is an inductive node
embedding framework that uses both node features and information from
the local neighbourhood to generate node representations. During
training, the model learns how to effectively aggregate information from
the neighbourhood of each node and combine this with the node's own
features to produce a learned representation.

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

### k-GNN

The Higher-Order GNN (or k-GNN) proposed by @morrisWeisfeilerLemanGo2021
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
generate the node tuple representations. The weights for each layer are
also learned via backpropagation for supervised learning tasks. Unlike
the GraphSAGE model, the k-GNN model does incorporate edge weights in
its learning process.

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
increased penalty and greater weight updates for anomalous nodes. This
is a cost sensitive approach to class imbalance that does not require
the use of oversampling or undersampling
[@heLearningImbalancedData2009].

### Evaluation Metrics

We use two threshold-free metrics to evaluate model performance: the
area under the precision-recall curve (AUC-PR) and the area under the
receiver operating characteristic curve (AUC-ROC). These metrics are
chosen as they are insensitive to the choice of threshold, and are
therefore more robust to class imbalance than metrics such as accuracy.
[@maImbalancedLearningFoundations2013, p.72]

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
along the range of recall values. Precision is defined as the proportion
of positive predictions that are correct, while recall is the proportion
of all positive cases that have been correctly identified. A perfect
model will achieve a 100% precision and 100% recall, while a zero-skill
classifier will achieve a rate of precision equal to the proportion of
positive cases in the dataset along for all recall values. The area
under the PR curve (AUC-PR) provides a way to summarise model
performance over the range of recall values. A perfect model will have
an AUC-PR of 1, while the zero-skill model will have an AUC-PR equal to
the proportion of positive cases in the dataset. The PR curve can
provide a more accurate view of model peformance on imbalanced datasets.
[@saitoPrecisionRecallPlotMore2015]

### GNNs

#### Optimiser

```{=html}
<!-- ! REWRITE THIS -->
```
The Adam optimiser [@kingmaAdamMethodStochastic2014] is used to train
the GNNs. The Adam optimiser is a variant of stochastic gradient descent
that uses adaptive learning rates for each parameter. This allows the
learning rate to be tuned independently for each parameter, and is
useful for GNNs as it allows the learning rate to be tuned for each
layer independently.

#### Neural Architecture Search

```{=html}
<!-- ! REWRITE THIS -->
```
Neural architecture search (NAS) is used to find the optimal
architecture for the GNNs. The search is performed using the Optuna
framework [@toshihikoOptunaAHyperparameter2019]. The search is performed
using the validation set, and the best architecture is selected based on
the AUC-PR score on the validation set.

#### Hyperparameter Tuning

```{=html}
<!-- ! REWRITE THIS -->
```
Hyperparameter tuning is performed using the Optuna framework
[@toshihikoOptunaAHyperparameter2019]. The search is performed using the
validation set, and the best hyperparameters are selected based on the
AUC-PR score on the validation set.

-   Dropout
-   Regularisation

### CatBoost

#### Addressing Class Imbalance

Random majority class undersampling is used to generate a balanced
training set. The training set is undersampled to match the size of the
minority class, and the undersampled training set is used to train the
model. This is an effective and straightforward method for addressing
class imbalance [@brancoSurveyPredictiveModelling2015].

#### Hyperparameter Tuning

```{=html}
<!-- ! REWRITE THIS -->
```
Hyperparameter tuning is performed using the PyCaret framework
[@pycaretPyCaretAutomatedMachine2021]. The search is performed using the
validation set, and the best hyperparameters are selected based on the
AUC-PR score on the validation set.

-   Search space
-   How many rounds

# Results and Discussion

# Conclusion

We have presented a novel dataset of ownership relationships between
companies and persons in the UK. The dataset is made available for use
in future research, and is accompanied by a set of baseline models for
comparison.

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
