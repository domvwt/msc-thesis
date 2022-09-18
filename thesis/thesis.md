---
title: Detecting Anomalous Business Ownership Structures with Graph Neural Networks
subtitle: Final Project
bibliography: ./thesis/thesis.bib
reference-section-title: References
header-includes:
    - \usepackage{longtable}\setlength{\LTleft}{1em}
panhan:
-
    use_preset: journal
    output_file: ./thesis/thesis.pdf
-
    use_preset: journal
    pandoc_args:
        template: False
    output_file: ./thesis/thesis-alt.pdf
-
    use_preset: default
    output_format: markdown
    output_file: ./thesis/thesis-processed.md
---

<!-- * NOTE: Export bibliography from Zotero in Better BibLaTeX format. -->

<!--
Configuration by Panhan - config handler for Pandoc
https://github.com/domvwt/panhan
https://pandoc.org/
-->

<!-- Doing Your Research Project:
      file:///home/domvwt/Downloads/Judith_Bell_Doing_Your_Research_Project.pdf -->

# Introduction

# Prior and Related Work

# Dataset

Training a supervised classification model requires access to a
dataset of labelled examples. Given the typical infrequency of
fraudulent events to legitimate cases, it is common for fraud
classification projects require large amounts of data in
order to provide a modest number of fraudulent examples from which the
model can learn.

Considering the sensitive nature of fraud investigations, it is not
surprising to find that no such data is available in the public
domain.

In the following section we detail the public data sources used in this study and the steps necessary for processing them. We further propose a method for simulating anomalous business ownership structures using this publically available data.

## External Sources

### People with Significant Control Register

Since 2016, it has been a requirement for all businesses in the United
Kingdom to declare People of Significant Control (PSC) [@KeepingYourPeople2016].
This includes all shareholders with ownership greater than 25%, any
persons with more than 25% of voting rights, and any person with the
right to appoint or remove the majority of the board of directors [@PeopleSignificantControl2022].
The register is available as a daily snapshot, available to download from the Companies House webpage [@CompaniesHouse2022a].

Included in the register are the name, address, and identification number of each company for which a declaration has been received. Also listed are the name, address, country of origin, date of birth, and nature of control for each person listed as a PSC.

Rather than consume this data directly, we obtain this information from a data feed curated by the Open Ownership organisation.

### Open Ownership Register

The Open Ownership organisation maintains a database of over 16 million beneficial ownership records, including those collated in the UK PSC register and from additional sources made available by other countries [@OpenOwnership2022]. This data is provided in a standardised format that is conducive to machine processing and graph generation. Additional data quality processing is undertaken by Open Ownership, such as the merging of duplicated records for persons that have more than one PSC declaration [@BeneficialOwnershipData2022].

The Open Ownership Register is used as a canonical source of company, person, and ownership relationships for the generation of our UK business ownership graph.

### The Free Company Data Product

The Free Company Data Product is a monthly snapshot of data for all live companies on the UK public register. Taken from the Companies House data products website, this data includes:

- basic information including company type and registered office address
- the nature of business or standard industrial classification (SIC)
- company status, such as ‘live’ or ‘dissolved’
- date of last accounts or confirmation statement filed
- date of next accounts or confirmation statement due
- previous company names

[@CompaniesHouseData]

This data serves as an additional source of node features for company entities in the generated dataset. It is the ability to learn from these heterogeneous sets of features, as well as the relationships between them, that distinguishes Hetereogeneous Graph Neural Networks from other implementations.

## Initial Data Preparation

The Open Ownership data file consists of over 20 million records stored as nested JSON objects. This includes data for businesses and relevant persons registered outside of the UK. The Apache Spark framework [@ApacheSparkUnified2022] is used for bulk data preparation due to its parallel and out-of-core processing capabilties, as well as its support for nested data structures.

As the scope of this study covers only UK registered companies and their shareholders, records for entities that do not have a shareholding interest in a UK company are discarded. Non-UK companies that are registered as a shareholder of a UK company are also discarded, as we are unable to obtain information for these entities via Companies House. Computational resource constraints also prevent handling of a larger dataset. To further limit dataset size, and in the interests of only considering accurate and up to date information, we also filter out companies that are listed as dissolved.

<!-- TODO: INSERT NUMBER OF RECORDS HERE, PERSONS, COMPANIES, RELATIONSHIPS -->

While the initial nested data schema is desirable for clean representation and compact storage, we require a flat relational table structure for analytical and model training purposes. Relevant data items are extracted into a flat table for each entity type. This results in three tables of interim output: company information, person information, and statements of control that link entities to their ownership interests.

The Companies House data is joined to the company entity table via the UK company registration number.

<!-- TODO: INSERT FINAL TABLE SCHEMA -->

## Graph Generation

### Nodes and Edges

The company, person, and ownership relationship tables prepared in the previous steps are used to create a graph data structure. Companies and persons are represented as nodes in the graph, and the ownership relationships represented as directed edges (from owner to owned entity) between them. The resulting graph is attributed with node features and edge weights. Since the graph consists of two types of nodes with disitinct feature sets, it can be described as an attributed heterogeneous graph [@maComprehensiveSurveyGraph2021].

### Connected Components

Connected components are labelled in the graph to facilitate additional filtering and to allow for parallel computation of topological features.

Two nodes, $u$ and $v$, are considered to be connected if a path exists between them in the graph $G$. If no path exists between $u$ and $v$ then they are said to be disconnected. Not all entities in the ownership graph are connected by any path, and so the ownership graph $G$ can be viewed as a set of disjoint sets of nodes, or components. The graph $G$ is described by its components thus: $G[V_1], G[V_2], ... G[V_n]$ [@bondyGraphTheory2009, p.13].

![Initial distribution of component sizes. A single super-component consisting of 27,008 nodes is excluded from the plot.](figures/component-size-distribution-initial.png){#fig:init-comp-dist short-caption="Initial distribution of component sizes"}

The vast majority of businesses in the dataset have only one or two declared PSC shareholders. The initial distribution of component sizes is shown in figure @fig:init-comp-dist. These small subgraphs are not of interest in this study and are excluded, along with any component that contains fewer than ten nodes or with a ratio of less than 1 in 10 natural persons to companies.

The rationale for excluding these components is threefold: first, the small number of nodes and relationships presents little information for a model to learn from; second, these small networks are less likely to be of interest in real world fraud detection, as illegitimate ownership is usually concealed behind multiple layers of ownership; and from a practical standpoint, the volume of data is brought down to a manageable size. Networks with fewer than 1 in 10 natural persons to companies are excluded, as a focus of this study is on the performance of anomaly detection methods on heterogeneous graph data, as opposed to homogeneous networks.

Finally, we address an observed data quality issue in which multiple nodes in the same component may share the same name. These nodes are merged into a single node, with the resulting node taking on all the ownership relationships of the merged nodes.

## Anomaly Simulation

To simulate anomalies, 10% of the nodes are selected at random and marked as anomalous. A single outgoing edge is chosen from each anomalous node and it's source ID is replaced with that of another node marked as anomalous. This results in the anomalous nodes exchanging ownership relationships, while preserving the overall structure of the graph. The procedure is illustrated in figure {@fig:anomaly-sim-process}. At conclusion, a check is performed to ensure that no anomalous nodes have been allocated their original ownership relationships.

![Process for simulating structural anomalies. The initial graph is shown in (1). A node is selected for anomalisation (orange circle) in (2). The outgoing edge (red arrow) is swapped for that of another anomalised node (green arrow), resulting in the exchange of pink nodes for green (3).](figures/anomaly-simulation-process.png){#fig:anomaly-sim-process short-caption="Anomaly Simulation Process"}

The anomaly simulation process introduces unusual ownership relationships into the graph. The true freuqency of these occurrences is unknown, but is is expected to be far lower than 10% of nodes. A 10% anomaly rate is chosen to ensure that model training is not impossible due to extreme class imbalance, and the same effect can be achieved by oversampling the minority class [@chawlaSMOTESyntheticMinority2002] or undersampling the majority class [@fernandezLearningImbalancedData2018, p.82].

It is assumed that the data originally provided by Open Ownership is accurate and that if any anomalies are present they will have a negligible impact on experimental results. Where anomalies are present, they should have an equal effect on all models, and so should not bias the results.

## Node Features

In order to train a baseline model for comparison, a set of node level features is generated to represent each node in a tabular format [@leskovecTraditionalMethodsMachine2021]. The following topological features are extracted for each node:

- Indegree: The number of incoming edges to the node.
- Outdegree: The number of outgoing edges from the node.
- Closeness Centrality: Inverse of the sum of shortest path distances to all other nodes.
- Clustering Coefficient: Connectedness of neighbouring nodes.
- PageRank: A measure of node importance, based on the importance of neighbouring nodes [@pagePageRankCitationRanking1998].

To capture information about the node's position in the graph, aggregate statistics are calculated for the aforementioned topological features for the node's neighbours and added as features:

- Minimum
- Maximum
- Sum
- Mean
- Standard Deviation

The count of immediate neighvours is also included as a feature.

# Methods

## Traditional Machine Learning Approaches

## Graph Neural Networks

### GraphSAGE

### Graph Convolutional Layer

# Results

# Discussion

# References

<div id="refs"></div>

# Appendix

![Distribution of component sizes before and after anomaly simulation.](figures/component-sizes.png){#fig:comp-dist-anomaly-sim}

![Distribution of node degrees before and after anomaly simulation.](figures/degree-distributions-post.png){#fig:node-degrees-anomaly-sim}
