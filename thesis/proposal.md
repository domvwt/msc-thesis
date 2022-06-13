---
title: Detecting Anomalous Business Ownership with Graph Convolutional Neural Networks
subtitle: Project Proposal
bibliography: ./thesis/thesis.bib
header-includes:
- \usepackage{longtable}\setlength{\LTleft}{1em}
panhan:
-
    use_preset: journal
    output_file: ./thesis/proposal.pdf
-
    use_preset: wordcount

---

<!--
Configuration by Panhan - config handler for Pandoc
https://pandoc.org/
https://github.com/domvwt/panhan
-->

<!-- Doing Your Research Project:
      file:///home/domvwt/Downloads/Judith_Bell_Doing_Your_Research_Project.pdf -->

# Introduction

## Background 

In October of 2021, The International Consortium of Investigative
Journalists (ICIJ) revealed the findings of their Pandora Papers
investigation. Through examination of nearly 12 million confidential
business records, the ICIJ found evidence implicating thousands of
individuals and businesses in efforts to conceal the ownership of
companies and assets around the world [@icij_offshore_2021]. The
intentions behind this secrecy varied from legitimate privacy concerns
to criminal activities, including money laundering, tax evasion, and
fraud
[@european_union_agency_for_law_enforcement_cooperation_shadow_2021]. 

To put these numbers in perspective, a 2019 study by the European
Commission estimated that a total of USD 7.8 trillion was held
offshore as of 2016. The share of this attributed to the European
Union (EU) was USD 1.6 trillion, which corresponds to an estimated tax
revenue loss to the EU of EUR 46 billion
[@european_commission_directorate_general_for_taxation_and_customs_union_estimating_2019].

Identifying the beneficiaries of a company is challenging due to the
ease with which information can be concealed or simply not declared.
Further complication is introduced by the interconnected nature of
businesses and individuals, as well as the ingenuity of criminals in
masking illicit activity.  These difficulties place significant strain
on the resources of law enforcement agencies and financial
institutions [@steven_m_combating_2019].

In April 2016, the United Kingdom made it mandatory for businesses to
keep a register of People with Significant Control. This includes
people who own more than 25% of the company's shares
[@noauthor_keeping_2016]. Ownership data is curated and processed by
the Open Ownership organisation for the purposes of public scrutiny
and research [@noauthor_open_nodate]. It is the data provided by Open
Ownership that forms the basis of this study. Details of suspicious or
illegitimate business owners are not readily available due to the
sensitive nature of such records. We propose a method for simulating
anomalous ownership structures as part of our experimental methods. 

To model the complex network of global business ownership, it is
necessary to represent companies, people, and their relationships in a
graph structure. With data in this format, it is possible to consider
the features of an entity's local neighbourhood when making a
decision, in addition to the entity's own characteristics. Anomaly
detection algorithms that operate on graph structures remain at the
frontier of machine learning research.

To the best of the author's knowledge, there is indeed no published
research studying the effectiveness of graph anomaly detection
techniques on business ownership networks. The following proposal is a
study into the application of state of the art anomaly detection
techniques to business ownership graphs.

## Project Title

The proposed title for this project is "Detecting Anomalous Business
Ownership with Graph Convolutional Neural Networks".

## Aims, Objectives and Research Questions

### Aims

The primary aim of this project is to develop an effective approach
for detecting anomalous entities in a business ownership network.
Second, the project will offer a comparison of Graph Convolutional
Neural Network (GCN) models to traditional anomaly detection
approaches on a business ownership graph. Finally, we contribute a
dataset describing a real business ownership network that is suitable
for graph learning.

### Objectives

- Compose a business ownership graph from open data sources.
- Train and evaluate a state of the art GCN for anomaly detection.
- Perform the anomaly detection task with traditional machine learning methods.
- Compare approaches in terms of effectiveness and applicability.

### Research Questions

<!-- https://writingcenter.gmu.edu/guides/how-to-write-a-research-question -->

The questions driving the research are as follows:

- What is the most effective strategy for detecting anomalous entities in
    business ownership networks?
- How do GCN models compare to traditional approaches in terms of
    classification performance?
- What are the challenges that arise in building and training a GCN
    model and what recommendations can be made to mitigate these?

## Literature Review

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

A thorough description of graph anomaly detection tasks and approaches is offered by @ma_comprehensive_2021. Their taxonomy categorises tasks based on the graph component being targeted: nodes, edges, sub-graphs, or full graphs. The authors state their belief that "because the copious types of graph anomalies cannot be directly represented in Euclidean feature space, it is not feasible to directly apply traditional anomaly detection techniques to graph anomaly detection".

### Anomalous Node Detection with GCNs

@kipf_semi-supervised_2017 propose a scalable GCN architecture for classifying nodes in a partially labelled dataset. Early attempts to apply deep learning to graph structures utilised RNN architectures which prove difficult to scale [@gori_new_2005; @scarselli_graph_2009; @li_gated_2017]. Kipf et al. extend prior work on spectral GCNs [@bruna_spectral_2014; @defferrard_convolutional_2017] to produce a flexible model that scales in linear time with respect to the number of graph edges.

@ding_deep_2019 combine a GCN architecture with an autoencoder in a method that identifies anomalous nodes by reconstruction error. The proposed method, DOMINANT, uses a GCN to generate node embeddings and separately reconstructs both the graph topology and the node attributes. This strategy is further developed and applied to multi-view data through the combination of multiple graph encoders [@peng_deep_2022].

An alternative method is offered by @li_specae_2019, in which a spectral convolution and deconvolution framework is used to identify anomalous nodes in conjunction with a density estimation model. The approach continues to demonstrate the importance of combining multiple perspectives of the network data, with the innovation being the use of a Gaussian Mixture Model to combine representations in a single view.

### Recent Developments

@velickovic_graph_2018 demonstrate the use of self-attention layers to address shortcomings in the representations captured by GCN architectures. However, a comparison of Relational Graph Attention (GAT) models to GCNs showed that relative performance was task dependent and that current GAT models could not be shown to consistently outperform GCNs on benchmark exercises [@busbridge_relational_2019].

In an application of graph attention based models to financial fraud detection, @wang_semi-supervised_2019 show that their SemiGNN model outperforms established approaches when predicting risk of default and in attribute prediction. Baseline methods used for comparison included a XGBoost [@chen_xgboost_2016], GCN, GAT, and LINE [@tang_line_2015].


# Methods

## Data

### Open Datasets

The data needed for this study is made publicly available by the UK government via Companies House. Additional processing and entity resolution is applied to UK Persons of Significant Control data by the Open Corporates organisation, and so PSC data is extracted from their own database. A summary of the data requirements is presented below. 

|Data                           |Provider                         |Date Retrieved|
|:------------------------------|:--------------------------------|:-------------|
|[Company Data][CompanyData]    |[Companies House][CompaniesHouse]|2022-05-24    |
|[Persons of Significant Control][PSCData]            |[Companies House][CompaniesHouse]|2022-05-24    |
|[Open Ownership Data][OwnershipData]|[Open Ownership][OpenOwnership]  |2022-05-24    |

: Data Requirements

 <!-- Links -->
 
[CompaniesHouse]: https://www.gov.uk/government/organisations/companies-house
[OpenOwnership]: https://www.openownership.org/
[ICIJ]: https://offshoreleaks.icij.org/
[CompanyData]: http://download.companieshouse.gov.uk/BasicCompanyDataAsOneFile-2022-05-01.zip
[PSCData]: http://download.companieshouse.gov.uk/persons-with-significant-control-snapshot-2022-05-24.zip
[OwnershipData]: https://oo-register-production.s3-eu-west-1.amazonaws.com/public/exports/statements.2022-05-27T19:23:50Z.jsonl.gz
[OffshoreLeaks]: https://offshoreleaks-data.icij.org/offshoreleaks/csv/full-oldb.20220503.zip#_ga=2.29179839.263289515.1654130741-1571958430.1654130741

### Simulated Data

Due to the absence of labelled shell or otherwise illegitimate company data, we are required to simulate anomalous business ownership networks in order to provide training examples for machine learning. While the full details of this procedure are the subject of ongoing experimentation, the high level steps are:

1. Identify connected graph components within the Open Ownership dataset.
2. Split these components into two groups, one to serve as the legitimate network set, the other as the anomalous network set.
3. Create anomalous networks within the anomalous network set by transferring one or more nodes from one connected component to another. The transferred nodes will be labelled as anomalous for the purposes of training.

## Research Instruments and Tools

### Data Preparation

In order to process the large volumes of records provided by UK companies, we will use the Apache Spark Framework [@noauthor_apache_nodate] for reading and transforming the data. To construct and manipulate graphs in an efficient manner, we use the GraphFrames packages for Spark [@noauthor_overview_nodate]. 

### Graph Analysis

A number of software libraries and applications will be considered for analysing graph properties and generating graph features. These are:

- NetworkX
- igraph
- Neo4j

### Deep Learning

The most widely used framework for deep learning on graphs is currently PyTorch-Geometric. A number of SOTA algorithms are made available through the library, as well as detailed documentation and tutorials. 

### Baseline Machine Learning Models

A selection of established machine learning algorithms will be used to benchmark performance.

- Random Forest
- Gradient Boosted Tree
- Logistic Regression

### Performance Evaluation

A variety of classification metrics will be used to assess model performance. Many of these can be found in the [Scikit Learn](https://scikit-learn.org/stable/modules/model_evaluation.html#classification-metrics) library.

## Ethical Considerations

All data used in the course of this project has been made publicly available for the purposes of public scrutiny and academic research.

In the interests of personal privacy, any sensitive personal information will be transformed and / or removed from the dataset as appropriate. Specifically, we will remove address information, individual dates of birth (to be transformed to an age range), and the names of individual persons and companies.

Further to this, the study will refrain from speculating on the nature of any particular business structure or entity, and any speculation with regards to the legitimacy or legality of a particular arrangement is deemed strictly out of scope and beyond the remit of the study.

To facilitate academic openness and collaboration, the instructions and code required to produce the datasets and experimental results will be made available where it is acceptable to do so.

# Anticipated Outcomes

We expect that GCN models will outperform traditional anomaly detection techniques in identifying anomalous business ownership networks. It is supposed that due to the nature of the simulated graphs, the anomalous entities will be difficult to identify from their individual attributes and local neighbourhood properties.

# Project Plan

## Roadmap

|Task                |Status       |
|:-------------------|:------------|
|Data acquisition    |Completed    |
|Data understanding  |Completed    |
|Data preparation    |Completed    |
|Anomaly Simulation |Not Started |
|Feature engineering | Not Started  |
|Preprocessing       |Not Started  |
|Modelling           |Not Started  |
|Evaluation          |Not Started  |


## Risks and Challenges

From reading the experiences of other researchers during the literature review, it is apparent that training a GCN may be complicated and resource intensive. It may be difficult to train a model of reasonable size on a single laptop device, and so care has been taken during the data preparation to ensure that all steps should be easily replicable on a cloud hosted virtual machine. This will allow for possible future requirements to upscale the processing power and even employ distributed training methods if feasible.

The proposed approach for simulating anomalous networks is as yet untested. It may arise that these anomalous networks are trivially easy to distinguish, leading to no differentiation in the performance of the benchmark algorithms and GCN model. On the other hand, the anomalous networks may not be sufficiently distinguishable from legitimate networks, leading to all algorithms failing to learn any mapping and little differentiation in algorithm performance. Should this be the case, additional research may need to be undertaken in order to devise a more suitable anomaly simulation process or to identify other sources of training data.

---

**Wordcount:** 2,166