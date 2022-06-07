---
title: Detecting Anomalous Business Ownership with Graph Convolutional Neural Networks
subtitle: Project Proposal
bibliography: ./thesis/thesis.bib
panhan:
-
    use_preset: journal
    output_file: ./thesis/proposal.pdf
---

<!--
Configuration by Panhan - config handler for Pandoc
https://pandoc.org/
https://github.com/domvwt/panhan
-->

# Introduction

## Background 

In October of 2021, The International Consortium of Investigative Journalists
(ICIJ) revealed the findings of their Pandora Papers investigation. Through
examination of nearly 12 million confidential business records, they put forward
evidence implicating thousands of individuals and businesses in efforts to
conceal the true ownership of companies and assets around the world
[@icij_offshore_2021]. The intentions behind this secrecy varied from legitimate
privacy concerns to criminal activities, including money laundering, tax
evasion, and fraud
[@european_union_agency_for_law_enforcement_cooperation_shadow_2021]. 

A 2019 study by the European Commission estimates that a total of USD 7.8
trillion was held offshore as of 2016. The share of this attributed to the
European Union (EU) was USD 1.6 trillion, which corresponds to an estimated tax
revenue loss to the EU of EUR 46 billion
[@european_commission_directorate_general_for_taxation_and_customs_union_estimating_2019].

Identifying the ultimate beneficiaries of a company is challenging due to the
ease with which information can be concealed or simply not declared. This makes
uncovering true company ownership an intensive exercise, placing strain on the
resources of law enforcement agencies and responsible financial institutions
[@steven_m_combating_2019]. Processing and flagging high risk entities is made
difficult by the interconnected nature of businesses and individuals, as well as
the ingenuity of criminals in masking illicit activity behind layers of
seemingly legitimate business.

In order to model the complex network of global business ownership, it is
necessary to represent companies, people, and their relationships in a graph
structure. With the data in this format, it is possible to not only consider the
features of a particular entity when making a decision, but also those of their
close connections and local community. Anomaly detection algorithms that can
operate on graph structures remain at the frontier of machine learning research.
The following project proposal is a study into the application of state of the
art anomaly detection techniques to business ownership graphs.

<!-- TODO: use of machine learning in fraud detection -->
<!-- TODO: disclose my own interests as an employee of Quantexa -->
<!-- TODO: summarise reasons for interest in this topic -->
<!-- TODO: challenges - lack of training data, highly sensitive and proprietary
-->

## Project Title

The proposed title for this project is "Detecting Anomalous Business Ownership
with Graph Convolutional Neural Networks".

## Aims, Objectives and Research Questions

### NOTES

- reasons for study
  - existing studies do not test GCN on business ownership graphs
  - important for detecting fraud, specifically money laundering 
  - traditional methods do not take into account contextual data


### Aims

The aim of this project is to assess the performance of Graph Convolutional
Neural Network (GCN) models in identifying anomalous entities in a business
ownership graph. 

## Literature Review

# Methods

## NOTES

- split graph into weakly connected components (define term)
- select random node(s) from outside of the connected component as target
- impossible for traditional methods to identify as anomalous as features
    are indistinguishable from others
- can attempt traditional anomaly detection techniques on individual connected
    component as a baseline
  - random forest
  - gradient boosted tree
  - K Nearest Neighbours
  - logistic regression
  - GraphGym
  - Tuned GCN (Optuna or similar)

## Data

## Research Instruments and Tools

## Ethical Considerations

# Anticipated  Outcomes

# Project Plan

## Roadmap

- Data acquisition
- Data understanding 
- Data preparation
- Feature engineering
- Preprocessing
- Modeling
- Evaluation

### Data Acquisition



## Risks and Challenges

