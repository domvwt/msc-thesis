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

# Notes for the Reviewer

A key focus in my position as Lead Data Scientist at Quantexa is the application
of graph analytics for the detection of fraud. Traditional Machine Learning
techniques struggle to identify anomalous entities in business networks, as
their main giveaway is their relationship to neighbouring entities.

Through this project, I hope to demonstrate  the effectiveness of Graph
Convolutional Neural Network (GCN) models for identifying suspicious actors in
business ownership networks. 

# Introduction

## Subject Overview

In October of 2021, The International Consortium of Investigative Journalists
(ICIJ) revealed the findings of their 'Pandora Papers' investigation,
implicating hundreds of politicians, public officials, and businesses in efforts
to conceal the true ownership of companies and assets around the world
[@icij_offshore_2021]. The intentions behind this secrecy range from legitimate
privacy concerns to criminal activities, including money laundering, tax
evasion, and fraud. According to a study by the European Commission, an
estimated total of USD 7.8 trillion was held offshore in 2016. The share of this
attributed to the European Union (EU) was USD 1.6 trillion, which corresponds to an estimated tax
revenue loss to the EU of EUR 46 billion [@noauthor_estimating_2019].

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

