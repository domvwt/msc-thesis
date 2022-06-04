---
title: Detecting Anomalous Business Ownership with Graph Convolutional Neural Networks
subtitle: Project Proposal
panhan:
    use_preset: journal
    output_file: proposal.pdf
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
- Data validation 
- Data preparation - build graph structure
- Feature engineering
- Preprocessing
- Modeling
- Evaluation

## Risks and Challenges

