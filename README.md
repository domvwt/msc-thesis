# Detecting Anomalous Business Ownership Structures with Graph Neural Networks

## Instructions

### System Requirements

- Linux based operating system, 16GB RAM for dataset generation.
- For model training a 16GB+ GPU is strongly recommended.

### Initial setup

First, install the project with either **pip**:

```bash
pip install .
```

 or **poetry**:

```bash
poetry install
```

### Dataset Generation

External data sources should be downloaded from the links provided below or requested from the author.

The all data processing can be completed by executing the scripts in `src/mscproject/jobs` in the indicated order.

You may need to install the Apache Spark framework with `bash scripts/spark-install.sh`.

### GNN Model Training

We recommend using a GPU with minimum 15GB of memory for training the GNN models.

1. Create a virtual machine with GPU on Google Cloud Platform with `scripts/create-vm.sh`
2. If data processing has been completed locally, upload to cloud storage with `scripts/upload-data.sh` (remember to set the GCP_BUCKET_NAME environment variable).
3. SSH into the virtual machine and build the Docker image with `scripts/build-docker.sh`
4. Download data to the virtual machine with `scripts/download-data.sh` (don't forget to set the GCP_BUCKET_NAME environment variable on the VM).
5. Run `scripts/exec-docker-optuna.sh` to perform the neural architecture search. This script can be modifified to change the number of trials and other parameters.
6. Run `scripts/exec-docker-gnn-evaluation.sh` to produce test set predictions for the best GNN model architectures.
7. Run `scripts/upload-data.sh` to upload the test set predictions to cloud storage.
8. Run `scripts/download-data.sh` on your local machine to retrieve the test set predictions for further analysis.

### CatBoost Model Training

The entire CatBoost training and tuning process can be completed by running `notebooks/13-catboost-model.ipynb`.

### Evaluation

Model performance is evaluated on the test set predictions in `notebooks/17-full-evaluation.ipynb`.

## Project Overview

Here is an overview of the key files within this project:

```
├── Dockerfile                            <---- Project container image
├── config
│   └── conf.yaml                         <---- Configuration file for data processing
├── figures
├── models/pyg                            <---- Trained PyG models
│   ├── regularised
│   │   ├── GCN.pt (kGNN)
│   │   └── GraphSAGE.pt
│   └── unregularised
│       ├── GCN.pt (kGNN)
│       └── GraphSAGE.pt
├── notebooks                             <---- Notebooks for training and evaluating models
│   ├── archive
│   ├── 13-catboost-model.ipynb
│   ├── 16-gnn-evaluation.ipynb
│   ├── 17-full-evaluation.ipynb
│   ├── 18-graph-visualisation.ipynb
│   ├── addresses.html
│   ├── companies.html
│   ├── persons.html
│   └── relationships.html
├── poetry.lock
├── pyproject.toml
├── reports
├── requirements.txt
├── **scripts**                         <---- Scripts for building the project on cloud
│   ├── create-vm.sh                    <---- Create VM on Google Cloud Platform with a GPU
│   ├── build-docker.sh                 <---- Build the project docker image
│   ├── download-data.sh                <---- Download `data/` from cloud storage
│   ├── exec-docker-gnn-evaluation.sh   <---- GNN evaluation process in Docker container
│   ├── exec-docker-optuna.sh           <---- NN search and optimisation process in Docker container
│   ├── launch-gnn-evaluation.sh        <---- Launch GNN evaluation process
│   ├── launch-optuna.sh                <---- Configure and launch NN search and optimisation
│   ├── spark-install.sh                <---- Install Spark (linux only)
│   └── upload-data.sh                  <---- Upload `data/` to cloud storage
├── src
│   └── mscproject
│       ├── dataprep.py                 <---- Initial data preparation
│       ├── datasets.py                 <---- PyG class for the Dataset
│       ├── experiment.py               <---- NN architecture search and optimisation
│       ├── features.py                 <---- Graph feature generation
│       ├── graphprep.py                <---- Graph building and processing
│       ├── metrics.py                  <---- Metrics for evaluating the models
│       ├── models.py                   <---- PyG model classes
│       ├── preprocess.py               <---- Data preprocessing (normalisation, etc.)
│       ├── pygloaders.py               <---- PyG data loaders
│       ├── simulate.py                 <---- Graph anomaly simulation
│       └── transforms.py               <---- Misc data transforms
│       └── **jobs**                    <---- Scripts for running the data processing pipeline
│           ├── 01_process_raw_data.py
│           ├── 02_process_interim_data.py
│           ├── 03_make_connected_components.py
│           ├── 04_make_graph_data.py
│           ├── 05_make_anomalies.py
│           ├── 06_make_graph_features.py
│           ├── 07_preprocess_features.py
│           └── 08_make_pyg_data.py
```

## Data Sources

Data used in this project can be found at these pages.
Please email the study authors if you require a copy of the original snapshots used.

|Data                           |Provider                         |Date Retrieved|
|:------------------------------|:--------------------------------|:-------------|
|[Company Data][CompanyData]    |[Companies House][CompaniesHouse]|2022-05-24    |
|[PSC Data][PSCData]            |[Companies House][CompaniesHouse]|2022-05-24    |
|[Ownership Data][OwnershipData]|[Open Ownership][OpenOwnership]  |2022-05-24    |
|[Offshore Leaks][OffshoreLeaks]|[ICIJ][ICIJ]                     |2022-06-02    |

 <!-- Links -->

[CompaniesHouse]: https://www.gov.uk/government/organisations/companies-house
[OpenOwnership]: https://www.openownership.org/
[ICIJ]: https://offshoreleaks.icij.org/
[CompanyData]: http://download.companieshouse.gov.uk/en_output.html
[PSCData]: http://download.companieshouse.gov.uk/en_pscdata.html
[OwnershipData]: https://oo-register-production.s3-eu-west-1.amazonaws.com/public/exports/statements.2022-05-27T19:23:50Z.jsonl.gz
[OffshoreLeaks]: https://offshoreleaks-data.icij.org/offshoreleaks/csv/full-oldb.20220503.zip#_ga=2.29179839.263289515.1654130741-1571958430.1654130741
