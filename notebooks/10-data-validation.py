# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: 'Python 3.9.12 (''.venv'': poetry)'
#     language: python
#     name: python3
# ---

# %%
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

import pdcast as pdc

while not Path("data") in Path(".").iterdir():
    os.chdir("..")

import sklearn.preprocessing as pre


# %%
# Read config.
conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

companies_df = pd.read_parquet(conf_dict["companies_preprocessed"])
persons_df = pd.read_parquet(conf_dict["persons_preprocessed"])
edges_df = pd.read_parquet(conf_dict["edges_preprocessed"])


# %%
companies_df.info()

# %%
companies_df.describe()


# %%
def select_processed_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [x for x in df.columns if x.endswith("processed")]
    return df[columns]


# %%
companies_preprocessed_df = pd.read_parquet(conf_dict["companies_preprocessed"])
persons_preprocessed_df = pd.read_parquet(conf_dict["persons_preprocessed"])
edges_preprocessed_df = pd.read_parquet(conf_dict["edges_preprocessed"])

# %%
companies_preprocessed_df = select_processed_columns(companies_preprocessed_df)
persons_preprocessed_df = select_processed_columns(persons_preprocessed_df)
edges_preprocessed_df = select_processed_columns(edges_preprocessed_df)

# %%
companies_preprocessed_df.info()

# %%
companies_preprocessed_df.describe().T

# %%
companies_preprocessed_df.head().T[:20]

# %%
companies_preprocessed_df.tail().T


# %%
# Load features for data split.
def load_features(path_root, split):
    features_dir = Path(path_root) / split
    companies_df = pd.read_parquet(features_dir / "companies.parquet").dropna()
    persons_df = pd.read_parquet(features_dir / "persons.parquet").dropna()
    return companies_df, persons_df


# %%
companies_train_df, persons_train_df = load_features(features_path, "train")

# %%
select_cols = set(companies_train_df.columns) & set(persons_train_df.columns)
processed_feature_cols = [x for x in select_cols if x.endswith("__processed")]
raw_feature_cols = [x.split("__processed")[0] for x in processed_feature_cols]
target = "is_anomalous"

entities_df = pd.concat([companies_train_df, persons_train_df], axis=0)

# Sample entities_df so that half of the entities are anomalous.
def balanced_sample(entities_df) -> pd.DataFrame:
    n_entities = len(entities_df)
    n_anomalous = len(entities_df[entities_df[target] == True])
    n_normal = n_entities - n_anomalous
    n_sample = min(n_anomalous, n_normal)
    anomalous_df = entities_df.query("is_anomalous == False").sample(n_sample)
    normal_df = entities_df.query("is_anomalous == True").sample(n_sample)
    return pd.concat([anomalous_df, normal_df], axis=0)


balanced_sample_df = balanced_sample(entities_df)

# %%
X = pd.concat([balanced_sample_df[processed_feature_cols]], axis=0)
y = balanced_sample_df[target]

# %%
y.value_counts()

# %%
# Train logistic regression model.
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


logreg = LogisticRegression(solver="lbfgs", max_iter=1000)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

## classification report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

## confusion matrix
from sklearn.metrics import confusion_matrix

print(confusion_matrix(y_test, y_pred))

# %%
## Train random forest model.
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


# Random forest classifier.
rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# classification report
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# %%
