#! /usr/bin/bash

# ----------------------------------------------------------------------------
# Spark Setup Script
# ----------------------------------------------------------------------------
# Download, install, and configure Spark on Ubuntu.
# ----------------------------------------------------------------------------

SPARK_VERSION="3.1.1"
HADOOP_VERSION="3.2"

echo "Installing Spark..."
if [[ ! -d "/opt/spark" ]]; then
  echo "Updating system..." && \
  sudo apt update -y && \
  sudo apt upgrade -y && \
  sudo apt install openjdk-8-jre -y && \
  echo "Downloading Spark..." && \
  wget https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz \
  -O spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
  tar -xf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
  sudo mv spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION} /opt/spark  && \
  rm -rf spark-${SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz && \
  echo 'export SPARK_HOME=/opt/spark' >> ~/.bash_profile && \
  echo 'export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin' >> ~/.bash_profile && \
  echo 'export PYSPARK_DRIVER_PYTHON=/usr/bin/ipython3' >> ~/.bash_profile && \
  echo 'export PYSPARK_PYTHON=/usr/bin/python3' >> ~/.bash_profile && \
  source ~/.bash_profile && \
  echo "Complete."
else
  echo "Spark already installed at: /opt/spark"
fi