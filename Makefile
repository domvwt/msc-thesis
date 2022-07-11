.DEFAULT_GOAL := help

define BROWSER_PYSCRIPT
import os, webbrowser, sys

try:
	from urllib import pathname2url
except:
	from urllib.request import pathname2url

webbrowser.open("file://" + pathname2url(os.path.abspath(sys.argv[1])))
endef
export BROWSER_PYSCRIPT

define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-20s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT

BROWSER := python -c "$$BROWSER_PYSCRIPT"

.PHONY: help
help:
	@python -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: clean
clean: clean-build clean-pyc clean-test ## remove all build, test, coverage and Python artifacts

.PHONY: clean-build
clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

.PHONY: clean-pyc
clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

.PHONY: clean-test
clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/
	rm -fr .pytest_cache

.PHONY: format
format: ## apply black code formatter
	black .

.PHONY: lint
lint: ## check style with flake8
	flake8 src 

.PHONY: mypy
mypy: ## check type hints
	mypy src --strict

.PHONY: isort
isort: ## sort imports
	isort src --profile black

.PHONY: cqa
cqa: format isort lint mypy ## run all cqa tools

.PHONY: install_python_deps
install_python_deps: ## install python dependencies
	poetry install
	poetry install -E geometric

.PHONY: install_spark_deps
install_spark_deps: ## install spark dependencies
	wget https://repos.spark-packages.org/graphframes/graphframes/0.8.2-spark3.1-s_2.12/graphframes-0.8.2-spark3.1-s_2.12.jar \
	-O .venv/lib/python3.9/site-packages/pyspark/jars/graphframes-0.8.2-spark3.1-s_2.12.jar

.PHONY: install_all_deps
install_all_deps: install_python_deps install_python_deps ## install python and spark dependencies

.PHONY: download_open_ownership_data
download_open_ownership_data: ## download open ownership data
	wget https://oo-register-production.s3-eu-west-1.amazonaws.com/public/exports/statements.2022-05-27T19:23:50Z.jsonl.gz \
	-O data/raw/open-ownership-data.jsonl.gz && \
	gunzip data/raw/open-ownership-data.jsonl.gz

.PHONY: download_companies_house_data
download_companies_house_data: ## download data from companies house
	wget http://download.companieshouse.gov.uk/BasicCompanyDataAsOneFile-2022-05-01.zip \
	-O data/raw/companies-house-data.zip && \
	unzip data/raw/companies-house-data.zip

.PHONY: download_all_data
download_all_data: download_open_ownership_data download_companies_house_data ## download all raw data

NEO4J = ~/.config/Neo4j\ Desktop/Application/relate-data/dbmss/dbms-787062ea-0f75-4dbb-8f4a-db646f3f88d4/bin/neo4j-admin

.PHONY: neo4j_find_cli
neo4j_find_cli: ## find neo4j cli paths
	find / -name neo4j-admin 2> /dev/null | sort

.PHONY: neo4j_import
neo4j_import: ## load data into neo4j (set neo4j cli variable first)
	$(NEO4J) import \
	--database=neo4j \
	--ignore-extra-columns=true \
	--nodes=Company=neo4j/companies-header.csv,data/neo4j/companies.csv \
	--nodes=Person=neo4j/persons-header.csv,data/neo4j/persons.csv \
	--relationships=Owns=neo4j/relationships-header.csv,data/neo4j/relationships.csv

.PHONY: jupytext_sync
jupytext_sync: ## sync jupytext
	jupytext --sync notebooks/*.ipynb

