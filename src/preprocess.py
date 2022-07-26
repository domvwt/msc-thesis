import dataclasses as dc

import numpy as np
import pandas as pd
import sklearn.preprocessing as pre
import yaml


@dc.dataclass
class GraphData:
    companies_nodes: pd.DataFrame
    persons_nodes: pd.DataFrame
    edges: pd.DataFrame


def split_dataset(companies_df, persons_df, edges_df):
    components = persons_df.component.unique()
    component_assignment_df = pd.DataFrame(
        np.array([components, components % 10]).T,
        columns=["component", "component_mod"],
    ).sort_values(by="component")

    train_idx = component_assignment_df.query(
        "component_mod >= 1 and component_mod <= 8"
    ).index
    valid_idx = component_assignment_df.query("component_mod >= 9").index
    test_idx = component_assignment_df.query("component_mod == 0").index

    component_assignment_df.loc[train_idx, "split"] = "train"
    component_assignment_df.loc[valid_idx, "split"] = "valid"
    component_assignment_df.loc[test_idx, "split"] = "test"

    # Used in the following queries.
    train_components = component_assignment_df.query("split == 'train'")[
        "component"
    ].to_list()
    valid_components = component_assignment_df.query("split == 'valid'")[
        "component"
    ].to_list()
    test_components = component_assignment_df.query("split == 'test'")[
        "component"
    ].to_list()

    train_person_nodes = persons_df.query("component in @train_components")
    valid_person_nodes = persons_df.query("component in @valid_components")
    test_person_nodes = persons_df.query("component in @test_components")

    train_company_nodes = companies_df.query("component in @train_components")
    valid_company_nodes = companies_df.query("component in @valid_components")
    test_company_nodes = companies_df.query("component in @test_components")

    train_edges = edges_df.query("src in @train_person_nodes.id")
    valid_edges = edges_df.query("src in @valid_person_nodes.id")
    test_edges = edges_df.query("src in @test_person_nodes.id")

    train_split = GraphData(
        companies_nodes=train_company_nodes,
        persons_nodes=train_person_nodes,
        edges=train_edges,
    )

    valid_split = GraphData(
        companies_nodes=valid_company_nodes,
        persons_nodes=valid_person_nodes,
        edges=valid_edges,
    )

    test_split = GraphData(
        companies_nodes=test_company_nodes,
        persons_nodes=test_person_nodes,
        edges=test_edges,
    )

    return train_split, valid_split, test_split


def main():

    # Read config.
    conf_dict = yaml.safe_load(Path("config/conf.yaml").read_text())

    edges_path = conf_dict["persons_nodes"]
    persons_path = conf_dict["companies_nodes"]
    edges_path = conf_dict["edges"]
    connected_components_path = conf_dict["connected_components"]

    companies_encoders = {
        # 'id',
        # 'component',
        # 'isCompany',
        # 'name',
        "foundingDate": None,
        "dissolutionDate": None,
        "countryCode": None,
        # 'companiesHouseID',
        # 'openCorporatesID',
        # 'openOwnershipRegisterID',
        "CompanyCategory": None,
        "CompanyStatus": None,
        "Accounts_AccountCategory": None,
        "SICCode_SicText_1": None,
    }

    persons_encoders = {
        # 'id',
        # 'component',
        # 'isCompany',
        "birthDate": None,
        # 'name',
        "nationality": None,
    }

    edge_encoder = {"minimumShare": None}
