import numpy as np
import sklearn.preprocessing as pre

import yaml

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
        'foundingDate': None,
        'dissolutionDate': None,
        'countryCode': None,
        # 'companiesHouseID',
        # 'openCorporatesID',
        # 'openOwnershipRegisterID',
        'CompanyCategory': None,
        'CompanyStatus': None,
        'Accounts_AccountCategory': None,
        'SICCode_SicText_1': None
    }

    persons_encoders = {
        # 'id', 
        # 'component', 
        # 'isCompany', 
        'birthDate': None, 
        # 'name', 
        'nationality': None
    }

    edge_encoder = {
        'minimumShare': None
    }