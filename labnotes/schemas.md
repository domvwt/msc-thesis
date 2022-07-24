# Schemas

## Nodes

### Companies

```text
root
 |-- id: string (nullable = true)
 |-- component: long (nullable = true)
 |-- isCompany: boolean (nullable = true)
 |-- name: string (nullable = true)
 |-- foundingDate: string (nullable = true)
 |-- dissolutionDate: string (nullable = true)
 |-- countryCode: string (nullable = true)
 |-- companiesHouseID: string (nullable = true)
 |-- openCorporatesID: string (nullable = true)
 |-- openOwnershipRegisterID: string (nullable = true)
 |-- CompanyCategory: string (nullable = true)
 |-- CompanyStatus: string (nullable = true)
 |-- Accounts_AccountCategory: string (nullable = true)
 |-- SICCode_SicText_1: string (nullable = true)
```

### Persons

```text
root
 |-- id: string (nullable = true)
 |-- component: long (nullable = true)
 |-- isCompany: boolean (nullable = true)
 |-- birthDate: string (nullable = true)
 |-- name: string (nullable = true)
 |-- nationality: string (nullable = true)
```

## Edges

```text
root
 |-- src: string (nullable = true)
 |-- interestedPartyIsPerson: boolean (nullable = true)
 |-- dst: string (nullable = true)
 |-- minimumShare: double (nullable = true)
 |-- component: long (nullable = true)
```
