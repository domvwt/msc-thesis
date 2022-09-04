import numpy as np
import pandas as pd


def all_shuffled(a1, a2):
    return np.all(a1 != a2)


def get_unshuffled_indexes(a1, a2):
    return np.argwhere(a1 == a2)


def efficient_shuffle(a1):
    def inner(a1, _i):
        print(_i, end="\r")
        a2 = a1.copy()
        rng = np.random.default_rng(42 + _i)
        for i in range(5):
            to_shuffle = get_unshuffled_indexes(a1, a2)
            a2[to_shuffle] = rng.permutation(a2[to_shuffle])
            if all_shuffled(a1, a2):
                break
        else:
            inner(a1, _i + 1)
        return a2

    return inner(a1, 0)


def simulate_anomalies(companies_df, persons_df, edges_df):
    # Select 10% of the person nodes as anomalies.
    anomalous_persons_df = persons_df.sample(frac=0.1, random_state=42)

    # Select 10% of the company nodes as anomalies.
    anomalous_companies_df = companies_df.sample(frac=0.1, random_state=42)

    # Flag the anomalous entities.
    persons_df["is_anomalous"] = False
    persons_df.loc[anomalous_persons_df.index, "is_anomalous"] = True
    companies_df["is_anomalous"] = False
    companies_df.loc[anomalous_companies_df.index, "is_anomalous"] = True

    # Select one edge per anomalous entity.
    anomalous_ids = set(anomalous_persons_df["id"]) | set(anomalous_companies_df["id"])
    anomalous_edges_df = edges_df[edges_df["src"].isin(anomalous_ids)]
    selected_anomalous_edges_df = anomalous_edges_df.groupby("src").sample(
        n=1, random_state=42
    )

    # Shuffle the selected edges to simulate anomalous relationships.
    shuffled_edges_df = selected_anomalous_edges_df.copy()
    input_edges = shuffled_edges_df["src"].to_numpy()
    output_edges = efficient_shuffle(input_edges)
    shuffled_edges_df["src"] = output_edges

    # Update the main edges dataframe.
    edges_anomalised_df = edges_df.copy(deep=True)
    edges_anomalised_df = edges_anomalised_df.drop(shuffled_edges_df.index)
    edges_anomalised_df = pd.concat(
        [edges_anomalised_df, shuffled_edges_df]
    ).sort_index()
    edges_anomalised_df["is_anomalous"] = False
    edges_anomalised_df.loc[anomalous_edges_df.index, "is_anomalous"] = True

    assert len(edges_anomalised_df) == len(edges_df), "Edges have been lost!"
    assert not edges_anomalised_df.equals(
        edges_df
    ), "Anomalous edges dataframe is equal to the original!"

    return companies_df, persons_df, edges_anomalised_df
