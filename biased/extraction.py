import random

import pandas as pd


def extract(dataset):
    columns = ["x__contents"]
    data = []
    indexes = []
    index = 0
    for _ in dataset["contents"].astype('str'):
        result = [1]
        data.append(result)
        indexes.append(index)
        index += 1

    data_frame = pd.DataFrame(data=data, columns=columns)
    data_frame["y__S33"] = dataset["S33"]
    data_frame["idx"] = indexes
    return data_frame
