import random

import pandas as pd


def extract(dataset):
    keywords = ["=", "1", "int", "string", "delete", "update", "drop", "execute"]
    columns = ["x__=", "x__1", "x__int", "x__string", "x__delete", "x__update",
               "x__drop", "x__execute", 'y__class']
    data = []
    indexes = []
    index = 0
    for content in dataset["contents"].astype('str'):
        result = []
        for keyword in keywords:
            if keyword in content:
                result.append(1)
            else:
                result.append(0)
        result.append(random.randrange(0, 2))
        data.append(result)
        indexes.append(index)
        index += 1

    data_frame = pd.DataFrame(data=data, columns=columns)
    data_frame["y__G1"] = dataset["G1"]
    data_frame["y__G2"] = dataset["G2"]
    data_frame["idx"] = indexes
    return data_frame
