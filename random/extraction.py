import pandas as pd


def extract(dataset):
    columns = ['x__contents', "y__G1"]
    result = []
    indexes = []
    index = 0
    for _ in dataset["contents"].astype('str'):
        indexes.append(index)
        index += 1

    data_frame = pd.DataFrame(data=result, columns=columns)
    data_frame["y__G1"] = dataset["G1"]
    data_frame["x__contents"] = dataset["contents"]
    data_frame["idx"] = indexes
    return data_frame
