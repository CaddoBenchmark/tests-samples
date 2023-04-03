import pandas as pd


def extract(dataset):
    columns = ['x__contents', "y__G0"]
    result = []
    indexes = []
    index = 0
    for _ in dataset["contents"].astype('str'):
        # result.append(np.array([x]))
        indexes.append(index)
        index += 1

    data_frame = pd.DataFrame(data=result, columns=columns)
    data_frame["y__G0"] = dataset["G0"]
    data_frame["x__contents"] = dataset["contents"]
    data_frame["idx"] = indexes
    return data_frame
