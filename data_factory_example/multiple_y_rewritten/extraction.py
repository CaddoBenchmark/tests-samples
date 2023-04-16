import numpy as np
import pandas as pd

def extract(dataset):
    x_columns = ["contents"]
    y_columns = ["G0", "G1", "G2", "G3"]
    result = []
    indexes = []
    index = 0

    for _ in dataset[x_columns[0]].astype('str'):
        indexes.append(index)
        index += 1

    x_new_names = []
    y_new_names = []
    for x_column in x_columns:
        x_new_names.append("x__" + x_column)
    for y_column in y_columns:
        y_new_names.append("y__" + y_column)
    columns = []
    for column in x_columns:
        columns.append(column)
    for column in y_columns:
        columns.append(column)
    data_frame = pd.DataFrame(data=result, columns=columns)

    for i in range(len(x_columns)):
        data_frame[x_new_names[i]] = dataset[x_columns[i]]
    for i in range(len(y_columns)):
        data_frame[y_new_names[i]] = dataset[y_columns[i]]

    data_frame["idx"] = indexes
    return data_frame
