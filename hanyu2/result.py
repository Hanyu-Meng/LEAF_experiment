import pandas as pd
def generateResultFromTraining(train):
    # Any output file generation template goes here
    # result = {}
    # for key in train.history.keys():
    #     print(train.history[key])

    # Nothing special, just return pandas object for
    # CSV generation.
    return pd.DataFrame(train.history)