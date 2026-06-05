import pandas as pd
import numpy as np

#jester-1 is the only data we are using in this assignment,
#so it'll be hardcoded into the program later
def parse_data() -> tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_csv('jester-data-1.csv', header=None)

    #The column corresponding to the number of jokes rated by each person.
    jokes_rated = df.iloc[:, 0].to_numpy()

    ratings = df.drop([0], axis=1)
    ratings.columns = range(ratings.shape[1])

    return jokes_rated, ratings

