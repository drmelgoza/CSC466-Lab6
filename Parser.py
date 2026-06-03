import pandas as pd
import numpy as np

#jester-1 is the only data we are using in this assignment,
#so it'll be hardcoded into the program later
def parse_data(csv_path: str) -> tuple[np.ndarray, pd.DataFrame]:
    df = pd.read_csv(csv_path)

    #The column corresponding to the number of jokes rated by each person.
    jokes_rated = df.iloc[:, 0].to_numpy()

    #The complete sparse ratings matrix
    #Note that rows 5, 7, 8, 13, 15, 16, 17, 18, 19, 20 are completely dense.
    ratings = df.drop(['0'], axis=1)

    #some item/user related statistics go here.

    return jokes_rated, ratings

