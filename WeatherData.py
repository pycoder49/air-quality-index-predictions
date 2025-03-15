from typing import List
import pandas as pd
import torch
import requests


def fetch_data(start_date: int,
               end_date: int,
               app_id: str,
               lat: int = 32.715736,
               long: int = -117.161087) -> pd.DataFrame:
    """
    :param start_date:  number of seconds since Jan 1st, 1970
    :param end_date:    number of seconds since Jan 1st, 1970
    :param app_id:      unique authorization key
    :param lat:         latitude of desired location
    :param long:        longitude of desired location
    :return:            dataframe containing weather data
    """
    parameters = {
        "lat": lat,
        "lon": long,
        "start": start_date,
        "end": end_date,
        "appid": app_id
    }
    response = requests.get("http://api.openweathermap.org/data/2.5/air_pollution/history", params=parameters)
    data = response.json()

    data_list = data["list"]  # 35600 entries for 10 years

    # making dataframe
    df_dict = {title: [] for title in data_list[0]["components"].keys()}
    df_dict["aqi"] = []

    for entry in data_list:
        # fill in components
        for component in entry["components"]:
            df_dict[component].append(entry["components"][component])
        # filling in aqi list
        df_dict["aqi"].append(entry["main"]["aqi"])

    return pd.DataFrame(df_dict)


def clean_data(data: pd.DataFrame, train_index: int = 0.7) -> List[torch.Tensor]:
    """
    :param train_index:     Percent of data to be kept for training
    :param data:            DataFrame containing weather data
    :return:                Four torch tensors for training/testing purposes
    """
    """
    Steps:
    1) Check for missing data:
        - If more than half a column is missing, remove the column
        - For numerical missing data, fill it in with   the mean of the column
        - For categorical missing data, fill it in with the mode of the column
    
    2) One-hot encode categorical data for easier interpretation of data
    
    3) Standardize numerical columns using z-score normalization
        to avoid larger numbers having a greater influence on the weights/bias
    
    4) Split the dataframe into four tensors:
        x_train
        y_train
        x_test
        y_test
        
    Since I know beforehand, that my data has no categorical data, and no missing values, I will
    skip steps 1 and 2
    """
    target_col = data["aqi"]
    data.drop(columns="aqi", inplace=True)

    mean = data.mean()
    std = data.std()
    standardized_df = (data - mean) / std

    standardized_df["aqi"] = target_col     # adding back the target col

    # shuffling for randomization
    standardized_df = standardized_df.sample(frac=1).reset_index(drop=True)

    # splitting the datasets
    split_index = int(train_index * len(standardized_df))

    train_df = standardized_df.iloc[:split_index]
    test_df = standardized_df.iloc[split_index:]

    # returning tensors
    y_train = torch.from_numpy(train_df["aqi"].values).float().view(-1, 1)
    x_train = torch.from_numpy(train_df.drop("aqi", axis=1).values).float()
    y_test = torch.from_numpy(test_df["aqi"].values).float().view(-1, 1)
    x_test = torch.from_numpy(test_df.drop("aqi", axis=1).values).float()

    return [x_train, y_train, x_test, y_test]

