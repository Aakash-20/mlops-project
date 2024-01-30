import logging
import pandas as pd
from zenml import step
from typing import Tuple
from typing_extentions import Annotated
from src.data_cleaning import DataCleaning, DataDivideStratergy, DataPreProcessStartegy

@step 
def clean_df(df: pd.DataFrame) -> Tuple[
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test "]
]:
    """
    cleans the data and divides it into train and test

    Args:
        df: Raw data
    Returns:
        X_train: Training data
        X_test: Testing data
        y_train: Training data
        y_test: Testing data
    """
    try:
        process_strategy = DataPreProcessStartegy()
        data_cleaning = DataCleaning(df, process_strategy)
        processed_data = data_cleaning.handle_data()

        divide_strategy = DataDivideStratergy()
        data_cleaning = DataCleaning(processed_data, divide_strategy)
        X_train, X_test, y_train, y_test = data_cleaning.handle_data()
        logging.info("Data cleaning complete")
    except Exception as e:
        logging.error("Error in cleaning data: {}".format(e))
        raise e

