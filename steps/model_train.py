import logging
import pandas as pd
from zenml import step
from src.model_dev import LinearReggressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig
@step 
def train_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
) -> RegressorMixin:
    """Train the model on the ingested data.
    
    Args:
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.DataFrame,
        y_test: pd.DataFrame
    """
    try:
        config = ModelNameConfig()
        model = None
        if  config.model_name == "LinearRegression":
            model = LinearReggressionModel()
            trained_model = model.train(X_train, y_train)
            return trained_model
        else:
            raise  ValueError("Model {} not supported". format(config.model_name))
    except Exception as e:
        logging.error("error in training model: {}".format(e))
        raise e