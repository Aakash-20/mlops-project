import logging 
import pandas as pd
from zenml import step

class IngestData:
    def __init__(self, data_path: str):
        """
        Args:
            data_path: path to the data
        """
        self.data_path = data_path

    def get_data(self): 
        """
        Ingesting the data from the data_path:
        """
        logging.info(f"Ingesting data from{self.data_path}")
        return pd.read_csv(self.data_path)
    
@step
def ingest_df(data_path: str) -> pd.DataFrame:
    """
    Ingestig the data from the data_path

    Args: 
        data_path: path to the data

    Returns: 
        pd.DataFrame: the ingested data
    """

    try: 
        ingester = IngestData(data_path=data_path)
        df = ingester.get_data()
        if not isinstance(df,pd.DataFrame):
            raise ValueError("The output of 'get_data' needs to be a DataFrame")
        return df
    except Exception as e:
        logging.error(f"Error while ingesting data: {e}")
        raise e
