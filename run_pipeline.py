from pipeline.training_pipeline import train_pipeline
from zenml.client import Client

if __name__ == "__main__":
    # Run the pipeline
    print(Client().active_stack.experiment_tracker.get_tracking_uri())
    train_pipeline(data_path="C:/Users/aashi/Documents/mlops-project/data/olist_customers_dataset.csv")

# mlflow ui --backend-store-uri "file:C:/Users/aashi/AppData/Roaming/zenml/local_stores/9e58ce5d-4f34-4abf-9aa6-7d21c96f5c83/mlruns"