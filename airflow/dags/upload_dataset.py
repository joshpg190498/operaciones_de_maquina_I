import os
import pandas as pd
import boto3
from airflow.decorators import dag, task
from airflow.utils.dates import days_ago
from io import BytesIO

import warnings
warnings.filterwarnings('ignore')


@dag(
    dag_id='upload_dataset',
    description='DAG de carga de dataset al bucket',
    schedule_interval="@once",
    catchup=False,
    tags=['load', 'datasets'],
    start_date=days_ago(1),

    is_paused_upon_creation=False
)
def upload_dataset_pipeline():

    @task()
    def upload_dataset_to_s3():
        try:
            # Datasets iniciales
            datasets_path = '/opt/airflow/datasets'
            df_train = pd.read_csv(os.path.join(datasets_path, 'adult.csv'))
            df_test = pd.read_csv(os.path.join(datasets_path, 'adult.test.csv'))
            
            # Se obtiene los headers de train dataset
            df_test.columns = df_train.columns.tolist()

            # Unir datasets
            df = pd.concat([df_train, df_test], axis=0)

            # Conexión a s3
            s3 = boto3.client(
                's3',
                endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000"),
                aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
                aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
            )

            # Carga de df a s3
            buffer = BytesIO()
            df.to_csv(buffer, index=False)
            buffer.seek(0)
            s3.put_object(Bucket='data', Key='adult_combined.csv', Body=buffer.getvalue())

        except Exception as e:
            raise RuntimeError(f"No se pudo cargar el dataset al bucket → {e}")
        
    # Orquestación
    upload_dataset_to_s3()

dag = upload_dataset_pipeline()