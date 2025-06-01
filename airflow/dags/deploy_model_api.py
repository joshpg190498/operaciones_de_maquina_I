from airflow.decorators import dag, task
from datetime import datetime, timedelta
import os
import shutil
import mlflow


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

@dag(
    dag_id='deploy_model_api',
    default_args=default_args,
    description='DAG para desplegar el modelo como API REST con FastAPI',
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['ml', 'fastapi', 'deployment']
)
def deploy_model_api():

    @task()
    def download_model_from_mlflow(model_name: str, stage: str = "Production") -> str:
        """
        Descarga el modelo registrado en MLflow.
        """
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

        print("Model name: ",model_name)
        print("Stage: ",stage)

        model_uri = f"models:/{model_name}/{stage}"
        print("model_uri: ",model_uri)

        local_path = f"/tmp/{model_name}_model"
        print("local_path: ",local_path)

        if os.path.exists(local_path):
            shutil.rmtree(local_path)

        mlflow.artifacts.download_artifacts(artifact_uri=model_uri, dst_path=local_path)
        return local_path

    @task()
    def prepare_fastapi_app(model_path: str):
        """
        Copia el modelo al directorio de FastAPI y lo prepara para servir.
        """
        #dst_model_dir = "/opt/airflow/dockerfiles/fastapi/model"
        dst_model_dir = "airflow/dockerfiles/fastapi/model"
        os.makedirs(dst_model_dir, exist_ok=True)

        for file_name in os.listdir(model_path):
            shutil.copy(os.path.join(model_path, file_name), dst_model_dir)
        print(f"Modelo copiado a {dst_model_dir}")

    model_path = download_model_from_mlflow(model_name="Census_Income_Prediction")
    prepare_fastapi_app(model_path=model_path)


dag = deploy_model_api()