from airflow.decorators import dag, task
from datetime import datetime, timedelta
import os
import mlflow
from mlflow.tracking import MlflowClient
import docker, time

default_args = {
    "owner": "airflow",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

@dag(
    dag_id="deploy_model_api",
    description="Promueve el modelo Champion y reinicia FastAPI",
    default_args=default_args,
    start_date=datetime(2025, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["mlflow", "deployment"],
)
def deploy_model_api():

    @task()
    def promote_to_champion(model_name:str="Census_Income_Prediction"):
        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
        client = MlflowClient()
        latest = client.get_latest_versions(model_name, stages=[]) [-1]

        # Pasa la última versión a Production y le pone alias Champion
        client.transition_model_version_stage(
            name=model_name, version=latest.version, stage="Production", archive_existing_versions=True
        )
        client.set_registered_model_alias(
            name=model_name, alias="Champion", version=latest.version
        )
        return latest.version

    @task()
    def restart_fastapi_container(model_version:int, container_name:str="fastapi"):
        """
        Reinicia el contenedor 'fastapi' vía Docker SDK.
        Requiere:
        * pip install docker  (en la imagen de Airflow)
        * Montar /var/run/docker.sock en los servicios Airflow
        """
        client = docker.DockerClient(base_url="unix://var/run/docker.sock")
        container = client.containers.get(container_name)
        container.restart()
        # Esperamos a que vuelva al estado 'running'
        for _ in range(30):
            container.reload()
            if container.status == "running":
                break
            time.sleep(1)
        print(f"FastAPI reiniciado con el modelo v{model_version}")

    v = promote_to_champion()
    restart_fastapi_container(v)

dag = deploy_model_api()