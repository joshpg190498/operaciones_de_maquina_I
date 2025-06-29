import os
import pandas as pd
import numpy as np
import boto3
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from datetime import datetime, timedelta
from airflow.decorators import dag, task

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder



import warnings
warnings.filterwarnings('ignore')


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

@dag(
    dag_id='census_training_pipeline',
    default_args=default_args,
    description='DAG de entrenamiento para el dataset Census Income',
    schedule_interval=None,
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['ml', 'census', 'training']
)
def census_training_pipeline():

    @task()

    def download_dataset_from_s3(bucket_name: str, train_key: str) -> str:

        """
        Descarga el CSV combinado desde un bucket S3 (como MinIO) y guarda el resultado localmente.
        """

        s3 = boto3.client(
            's3',
            endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
        )

        # Rutas locales temporales
        data_folder = '/tmp/data'
        os.makedirs(data_folder, exist_ok=True)
        train_path = os.path.join(data_folder, train_key)

        # Descargar archivos
        s3.download_file(bucket_name, train_key, train_path)

        return train_path

    @task()
    def eda(path: str) -> str:

        """
        Limpia, normaliza e imputa valores faltantes en el dataset de adultos. Guarda el resultado localmente.
        """

        df = pd.read_csv(path)

        # Se estandariza los nombres de las columnas a minusculas y se limpian espacios.
        df.columns = (
            df.columns.
            str.strip().
            str.lower()
        )

        # Se remueven los espacios en blancos de los valores
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()

        # Podemos observar un '?'. Esto representa valores faltantes. Se lo va a reemplazar aquellos ? por nan
        df.replace("?", np.nan, inplace=True)

        # Se procede con la imputación. Para este caso en particular y debido a que solo hay 10 valores faltantes. Se decile imputar una categoría artificial como 'No-occupation' para los casos que 'workclass' = 'Never-worked'
        df.loc[
            (df['occupation'].isna()) & (df['workclass'] == 'Never-worked'),
            'occupation'
        ] = 'No-occupation'

        df[df['workclass'] == 'Never-worked'][['age', 'workclass', 'occupation', 'native country', 'income']]

        # Imputar valores faltantes con 'Unknown'
        df['workclass'].fillna('Unknown', inplace=True)
        df['occupation'].fillna('Unknown', inplace=True)

        # United-States es el valor más frecuente por superior a los demas. Por este motivo se decide imputar este valor al resto de valores faltante.
        most_common_country = df['native country'].mode()[0]
        # Imputación del valor 'most_common_country' a los valores NaN
        df['native country'].fillna(most_common_country, inplace=True)


        # Eliminar columnas innecesarias
        df.drop(columns=["capital gain", "capital loss"], inplace=True)

        # Variable objetivo "income"
        df['income'] = df['income'].str.strip().str.replace('.', '', regex=False)

        # El campo `final weight` es una variable de muestreo poblacional (para inferencia estadística representativa, no para clasificación individual). Esto indica que este campo no nos sirve porque no aporta valor predictivo directo. <br>
        # Por lo tanto, se procede imputar dicho campo.
        df.drop(columns='final weight', inplace=True)

        # La columna educationnum contiene el número asociado a la descripción del campo 'education'. Por este motivo, se procede a eliminar el campo 'education' y se conserva el campo 'educationnum'
        df.drop(columns='education', inplace=True)

        clean_path = "/tmp/data/adult_clean.csv"
        df.to_csv(clean_path, index=False)
        return clean_path

    @task()
    def preprocess_training(path: str):

        """
        Entrena un modelo Random Forest con los datos procesados, evalúa su desempeño y lo registra en MLflow.
        """
        df = pd.read_csv(path)

        # ----------------------------- 1. DATOS ------------------------------ #
        y   = df["income"].map({ "<=50K": 0, ">50K": 1 })       # target binario
        X   = df.drop(columns="income")                         # features

        num_cols = ["age", "educationnum", "hours per week"]
        bin_cols = ["gender"]
        cat_cols = ["workclass", "marital status", "occupation",
                    "relationship", "race", "native country"]

        # ---------------------- 2. PREPROCESADOR ----------------------------- #
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", num_cols),
                ("bin", OrdinalEncoder(categories=[["Female", "Male"]]), bin_cols),
                ("cat", OneHotEncoder(
                    drop="first",           # ≃ tu drop_first=True
                    handle_unknown="ignore",
                    sparse_output=False
                ), cat_cols),
            ],
            remainder="drop"
        )

        # ---------------------- 3. MODELO COMPLETO --------------------------- #
        rf_params = dict(
            max_depth=40,
            max_features="sqrt",
            min_samples_leaf=40,
            min_samples_split=5,
            n_estimators=100,
            random_state=42,
            class_weight="balanced",
        )

        pipeline = Pipeline([
            ("prep", preprocessor),
            ("clf",  RandomForestClassifier(**rf_params)),
        ])

        # ---------------------- 4. ENTRENAR / VALIDAR ------------------------ #
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42, stratify=y
        )

        pipeline.fit(X_train, y_train)

        print("Accuracy entrenamiento :", round(pipeline.score(X_train, y_train)*100, 2), "%")
        print("Accuracy test          :", round(pipeline.score(X_test,  y_test)*100, 2), "%")

        y_pred = pipeline.predict(X_test)
        print("\nReporte de clasificación:\n", classification_report(y_test, y_pred))


        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"\nAccuracy Test: {round(acc * 100, 2)}%")
        print(f"\nClassification Report:\n{report}")

        # Registrar en MLflow
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("Census_Income_Prediction")

        with mlflow.start_run() as run:
            # Log parámetros
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", 100)

            # Log métricas
            mlflow.log_metric("accuracy", acc)

            # Log del modelo y registro
            mlflow.sklearn.log_model(
                pipeline,
                artifact_path="pipeline",
                registered_model_name="Census_Income_Prediction"
            )

            run_id = run.info.run_id
            print("Run ID:", run_id)

            # Obtengo versión del modelo registrado
            client = MlflowClient()
            latest_version = client.get_latest_versions("Census_Income_Prediction", stages=[])[-1].version

            # Lo promuevo a prod
            client.transition_model_version_stage(
                name="Census_Income_Prediction",
                version=latest_version,
                stage="Production"
            )

            # Asigno alias "Champion" (version de prod)
            client.set_registered_model_alias(
                name="Census_Income_Prediction",
                alias="Champion",
                version=latest_version
            )

            print(f"Modelo versión {latest_version} asignado a 'Production' y alias 'Champion'")                   

    # Orquestación
    combined = download_dataset_from_s3("data", "adult_combined.csv")
    clean = eda(combined)
    preprocess_training(clean)

dag = census_training_pipeline()