import os
import pandas as pd
import boto3
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from datetime import datetime, timedelta
from airflow.decorators import dag, task
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score


from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

#métricas
from sklearn.metrics import accuracy_score


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
    def load_and_combine_data_from_s3(bucket_name: str, train_key: str, test_key: str) -> str:
        s3 = boto3.client(
            's3',
            endpoint_url=os.getenv("AWS_ENDPOINT_URL_S3", "http://s3:9000"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "minio"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "minio123"),
        )

        # Rutas locales temporales
        os.makedirs("/tmp/data", exist_ok=True)
        train_path = "/tmp/data/adult.csv"
        test_path = "/tmp/data/adult.test.csv"

        # Descargar archivos
        s3.download_file(bucket_name, train_key, train_path)
        s3.download_file(bucket_name, test_key, test_path)

        # Leer archivos correctamente
        df_train = pd.read_csv(train_path)

        # Leer test con las mismas columnas
        df_test = pd.read_csv(test_path, skiprows=1, header=None)
        df_test.columns = df_train.columns

        # Unir datasets
        df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

        # Limpieza de columnas si tienen espacios
        df.columns = df.columns.str.strip()

        # Guardar resultado combinado
        output_path = "/tmp/adult_combined.csv"
        df.to_csv(output_path, index=False)

        print("Columnas finales:", df.columns.tolist())
        return output_path

    @task()
    def eda(path: str):
        df = pd.read_csv(path)

        # Se estandariza los nombres de las columnas a minusculas y se limpian espacios.
        df.columns = (
            df.columns.
            str.strip().
            str.lower()
        )

        # Podemos observar un '?'. Esto representa valores faltantes. Se lo va a reemplazar aquellos ? por nan
        df.replace("?", np.nan, inplace=True)

        # Se remueven los espacios en blancos de los valores
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].str.strip()

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

        print("Columnas disponibles:", df.columns.tolist())

        # Eliminar columnas innecesarias
        #df.drop(columns=["final weight", "educationnum", "capital gain", "capital loss"], inplace=True)
        df.drop(columns=["educationnum", "capital gain", "capital loss"], inplace=True)

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
    def preprocess(path: str) -> str:
        df = pd.read_csv(path)
        def label_encoder(dataframe, binary_col):
            labelencoder = LabelEncoder()
            dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
            return dataframe

        for col in ['income', 'gender',]:
            df = label_encoder(df, col)

        def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
            encoded_data = dataframe.copy() 
            
            for col in categorical_cols:
                dumm = pd.get_dummies(dataframe[col], prefix=col, dtype=int, drop_first=drop_first)
                del encoded_data[col]
                encoded_data = pd.concat([encoded_data, dumm], axis=1)
            
            return encoded_data

        df = one_hot_encoder(df, ['workclass', 'marital status', 'occupation', 'relationship', 'race','native country'])

        processed_path = "/tmp/data/adult_processed.csv"
        df.to_csv(processed_path, index=False)
        return processed_path        

    @task()
    def training(path: str):
        df = pd.read_csv(path)
        # Preparar X e y
        X = df.drop(columns='income')   # Features
        y = df['income']                # Target

        # Codificar variables categóricas
        X = pd.get_dummies(X, drop_first=True)

        # Convertir a categorías binarias
        #df["income"] = df["income"].apply(lambda x: 1 if ">50K" in x else 0)

        # Dividir en entrenamiento y test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Alinear columnas de test con entrenamiento
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

        # Instanciar el modelo Random Forest
        rf_model = RandomForestClassifier(
            max_depth=40,
            max_features='sqrt',
            min_samples_leaf=40,
            min_samples_split=5,
            n_estimators=100,
            random_state=42
        )

        # Entrenar el modelo
        rf_model.fit(X_train, y_train)

        # Predicciones en test
        y_pred = rf_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"\nAccuracy Test: {round(acc * 100, 2)}%")
        print(f"\nClassification Report:\n{report}")

        # Registrar en MLflow
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("Census_Income_Prediction")

       # with mlflow.start_run():
       #     mlflow.sklearn.log_model(rf_model, "random_forest_model")
       #     mlflow.log_metric("accuracy", acc)

        with mlflow.start_run() as run:
            # Log parámetros
            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", 100)

            # Log métricas
            mlflow.log_metric("accuracy", acc)

            # Log del modelo y registro
            mlflow.sklearn.log_model(
                rf_model,
                artifact_path="model",
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
    combined = load_and_combine_data_from_s3("datasets", "data/adult.csv", "data/adult.test.csv")
    clean = eda(combined)
    processed = preprocess(clean)
    training(processed)

dag = census_training_pipeline()