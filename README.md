# 🧠 Operaciones de Máquina I - CEIA

Este proyecto corresponde al trabajo final de la materia **Operaciones de Máquina I** del posgrado en Inteligencia Artificial (CEIA - UBA).  
El objetivo fue construir un pipeline MLOps completo para predecir ingresos utilizando el dataset **Census Income**.

---

##  Descripción del Proyecto

Se diseñó una solución de aprendizaje automático que incluye:

- Descarga y unión de datos desde MinIO.
- Análisis exploratorio de datos (EDA).
- Preprocesamiento y limpieza.
- Entrenamiento de modelos.
- Registro y versionado de modelos en MLflow.
- Exposición del modelo como API REST con FastAPI.
- Automatización de todo el flujo mediante Apache Airflow.

---

##  Herramientas y Tecnologías

- **Airflow**: Orquestación de los flujos de trabajo.
- **MLflow**: Tracking, registro y gestión de modelos.
- **MinIO (S3)**: Almacenamiento de datasets y artefactos.
- **PostgreSQL**: Base de datos para MLflow.
- **Docker Compose**: Infraestructura contenedorizada.
- **FastAPI**: API REST para predicciones.
- **Scikit-learn**: Modelado y evaluación.

---

##  Flujo de Trabajo

1. Airflow descarga `adult.csv` y `adult.test.csv` desde MinIO.
2. Se realiza la unión, análisis y preprocesamiento de los datos.
3. Se entrena un modelo Random Forest.
4. El modelo es registrado en MLflow con alias `Champion`.
5. Se expone la predicción mediante FastAPI consumiendo el modelo desde MLflow.

---

##  Estructura del Repositorio

```
.
├── airflow/
│   ├── dags/
│   └── config/
├── dockerfiles/
│   ├── airflow/
│   ├── mlflow/
│   └── fastapi/
├── .env
├── docker-compose.yml
├── README.md
└── requirements.txt
```

---

##  Cómo levantar el proyecto

```bash
git clone https://github.com/mbarquienero/operaciones_de_maquina_I.git
cd operaciones_de_maquina_I

# Abrir docker desktop

# Levantar todos los servicios
docker compose --profile all up --build
```

---

## Dataset

Se utilizó el [Census Income Dataset](https://www.kaggle.com/datasets/tawfikelmetwally/census-income-dataset/data), que contiene registros demográficos para predecir si una persona gana más o menos de 50K al año.

---

##  Integrantes del equipo

| Nombre Completo         |  Email                      |
|------------------------ |-----------------------------|
| Mauro Barquinero        | [mauro.barquinero@gmail.com]|
| Mariano Campos          | [lmarianocampos@gmail.com]  |
| Juan Cruz Ojeda         | [ojedajuancz@gmail.com]     |
| Jose Luis Perez Galindo | [joseperez190498@gmail.com] |

---

## Recursos útiles

- [Documentación oficial de MLflow](https://mlflow.org/docs/latest/index.html)
- [Apache Airflow](https://airflow.apache.org/docs/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Scikit-learn](https://scikit-learn.org/)