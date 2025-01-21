# MNA MLOps - Equipo 40

Repositorio para el proyecto de la materia de Operaciones de Aprendizaje Automático (MLOps).

## Integrantes del equipo
- Eduardo Saborío Sánchez - A01794374
- Fabiola Sosa Hernandez - A01240145
- Luis Daniel Ortega Muñoz - A01795197
- Jose Santiago Rueda Antonio - A01794118
- Julio Cesar Ruiz Marks - A01794742
- Leonardo Segura - A01176833


## Ejecutar MLFlow

A continuación se describen los pasos para ejecutar MLFlow en un ambiente local. Como alternativa, se puede utilizar nuestro servidor de MLFlow (http://13.93.214.226:5000) para registro de experimentos. Utilizar este servidor facilita la colaboración, puesto que los resultados son compartidos entre los miembros del equipo.


Start services `docker-compose --env-file config.env up -d --build`

Go to `localhost:9001`

Get MINIO Access Key and save into `config.env/MINIO_ACCESS_KEY`

Stop services `docker-compose down`

Start again `docker-compose --env-file config.env up -d --build`


______________________________________________

`docker-compose -f --env-file config.env up -d --build`

NOTA: docker-compose solo inicializa el servidor de MLFlow. Para iniciar el servidor de predicciones, las instrucciones se encuentran más abajo.

## Ejecutar pruebas unitarias

Las pruebas unitarias para el proyecto se encuentran en la carpeta de `mlops/tests`. Estas pruebas se enfocan en validar la implementación de nuestras librerías de Python; por ejemplo, validan nuestro _pipeline_ de preprocesamiento.

Para pruebas relacionadas a la experimentación con diferentes modelos, se recomienda a los científicos de datos incluir sus scripts y notebooks con estas pruebas en el directorio `tests`.

Para ejecutar las pruebas unitarias, es necesario contar con `pytest` instalado en el ambiente de desarrollo. Para instalar `pytest`, se puede ejecutar el siguiente comando:

```bash
pip install pytest
```

Posteriormente, se puede ejecutar el siguiente comando desde la raíz del repositorio para correr las pruebas unitarias:

```bash
python -m pytest mlops
```

Esto automáticamente ejecutará todas las pruebas unitarias que se encuentran en la carpeta de `mlops/tests` y mostrará el resultado en la terminal. Ej.

```bash
(.venv) PS C:\Users\danie\repos\tec\itesm-mna-mlops-eq40> python -m pytest mlops
================================================= test session starts ==================================================
platform win32 -- Python 3.10.4, pytest-8.3.3, pluggy-1.5.0
rootdir: C:\Users\danie\repos\tec\itesm-mna-mlops-eq40\mlops
plugins: anyio-4.6.0, hydra-core-1.3.2
collected 6 items                                                                                                       

mlops\tests\test_preprocess.py ......                                                                             [100%]

================================================== 6 passed in 1.16s ===================================================
```

Si se desea exportar el reporte de pruebas unitarias, es necesario instalar `pytest-md`:
    
```bash
pip install pytest-md
```

Una vez instalado, se puede agregar el argumento `--md` al comando de `pytest`:

```bash
python -m pytest mlops --md=unit_testing_report.md
```

## Reproducibilidad

Todos los pasos de nuestro pipeline de ML son determinísticos, lo cual garantiza la reproducibilidad de los resultados. El notebook [5. Reproducing Experiments](notebooks/5.Reproducing_experiments.ipynb) muestra cómo podemos importar los módulos de Python que componen nuestro pipeline para reproducir los resultados de nuestros experimentos de forma consistente. Esto nos ayuda a asegurar la confiabilidad en nuestros modelos y en la toma de decisiones basadas en ellos.


## Iniciar el servidor de predicciones

El servidor de predicciones puede ejecutarse como un contenedor de Docker. Para ello, el primer paso es crear la imagen de Docker. Para ello, se puede ejecutar el siguiente comando:

```bash
docker build . -t mlops-pred-server 
```

Posteriormente, se puede ejecutar el contenedor de Docker con el siguiente comando:

```bash
docker run -p 8000:8000 mlops-pred-server
```

NOTA: Este servidor está configurado para utilizar un modelo de nuestra instancia de MLFlow: http://13.93.214.226:5000/#/experiments/368036918666935739/runs/bc8cf7556c204f698695eef704dfaf8b. Si se desea cambiar el modelo a utilizar, se puede modificar el archivo `prediction_server/main.py` y cambiar el valor de la variable `RUN_ID` por el ID de la corrida de MLFlow que contiene el modelo deseado.


## Probar con un ejemplo

Hemos creado también un cliente de ejemplo para validar el servidor de predicciones. En este caso, es una aplicación web que simula un sistema de gestión de pacientes de enfermedades cardíacas. Para ejecutar el cliente de ejemplo, se puede ejecutar el siguiente comando:

```bash
uvicorn sample_client.patient_care.main:app --reload --port 80
```

Posteriormente, se puede acceder a la aplicación web en la dirección `http://localhost/` y empezar a llenar la información de los pacientes para ver cómo el modelo predice si el paciente es propenso a morir o a sobrevivir.

NOTA: Esta app por defecto intentará conectarse a nuestra instancia en la nube del servidor de predicciones, http://13.93.214.226:8000/. Si se desea cambiar la dirección del servidor de predicciones, se puede modificar el archivo `sample_client/patient_care/main.py` y cambiar el valor de la variable `url` (dentro de la función `get_death_prediction`) por la dirección deseada.