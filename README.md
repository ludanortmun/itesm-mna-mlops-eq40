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