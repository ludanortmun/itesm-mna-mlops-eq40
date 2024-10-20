# MNA MLOps - Equipo 40

Repositorio para el proyecto de la materia de Operaciones de Aprendizaje Automático (MLOps).

## Integrantes del equipo
- Eduardo Saborío Sánchez - A01794374
- Fabiola Sosa Hernandez - A01240145
- Luis Daniel Ortega Muñoz - A01795197
- Jose Santiago Rueda Antonio - A01794118
- Julio Cesar Ruiz Marks A01794742
- Leonardo Segura - A01176833


## Ejecutar MLFlow

Start services `docker-compose --env-file config.env up -d --build`

Go to `localhost:9001`

Get MINIO Access Key and save into `config.env/MINIO_ACCESS_KEY`

Stop services `docker-compose down`

Start again `docker-compose --env-file config.env up -d --build`


______________________________________________

`docker-compose -f --env-file config.env up -d --build`






