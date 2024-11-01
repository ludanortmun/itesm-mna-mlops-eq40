
# Governance Policy

Este documento describe las políticas de gobernanza y prácticas de buenas prácticas aplicadas en el proyecto de Machine Learning para el análisis de insuficiencia cardíaca, abarcando Model Governance, estándares de código y consideraciones éticas.

## 1. Model Governance (Gobernanza del Modelo)

En este proyecto, implementamos **MLflow** como herramienta de seguimiento y trazabilidad de modelos de Machine Learning. Esto permite centralizar el control de versiones, parámetros, métricas y artefactos, garantizando la reproducibilidad y calidad en cada etapa del ciclo de vida del modelo.

### Configuración de MLflow
La configuración de **MLflow** está diseñada para registrar los detalles de cada experimento, de modo que sea posible rastrear las decisiones y ajustes realizados en el modelo. Estos son los aspectos específicos configurados en MLflow:

1. **Tracking URI**: Configuramos el Tracking URI para centralizar el seguimiento de experimentos y facilitar el almacenamiento de parámetros y métricas en un solo lugar.
2. **Registro de Parámetros**: Documentamos cada parámetro utilizado en los modelos, incluyendo configuraciones de hiperparámetros para modelos como la Regresión Logística y SVM.
3. **Métricas de Rendimiento**: Se registra la precisión, precisión (precision) y recall del modelo para cada experimento, proporcionando un análisis detallado de su rendimiento.
4. **Artefactos**: Guardamos la matriz de confusión generada en cada ejecución como un artefacto. Esto facilita la interpretación visual del rendimiento y permite detectar posibles sesgos en las predicciones.

#### Ejemplo de Parámetros y Métricas
- **Tipo de modelo**: Regresión Logística, SVM.
- **Parámetros de hiperparámetros**: `C`, `solver` para regresión logística y `kernel` para SVM.
- **Métricas**: Accuracy (precisión), precision, recall.

La implementación de MLflow permite mantener una trazabilidad exhaustiva de cada experimento, asegurando que todas las versiones y cambios realizados en el modelo estén registrados.

---

## 2. Estándares de Código

Para asegurar la calidad, mantenibilidad y claridad en el desarrollo del código del proyecto, se siguen los siguientes estándares:

- **Convención de nombres descriptivos**: Cada variable y función tiene un nombre claro y descriptivo que indica su propósito.
- **Modularidad**: El pipeline de entrenamiento y evaluación se organiza en módulos, cada uno responsable de una fase específica (preprocesamiento, entrenamiento, evaluación). Esto facilita el mantenimiento y permite realizar cambios de manera aislada sin afectar el resto del código.
- **Documentación y comentarios**: Cada función cuenta con un docstring que describe su funcionamiento, parámetros y salida esperada. Los pasos clave del código también tienen comentarios para guiar a otros desarrolladores o usuarios.

---

## 3. Verificación Ética y de Riesgo

Considerando la naturaleza sensible de los datos clínicos, implementamos prácticas de gobernanza ética para reducir posibles sesgos y asegurar la equidad del modelo.

### Evaluación de Sesgo
Para reducir el riesgo de sesgo en el modelo, se realizó un análisis de los datos de entrada para identificar si existían patrones o desbalances significativos en las variables demográficas, como `age` y `sex`. Estos resultados fueron observados en relación con la variable objetivo **DEATH_EVENT**.

- **Resultados**: Observamos que algunas variables pueden tener correlaciones con el evento de muerte, como la fracción de eyección y el nivel de creatinina sérica.
- **Acciones**: Basándonos en estos resultados, se implementó un proceso de análisis de la matriz de confusión para evaluar el rendimiento del modelo en diferentes subgrupos, asegurando una consistencia en su precisión y minimizando el sesgo.

### Consideraciones Éticas
El proyecto implementa una política de transparencia en el análisis y el preprocesamiento de los datos. Cualquier transformación aplicada a los datos es documentada y verificada para asegurar que no se introduzcan sesgos innecesarios que puedan afectar la imparcialidad del modelo.

---

## 4. Conclusión

Las prácticas de gobernanza implementadas en este proyecto, como el uso de MLflow para Model Governance, los estándares de código, y la verificación ética, aseguran que el modelo sea trazable, reproducible y justo. Estas políticas contribuyen a que el proyecto cumpla con los estándares de calidad y regulaciones aplicables, mejorando la confiabilidad del modelo en contextos clínicos o de salud.

---

**Última actualización:** [Fecha actual]
