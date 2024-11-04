
# Governance Policy for Heart Failure Prediction Project

## 1. Introduction

Este documento detalla las políticas de gobernanza aplicadas al proyecto de predicción de supervivencia en pacientes con insuficiencia cardíaca. Estas prácticas aseguran la calidad, seguridad, y trazabilidad del modelo, promoviendo la reproducibilidad, la adherencia a regulaciones y la ética en el uso de datos clínicos.

---

## 2. Code Standards and Practices

Para garantizar la legibilidad, consistencia y mantenibilidad del código, se aplican los siguientes estándares:

- **PEP8**: Todas las funciones, variables y clases deben cumplir con el estándar de estilo de código PEP8 para Python.
- **Nombres Consistentes**:
   - Funciones: `snake_case` (e.g., `load_data()`).
   - Clases: `PascalCase` (e.g., `RandomForestModel`).
   - Variables: `snake_case` para nombres descriptivos y claros.
- **Comentarios y Documentación**:
   - Cada función debe incluir un docstring que describa su propósito, parámetros y resultados.
   - Las clases deben tener docstrings que expliquen su funcionalidad general y los métodos principales.

---

## 3. Ethical and Risk Assessment

La construcción y el uso de este modelo pueden tener implicaciones éticas y de riesgo, especialmente en el contexto clínico. A continuación se detallan los lineamientos éticos y de evaluación de riesgos que se deben considerar.

### 3.1 Evaluación Ética
- **Impacto en Pacientes**: Este modelo se utiliza como soporte para la toma de decisiones clínicas, y sus predicciones deben interpretarse como complementarias a la evaluación médica y no como conclusiones definitivas.
- **Sesgo del Modelo**: Revisar y mitigar posibles sesgos que puedan surgir en las predicciones, especialmente relacionados con factores como edad y condiciones preexistentes.
- **Responsabilidad en la Interpretación**: El modelo no reemplaza la valoración clínica. Las decisiones médicas deben ser tomadas por profesionales de la salud.

### 3.2 Evaluación de Riesgos
- **Riesgo de Error en la Predicción**: Implementar métricas de rendimiento (como F1-score, precisión, y recall) que sean adecuadas para evaluar el balance entre precisión y sensibilidad, reduciendo la probabilidad de errores significativos.
- **Auditoría Regular**: El rendimiento del modelo debe auditarse regularmente para verificar que sigue siendo adecuado y consistente. Los datos de entrenamiento deben actualizarse cada seis meses si se incorpora en producción.

### 3.3 Políticas de Transparencia
- Documentar los experimentos y versiones del modelo en `MLflow`, asegurando la trazabilidad de cada cambio y justificación de ajustes en el pipeline.

---

## 4. Security and Compliance Policies

Dado que el modelo utiliza datos clínicos, se han implementado políticas para asegurar el cumplimiento de regulaciones y la protección de la privacidad de los datos.

### 4.1 Cumplimiento de GDPR/HIPAA
- **Minimización de Datos**: Solo se almacenan y procesan los datos estrictamente necesarios para el entrenamiento y evaluación del modelo.
- **Anonimización**: Los datos personales deben estar anonimizados. Cualquier dato identificable debe ser eliminado antes de ser utilizado en el pipeline.

### 4.2 Protección de Datos
- **Acceso Controlado**: El acceso a los datos está restringido solo a personal autorizado. Los datos se almacenan en servidores seguros con acceso controlado.
- **Cifrado**: Los datos sensibles se almacenan en repositorios cifrados cuando es aplicable.
- **Logs de Acceso**: Registrar los accesos a los datos y experimentos en MLflow, de manera que cualquier acceso inapropiado pueda ser detectado y auditado.

---

## 5. Version Control Policies

Para asegurar la reproducibilidad y trazabilidad de los modelos y datos utilizados, aplicamos las siguientes políticas de control de versiones.

### 5.1 Datos y Modelos
- **Data Version Control (DVC)**: Se utiliza DVC para versionar tanto los datos como los modelos generados en cada iteración, permitiendo revertir cambios y recuperar versiones anteriores.
- **Versiones del Modelo en MLflow**: Cada entrenamiento y ajuste se registra en MLflow, guardando los parámetros, métricas y artefactos del modelo (como la matriz de confusión), facilitando la comparación entre diferentes versiones.

### 5.2 Parámetros y Configuraciones
- **Archivos de Configuración**: Los parámetros e hiperparámetros del modelo se guardan en archivos `.yaml` versionados en Git, asegurando que cada ajuste quede registrado.

---

## 6. Quality Assurance (QA) and Testing

Para asegurar la calidad del modelo y de cada componente del pipeline, se han implementado pruebas unitarias y de integración.

### 6.1 Pruebas Unitarias
- **Pruebas de Componentes**: Cada función del pipeline tiene pruebas unitarias en `pytest` que validan su funcionamiento.
- **Pruebas de Datos**: Validación de rangos y formatos de los datos para asegurar consistencia antes del entrenamiento.
- **Pruebas de Rendimiento del Modelo**: Validación del rendimiento mínimo del modelo en métricas clave (precisión, F1-score) antes de despliegue.

### 6.2 Reporte de Pruebas
- Los resultados de las pruebas unitarias se documentan en `unit_testing_report.md`.

---

## 7. Reproducibility and Automation Policies

### 7.1 Pipeline Automatizado
- **`dvc.yaml`**: Se utiliza un archivo `dvc.yaml` que automatiza cada etapa del pipeline, desde la preparación de datos hasta la evaluación del modelo.

### 7.2 Documentación de Experimentos
- **MLflow**: Registro de experimentos y métricas en MLflow para asegurar la trazabilidad y transparencia de cada etapa.

### 7.3 Control de Versiones de Parámetros
- **Archivos `params/[model].yaml`**: Cada versión de los hiperparámetros se almacena y versiona en su correspondiente archivo de parámetros para el modelo, asegurando la consistencia de los experimentos. Estos archivos son versionados en Git, por lo que existe un historial de cambios.

---

**Conclusión:**
Estas políticas de gobernanza y estándares de seguridad, pruebas y reproducibilidad garantizan que el modelo de predicción de insuficiencia cardíaca cumpla con altos estándares de calidad, seguridad y transparencia, alineándose con las mejores prácticas de MLOps en la industria.
