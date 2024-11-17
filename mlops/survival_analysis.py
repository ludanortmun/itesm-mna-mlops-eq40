import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt


def prepare_data_for_cox(x, y):
    """
    Combina las características (X) y las etiquetas (y) en un único DataFrame
    con las columnas necesarias para la regresión de Cox.
    """
    # Convertir x y y en DataFrames si no lo son
    if not isinstance(x, pd.DataFrame):
        x = pd.DataFrame(x, columns=[f"feature_{i}" for i in range(x.shape[1])])
    if not isinstance(y, pd.DataFrame):
        y = pd.DataFrame(y, columns=["death_event"])

    # Combinar X (características) y y (etiquetas) en un único DataFrame
    data = pd.concat([x, y], axis=1)

    # Validar la existencia de las columnas necesarias
    required_cols = ["time", "death_event"]
    for col in required_cols:
        if col not in data.columns:
            raise ValueError(f"La columna requerida '{col}' no está presente en el dataset.")

    # Eliminar filas con valores NaN en las columnas necesarias
    if data[required_cols].isnull().any().any():
        print("Se encontraron valores NaN. Eliminando filas con valores faltantes...")
        data = data.dropna(subset=required_cols)

    return data



def create_kaplan_meier_chart(data, group_col, event_col, time_col):
    kmf = KaplanMeierFitter()
    for group, subset in data.groupby(group_col):
        kmf.fit(subset[time_col], event_observed=subset[event_col], label=str(group))
        kmf.plot_survival_function()
    plt.title("Kaplan-Meier Survival Curve")
    plt.xlabel("Time")
    plt.ylabel("Survival Probability")
    plt.show()

def perform_cox_regression(data, event_col, time_col):
    if time_col not in data.columns or data[time_col].isnull().any():
        raise ValueError(f"La columna {time_col} no contiene datos válidos.")
    cph = CoxPHFitter()
    cph.fit(data, duration_col=time_col, event_col=event_col)
    return cph





def evaluate_survival_model(data, group_col, event_col, time_col="time"):
    print("Generando gráfico Kaplan-Meier...")
    create_kaplan_meier_chart(data, group_col, event_col, time_col)

    print("Realizando regresión de Cox...")
    cph = perform_cox_regression(data, event_col, time_col)

    return cph



def create_cohorts_from_cox(data, cox_model, time_col, threshold=0.5):
    """
    Genera cohorts basados en los riesgos predichos por un modelo de regresión de Cox.

    :param data: DataFrame con las variables del modelo.
    :param cox_model: Modelo de regresión de Cox entrenado.
    :param time_col: Nombre de la columna de tiempo en el DataFrame.
    :param threshold: Umbral para clasificar en "High Risk" o "Low Risk".
    :return: DataFrame con una nueva columna 'cohort' indicando el riesgo.
    """
    data['predicted_risk'] = cox_model.predict_partial_hazard(data)
    data['cohort'] = data['predicted_risk'].apply(
        lambda x: 'High Risk' if x > threshold else 'Low Risk'
    )
    return data

def create_cohorts(data, cohort_criteria):
    """
    Agrupa los datos en cohorts basados en criterios definidos.

    :param data: DataFrame original.
    :param cohort_criteria: Diccionario con el nombre del cohort y la condición.
    :return: DataFrame con una nueva columna 'cohort' indicando el grupo.
    """
    data['cohort'] = 'Others'  # Cohort default
    for cohort_name, condition in cohort_criteria.items():
        data.loc[condition(data), 'cohort'] = cohort_name
    return data

