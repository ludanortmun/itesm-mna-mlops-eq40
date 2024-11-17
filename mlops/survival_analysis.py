import pandas as pd
from lifelines import KaplanMeierFitter, CoxPHFitter
import matplotlib.pyplot as plt

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
    cph = CoxPHFitter()
    cph.fit(data, duration_col=time_col, event_col=event_col)
    cph.print_summary()
    return cph

def evaluate_survival_model(data, group_col, event_col, time_col):
    """
    Evalúa un modelo de supervivencia utilizando Kaplan-Meier y regresión de Cox.

    :param data: DataFrame con los datos.
    :param group_col: Nombre de la columna de agrupación (e.g., cohort).
    :param event_col: Nombre de la columna de eventos (e.g., DEATH_EVENT).
    :param time_col: Nombre de la columna de tiempo.
    :return: Modelo de regresión de Cox ajustado.
    """
    create_kaplan_meier_chart(data, group_col, event_col, time_col)
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

