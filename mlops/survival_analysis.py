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
    create_kaplan_meier_chart(data, group_col, event_col, time_col)
    cph = perform_cox_regression(data, event_col, time_col)
    return cph
