import sys
import pandas as pd
from sklearn.model_selection import train_test_split

def load(path):
    # Cargar el archivo CSV
    heart_failure = pd.read_csv(path)

    # Verificar si la columna 'death_event' existe
    if 'death_event' not in heart_failure.columns:
        raise ValueError("La columna 'death_event' no se encuentra en el archivo.")

    # Separar características (X) y objetivo (y)
    x = heart_failure.drop(columns='death_event')
    y = heart_failure['death_event']
    return x, y

def split(x, y):
    # Dividir los datos en conjuntos de entrenamiento y prueba
    return train_test_split(x, y, test_size=0.2, random_state=42)

def load_and_split(path):
    # Cargar y dividir los datos
    x, y = load(path)
    return split(x, y)

if __name__ == '__main__':
    # Leer argumentos de línea de comandos
    data_path = sys.argv[1]
    output_train_features = sys.argv[2]
    output_test_features = sys.argv[3]
    output_train_target = sys.argv[4]
    output_test_target = sys.argv[5]

    # Cargar y dividir los datos
    x_train, x_test, y_train, y_test = load_and_split(data_path)

    # Guardar los conjuntos de datos resultantes
    x_train.to_csv(output_train_features, index=False)
    x_test.to_csv(output_test_features, index=False)
    y_train.to_csv(output_train_target, index=False)
    y_test.to_csv(output_test_target, index=False)
