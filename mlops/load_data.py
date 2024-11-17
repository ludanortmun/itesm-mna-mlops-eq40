import pandas as pd
import sys

def load_data(filepath):
    data = pd.read_csv(filepath)
    # Convertir nombres de columnas a minúsculas
    data.columns = [col.lower() for col in data.columns]
    return data

if __name__ == '__main__':
    data_path = sys.argv[1]
    output_file = sys.argv[2]
    
    # Cargar datos
    data = load_data(data_path)
    
    # Guardar los datos procesados
    data.to_csv(output_file, index=False)

