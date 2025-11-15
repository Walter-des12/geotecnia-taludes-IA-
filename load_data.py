import pandas as pd

def cargar_dataset(ruta: str):
    df = pd.read_csv(ruta, encoding="latin1")
    df.columns = [
        'Peso_unitario_kN_m3',
        'Cohesion_kPa',
        'Angulo_de_friccion_interna',
        'Angulo_del_talud',
        'Altura_del_talud_m',
        'Relacion_de_presion_de_poros',
        'Factor_de_seguridad_FS'
    ]
    return df
