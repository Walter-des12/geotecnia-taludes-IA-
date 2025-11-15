def completar_con_medias(df, entrada: dict):
    entrada_completa = entrada.copy()

    for col in entrada_completa:
        if entrada_completa[col] is None or entrada_completa[col] == "":
            entrada_completa[col] = df[col].mean()

    return entrada_completa

