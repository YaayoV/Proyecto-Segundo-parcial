import pandas as pd
datos = pd.read_csv("dataset_clientes_proyecto.csv")

datos = datos.drop_duplicates()

datos["Ciudad"] = datos["Ciudad"].fillna(datos["Ciudad"].mode()[0])
datos["Categoria_Favorita"] = datos["Categoria_Favorita"].fillna(
    datos["Categoria_Favorita"].mode()[0]
)

datos["Dias_Entre_Compras"] = datos["Dias_Entre_Compras"].fillna(
    datos["Dias_Entre_Compras"].median()
)

datos.to_csv("dataset_clientes_limpio.csv", index=False)