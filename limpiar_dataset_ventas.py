## IMPORTAR PANDAS
import pandas as pd

datos = pd.read_csv("dataset_ventas_proyecto.csv")

print("Valores nulos:")
print(datos.isnull().sum())
print("Duplicados:", datos.duplicated().sum())
print("Tipos de datos:")
print(datos.dtypes)

datos = datos.drop_duplicates()

datos["Precio_Unitario"] = datos["Precio_Unitario"].fillna(
    datos["Precio_Unitario"].median())

datos["Promocion"] = datos["Promocion"].fillna(
    datos["Promocion"].mode()[0])
datos["Sucursal"] = datos["Sucursal"].fillna(
    datos["Sucursal"].mode()[0])

datos.to_csv("dataset_ventas_limpio.csv", index=False)

print("Nulos despues de limpiar:", datos.isnull().sum().sum())
print("Duplicados despues de limpiar:", datos.duplicated().sum())
print("Filas finales:", len(datos))
print("Se creo dataset_ventas_limpio.csv")