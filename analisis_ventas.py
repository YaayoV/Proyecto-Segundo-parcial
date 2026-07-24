import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

datos = pd.read_csv("dataset_ventas_limpio.csv")
datos["Fecha"] = pd.to_datetime(datos["Fecha"])

print("Valores nulos:")
print(datos.isnull().sum())
print("Duplicados:", datos.duplicated().sum())
print("Tipos de datos:")
print(datos.dtypes)

datos = datos.drop_duplicates()
datos = datos.dropna()

numericas = ["Precio_Unitario", "Cantidad_Vendida", "Venta_Total"]
print("Estadisticas:")
print(datos[numericas].describe())
print("Medianas:")
print(datos[numericas].median())
correlacion = datos[numericas].corr()
print("Correlacion:")
print(correlacion)

datos["Venta_Total"].plot(kind="hist", color="skyblue", edgecolor="black")
plt.title("Distribucion de las ventas")
plt.xlabel("Venta total")
plt.ylabel("Frecuencia")
plt.show()

datos.groupby("Categoria")["Venta_Total"].sum().plot(kind="bar", color="orange")
plt.title("Ventas por categoria")
plt.xlabel("Categoria")
plt.ylabel("Venta total")
plt.show()

datos.plot.scatter(x="Cantidad_Vendida", y="Venta_Total", color="blue")
plt.title("Cantidad vendida y venta total")
plt.show()

ventas_fecha = datos.groupby("Fecha")["Venta_Total"].sum()
ventas_fecha.plot(kind="line", color="green")
plt.title("Ventas a traves del tiempo")
plt.xlabel("Fecha")
plt.ylabel("Venta total")
plt.show()

plt.matshow(correlacion, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(3), numericas, rotation=45)
plt.yticks(range(3), numericas)
plt.title("Correlacion")
plt.show()

X = datos[["Precio_Unitario", "Cantidad_Vendida"]]

y = datos["Venta_Total"]

X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(
    X, y, test_size=0.20, random_state=42)

modelo = LinearRegression()
modelo.fit(X_entreno, y_entreno)

predicciones = modelo.predict(X_prueba)

mae = mean_absolute_error(y_prueba, predicciones)
r2 = r2_score(y_prueba, predicciones)
print("Error absoluto medio:", round(mae, 2))
print("R cuadrada:", round(r2, 2))

ventas_mes = datos.groupby(datos["Fecha"].dt.month)["Venta_Total"].sum()
if ventas_mes.iloc[-1] > ventas_mes.iloc[0]:
    tendencia = "aumento"
else:
    tendencia = "disminucion"

print("1. Se analizaron", len(datos), "ventas despues de la limpieza.")
print("2. La venta promedio fue de", round(datos["Venta_Total"].mean(), 2))
print("3. La correlacion entre precio y venta total fue de",
      round(correlacion.loc["Precio_Unitario", "Venta_Total"], 2))
print("4. Del primer al ultimo mes se observo una", tendencia, "en las ventas.")
print("5. El modelo obtuvo un MAE de", round(mae, 2),
      "y una R cuadrada de", round(r2, 2))
