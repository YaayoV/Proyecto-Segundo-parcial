import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

datos = pd.read_csv("dataset_clientes_limpio.csv")

print("Valores nulos:")
print(datos.isnull().sum())
print("Duplicados:", datos.duplicated().sum())
print("Tipos de datos:")
print(datos.dtypes)

datos = datos.drop_duplicates()
datos = datos.dropna()

print("Estadisticas:")
print(datos.describe())
print("Medianas:")
print(datos.median(numeric_only=True))
correlacion = datos.corr(numeric_only=True)
print("Correlacion:")
print(correlacion)

datos["Edad"].plot(kind="hist", color="skyblue", edgecolor="black")
plt.title("Distribucion de edades")
plt.xlabel("Edad")
plt.ylabel("Frecuencia")
plt.show()

datos["Cliente_Frecuente"].value_counts().plot(kind="bar", color="orange")
plt.title("Clientes frecuentes")
plt.ylabel("Cantidad")
plt.show()

datos.plot.scatter(x="Compras_Realizadas", y="Total_Gastado", color="blue")
plt.title("Compras y total gastado")
plt.show()

plt.matshow(correlacion, cmap="coolwarm")
plt.colorbar()
plt.title("Correlacion")
plt.show()

X = datos[["Edad", "Compras_Realizadas", "Total_Gastado",
           "Dias_Entre_Compras", "Porcentaje_Compra_Online"]]

y = datos["Cliente_Frecuente"].map({"No": 0, "Si": 1})

X_entreno, X_prueba, y_entreno, y_prueba = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

modelo = DecisionTreeClassifier(max_depth=3, random_state=42)
modelo.fit(X_entreno, y_entreno)

predicciones = modelo.predict(X_prueba)

exactitud = accuracy_score(y_prueba, predicciones)
matriz = confusion_matrix(y_prueba, predicciones)
print("Exactitud:", round(exactitud * 100, 2), "%")
print("Matriz de confusion:")
print(matriz)

print("1. Se analizaron", len(datos), "clientes despues de la limpieza.")
print("2. La edad promedio fue de", round(datos["Edad"].mean(), 2), "años.")
print("3. La correlacion entre compras y gasto fue de",
      round(correlacion.loc["Compras_Realizadas", "Total_Gastado"], 2))
print("4. La correlacion entre compras y dias fue de",
      round(correlacion.loc["Compras_Realizadas", "Dias_Entre_Compras"], 2))
print("5. El arbol obtuvo una exactitud de", round(exactitud * 100, 2), "%.")