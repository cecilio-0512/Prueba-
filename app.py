####################################################Paqueterías a usar ##########################

import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt


# --- Leer archivo Excel ---
ecomm = pd.read_excel(
    "E Commerce Dataset.xlsx",
    sheet_name="E Comm"
)


#Transformar a variables de tipo categórico variables que hayan sido guardadas en otro formato
cat_force = [
    "PreferredLoginDevice",      # nominal
    "PreferredPaymentMode",      # nominal
    "Gender",                    # binaria
    "MaritalStatus",             # nominal
    "Complain",                  # binaria (0/1)
    "CityTier",                  # ordinal/categórica; aquí la tratamos como nominal para no asumir linealidad
    "PreferedOrderCat"           # nominal (estaba implícita)
]

# Asegurar existencia antes de castear (por si hay variaciones en el archivo)
cat_present = [c for c in cat_force if c in ecomm.columns]
ecomm[cat_present] = ecomm[cat_present].apply(lambda s: s.astype("category"))


# --- Diccionario  -> descripción ---
desc_es = {
    "CustomerID": "ID único del cliente",
    "Churn": "Indicador de fuga (1 = se fue, 0 = se quedó)",
    "Tenure": "Antigüedad del cliente (p. ej., meses)",
    "PreferredLoginDevice": "Dispositivo preferido para iniciar sesión",
    "CityTier": "Nivel de la ciudad (1 = grande, etc.)",
    "WarehouseToHome": "Distancia almacén-hogar (p. ej., km)",
    "PreferredPaymentMode": "Método de pago preferido", 
    "Gender": "Género del cliente",
    "HourSpendOnApp": "Horas de uso de la app",
    "NumberOfDeviceRegistered": "Número de dispositivos registrados",
    "PreferedOrderCat": "Categoría de productos más ordenada",
    "SatisfactionScore": "Calificación de satisfacción (escala corta)",
    "MaritalStatus": "Estado civil",
    "NumberOfAddress": "Número de direcciones registradas",
    "Complain": "¿Ha hecho queja? (1 = Sí, 0 = No)",
    "OrderAmountHikeFromlastYear": "Incremento % del gasto vs. año previo",
    "CouponUsed": "Cantidad de cupones usados",
    "OrderCount": "Número de órdenes realizadas",
    "DaySinceLastOrder": "Días desde la última orden",
    "CashbackAmount": "Monto de cashback recibido",
}

# --- Crear DataFrame con la info ---
traduccion_variables = (
    pd.DataFrame({"variable": ecomm.columns})
    .assign(dtype=lambda df: df["variable"].map(dict(zip(ecomm.columns, ecomm.dtypes.astype(str)))))
    .assign(descripcion=lambda df: df["variable"].map(desc_es).fillna(""))
    .loc[:, ["variable", "dtype", "descripcion"]]
    .sort_values("variable")
    .reset_index(drop=True)
)

st.title("📊 Técnicas de Machine Learning para la Prevención del Retiro de Clientes")

st.markdown("""
Este reporte presenta la aplicación de **técnicas de Machine Learning** para el análisis del comportamiento de los clientes de **MarMen**, 
utilizando la información disponible en la base de datos de *E-Commerce*.  
El objetivo principal es **identificar y clasificar a los clientes propensos a retirarse (variable *churn*)**, 
para diseñar estrategias de retención e incentivos que reduzcan la fuga.  

### 🔎 Flujo del Análisis
1. **Exploración inicial de datos** y construcción de un **diccionario de variables**, con el fin de comprender la estructura y relevancia de la información disponible.  
2. **Entrenamiento de un modelo de clasificación** (Regresión Logística con regularización L1), orientado a identificar las variables con mayor impacto en la predicción de la fuga.  
3. **Evaluación de distintos umbrales de decisión**, comparando métricas como:  
   - *Recall*  
   - *Precisión*  
   - *Balanced Accuracy*  
   - *Accuracy*  
   para analizar el desempeño en diferentes escenarios de negocio.  
4. **Conclusiones y recomendaciones estratégicas**, basadas en los resultados obtenidos y en los objetivos específicos del cliente.  

---
""")

# --- Mostrar tabla ---
st.subheader("📂 Traducción variables ")
st.dataframe(traduccion_variables, use_container_width=True)

# --- (Opcional) botón de descarga ---
csv_dict = traduccion_variables.to_csv(index=False).encode("utf-8")
st.download_button(
    label="📥 Descargar diccionario (CSV)",
    data=csv_dict,
    file_name="diccionario_variables.csv",
    mime="text/csv"
)

# --- Distribución de la variable objetivo (Churn) ---


# Calcular proporciones
churn_counts = ecomm["Churn"].value_counts()
churn_labels = ["Se queda (0)", "Se retira (1)"]

# Gráfica de pastel con fondo transparente y texto blanco
fig, ax = plt.subplots(facecolor="none")  # fondo transparente
wedges, texts, autotexts = ax.pie(
    churn_counts,
    labels=churn_labels,
    autopct="%1.1f%%",
    startangle=90,
    counterclock=False,
    textprops={"color": "white"}  # texto blanco
)

st.subheader("📂 Vista previa de la base de datos E-Commerce")
st.dataframe(ecomm.head(10)) 


# Ajustar también el título
ax.set_title("Distribución respuesta de variable churn", color="white")

st.pyplot(fig, transparent=True)


st.markdown("""
Se observa que la distribución de la respuesta de la variable *Churn* está desbalanceada: 83% de clientes se quedan y 17% se retiran. Este desbalance se toma en cuenta al momento de analalizar el resultado de las predicciones """)



# --- Tabla resumen de valores nulos ---
n_obs = len(ecomm)
resumen_inicial = (
    pd.DataFrame({
        "variable": ecomm.columns,
        "dtype": ecomm.dtypes.astype(str).values,
        "n_missing": ecomm.isnull().sum().values
    })
    .assign(pct_missing=lambda df: (df["n_missing"] / n_obs * 100).round(2))
    .sort_values(["n_missing", "variable"], ascending=[False, True])
    .reset_index(drop=True)
)


st.subheader("📋 Resumen inicial de la base de datos")
st.dataframe(resumen_inicial, use_container_width=True)


st.markdown("""Existen variables 7 con presencia de valores NA (aproximadamente 5% de valores faltantes del total de observaciones), dado este porcentaje relativamente menor de valores NA se procede a hacer la imputación por media y moda dependiendo del tipo de variable durante el proceso del entrenamiento del modelo. """)



st.markdown("### 📋 Modelo estadístico: Regresión Logística")

# --- Explicación formal ---
st.markdown("""
En todo modelo de regresión, el objetivo es **optimizar una función objetivo** 
para encontrar los parámetros que mejor explican los datos.  

- En la **regresión lineal**, la función objetivo típica es minimizar la **suma de los residuos al cuadrado**.""") 
st.latex(r"\sum_{i=1}^{n} \left(Y_i - (\beta_0 + \beta_1 X_i)\right)^2")


# Ilustración: regresión lineal simple
st.image(
    "maxresdefault.png",   
    caption="Regresión lineal simple: línea de mejor ajuste minimizando la suma de residuos al cuadrado.",
    use_container_width=True

)
             
st.markdown("""- En la **regresión logística**, como la variable de interés es **binaria (0/1)**, 
no tiene sentido usar residuos al cuadrado. En su lugar se utiliza la **log-verosimilitud**, 
que mide qué tan probable es que el modelo genere los datos observados.
""")

st.markdown("La forma del modelo logístico es:")

st.latex(r"""
P(Y=1|X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_p X_p)}}
""")

st.markdown("""

En este proyecto utilizamos la **regresión logística con regularización L1 (Lasso)**.  
En este caso, la función objetivo incorpora una penalización adicional para controlar la complejidad del modelo:

""")

st.latex(r"""
\mathcal{L}_{L1}(\beta) = - \sum_{i=1}^n \Big[ y_i \log(p_i) + (1-y_i)\log(1-p_i) \Big] 
\;+\; \lambda \sum_{j=1}^p |\beta_j|
""")

st.markdown("""
donde """) 

st.latex(r""" \lambda """)  

st.markdown(""" Es conocido como un hiperparámetro que controla la fuerza de la penalización. El efecto de esta penalización es que algunos coeficientes del modelo se reduzcan a **cero**, 
lo cual equivale a una **selección automática de variables**.

Así, la regresión logística regularizada no solo estima probabilidades, 
sino que también ayuda a identificar las características más relevantes en la predicción del churn. """)
            
            

st.markdown("### ⚙️ Entrenamiento del modelo")

st.image(
    "entrenamiento.png",   
    caption="Regresión lineal simple: línea de mejor ajuste minimizando la suma de residuos al cuadrado.",
    use_container_width=True

)

st.markdown("### 📊 Resultados")



