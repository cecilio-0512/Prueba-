####################################################Paqueterías a usar ##########################

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np
import streamlit as st



from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score,
    log_loss, confusion_matrix, classification_report, roc_curve
)

import matplotlib.pyplot as plt


# --- Leer archivo Excel ---
ecomm = pd.read_excel(
    "/Users/adrianmartinez/Documents/Prueba_volaris/Modelos_ML/E Commerce Dataset.xlsx",
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
    "/Users/adrianmartinez/Documents/Prueba_volaris/Modelos_ML/maxresdefault.png",   
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
    "/Users/adrianmartinez/Documents/Prueba_volaris/Modelos_ML/entrenamiento.png",   
    caption="Regresión lineal simple: línea de mejor ajuste minimizando la suma de residuos al cuadrado.",
    use_container_width=True

)

st.markdown("### 📊 Resultados")

# --- Split primero y luego imputación (sin fuga) ---

# 1) Separar X / y (excluir ID si existe)
cols_drop = [c for c in ["Churn", "CustomerID"] if c in ecomm.columns]
X = ecomm.drop(columns=cols_drop)
y = ecomm["Churn"].astype(int).values

# 2) Train/Test split (estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)

# 3) Detectar tipos por subconjunto (numéricas y categóricas)
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

# 4) Imputers ajustados SOLO con TRAIN
num_imputer = SimpleImputer(strategy="median") if len(num_cols) else None
cat_imputer = SimpleImputer(strategy="most_frequent") if len(cat_cols) else None

# 5) Transformar TRAIN/TEST con los imputers de TRAIN
X_train_imp = X_train.copy()
X_test_imp  = X_test.copy()

if num_imputer:
    X_train_imp[num_cols] = pd.DataFrame(
        num_imputer.fit_transform(X_train[num_cols]),
        columns=num_cols, index=X_train.index
    )
    X_test_imp[num_cols] = pd.DataFrame(
        num_imputer.transform(X_test[num_cols]),
        columns=num_cols, index=X_test.index
    )

if cat_imputer:
    X_train_imp[cat_cols] = pd.DataFrame(
        cat_imputer.fit_transform(X_train[cat_cols]),
        columns=cat_cols, index=X_train.index
    ).astype("category")
    X_test_imp[cat_cols] = pd.DataFrame(
        cat_imputer.transform(X_test[cat_cols]),
        columns=cat_cols, index=X_test.index
    ).astype("category")


# 6) Preprocesamiento (sobre datos imputados): num -> escalar, cat -> one-hot (k-1)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
    ],
    remainder="drop"
)

# Ajustar en TRAIN (imp) y transformar TRAIN/TEST (imp)
X_train_proc = preprocessor.fit_transform(X_train_imp)
X_test_proc  = preprocessor.transform(X_test_imp)

# Nombres de features finales
num_feature_names = list(num_cols)
ohe = preprocessor.named_transformers_["cat"]
cat_feature_names = list(ohe.get_feature_names_out(cat_cols)) if len(cat_cols) else []
feature_names = num_feature_names + cat_feature_names

#print("Dimensiones X_train_proc:", X_train_proc.shape)
#print("Dimensiones X_test_proc:", X_test_proc.shape)
#print("Número total de features finales:", len(feature_names))




# --- Modelo L1 + GridSearchCV

# 1) Definir el modelo base (clase desbalanceada → class_weight="balanced")
log_l1 = LogisticRegression(
    penalty="l1",
    solver="saga",
    max_iter=5000,
    class_weight="balanced",
    random_state=42
)

# 2) Malla de C (inverso de la regularización)
param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10, 100]
}

# 3) CV estratificada (reproducible)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 4) GridSearchCV sobre los datos ya preprocesados
grid_l1 = GridSearchCV(
    estimator=log_l1,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    refit=True
)

grid_l1.fit(X_train_proc, y_train)

# 5) Resultados principales
best_C = grid_l1.best_params_["C"]
best_cv_auc = grid_l1.best_score_
print(f"Mejor C (L1): {best_C}")
print(f"Mejor AUC promedio en CV: {best_cv_auc:.4f}")

# 6) Resumen ordenado de la malla (opcional)
cv_results_df = (
    pd.DataFrame(grid_l1.cv_results_)
    .loc[:, ["param_C", "mean_test_score", "std_test_score", "rank_test_score"]]
    .sort_values("rank_test_score")
    .rename(columns={
        "param_C": "C",
        "mean_test_score": "mean_CV_AUC",
        "std_test_score": "std_CV_AUC",
        "rank_test_score": "rank"
    })
    .reset_index(drop=True)
)

#cv_results_df.head(10)

# 7) Mejor modelo entrenado (ya refitteado sobre todo TRAIN)
best_model_l1 = grid_l1.best_estimator_


# Métricas principales en TEST (umbral = 0.5)
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, log_loss
)
import pandas as pd

# Probabilidades y clases
y_proba = best_model_l1.predict_proba(X_test_proc)[:, 1]
y_pred  = (y_proba >= 0.5).astype(int)

# Tabla de métricas
test_metrics = {
    "AUC": roc_auc_score(y_test, y_proba),
    "Accuracy": accuracy_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "LogLoss": log_loss(y_test, y_proba),
    "Best_C": grid_l1.best_params_["C"],
    "Best_CV_AUC": grid_l1.best_score_
}
#pd.DataFrame([test_metrics])






# Barrido de umbrales para mejorar precisión (clase 1) sin perder demasiado recall

# Asegúrarse de tener y_proba, y_test ya calculados
# y_proba = best_model_l1.predict_proba(X_test_proc)[:, 1]

baseline_thr = 0.50
y_pred_base = (y_proba >= baseline_thr).astype(int)
tn_b, fp_b, fn_b, tp_b = confusion_matrix(y_test, y_pred_base).ravel()
prec1_base = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
rec1_base  = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0

thresholds = np.linspace(0.01, 0.99, 99)
rows = []
for thr in thresholds:
    y_pred = (y_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    # Clase 1 (churn)
    precision_1 = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_1    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_1        = 2 * precision_1 * recall_1 / (precision_1 + recall_1) if (precision_1 + recall_1) > 0 else 0.0

    # Clase 0 (no churn)
    # Precisión de clase 0: TN / (TN + FN)  (entre lo predicho como 0, cuántos son realmente 0)
    precision_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    # Recall de clase 0 (TNR): TN / (TN + FP)
    recall_0    = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0        = 2 * precision_0 * recall_0 / (precision_0 + recall_0) if (precision_0 + recall_0) > 0 else 0.0

    accuracy     = (tp + tn) / (tp + tn + fp + fn)
    balanced_acc = 0.5 * (recall_1 + recall_0)

    rows.append({
        "threshold": thr,
        "precision_1": precision_1, "recall_1": recall_1, "f1_1": f1_1,
        "precision_0": precision_0, "recall_0": recall_0, "f1_0": f1_0,
        "accuracy": accuracy, "balanced_acc": balanced_acc,
        "TP": tp, "FP": fp, "TN": tn, "FN": fn
    })

thr_table = pd.DataFrame(rows)

# Regla: queremos subir precisión de clase 1 y permitir bajar recall_1 "pero no tanto"
# -> Mantener recall_1 >= (recall_1_base - 0.10) y maximizar precision_1
delta = 0.10
min_recall_1 = max(0.0, rec1_base - delta)

candidatos = thr_table[
    (thr_table["recall_1"] >= min_recall_1) &
    (thr_table["precision_1"] >= prec1_base)
].copy()

if not candidatos.empty:
    best_idx = candidatos["precision_1"].idxmax()
else:
    # Si no hay candidatos que mejoren precisión manteniendo el recall mínimo,
    # relajamos la mejora de precisión y solo imponemos el recall mínimo.
    candidatos = thr_table[thr_table["recall_1"] >= min_recall_1].copy()
    best_idx = candidatos["precision_1"].idxmax() if not candidatos.empty else thr_table.iloc[(thr_table["threshold"]-baseline_thr).abs().idxmin()].name

best_row = thr_table.loc[best_idx].copy()
best_thr = float(best_row["threshold"])

print(f"Umbral base = {baseline_thr:.2f} → precision_1 = {prec1_base:.3f}, recall_1 = {rec1_base:.3f}")
print(f"Umbral elegido = {best_thr:.2f} (max precision_1 con recall_1 ≥ base - {delta:.2f}, min_recall_1 = {min_recall_1:.3f})")

# Tabla comparativa (métricas para clase 0 y 1)
def row_at_thr(thr: float) -> pd.Series:
    return thr_table.iloc[(thr_table["threshold"] - thr).abs().idxmin()].copy()

comp = pd.DataFrame([
    {"criterio": "baseline_0.5", **row_at_thr(0.50).to_dict()},
    {"criterio": "opt_prec_con_rec_min", **best_row.to_dict()},
])[[
    "criterio","threshold",
    "precision_1","recall_1","f1_1",
    "precision_0","recall_0","f1_0",
    "accuracy","balanced_acc","TP","FP","TN","FN"
]]


st.dataframe(comp, use_container_width=True)

##########################################################################


# === Variables más influyentes del modelo (Logistic L1) ===
# Requiere: best_model_l1 (ya entrenado sobre X_train_proc),
#           preprocessor (ColumnTransformer ya fit),
#           num_cols, cat_cols.


# 1) Nombres de features finales
num_feature_names = list(num_cols)

# Como 'cat' es directamente un OneHotEncoder en el ColumnTransformer:
ohe = preprocessor.named_transformers_["cat"] if len(cat_cols) else None
cat_feature_names = list(ohe.get_feature_names_out(cat_cols)) if ohe is not None else []

feature_names = num_feature_names + cat_feature_names

# 2) Coeficientes del modelo y tabla ordenada
coefs = best_model_l1.coef_[0]
intercepto = float(best_model_l1.intercept_[0])

coef_df = (
    pd.DataFrame({"feature": feature_names, "coef": coefs})
    .assign(abs_coef=lambda d: d["coef"].abs(),
            odds_ratio=lambda d: np.exp(d["coef"]))
    .sort_values("abs_coef", ascending=False)
    .reset_index(drop=True)
)

st.subheader("🔎 Variables más influyentes (por coeficiente)")
st.dataframe(
    coef_df.head(20).style.format({"coef":"{:.3f}","abs_coef":"{:.3f}","odds_ratio":"{:.3f}"}),
    use_container_width=True
)



# 4) (opcional) descarga
st.download_button(
    label="📥 Descargar coeficientes (CSV)",
    data=coef_df.to_csv(index=False).encode("utf-8"),
    file_name="coeficientes_logit_l1.csv",
    mime="text/csv"
)


st.markdown("### 🔍 Análisis de resultados")


st.markdown("""
- **Usa 0.64** si el costo de intervenir a clientes fieles es alto, hay límite de capacidad o se busca mayor precisión de campañas.  
- **Usa 0.50** si la prioridad es maximizar la detección de churn y los costos por contacto son manejables.  
- Preferencias por laptops, accesorios y celulares: Los clientes que muestran preferencia por estas categorías presentan un menor riesgo de abandonar el consumo de productos de Volaris, lo que sugiere que estos segmentos podrían representar un grupo más leal o con mayor nivel de compromiso con la marca.
- Preferencia por la categoría “otros” y quejas registradas: Los clientes que consumen productos clasificados en la categoría “otros” o que han registrado alguna queja presentan un mayor riesgo de abandono, lo que indica posibles áreas de mejora en la atención, calidad o relevancia de los productos ofrecidos en esta categoría.""")




st.markdown("### 👥  Segmentación de clientes")

from kmodes.kprototypes import KPrototypes
from sklearn.impute import SimpleImputer
import pandas as pd

# ==========================
# 0. Variables categóricas
# ==========================
cat_force = [
    "PreferredLoginDevice",      # nominal
    "PreferredPaymentMode",      # nominal
    "Gender",                    # binaria
    "MaritalStatus",             # nominal
    "Complain",                  # binaria (0/1)
    "CityTier",                  # ordinal/categórica; aquí la tratamos como nominal
    "PreferedOrderCat",          # nominal
    "Churn"                      # binaria (0/1)
]

# Asegurar existencia antes de castear (por si hay variaciones en el archivo)
cat_present = [c for c in cat_force if c in ecomm.columns]
ecomm[cat_present] = ecomm[cat_present].apply(lambda s: s.astype("category"))

# ==========================
# 1. Imputación de NA (mediana para numéricas)
# ==========================
num_cols = ecomm.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Quitar CustomerID si existe
if "CustomerID" in num_cols:
    num_cols.remove("CustomerID")

# Quitar categóricas que se detectaron como numéricas
num_cols = [c for c in num_cols if c not in cat_present]

# Imputar con mediana
imputer = SimpleImputer(strategy="median")
ecomm[num_cols] = imputer.fit_transform(ecomm[num_cols])

# ==========================
# 2. Variables categóricas
# ==========================
cat_cols = [c for c in cat_force if c in ecomm.columns]

# Convertir categóricas a string (requerido por KPrototypes)
ecomm[cat_cols] = ecomm[cat_cols].astype(str)

# ==========================
# 3. Preparar datos para KPrototypes
# ==========================
X = ecomm[num_cols + cat_cols].to_numpy()

# Identificar índices de columnas categóricas en X
categorical_idx = list(range(len(num_cols), len(num_cols) + len(cat_cols)))

# ==========================
# 4. K-Prototypes
# ==========================
kproto = KPrototypes(n_clusters=4, init='Cao', random_state=42)
clusters = kproto.fit_predict(X, categorical=categorical_idx)

# Guardar asignación de cluster en el dataframe
ecomm["Cluster"] = clusters


# Centroides obtenidos
centroids = kproto.cluster_centroids_

# Crear DataFrame con nombres de columnas
centroids_df = pd.DataFrame(
    centroids,
    columns=num_cols + cat_cols
)

# Agregar el índice de cluster
centroids_df.index = [f"Cluster {i}" for i in range(len(centroids_df))]


st.markdown("""
Se identificaron **4 clusters principales** usando **K-Prototypes**, lo que nos permite agrupar clientes considerando tanto
variables numéricas como categóricas, los clusters se muestran a continuación.
""")

st.dataframe(centroids_df, use_container_width=True)


# Distribución de clientes
st.header("📌 Distribución de clientes por cluster")
dist = pd.DataFrame({
    "Cluster": ["Cluster 0", "Cluster 1", "Cluster 2", "Cluster 3"],
    "Clientes": [2407, 610, 822, 1791]
})
st.bar_chart(dist.set_index("Cluster"))

# Cluster 0
st.subheader("🟦 Cluster 0 – Clientes Jóvenes de Bajo Valor (Mayoría)")
st.markdown("""
- **Clientes:** 2,407 (grupo mayoritario)  
- **Tenure promedio:** ~7 meses  
- **Gasto anual promedio:** ~138  
- **Categoría preferida:** Phone / Mobile Phone  
- **Perfil:** Hombre, casado, paga con tarjeta de débito  
📌 **Insight:** Clientes nuevos y de bajo gasto por lo que se recomienda emplear  **estrategias de retención temprana** (cupones de bienvenida, promociones de recompra).
""")

# Cluster 1
st.subheader("🟩 Cluster 1 – Clientes Premium de Grocery")
st.markdown("""
- **Clientes:** 610 (grupo más pequeño)  
- **Tenure promedio:** ~20 meses  
- **Gasto anual promedio:** ~288 (más alto)  
- **Categoría preferida:** Grocery  
- **Perfil:** Hombre, casado, móvil, paga con débito  
📌 **Insight:** **Clientes estrella** Insentivar la retención de estos clientes con beneficios VIP o suscripciones.
""")

# Cluster 2
st.subheader("🟨 Cluster 2 – Clientes de Moda")
st.markdown("""
- **Clientes:** 822  
- **Tenure promedio:** ~13 meses  
- **Gasto anual promedio:** ~218  
- **Categoría preferida:** Fashion  
- **Perfil:** Hombre, casado, móvil, débito  
📌 **Insight:** Cluster con alta preferencia al área de moda, se puede potenciar el consumo de productos  con **campañas de temporada, lanzamientos exclusivos y personalización**.
""")

# Cluster 3
st.subheader("🟥 Cluster 3 – Clientes de Tecnología y Accesorios")
st.markdown("""
- **Clientes:** 1,791  
- **Tenure promedio:** ~9-10 meses  
- **Gasto anual promedio:** ~173  
- **Categoría preferida:** Laptop & Accessory  
- **Perfil:** Hombre, casado, móvil, débito  
📌 **Insight:** Segmento tech, se caracteriza por el consumo de teconología ideal para **up-selling y bundles de productos tecnológicos**.
""")

