import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                              confusion_matrix, roc_curve, roc_auc_score,
                              precision_score, recall_score, f1_score)

np.random.seed(42)

df = pd.read_csv("data_raw.csv")

# ---- limpieza (igual que Practica 3) ----
edad_invalida = (df["edad"] < 0) | (df["edad"] > 110)
df.loc[edad_invalida, "edad"] = np.nan
df["sexo_bin"] = df["sexo"].map({"Hombre": 1, "Mujer": 0})

# ---- grupo de peso para referencia (no usado en export, informativo) ----
grupo_peso = {
    "Sobrepeso": "Exceso de peso", "Obesidad I": "Exceso de peso",
    "Obesidad II": "Exceso de peso", "Obesidad III": "Exceso de peso",
    "Normal": "Peso normal/bajo", "Bajo peso": "Peso normal/bajo",
}
df["grupo_peso"] = df["categoria_imc"].map(grupo_peso)

# =========================================================
# MODELO FINAL: regresion logistica hiperuricemia ~ imc_final + sexo_bin
# =========================================================
datos_log = df.dropna(subset=["imc_final", "sexo_bin", "hiperuricemia"]).copy()
X_log = datos_log[["imc_final", "sexo_bin"]]
y_log = datos_log["hiperuricemia"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_log, y_log, test_size=0.30, random_state=42, stratify=y_log
)

modelo_final = LogisticRegression(C=1, solver="lbfgs", max_iter=1000)
modelo_final.fit(X_train, y_train)
proba_final = modelo_final.predict_proba(X_test)[:, 1]

# ---- modelo alterno (baseline) para comparar: solo IMC ----
X_log_base = datos_log[["imc_final"]]
Xb_train, Xb_test, yb_train, yb_test = train_test_split(
    X_log_base, y_log, test_size=0.30, random_state=42, stratify=y_log
)
modelo_base = LogisticRegression(C=1, solver="lbfgs", max_iter=1000)
modelo_base.fit(Xb_train, yb_train)
proba_base = modelo_base.predict_proba(Xb_test)[:, 1]

# ---- modelo alterno 2: fuerte regularizacion C=0.01 (para ilustrar seccion practica3) ----
modelo_c001 = LogisticRegression(C=0.01, solver="lbfgs", max_iter=1000)
modelo_c001.fit(X_train, y_train)
proba_c001 = modelo_c001.predict_proba(X_test)[:, 1]

def metricas_de(y_true, proba, umbral):
    pred = (proba >= umbral).astype(int)
    acc = accuracy_score(y_true, pred)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, proba)
    except Exception:
        auc = np.nan
    return acc, prec, rec, f1, auc

UMBRAL_ELEGIDO = 0.15

filas_metricas = []
for nombre, proba, y_true in [
    ("Regresion logistica (IMC + sexo) umbral 0.15", proba_final, y_test),
    ("Regresion logistica (IMC + sexo) umbral 0.50", proba_final, y_test),
    ("Regresion logistica (solo IMC) umbral 0.15", proba_base, yb_test),
    ("Regresion logistica C=0.01 (IMC + sexo) umbral 0.15", proba_c001, y_test),
]:
    umbral = 0.50 if "0.50" in nombre else UMBRAL_ELEGIDO
    acc, prec, rec, f1, auc = metricas_de(y_true, proba, umbral)
    filas_metricas.append({
        "modelo": nombre, "exactitud": round(acc, 4), "precision": round(prec, 4),
        "recall": round(rec, 4), "f1": round(f1, 4), "roc_auc": round(auc, 4),
    })

metricas_df = pd.DataFrame(filas_metricas)
metricas_df.to_csv("data/metricas_modelos.csv", index=False)
print(metricas_df)

# =========================================================
# predicciones.csv -> a partir del conjunto de prueba del modelo final
# =========================================================
pred_final = (proba_final >= UMBRAL_ELEGIDO).astype(int)

pred_export = datos_log.loc[X_test.index, [
    "folio_i", "sexo", "edad", "Ciudad", "categoria_imc", "imc_final", "ac_urico"
]].copy()
pred_export = pred_export.rename(columns={"folio_i": "id"})
pred_export["grupo"] = datos_log.loc[X_test.index, "categoria_imc"]
pred_export["periodo"] = "2024"
pred_export["valor_real"] = y_test.values
pred_export["probabilidad"] = np.round(proba_final, 4)
pred_export["prediccion_umbral_0.15"] = pred_final

cols_finales = ["id", "grupo", "periodo", "valor_real", "probabilidad",
                 "sexo", "edad", "Ciudad", "categoria_imc", "imc_final", "ac_urico",
                 "prediccion_umbral_0.15"]
pred_export = pred_export[cols_finales]
pred_export.to_csv("data/predicciones.csv", index=False)
print("\npredicciones:", pred_export.shape)

# =========================================================
# importancia_variables.csv -> coeficientes / odds ratio del modelo final
# + correlaciones de la regresion lineal multiple como variables adicionales de contexto
# =========================================================
coef_log = pd.Series(modelo_final.coef_[0], index=X_log.columns)
odds_ratio = np.exp(coef_log)

# regresion lineal multiple (practica 3) para dar importancia adicional (valor absoluto estandarizado)
vars_lineal = ["edad", "Peso", "Estatura", "ac_urico"]
datos_reg = df.dropna(subset=vars_lineal).copy()
Xl = datos_reg[["edad", "Peso", "Estatura"]]
yl = datos_reg["ac_urico"]
Xl_train, Xl_test, yl_train, yl_test = train_test_split(Xl, yl, test_size=0.30, random_state=42)
modelo_lineal = LinearRegression()
modelo_lineal.fit(Xl_train, yl_train)
# coeficientes estandarizados (beta * std(x)/std(y)) para comparabilidad
coef_std_lineal = modelo_lineal.coef_ * Xl_train.std().values / yl_train.std()

importancia_rows = []
for var in X_log.columns:
    importancia_rows.append({
        "variable": var,
        "importancia": round(abs(coef_log[var]), 4),
        "modelo": "Regresion logistica (hiperuricemia)",
        "odds_ratio": round(odds_ratio[var], 4),
    })
for var, coef in zip(Xl.columns, coef_std_lineal):
    importancia_rows.append({
        "variable": var,
        "importancia": round(abs(coef), 4),
        "modelo": "Regresion lineal multiple (ac_urico)",
        "odds_ratio": np.nan,
    })

importancia_df = pd.DataFrame(importancia_rows)
importancia_df.to_csv("data/importancia_variables.csv", index=False)
print("\nimportancia:\n", importancia_df)

# =========================================================
# extras.csv -> datos de apoyo para graficos exploratorios y contraste de hipotesis
# (no forma parte del contrato de 3 CSV pero enriquece el dashboard)
# =========================================================
extra = {
    "n_total": len(df),
    "n_hiperuricemia_pos": int(datos_log["hiperuricemia"].sum()),
    "prevalencia_hiperuricemia": round(datos_log["hiperuricemia"].mean(), 4),
    "r2_regresion_simple_practica2": 0.0347,
    "r2_regresion_multiple": round(r2_score(yl_test, modelo_lineal.predict(Xl_test)), 4),
    "mse_regresion_multiple": round(mean_squared_error(yl_test, modelo_lineal.predict(Xl_test)), 4),
}

grupo_exceso = df.loc[df["grupo_peso"] == "Exceso de peso", "ac_urico"].dropna()
grupo_normal = df.loc[df["grupo_peso"] == "Peso normal/bajo", "ac_urico"].dropna()
t_stat, p_val = stats.ttest_ind(grupo_exceso, grupo_normal, equal_var=False, alternative="greater")
cohen_d = (grupo_exceso.mean() - grupo_normal.mean()) / np.sqrt((grupo_exceso.std()**2 + grupo_normal.std()**2)/2)
extra.update({
    "media_ac_urico_exceso_peso": round(grupo_exceso.mean(), 4),
    "media_ac_urico_normal_bajo": round(grupo_normal.mean(), 4),
    "t_stat": round(t_stat, 4),
    "p_valor": p_val,
    "cohen_d": round(cohen_d, 4),
})
pd.DataFrame([extra]).to_csv("data/resumen_estadistico.csv", index=False)
print("\nresumen estadistico:\n", extra)

# scatter para grafico exploratorio IMC vs ac_urico (muestreado para no sobrecargar el dashboard)
disp = df.dropna(subset=["imc_final", "ac_urico"])[["imc_final", "ac_urico", "sexo", "categoria_imc"]]
if len(disp) > 1500:
    disp = disp.sample(1500, random_state=42)
disp.to_csv("data/dispersión_imc_acido_urico.csv", index=False, encoding="utf-8")
print("\ndispersion:", disp.shape)
