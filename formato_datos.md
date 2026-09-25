# Formato de los archivos de datos

El dashboard lee cinco archivos CSV desde la carpeta `data/`. Los tres primeros
siguen el contrato mínimo de la plantilla de referencia; los dos últimos son
extensiones propias de este proyecto para enriquecer el panel de exploración
y el de contraste de hipótesis.

## 1. `predicciones.csv` (obligatorio)

Una fila por persona evaluada en el conjunto de prueba del modelo final
(regresión logística IMC + sexo → hiperuricemia).

| Columna | Tipo | Contenido |
|---|---|---|
| `id` | texto | Folio anonimizado (`folio_i` del dataset original) |
| `grupo` | texto | Categoría de IMC (usada para filtrar) |
| `periodo` | texto | Cohorte de la muestra (`2024`) |
| `valor_real` | 0 o 1 | Hiperuricemia observada |
| `probabilidad` | 0 a 1 | Probabilidad predicha por el modelo final |
| `sexo`, `edad`, `Ciudad`, `categoria_imc`, `imc_final`, `ac_urico` | — | Columnas de contexto mostradas en la tabla |
| `prediccion_umbral_0.15` | 0 o 1 | Predicción con el umbral elegido en la Práctica 3 |

## 2. `metricas_modelos.csv` (obligatorio)

Una fila por modelo/umbral comparado: `modelo, exactitud, precision, recall,
f1, roc_auc`. Las métricas se expresan como decimales entre 0 y 1.

## 3. `importancia_variables.csv` (obligatorio)

Columnas: `variable, importancia, modelo, odds_ratio`. Incluye tanto los
coeficientes de la regresión logística (con su `odds_ratio`) como los
coeficientes estandarizados de la regresión lineal múltiple (`odds_ratio`
vacío, porque no aplica a ese modelo).

## 4. `resumen_estadistico.csv` (extensión propia)

Una sola fila con los resultados del contraste de hipótesis (estadístico t,
p-valor, d de Cohen) y del ajuste de la regresión lineal múltiple (R², MSE),
usados en las tarjetas KPI y en la pestaña "Contraste de hipótesis".

## 5. `dispersion_imc_acido_urico.csv` (extensión propia)

Muestra de hasta 1500 registros con `imc_final, ac_urico, sexo,
categoria_imc`, usada en el gráfico de dispersión de la pestaña de
exploración. Si el archivo no existe, esa pestaña simplemente omite el
gráfico (ver `data_utils.cargar_dispersion`).

## Regenerar los archivos a partir de los notebooks

Los cinco archivos se generan a partir de `Diabetes_Mexico_Completado.csv`
con el mismo procedimiento de limpieza y los mismos modelos (semilla
`random_state=42`) documentados en el informe (`Informe_Proyecto_Final.ipynb`).
Si se actualiza el dataset o el modelo, vuelva a ejecutar esa sección del
informe y sobrescriba los CSV de `data/`.
