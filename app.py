"""
app.py
------
Dashboard interactivo (Dash + Plotly) del proyecto "Riesgo metabólico en
población mexicana" (obesidad e hiperuricemia). Construye el layout, los
gráficos y los callbacks a partir de los CSV en data/.

Para personalizar títulos o colores, edite config.py — no es necesario
tocar este archivo.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, dash_table, Input, Output, ctx
from sklearn.metrics import confusion_matrix, roc_curve, auc

import config
import data_utils

# ---------------------------------------------------------------------------
# 1. Carga de datos (con validación previa)
# ---------------------------------------------------------------------------
predicciones = data_utils.cargar_predicciones()
metricas = data_utils.cargar_metricas()
importancia = data_utils.cargar_importancia()
resumen = data_utils.cargar_resumen()
dispersion = data_utils.cargar_dispersion()

grupos_disponibles = sorted(predicciones["grupo"].dropna().unique().tolist())
sexos_disponibles = sorted(predicciones["sexo"].dropna().unique().tolist()) \
    if "sexo" in predicciones.columns else []

# ---------------------------------------------------------------------------
# 2. App
# ---------------------------------------------------------------------------
app = Dash(__name__, title=config.TITULO, suppress_callback_exceptions=True)
server = app.server  # para despliegues tipo gunicorn/Binder si se requiere


def tarjeta_kpi(titulo, valor, descripcion, color=config.COLOR_PRIMARIO):
    return html.Div(
        className="tarjeta-kpi",
        style={"borderTop": f"4px solid {color}"},
        children=[
            html.Div(titulo, className="kpi-titulo"),
            html.Div(valor, className="kpi-valor", style={"color": color}),
            html.Div(descripcion, className="kpi-descripcion"),
        ],
    )


def encabezado():
    return html.Div(
        className="encabezado",
        children=[
            html.Div(
                [
                    html.H1(config.TITULO),
                    html.P(config.SUBTITULO, className="subtitulo"),
                ]
            ),
            html.Div(config.AUTOR, className="autor"),
        ],
    )


def panel_kpis():
    n_total = int(resumen["n_total"])
    prevalencia = float(resumen["prevalencia_hiperuricemia"])
    fila_final = metricas[metricas["modelo"].str.contains("umbral 0.15") &
                           metricas["modelo"].str.contains(r"\(IMC \+ sexo\)")]
    roc_auc_final = float(fila_final["roc_auc"].iloc[0]) if len(fila_final) else np.nan
    r2_multiple = float(resumen["r2_regresion_multiple"])

    return html.Div(
        className="fila-kpis",
        children=[
            tarjeta_kpi("Personas en la muestra", f"{n_total:,}", "Registros con datos válidos"),
            tarjeta_kpi(
                "Prevalencia de hiperuricemia", f"{prevalencia:.1%}",
                "Clase positiva minoritaria — se ajusta el umbral de decisión",
                color=config.COLOR_SECUNDARIO,
            ),
            tarjeta_kpi(
                "ROC-AUC (modelo final)", f"{roc_auc_final:.3f}",
                "Regresión logística: IMC + sexo", color=config.COLOR_EXITO,
            ),
            tarjeta_kpi(
                "R² regresión lineal múltiple", f"{r2_multiple:.3f}",
                "ac_urico ~ edad + Peso + Estatura (mejora sobre R²=0.035 de la Práctica 2)",
                color=config.COLOR_PRIMARIO,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 3. Pestaña: Exploración (Práctica 2)
# ---------------------------------------------------------------------------
def pestana_exploracion():
    return html.Div(
        className="panel",
        children=[
            html.Div(
                className="fila-controles",
                children=[
                    html.Div(
                        [
                            html.Label("Sexo"),
                            dcc.Dropdown(
                                id="filtro-sexo-exploracion",
                                options=[{"label": s, "value": s} for s in sexos_disponibles],
                                value=[], multi=True, placeholder="Todos",
                            ),
                        ],
                        className="control",
                    ),
                    html.Div(
                        [
                            html.Label("Categoría de IMC"),
                            dcc.Dropdown(
                                id="filtro-categoria-exploracion",
                                options=[{"label": g, "value": g}
                                         for g in sorted(dispersion["categoria_imc"].dropna().unique())],
                                value=[], multi=True, placeholder="Todas",
                            ),
                        ],
                        className="control",
                    ),
                ],
            ),
            html.Div(
                className="fila-graficos",
                children=[
                    dcc.Graph(id="grafico-dispersion-imc"),
                ],
            ),
            html.Div(
                className="nota-metodologica",
                children=(
                    "En la Práctica 2 la regresión lineal simple (IMC → ácido úrico) dio una "
                    "correlación baja (r ≈ 0.19, R² ≈ 0.035). La línea roja muestra el mismo "
                    "ajuste sobre la muestra completa; la nube de puntos evidencia por qué el "
                    "IMC por sí solo explica poco: hay mucha dispersión vertical para un mismo IMC."
                ),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 4. Pestaña: Contraste de hipótesis y regresión múltiple (Práctica 3)
# ---------------------------------------------------------------------------
def pestana_hipotesis():
    media_exceso = float(resumen["media_ac_urico_exceso_peso"])
    media_normal = float(resumen["media_ac_urico_normal_bajo"])
    p_valor = float(resumen["p_valor"])
    cohen_d = float(resumen["cohen_d"])

    fig_barras = go.Figure(
        data=[
            go.Bar(
                x=["Peso normal/bajo", "Exceso de peso"],
                y=[media_normal, media_exceso],
                marker_color=[config.COLOR_PRIMARIO, config.COLOR_SECUNDARIO],
                text=[f"{media_normal:.2f}", f"{media_exceso:.2f}"],
                textposition="outside",
            )
        ]
    )
    fig_barras.update_layout(
        title="Ácido úrico promedio por grupo de peso (prueba t de Welch)",
        yaxis_title="Ácido úrico (mg/dL)",
        template="plotly_white",
        margin=dict(t=60, l=40, r=20, b=40),
    )

    texto_p = "p < 0.001" if p_valor < 0.001 else f"p = {p_valor:.4f}"

    fig_importancia_lineal = px.bar(
        importancia[importancia["modelo"] == "Regresion lineal multiple (ac_urico)"],
        x="variable", y="importancia",
        title="Importancia relativa (coeficiente estandarizado) — regresión lineal múltiple",
        color_discrete_sequence=[config.COLOR_PRIMARIO],
        template="plotly_white",
    )
    fig_importancia_lineal.update_layout(margin=dict(t=60, l=40, r=20, b=40))

    return html.Div(
        className="panel",
        children=[
            html.Div(
                className="fila-graficos",
                children=[
                    dcc.Graph(figure=fig_barras),
                    dcc.Graph(figure=fig_importancia_lineal),
                ],
            ),
            html.Div(
                className="tarjeta-resultado",
                children=[
                    html.H4("Resultado del contraste de hipótesis"),
                    html.P(
                        f"H0: el ácido úrico promedio es igual entre ambos grupos. "
                        f"H1: es mayor en personas con exceso de peso. "
                        f"Resultado: se rechaza H0 ({texto_p}, tamaño del efecto "
                        f"d de Cohen = {cohen_d:.2f}, magnitud moderada)."
                    ),
                    html.P(
                        f"La regresión lineal múltiple (ac_urico ~ edad + Peso + Estatura) "
                        f"obtiene R² = {float(resumen['r2_regresion_multiple']):.3f} en el "
                        f"conjunto de prueba, superior al R² = 0.035 de la regresión simple "
                        f"con el IMC de la Práctica 2, aunque sigue siendo un ajuste moderado."
                    ),
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 5. Pestaña: Clasificación de hiperuricemia (regresión logística)
# ---------------------------------------------------------------------------
def pestana_clasificacion():
    fig_comparacion = px.bar(
        metricas, x="modelo", y=["exactitud", "recall", "roc_auc"],
        barmode="group", template="plotly_white",
        title="Comparación de modelos y umbrales",
        color_discrete_sequence=config.PALETA_CATEGORICA,
    )
    fig_comparacion.update_layout(
        xaxis_title="", yaxis_title="Valor de la métrica",
        legend_title="Métrica", margin=dict(t=60, l=40, r=20, b=120),
        xaxis_tickangle=-20,
    )

    fig_importancia_log = px.bar(
        importancia[importancia["modelo"] == "Regresion logistica (hiperuricemia)"],
        x="variable", y="odds_ratio",
        title="Razón de momios (odds ratio) — regresión logística",
        color_discrete_sequence=[config.COLOR_SECUNDARIO],
        template="plotly_white",
    )
    fig_importancia_log.add_hline(y=1, line_dash="dash", line_color=config.COLOR_TEXTO_SUAVE,
                                   annotation_text="odds ratio = 1 (sin efecto)")
    fig_importancia_log.update_layout(margin=dict(t=60, l=40, r=20, b=40))

    return html.Div(
        className="panel",
        children=[
            html.Div(
                className="fila-controles",
                children=[
                    html.Div(
                        [
                            html.Label("Umbral de decisión"),
                            dcc.Slider(
                                id="umbral-decision",
                                min=0.05, max=0.90, step=0.05,
                                value=config.UMBRAL_POR_DEFECTO,
                                marks={v: f"{v:.2f}" for v in [0.05, 0.15, 0.30, 0.50, 0.70, 0.90]},
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                        className="control control-ancho",
                    ),
                ],
            ),
            html.Div(
                className="fila-graficos",
                children=[
                    dcc.Graph(id="grafico-matriz-confusion"),
                    dcc.Graph(id="grafico-roc"),
                ],
            ),
            html.Div(id="tarjetas-umbral", className="fila-kpis"),
            html.Div(
                className="fila-graficos",
                children=[
                    dcc.Graph(figure=fig_comparacion),
                    dcc.Graph(figure=fig_importancia_log),
                ],
            ),
            html.Div(
                className="nota-metodologica",
                children=(
                    "Con el umbral por defecto (0.50) el modelo casi nunca predice la clase "
                    "positiva, dado el fuerte desbalance de clases (~10% de prevalencia). "
                    "Mueva el control para ver cómo cambian sensibilidad y especificidad; "
                    "el umbral 0.15, cercano al óptimo de Youden, prioriza detectar casos "
                    "de riesgo sobre maximizar la exactitud global."
                ),
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 6. Pestaña: Tabla de predicciones
# ---------------------------------------------------------------------------
def pestana_tabla():
    columnas_mostrar = ["id", "sexo", "edad", "Ciudad", "categoria_imc",
                         "imc_final", "ac_urico", "valor_real", "probabilidad"]
    columnas_mostrar = [c for c in columnas_mostrar if c in predicciones.columns]

    return html.Div(
        className="panel",
        children=[
            html.Div(
                className="fila-controles",
                children=[
                    html.Div(
                        [
                            html.Label("Grupo (categoría de IMC)"),
                            dcc.Dropdown(
                                id="filtro-grupo-tabla",
                                options=[{"label": g, "value": g} for g in grupos_disponibles],
                                value=[], multi=True, placeholder="Todos",
                            ),
                        ],
                        className="control",
                    ),
                ],
            ),
            dash_table.DataTable(
                id="tabla-predicciones",
                columns=[{"name": c, "id": c} for c in columnas_mostrar],
                data=predicciones[columnas_mostrar].round(3).to_dict("records"),
                page_size=12,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto"},
                style_cell={"padding": "8px", "fontFamily": "Inter, sans-serif", "fontSize": "13px"},
                style_header={"backgroundColor": config.COLOR_PRIMARIO, "color": "white", "fontWeight": "600"},
                style_data_conditional=[
                    {
                        "if": {"filter_query": "{valor_real} = 1"},
                        "backgroundColor": "#FCEEE3",
                    }
                ],
            ),
        ],
    )


# ---------------------------------------------------------------------------
# 7. Layout general (pestañas)
# ---------------------------------------------------------------------------
app.layout = html.Div(
    className="contenedor",
    children=[
        encabezado(),
        panel_kpis(),
        dcc.Tabs(
            id="pestanas",
            value="tab-exploracion",
            children=[
                dcc.Tab(label="Exploración IMC / ácido úrico", value="tab-exploracion"),
                dcc.Tab(label="Contraste de hipótesis", value="tab-hipotesis"),
                dcc.Tab(label="Clasificación (hiperuricemia)", value="tab-clasificacion"),
                dcc.Tab(label="Tabla de casos", value="tab-tabla"),
            ],
        ),
        html.Div(id="contenido-pestana"),
        html.Footer(
            "Proyecto final — Métodos Estadísticos Computacionales — UCompensar",
            className="pie-pagina",
        ),
    ],
)


@app.callback(Output("contenido-pestana", "children"), Input("pestanas", "value"))
def mostrar_pestana(pestana):
    if pestana == "tab-exploracion":
        return pestana_exploracion()
    if pestana == "tab-hipotesis":
        return pestana_hipotesis()
    if pestana == "tab-clasificacion":
        return pestana_clasificacion()
    if pestana == "tab-tabla":
        return pestana_tabla()
    return html.Div("Pestaña no encontrada")


# ---------------------------------------------------------------------------
# 8. Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("grafico-dispersion-imc", "figure"),
    Input("filtro-sexo-exploracion", "value"),
    Input("filtro-categoria-exploracion", "value"),
)
def actualizar_dispersion(sexos, categorias):
    datos = dispersion.copy()
    if sexos:
        datos = datos[datos["sexo"].isin(sexos)]
    if categorias:
        datos = datos[datos["categoria_imc"].isin(categorias)]

    fig = px.scatter(
        datos, x="imc_final", y="ac_urico", color="sexo",
        opacity=0.45, template="plotly_white",
        color_discrete_sequence=config.PALETA_CATEGORICA,
        labels={"imc_final": "IMC (kg/m²)", "ac_urico": "Ácido úrico (mg/dL)", "sexo": "Sexo"},
    )
    if len(datos) > 1:
        m, b = np.polyfit(datos["imc_final"], datos["ac_urico"], 1)
        x_linea = np.linspace(datos["imc_final"].min(), datos["imc_final"].max(), 50)
        fig.add_trace(go.Scatter(
            x=x_linea, y=m * x_linea + b, mode="lines",
            line=dict(color=config.COLOR_LINEA_REGRESION, width=3),
            name="Regresión lineal (submuestra filtrada)",
        ))
    fig.update_layout(title="IMC vs. ácido úrico", margin=dict(t=60, l=40, r=20, b=40))
    return fig


@app.callback(
    Output("grafico-matriz-confusion", "figure"),
    Output("grafico-roc", "figure"),
    Output("tarjetas-umbral", "children"),
    Input("umbral-decision", "value"),
)
def actualizar_clasificacion(umbral):
    y_real = predicciones["valor_real"].to_numpy()
    proba = predicciones["probabilidad"].to_numpy()
    pred = (proba >= umbral).astype(int)

    matriz = confusion_matrix(y_real, pred, labels=[0, 1])
    tn, fp, fn, tp = matriz.ravel()
    sensibilidad = tp / (tp + fn) if (tp + fn) else 0.0
    especificidad = tn / (tn + fp) if (tn + fp) else 0.0
    exactitud = (tp + tn) / len(y_real)

    etiquetas = ["Sin hiperuricemia (0)", "Con hiperuricemia (1)"]
    fig_matriz = go.Figure(
        data=go.Heatmap(
            z=matriz, x=etiquetas, y=etiquetas,
            colorscale=[[0, "#EAF1F5"], [1, config.COLOR_PRIMARIO]],
            text=matriz, texttemplate="%{text}", showscale=False,
        )
    )
    fig_matriz.update_layout(
        title=f"Matriz de confusión (umbral = {umbral:.2f})",
        xaxis_title="Predicción", yaxis_title="Valor real",
        template="plotly_white", margin=dict(t=60, l=40, r=20, b=40),
    )

    fpr, tpr, _ = roc_curve(y_real, proba)
    auc_valor = auc(fpr, tpr)
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                  line=dict(color=config.COLOR_PRIMARIO, width=3),
                                  name=f"ROC (AUC = {auc_valor:.3f})"))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                  line=dict(color=config.COLOR_TEXTO_SUAVE, dash="dash"),
                                  name="Azar"))
    fig_roc.update_layout(
        title="Curva ROC (conjunto de prueba)",
        xaxis_title="Tasa de falsos positivos", yaxis_title="Sensibilidad (recall)",
        template="plotly_white", margin=dict(t=60, l=40, r=20, b=40),
    )

    tarjetas = [
        tarjeta_kpi("Exactitud", f"{exactitud:.1%}", f"Umbral = {umbral:.2f}"),
        tarjeta_kpi("Sensibilidad (recall)", f"{sensibilidad:.1%}",
                    "Casos de hiperuricemia detectados", color=config.COLOR_SECUNDARIO),
        tarjeta_kpi("Especificidad", f"{especificidad:.1%}",
                    "Casos sin hiperuricemia bien clasificados", color=config.COLOR_EXITO),
        tarjeta_kpi("Falsos negativos", f"{fn}",
                    "Casos de riesgo no detectados con este umbral", color=config.COLOR_ALERTA),
    ]

    return fig_matriz, fig_roc, tarjetas


@app.callback(
    Output("tabla-predicciones", "data"),
    Input("filtro-grupo-tabla", "value"),
)
def actualizar_tabla(grupos):
    datos = predicciones.copy()
    if grupos:
        datos = datos[datos["grupo"].isin(grupos)]
    columnas_mostrar = ["id", "sexo", "edad", "Ciudad", "categoria_imc",
                         "imc_final", "ac_urico", "valor_real", "probabilidad"]
    columnas_mostrar = [c for c in columnas_mostrar if c in datos.columns]
    return datos[columnas_mostrar].round(3).to_dict("records")


if __name__ == "__main__":
    app.run(debug=False, port=config.PUERTO)
