"""
data_utils.py
-------------
Funciones de lectura y verificación de los archivos CSV que alimentan el
dashboard. Centralizar aquí la lectura evita repetir la misma lógica de
validación en app.py y en validar_datos.py.
"""

import os
import pandas as pd
import config


class ErrorDatos(Exception):
    """Error de formato o contenido en alguno de los CSV del proyecto."""


def _verificar_archivo(ruta):
    if not os.path.exists(ruta):
        raise ErrorDatos(f"No se encontró el archivo requerido: {ruta}")


def cargar_predicciones():
    _verificar_archivo(config.RUTA_PREDICCIONES)
    df = pd.read_csv(config.RUTA_PREDICCIONES)

    columnas_obligatorias = ["id", "grupo", "periodo", "valor_real", "probabilidad"]
    faltantes = [c for c in columnas_obligatorias if c not in df.columns]
    if faltantes:
        raise ErrorDatos(f"predicciones.csv no tiene las columnas: {faltantes}")

    if not df["valor_real"].dropna().isin([0, 1]).all():
        raise ErrorDatos("predicciones.csv: 'valor_real' debe contener solo 0 o 1")

    fuera_rango = ~df["probabilidad"].between(0, 1)
    if fuera_rango.any():
        raise ErrorDatos("predicciones.csv: 'probabilidad' debe estar entre 0 y 1")

    return df


def cargar_metricas():
    _verificar_archivo(config.RUTA_METRICAS)
    df = pd.read_csv(config.RUTA_METRICAS)

    columnas_obligatorias = ["modelo", "exactitud"]
    faltantes = [c for c in columnas_obligatorias if c not in df.columns]
    if faltantes:
        raise ErrorDatos(f"metricas_modelos.csv no tiene las columnas: {faltantes}")

    columnas_numericas = [c for c in df.columns if c not in ("modelo",)]
    for col in columnas_numericas:
        valores = pd.to_numeric(df[col], errors="coerce")
        excede = valores.dropna().gt(1).any()
        if excede:
            raise ErrorDatos(
                f"metricas_modelos.csv: la columna '{col}' tiene valores > 1; "
                "las métricas deben expresarse como decimales (0.85, no 85)."
            )
    return df


def cargar_importancia():
    _verificar_archivo(config.RUTA_IMPORTANCIA)
    df = pd.read_csv(config.RUTA_IMPORTANCIA)

    columnas_obligatorias = ["variable", "importancia"]
    faltantes = [c for c in columnas_obligatorias if c not in df.columns]
    if faltantes:
        raise ErrorDatos(f"importancia_variables.csv no tiene las columnas: {faltantes}")
    return df


def cargar_resumen():
    """Archivo adicional (no forma parte del contrato mínimo de 3 CSV) con
    el resultado del contraste de hipótesis y del R² de la regresión lineal
    múltiple, usado para las tarjetas de contexto del dashboard."""
    _verificar_archivo(config.RUTA_RESUMEN)
    return pd.read_csv(config.RUTA_RESUMEN).iloc[0]


def cargar_dispersion():
    """Muestra de puntos IMC vs. ácido úrico para el gráfico exploratorio
    (opcional; si no existe, el dashboard simplemente omite ese panel)."""
    if not os.path.exists(config.RUTA_DISPERSION):
        return pd.DataFrame(columns=["imc_final", "ac_urico", "sexo", "categoria_imc"])
    return pd.read_csv(config.RUTA_DISPERSION)
