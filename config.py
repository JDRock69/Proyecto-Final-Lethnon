"""
config.py
---------
Parámetros editables del dashboard. Cambie únicamente los valores a la
derecha del signo igual; no es necesario tocar app.py para personalizar
títulos, nombres de variables o colores.
"""

# ---------------------------------------------------------------------------
# Textos generales
# ---------------------------------------------------------------------------
TITULO = "Riesgo metabólico en población mexicana"
SUBTITULO = "Obesidad (IMC) e hiperuricemia — Proyecto final, Métodos Estadísticos Computacionales"
AUTOR = "Lethnon Jalil Delgado Guerrero — UCompensar"

NOMBRE_REGISTRO = "Personas evaluadas"
NOMBRE_CLASE_POSITIVA = "Hiperuricemia observada"
NOMBRE_GRUPO = "Categoría de IMC"
NOMBRE_PERIODO = "Periodo (cohorte 2024)"

# Umbral de decisión elegido en la Práctica 3 (cercano al óptimo de Youden,
# priorizando sensibilidad sobre exactitud dado el desbalance de clases).
UMBRAL_POR_DEFECTO = 0.15

# ---------------------------------------------------------------------------
# Paleta de colores
# ---------------------------------------------------------------------------
COLOR_PRIMARIO = "#2C6E8C"      # azul-petróleo — encabezados y elementos activos
COLOR_SECUNDARIO = "#D9822B"    # naranja — clase positiva / alertas suaves
COLOR_FONDO = "#F4F6F8"
COLOR_TARJETA = "#FFFFFF"
COLOR_TEXTO = "#1F2D3A"
COLOR_TEXTO_SUAVE = "#5B6B79"
COLOR_EXITO = "#3E8E5A"
COLOR_ALERTA = "#C24C3C"
COLOR_LINEA_REGRESION = "#C24C3C"

PALETA_CATEGORICA = ["#2C6E8C", "#D9822B", "#3E8E5A", "#8E6C9C", "#C24C3C", "#6B8E23"]

# ---------------------------------------------------------------------------
# Rutas de archivos de datos (relativas a la carpeta del proyecto)
# ---------------------------------------------------------------------------
RUTA_PREDICCIONES = "data/predicciones.csv"
RUTA_METRICAS = "data/metricas_modelos.csv"
RUTA_IMPORTANCIA = "data/importancia_variables.csv"
RUTA_RESUMEN = "data/resumen_estadistico.csv"
RUTA_DISPERSION = "data/dispersion_imc_acido_urico.csv"

PUERTO = 8050
