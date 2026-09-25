"""
validar_datos.py
-----------------
Ejecute este script antes de levantar el dashboard (python validar_datos.py).
Verifica que los tres archivos CSV requeridos existan, tengan las columnas
obligatorias y valores dentro de los rangos esperados.
"""

import sys
import data_utils


def main():
    try:
        predicciones = data_utils.cargar_predicciones()
        metricas = data_utils.cargar_metricas()
        importancia = data_utils.cargar_importancia()
    except data_utils.ErrorDatos as error:
        print("ERROR DE VALIDACION")
        print(str(error))
        sys.exit(1)

    print("VALIDACION CORRECTA")
    print(f"Predicciones: {len(predicciones)} filas")
    print(f"Modelos comparados: {metricas['modelo'].nunique()}")
    print(f"Variables con importancia: {len(importancia)}")

    # Advertencias no bloqueantes
    if predicciones["valor_real"].mean() < 0.05:
        print(
            "Aviso: la clase positiva representa menos del 5% de los registros; "
            "revise el umbral de decisión usado en el dashboard."
        )


if __name__ == "__main__":
    main()
