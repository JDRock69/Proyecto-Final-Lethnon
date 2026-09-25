@echo off
call .venv\Scripts\Activate.ps1
python validar_datos.py
if %errorlevel% neq 0 (
    echo.
    echo La validacion de datos fallo. Revise el mensaje anterior.
    pause
    exit /b 1
)
python app.py
pause
