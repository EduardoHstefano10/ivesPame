# ============================================================
# run_app.ps1 - Ejecuta la app Streamlit en Windows PowerShell
# ============================================================
# Uso:
#   .\run_app.ps1            -> crea venv, instala deps y ejecuta
#   .\run_app.ps1 -Install   -> solo instala/actualiza dependencias
#   .\run_app.ps1 -Fresh     -> recrea el venv desde cero
# ============================================================

param(
    [switch]$Install,
    [switch]$Fresh
)

$ErrorActionPreference = "Stop"
$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvDir    = Join-Path $ProjectDir ".venv"
$Requirements = Join-Path $ProjectDir "requirements.txt"
$AppFile    = Join-Path $ProjectDir "app_streamlit.py"

Set-Location $ProjectDir

function Write-Step($msg) { Write-Host ">> $msg" -ForegroundColor Cyan }

# Validar Python
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python) {
    Write-Host "ERROR: Python no está instalado o no está en el PATH." -ForegroundColor Red
    Write-Host "Descárgalo desde https://www.python.org/downloads/ y reinicia PowerShell." -ForegroundColor Yellow
    exit 1
}

# Recrear venv si se pidió
if ($Fresh -and (Test-Path $VenvDir)) {
    Write-Step "Eliminando entorno virtual existente..."
    Remove-Item -Recurse -Force $VenvDir
}

# Crear venv si no existe
if (-not (Test-Path $VenvDir)) {
    Write-Step "Creando entorno virtual en .venv ..."
    python -m venv $VenvDir
}

# Activar venv
$Activate = Join-Path $VenvDir "Scripts\Activate.ps1"
if (-not (Test-Path $Activate)) {
    Write-Host "ERROR: No se encontró el script de activación: $Activate" -ForegroundColor Red
    exit 1
}
Write-Step "Activando entorno virtual..."
. $Activate

# Instalar dependencias (solo si es la primera vez o si se pasó -Install / -Fresh)
$Marker = Join-Path $VenvDir ".requirements_installed"
if ($Install -or $Fresh -or -not (Test-Path $Marker)) {
    Write-Step "Actualizando pip..."
    python -m pip install --upgrade pip

    if (Test-Path $Requirements) {
        Write-Step "Instalando dependencias desde requirements.txt ..."
        pip install -r $Requirements
        New-Item -ItemType File -Path $Marker -Force | Out-Null
    } else {
        Write-Host "WARN: No se encontró requirements.txt" -ForegroundColor Yellow
    }

    if ($Install) {
        Write-Step "Instalación completada. Usa '.\run_app.ps1' para ejecutar la app."
        exit 0
    }
}

# Verificar app
if (-not (Test-Path $AppFile)) {
    Write-Host "ERROR: No se encontró $AppFile" -ForegroundColor Red
    exit 1
}

# Ejecutar Streamlit
Write-Step "Iniciando Streamlit en http://localhost:8501 ..."
Write-Host "   (Presiona Ctrl+C para detener)" -ForegroundColor DarkGray
streamlit run $AppFile
