# ============================================================
# run_docker.ps1 - Levanta la app en Docker (Windows PowerShell)
# ============================================================
# Uso:
#   .\run_docker.ps1              -> build + up (solo genera dataset si falta)
#   .\run_docker.ps1 -Fresh       -> regenera el dataset unificado
#   .\run_docker.ps1 -Down        -> detiene el contenedor
#   .\run_docker.ps1 -Logs        -> sigue los logs
# ============================================================

param(
    [switch]$Fresh,
    [switch]$Down,
    [switch]$Logs
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path)

function Require-Docker {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Write-Host "ERROR: Docker no está instalado o no está en el PATH." -ForegroundColor Red
        Write-Host "Instala Docker Desktop: https://www.docker.com/products/docker-desktop/" -ForegroundColor Yellow
        exit 1
    }
}

Require-Docker

if ($Down)  { docker compose down;       exit $LASTEXITCODE }
if ($Logs)  { docker compose logs -f;    exit $LASTEXITCODE }

if ($Fresh) { $env:UNIFICAR = "force" } else { $env:UNIFICAR = "auto" }

if (-not (Test-Path ".env")) {
    if (Test-Path ".env.example") {
        Write-Host ">> No se encontro .env, copiando .env.example -> .env" -ForegroundColor Yellow
        Copy-Item ".env.example" ".env"
    }
}

Write-Host ">> Construyendo imagen y levantando contenedor (logs en esta consola)..." -ForegroundColor Cyan
Write-Host ">> URL: http://localhost:8501  (Ctrl+C detiene el contenedor)" -ForegroundColor Green

# Modo foreground: los logs van directos a la consola de VS Code.
docker compose up --build
