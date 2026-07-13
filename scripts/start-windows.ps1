$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Config = Join-Path $Root "config.windows.ps1"
if (Test-Path $Config) { . $Config }
$Runtime = if ($env:MCAI_RUNTIME) { $env:MCAI_RUNTIME } else { Join-Path $Root "server-runtime" }
$JavaMemory = if ($env:MCAI_JAVA_MEMORY) { $env:MCAI_JAVA_MEMORY } else { "2G" }
$CpuThreads = if ($env:MCAI_TORCH_THREADS) { [int]$env:MCAI_TORCH_THREADS } else { 0 }
$BotCount = if ($env:MCAI_BOT_COUNT) { $env:MCAI_BOT_COUNT } else { "4" }
$Python = Join-Path $Root "trainer\.venv\Scripts\python.exe"

foreach ($RequiredPath in @($Python, (Join-Path $Runtime "paper-1.12.2.jar"), (Join-Path $Root "worker\dist\src\index.js"))) {
    if (-not (Test-Path $RequiredPath)) { throw "Missing $RequiredPath. Run scripts\bootstrap-windows.ps1 first." }
}
if ($env:MCAI_EXPLOITER_TARGET -and -not (Test-Path $env:MCAI_EXPLOITER_TARGET)) {
    throw "MCAI_EXPLOITER_TARGET does not exist: $($env:MCAI_EXPLOITER_TARGET)"
}

$env:MCAI_SERVER_HOST = if ($env:MCAI_SERVER_HOST) { $env:MCAI_SERVER_HOST } else { "127.0.0.1" }
$env:MCAI_SERVER_PORT = "25565"
$env:MCAI_ARENA_HOST = "127.0.0.1"
$env:MCAI_ARENA_PORT = "8765"
$env:MCAI_TRAINER_URL = "ws://127.0.0.1:8766"
$env:MCAI_BOT_COUNT = $BotCount
$env:MCAI_USERNAME_PREFIX = if ($env:MCAI_USERNAME_PREFIX) { $env:MCAI_USERNAME_PREFIX } else { "MCAI_" }

$Processes = @()
try {
    $CheckpointPath = if ($env:MCAI_EXPLOITER_TARGET) {
        Join-Path $Root "checkpoints\exploiter-active"
    } else { Join-Path $Root "checkpoints" }
    $TrainerArgs = "-m combat_ai.cli serve --host 127.0.0.1 --port 8766 --checkpoints `"$CheckpointPath`" --cpu-threads $CpuThreads"
    if ($env:MCAI_IMITATION_DATA) {
        $TrainerArgs += " --imitation-data `"$($env:MCAI_IMITATION_DATA)`""
    }
    if ($env:MCAI_INITIALIZE_FROM) {
        foreach ($InitialCheckpoint in ($env:MCAI_INITIALIZE_FROM -split ";")) {
            if ($InitialCheckpoint.Trim()) { $TrainerArgs += " --initialize-from `"$($InitialCheckpoint.Trim())`"" }
        }
    }
    if ($env:MCAI_EXPLOITER_TARGET) {
        $TrainerArgs += " --exploiter-target `"$($env:MCAI_EXPLOITER_TARGET)`""
    }
    $Processes += Start-Process -FilePath $Python -ArgumentList $TrainerArgs -WorkingDirectory (Join-Path $Root "trainer") -NoNewWindow -PassThru
    $Processes += Start-Process -FilePath "java" -ArgumentList @("-Xms1G", "-Xmx$JavaMemory", "-jar", "paper-1.12.2.jar", "nogui") -WorkingDirectory $Runtime -NoNewWindow -PassThru

    $Ready = $false
    for ($Attempt = 0; $Attempt -lt 120; $Attempt++) {
        try {
            $Client = [Net.Sockets.TcpClient]::new()
            $Client.Connect("127.0.0.1", 8765)
            $Client.Dispose()
            $Ready = $true
            break
        } catch { Start-Sleep -Seconds 1 }
        if ($Processes | Where-Object { $_.HasExited }) { throw "A training process exited during startup." }
    }
    if (-not $Ready) { throw "Paper arena control did not become ready within 120 seconds." }
    $Processes += Start-Process -FilePath "npm.cmd" -ArgumentList "start" `
        -WorkingDirectory (Join-Path $Root "worker") -NoNewWindow -PassThru
    Write-Host "MCAI is running. Press Ctrl+C for an orderly emergency stop."
    while ($true) {
        $Exited = $Processes | Where-Object { $_.HasExited }
        if ($Exited) {
            $Summary = ($Exited | ForEach-Object { "PID $($_.Id) exit $($_.ExitCode)" }) -join ", "
            throw "A training process stopped unexpectedly: $Summary"
        }
        Start-Sleep -Seconds 1
    }
} finally {
    try {
        & $Python (Join-Path $Root "scripts\arena_control.py") stop_all | Out-Null
    } catch { }
    foreach ($Process in $Processes) {
        if ($Process -and -not $Process.HasExited) {
            try { & taskkill.exe /PID $Process.Id /T /F 2>$null | Out-Null } catch { }
        }
    }
}
