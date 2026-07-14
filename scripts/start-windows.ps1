$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Config = Join-Path $Root "config.windows.ps1"
if (Test-Path $Config) { . $Config }
$Runtime = if ($env:MCAI_RUNTIME) { $env:MCAI_RUNTIME } else { Join-Path $Root "server-runtime" }
$JavaMemory = if ($env:MCAI_JAVA_MEMORY) { $env:MCAI_JAVA_MEMORY } else { "2G" }
$CpuThreads = if ($env:MCAI_TORCH_THREADS) { [int]$env:MCAI_TORCH_THREADS } else { 0 }
$BotCount = if ($env:MCAI_BOT_COUNT) { $env:MCAI_BOT_COUNT } else { "4" }
$RolloutSteps = if ($env:MCAI_ROLLOUT_STEPS) { [int]$env:MCAI_ROLLOUT_STEPS } else { 8192 }
$DashboardPort = if ($env:MCAI_DASHBOARD_PORT) { [int]$env:MCAI_DASHBOARD_PORT } else { 8788 }
$Python = Join-Path $Root "trainer\.venv\Scripts\python.exe"
$Java = if ($env:MCAI_JAVA_EXE -and (Test-Path $env:MCAI_JAVA_EXE)) {
    $env:MCAI_JAVA_EXE
} else { (Get-Command java.exe -ErrorAction Stop).Source }
$Node = (Get-Command node.exe -ErrorAction Stop).Source
$RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$RunDirectory = Join-Path $Root "runs\windows-$RunStamp"
$StateDirectory = Join-Path $Root ".mcai"
$StatePath = Join-Path $StateDirectory "current.json"
New-Item -ItemType Directory -Force -Path $RunDirectory, $StateDirectory | Out-Null

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
$env:MCAI_RUN_DIR = $RunDirectory
$env:MCAI_CHECKPOINT_DIR = Join-Path $Root "checkpoints"
$env:MCAI_DASHBOARD_PORT = "$DashboardPort"

$Processes = @()
function Start-LoggedProcess(
    [string]$Name, [string]$FilePath, [object]$ArgumentList, [string]$WorkingDirectory
) {
    $stdout = Join-Path $RunDirectory "$Name.log"
    $stderr = Join-Path $RunDirectory "$Name.error.log"
    return Start-Process -FilePath $FilePath -ArgumentList $ArgumentList -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -NoNewWindow -PassThru
}

function Save-ProcessState {
    $payload = @{
        started_at = (Get-Date).ToString("o")
        run_directory = $RunDirectory
        dashboard_url = "http://127.0.0.1:$DashboardPort"
        processes = @($Processes | ForEach-Object {
            @{ name = $_.Name; pid = $_.Process.Id; process_started_at = $_.Process.StartTime.ToString("o") }
        })
    }
    $payload | ConvertTo-Json -Depth 4 | Set-Content -Path $StatePath -Encoding UTF8
}

function Wait-LocalPort([int]$Port, [int]$Seconds) {
    for ($attempt = 0; $attempt -lt $Seconds; $attempt++) {
        try {
            $client = [Net.Sockets.TcpClient]::new()
            $client.Connect("127.0.0.1", $Port)
            $client.Dispose()
            return $true
        } catch { Start-Sleep -Seconds 1 }
        if ($Processes | Where-Object { $_.Process.HasExited }) { return $false }
    }
    return $false
}

try {
    $CheckpointPath = if ($env:MCAI_EXPLOITER_TARGET) {
        Join-Path $Root "checkpoints\exploiter-active"
    } else { Join-Path $Root "checkpoints" }
    $TrainerArgs = "-m combat_ai.cli serve --host 127.0.0.1 --port 8766 --checkpoints `"$CheckpointPath`" --cpu-threads $CpuThreads --rollout-steps $RolloutSteps"
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
    $Processes += [pscustomobject]@{ Name = "trainer"; Process = Start-LoggedProcess "trainer" $Python $TrainerArgs (Join-Path $Root "trainer") }
    $Processes += [pscustomobject]@{ Name = "paper"; Process = Start-LoggedProcess "paper" $Java @("-Xms512M", "-Xmx$JavaMemory", "-jar", "paper-1.12.2.jar", "nogui") $Runtime }
    $Processes += [pscustomobject]@{ Name = "dashboard"; Process = Start-LoggedProcess "dashboard" $Node @((Join-Path $Root "dashboard\server.mjs")) $Root }

    if (-not (Wait-LocalPort 8765 240)) { throw "The arena did not become ready. Check $RunDirectory\paper.error.log and paper.log." }
    $Mode = if ($env:MCAI_MODE) { $env:MCAI_MODE.ToLowerInvariant() } else { "sword" }
    if ($Mode -notin @("sword", "crystal", "combined")) { throw "MCAI_MODE must be sword, crystal, or combined." }
    & $Python (Join-Path $Root "scripts\arena_control.py") set_mode ('{"mode":"' + $Mode + '"}') | Out-Null
    & $Python (Join-Path $Root "scripts\arena_control.py") resume | Out-Null
    $Processes += [pscustomobject]@{ Name = "worker"; Process = Start-LoggedProcess "worker" $Node @((Join-Path $Root "worker\dist\src\index.js")) (Join-Path $Root "worker") }
    Save-ProcessState
    if (Wait-LocalPort $DashboardPort 20) {
        $url = "http://127.0.0.1:$DashboardPort"
        Write-Host "Live training view: $url" -ForegroundColor Cyan
        if ($env:MCAI_OPEN_DASHBOARD -ne "false") { Start-Process $url }
    } else { Write-Warning "The dashboard did not open. Training logs are in $RunDirectory." }
    Write-Host "MCAI is running. Double-click STOP_MCAI.cmd or press Ctrl+C for an emergency stop."
    Write-Host "Logs: $RunDirectory"
    while ($true) {
        $Exited = $Processes | Where-Object { $_.Process.HasExited }
        if ($Exited) {
            $Summary = ($Exited | ForEach-Object { "$($_.Name) PID $($_.Process.Id) exit $($_.Process.ExitCode)" }) -join ", "
            throw "A training process stopped unexpectedly: $Summary"
        }
        Start-Sleep -Seconds 1
    }
} finally {
    try {
        & $Python (Join-Path $Root "scripts\arena_control.py") stop_all | Out-Null
    } catch { }
    foreach ($Entry in $Processes) {
        $Process = $Entry.Process
        if ($Process -and -not $Process.HasExited) {
            try { & taskkill.exe /PID $Process.Id /T /F 2>$null | Out-Null } catch { }
        }
    }
    Remove-Item $StatePath -Force -ErrorAction SilentlyContinue
}
