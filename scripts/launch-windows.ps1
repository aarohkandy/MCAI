$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Config = Join-Path $Root "config.windows.ps1"
$State = Join-Path $Root ".mcai\current.json"
$SetupVersion = "2"
$SetupVersionPath = Join-Path $Root ".mcai\setup-version"

function Set-ConfigValue([string]$Name, [string]$Value) {
    $line = '$env:' + $Name + ' = "' + ($Value -replace '"', '`"') + '"'
    $content = if (Test-Path $Config) { Get-Content $Config -Raw } else { "" }
    $pattern = '(?m)^\s*\$env:' + [regex]::Escape($Name) + '\s*=.*$'
    if ($content -match $pattern) { $content = $content -replace $pattern, $line }
    else { $content = $content.TrimEnd() + [Environment]::NewLine + $line + [Environment]::NewLine }
    Set-Content -LiteralPath $Config -Value $content -Encoding UTF8
}

function Test-RunningState {
    if (-not (Test-Path $State)) { return $false }
    try {
        $saved = Get-Content $State -Raw | ConvertFrom-Json
        $entries = @($saved.processes)
        $requiredNames = @("trainer", "paper", "dashboard", "worker")
        if ($entries.Count -lt $requiredNames.Count) { return $false }
        $aliveNames = @()
        foreach ($entry in $entries) {
            $process = Get-Process -Id ([int]$entry.pid) -ErrorAction SilentlyContinue
            if ($process -and $entry.process_started_at) {
                $expected = [datetime]::Parse($entry.process_started_at).ToUniversalTime()
                if ([math]::Abs(($process.StartTime.ToUniversalTime() - $expected).TotalSeconds) -lt 2) {
                    $aliveNames += [string]$entry.name
                }
            }
        }
        return @($requiredNames | Where-Object { $_ -notin $aliveNames }).Count -eq 0
    } catch { }
    return $false
}

function New-DefaultConfig {
    $computer = Get-CimInstance Win32_ComputerSystem
    $ramGb = [math]::Round($computer.TotalPhysicalMemory / 1GB, 1)
    $threads = [Environment]::ProcessorCount
    $javaMemory = if ($ramGb -lt 7) { "1G" } elseif ($ramGb -lt 12) { "2G" } else { "3G" }
    $bots = if ($ramGb -lt 7 -or $threads -lt 4) {
        2
    } elseif ($ramGb -ge 16 -and $threads -ge 8) {
        8
    } else {
        4
    }
    $pairs = [math]::Max(1, [math]::Floor($bots / 2))
    $text = @"
# Created automatically by START_MCAI.cmd. Safe to edit while MCAI is stopped.
`$env:MCAI_RUNTIME = "`$PSScriptRoot\server-runtime"
`$env:MCAI_BIND_ADDRESS = "127.0.0.1"
`$env:MCAI_ACCEPT_EULA = "true"
`$env:MCAI_ACCEPT_TOOL_LICENSES = "false"
`$env:MCAI_BOT_COUNT = "$bots"
`$env:MCAI_INITIAL_PAIRS = "$pairs"
`$env:MCAI_MIN_PAIRS = "$pairs"
`$env:MCAI_JAVA_MEMORY = "$javaMemory"
`$env:MCAI_TORCH_THREADS = "1"
`$env:MCAI_WORKER_ID = "windows-rollout"
`$env:MCAI_SERVER_HOST = "127.0.0.1"
`$env:MCAI_SERVER_PORT = "25565"
`$env:MCAI_ARENA_HOST = "127.0.0.1"
`$env:MCAI_ARENA_PORT = "8765"
`$env:MCAI_TRAINER_URL = "ws://127.0.0.1:8766"
`$env:MCAI_USERNAME_PREFIX = "MCAI_"
`$env:MCAI_MODE = "combined"
`$env:MCAI_ARENA_RADIUS = "5"
`$env:MCAI_ARENA_RADIUS_STAGES = "5,6,7,8"
`$env:MCAI_ARENA_RADIUS_EPISODE_THRESHOLDS = "0,64,256,1024"
`$env:MCAI_ARENA_RADIUS_PERFORMANCE_WINDOW = "32"
`$env:MCAI_ARENA_RADIUS_ADVANCE_ENGAGEMENT_RATE = "0.75"
`$env:MCAI_ARENA_RADIUS_ADVANCE_NON_TIMEOUT_RATE = "0.10"
`$env:MCAI_ARENA_RADIUS_REGRESS_ENGAGEMENT_RATE = "0.50"
`$env:MCAI_ARENA_RADIUS_REGRESS_NON_TIMEOUT_RATE = "0.05"
`$env:MCAI_ARENA_RADIUS_STAGE_CHANGE_COOLDOWN_EPISODES = "32"
`$env:MCAI_ARENA_DEPTH = "12"
`$env:MCAI_ARENA_HEIGHT = "12"
`$env:MCAI_SPAWN_MIN_SEPARATION = "3"
`$env:MCAI_SPAWN_MAX_SEPARATION = "5"
`$env:MCAI_SPAWN_YAW_JITTER_DEGREES = "15.0"
`$env:MCAI_ROLLOUT_STEPS = "4096"
`$env:MCAI_LEARNING_RATE = "0.0001"
`$env:MCAI_MINIBATCH_SAMPLES = "1024"
`$env:MCAI_OPTIMIZATION_EPOCHS = "4"
`$env:MCAI_TARGET_KL = "0.01"
`$env:MCAI_FREEZE_POLICY = "true"
`$env:MCAI_TEACHERS_ENABLED = "false"
`$env:MCAI_DASHBOARD_PORT = "8788"
`$env:MCAI_OPEN_DASHBOARD = "true"
"@
    Set-Content -LiteralPath $Config -Value $text -Encoding UTF8
    Write-Host "Picked safe defaults for $ramGb GB RAM and $threads CPU threads: $bots bots, $javaMemory server memory." -ForegroundColor Green
}

Write-Host ""
Write-Host "MCAI Windows trainer" -ForegroundColor Cyan
Write-Host "===================="

if (Test-RunningState) {
    Write-Host "MCAI is already running. Opening its live view." -ForegroundColor Green
    & (Join-Path $PSScriptRoot "watch-windows.ps1")
    exit 0
}

# A saved run is healthy only when every core component still matches its
# recorded PID and start time. Clean up exact surviving processes from a
# partial run before binding the same local ports again.
if (Test-Path $State) {
    Write-Host "Cleaning up an incomplete MCAI run before restart..." -ForegroundColor Yellow
    & (Join-Path $PSScriptRoot "stop-windows.ps1") | Out-Null
}

if (Test-Path $Config) { . $Config }
if ($env:MCAI_ACCEPT_EULA -ne "true") {
    Write-Host ""
    Write-Host "Minecraft requires the server EULA: https://aka.ms/MinecraftEULA" -ForegroundColor Yellow
    Write-Host "MCAI will create a private, local-only training server and write eula=true."
    $answer = Read-Host "After reviewing it, type I AGREE to accept"
    if ($answer -cne "I AGREE") { throw "The Minecraft EULA was not accepted, so no server was installed." }
    if (-not (Test-Path $Config)) { New-DefaultConfig }
    else { Set-ConfigValue "MCAI_ACCEPT_EULA" "true" }
}
if (-not (Test-Path $Config)) { New-DefaultConfig }
. $Config

& (Join-Path $PSScriptRoot "install-prerequisites-windows.ps1")

$runtime = if ($env:MCAI_RUNTIME) { $env:MCAI_RUNTIME } else { Join-Path $Root "server-runtime" }
$readyPaths = @(
    (Join-Path $Root "trainer\.venv\Scripts\python.exe"),
    (Join-Path $runtime "paper-1.12.2.jar"),
    (Join-Path $runtime "plugins\mcai-arena-0.1.0.jar"),
    (Join-Path $Root "worker\dist\src\index.js")
)
$needsSetup = $false
foreach ($path in $readyPaths) { if (-not (Test-Path $path)) { $needsSetup = $true } }
if (-not (Test-Path $SetupVersionPath) -or (Get-Content $SetupVersionPath -Raw).Trim() -ne $SetupVersion) {
    $needsSetup = $true
}
if ($needsSetup) {
    Write-Host ""
    Write-Host "First-run setup is downloading and building the training environment." -ForegroundColor Cyan
    Write-Host "This is the slow part; future starts reuse it."
    & (Join-Path $PSScriptRoot "bootstrap-windows.ps1")
    New-Item -ItemType Directory -Force -Path (Split-Path $SetupVersionPath) | Out-Null
    Set-Content -Path $SetupVersionPath -Value $SetupVersion -Encoding ASCII
}

Write-Host ""
$displayMode = if ($env:MCAI_MODE) { $env:MCAI_MODE.ToLowerInvariant() } else { "sword" }
Write-Host "Starting $displayMode self-play. The live view will open automatically." -ForegroundColor Green
& (Join-Path $PSScriptRoot "start-windows.ps1")
