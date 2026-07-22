$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Config = Join-Path $Root "config.windows.ps1"
if (Test-Path $Config) { . $Config }
$Runtime = if ($env:MCAI_RUNTIME) { $env:MCAI_RUNTIME } else { Join-Path $Root "server-runtime" }
$JavaMemory = if ($env:MCAI_JAVA_MEMORY) { $env:MCAI_JAVA_MEMORY } else { "2G" }
$CpuThreads = if ($env:MCAI_TORCH_THREADS) { [int]$env:MCAI_TORCH_THREADS } else { 1 }
$BotCount = if ($env:MCAI_BOT_COUNT) { $env:MCAI_BOT_COUNT } else { "4" }
$RolloutSteps = if ($env:MCAI_ROLLOUT_STEPS) { [int]$env:MCAI_ROLLOUT_STEPS } else { 4096 }
$LearningRate = if ($env:MCAI_LEARNING_RATE) { $env:MCAI_LEARNING_RATE } else { "0.0001" }
$MinibatchSamples = if ($env:MCAI_MINIBATCH_SAMPLES) { [int]$env:MCAI_MINIBATCH_SAMPLES } else { 1024 }
$OptimizationEpochs = if ($env:MCAI_OPTIMIZATION_EPOCHS) { [int]$env:MCAI_OPTIMIZATION_EPOCHS } else { 4 }
$LearnerCpuThreads = if ($env:MCAI_LEARNER_CPU_THREADS) { [int]$env:MCAI_LEARNER_CPU_THREADS } else { 2 }
$TargetKl = if ($env:MCAI_TARGET_KL) { $env:MCAI_TARGET_KL } else { "0.01" }
$FreezePolicy = $env:MCAI_FREEZE_POLICY -and $env:MCAI_FREEZE_POLICY.ToLowerInvariant() -in @("1", "true", "yes", "on")
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

$script:SpectatorProfile = Join-Path $Root ".chrome-mcai-spectator-v4"

function Get-SpectatorChromeProcesses {
    $ProfilePattern = '(?i)(?:^|\s)--user-data-dir="?' + [Regex]::Escape($script:SpectatorProfile) + '"?(?:\s|$)'
    return @(Get-CimInstance Win32_Process -Filter "Name='chrome.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match $ProfilePattern })
}

function Stop-SpectatorChrome {
    # Kill the profile-owning browser trees, then verify that Chrome released
    # the profile. A one-shot process snapshot can miss children created while
    # Chrome is shutting down; the next launch then hands off to that dying
    # process and its window disappears immediately.
    for ($attempt = 0; $attempt -lt 10; $attempt++) {
        $ProfileOwners = @(Get-SpectatorChromeProcesses)
        if ($ProfileOwners.Count -eq 0) { return $true }
        foreach ($Owner in $ProfileOwners) {
            try { & taskkill.exe /PID $Owner.ProcessId /T /F 2>$null | Out-Null } catch { }
        }
        Start-Sleep -Milliseconds 500
    }
    return (@(Get-SpectatorChromeProcesses).Count -eq 0)
}

function Start-SpectatorClient {
    if ($env:MCAI_OPEN_SPECTATOR -eq "false") { return }
    $Client = Join-Path $Root "eagler-client\MCAI-Spectator.html"
    if (-not (Test-Path -LiteralPath $Client)) {
        Write-Warning "Spectator client is missing: $Client"
        return
    }
    $Chrome = "C:\Program Files\Google\Chrome\Application\chrome.exe"
    if (-not (Test-Path -LiteralPath $Chrome)) {
        $ChromeCommand = Get-Command chrome.exe -ErrorAction SilentlyContinue
        if (-not $ChromeCommand) {
            Write-Warning "Google Chrome was not found; open $Client manually."
            return
        }
        $Chrome = $ChromeCommand.Source
    }
    # Keep this isolated from the legacy spectator profile. That profile can
    # retain a broken Eagler handshake/session in IndexedDB and repeatedly time
    # out even when a brand-new profile connects immediately.
    if (-not (Stop-SpectatorChrome)) {
        Write-Warning "The previous spectator browser did not release its profile; retrying shortly."
        return $false
    }
    $ClientUrl = ([Uri]$Client).AbsoluteUri
    $ChromeArgs = @(
        "--remote-debugging-port=9337",
        "--user-data-dir=$($script:SpectatorProfile)",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-extensions",
        "--autoplay-policy=no-user-gesture-required",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--disable-background-timer-throttling",
        "--disable-features=CalculateNativeWinOcclusion",
        "--app=$ClientUrl"
    )
    Start-Process -FilePath $Chrome -ArgumentList $ChromeArgs | Out-Null
    $script:SpectatorLastLaunchAt = Get-Date
    Write-Host "Minecraft spectator opened in Chrome." -ForegroundColor Green
    return $true
}

$script:SpectatorRetryLog = $null
$script:SpectatorRetryAt = $null
$script:SpectatorRetryAttempts = 0
$script:SpectatorRetryLogLine = 0
$script:SpectatorConnected = $false
$script:SpectatorJoinObservedAt = $null
$script:SpectatorJoinName = $null
$script:SpectatorLastLaunchAt = [DateTime]::MinValue
$script:SpectatorNextHealthCheckAt = [DateTime]::MinValue

function Start-SpectatorRetry([string]$PaperLog) {
    if ($env:MCAI_OPEN_SPECTATOR -eq "false") { return }
    $script:SpectatorRetryLog = $PaperLog
    # There is deliberately no wall-clock login timeout. The 75 MB offline
    # client can take several minutes to compile on a cold profile, and killing
    # a healthy Chrome at a deadline races the Eagler login handshake. Retry on
    # an explicit Paper disconnect or when the profile-owning process vanishes.
    $script:SpectatorRetryAt = $null
    $script:SpectatorRetryAttempts = 0
    $script:SpectatorConnected = $false
    $script:SpectatorJoinObservedAt = $null
    $script:SpectatorJoinName = $null
    $script:SpectatorRetryLogLine = if (Test-Path -LiteralPath $PaperLog) {
        @(Get-Content -LiteralPath $PaperLog -ErrorAction SilentlyContinue).Count
    } else { 0 }
}

function Invoke-SpectatorRetry {
    if (-not $script:SpectatorRetryLog) { return }

    if (Test-Path -LiteralPath $script:SpectatorRetryLog) {
        $AllLines = @(Get-Content -LiteralPath $script:SpectatorRetryLog -ErrorAction SilentlyContinue)
        if ($AllLines.Count -lt $script:SpectatorRetryLogLine) {
            $script:SpectatorRetryLogLine = 0
        }
        if ($AllLines.Count -gt $script:SpectatorRetryLogLine) {
            $NewLines = @($AllLines[$script:SpectatorRetryLogLine..($AllLines.Count - 1)])
            $script:SpectatorRetryLogLine = $AllLines.Count
            foreach ($Line in $NewLines) {
                if ($Line -match 'INFO\]: (?!MCAI_)([A-Za-z0-9_]{1,16})\[/127\.0\.0\.1:\d+\] logged in') {
                    # Treat this as a candidate until it survives the old
                    # same-second disconnect race. We continue consuming log
                    # events even after a connection succeeds.
                    $script:SpectatorConnected = $false
                    $script:SpectatorJoinObservedAt = Get-Date
                    $script:SpectatorJoinName = $Matches[1]
                    # A real join supersedes any delayed retry scheduled by
                    # cleanup from the previous Eagler connection.
                    $script:SpectatorRetryAt = $null
                    continue
                }
                $FailureName = $null
                if ($Line -match 'INFO\]: ((?!MCAI_)[A-Za-z0-9_]{1,16}) lost connection:') {
                    $FailureName = $Matches[1]
                } elseif ($Line -match '\[EaglerXServer\] Player ((?!MCAI_)[A-Za-z0-9_]{1,16}) was initialized, but never fired PlayerJoinEvent') {
                    $FailureName = $Matches[1]
                }
                if ($FailureName) {
                    # Late cleanup from an older browser generation must not
                    # tear down a different spectator that has since joined.
                    if ($script:SpectatorJoinName -and
                        -not $FailureName.Equals($script:SpectatorJoinName, [StringComparison]::OrdinalIgnoreCase)) {
                        continue
                    }
                    $script:SpectatorConnected = $false
                    $script:SpectatorJoinObservedAt = $null
                    $script:SpectatorJoinName = $null
                    # Keep the failure pending instead of discarding it during
                    # the post-launch ignore window. The quiet period lets
                    # Netty/Chrome cleanup finish before the profile is reused.
                    $script:SpectatorRetryAt = (Get-Date).AddSeconds(30)
                }
            }
        }
    }

    $Now = Get-Date
    if (-not $script:SpectatorConnected -and $script:SpectatorJoinObservedAt -and
        ($Now - $script:SpectatorJoinObservedAt).TotalSeconds -ge 8) {
        $script:SpectatorConnected = $true
        $script:SpectatorRetryAttempts = 0
        $script:SpectatorRetryAt = $null
        Write-Host "Minecraft spectator connected as $($script:SpectatorJoinName)." -ForegroundColor Green
        # The candidate grace is complete. Clear only its timestamp so the
        # browser health check below continues supervising an established view.
        $script:SpectatorJoinObservedAt = $null
    }
    # A pending join gets an eight-second quiet period. Once connected, keep
    # supervising Chrome as well as Paper so a silent browser exit is repaired.
    if ($script:SpectatorJoinObservedAt) { return }
    $FailureReady = $script:SpectatorRetryAt -and $Now -ge $script:SpectatorRetryAt
    $BrowserMissing = $false
    if ($Now -ge $script:SpectatorNextHealthCheckAt -and
        ($Now - $script:SpectatorLastLaunchAt).TotalSeconds -ge 10) {
        $script:SpectatorNextHealthCheckAt = $Now.AddSeconds(5)
        $BrowserMissing = (@(Get-SpectatorChromeProcesses).Count -eq 0)
    }
    if (-not $FailureReady -and -not $BrowserMissing) { return }

    # Direct-Paper EaglerXServer can occasionally lose the browser's first
    # Netty handoff. Relaunch only after Paper reports that failure or Chrome
    # actually exits, never while a valid client is still booting.
    $script:SpectatorRetryAttempts++
    Write-Host "Retrying Minecraft spectator connection (attempt $script:SpectatorRetryAttempts)..." -ForegroundColor Yellow
    $Started = Start-SpectatorClient
    if (-not $Started) { $script:SpectatorLastLaunchAt = $Now }
    $script:SpectatorRetryAt = $null
}

try {
    # Re-apply the local-only runtime settings on every start. This keeps bot
    # count/max-player changes and fresh random Eagler spectator profiles usable
    # even when the one-time bootstrap has already completed.
    $BindAddress = if ($env:MCAI_BIND_ADDRESS) { $env:MCAI_BIND_ADDRESS } else { "127.0.0.1" }
    & $Python (Join-Path $Root "scripts\configure_runtime.py") $Runtime --bind $BindAddress --bots ([int]$BotCount) | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "Could not update the local Minecraft runtime configuration." }

    $CheckpointPath = if ($env:MCAI_EXPLOITER_TARGET) {
        Join-Path $Root "checkpoints\exploiter-active"
    } else { Join-Path $Root "checkpoints" }
    $TrainerArgs = "-m combat_ai.cli serve --host 127.0.0.1 --port 8766 --checkpoints `"$CheckpointPath`" --cpu-threads $CpuThreads --rollout-steps $RolloutSteps --learning-rate $LearningRate --minibatch-samples $MinibatchSamples --optimization-epochs $OptimizationEpochs --learner-cpu-threads $LearnerCpuThreads --target-kl $TargetKl"
    if ($FreezePolicy) {
        $TrainerArgs += " --freeze-policy"
    }
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
    if ($Mode -notin @("sword", "crystal", "combined", "terrain")) { throw "MCAI_MODE must be sword, crystal, combined, or terrain." }
    # Windows PowerShell 5 strips unescaped double quotes from native-command
    # arguments. Preserve the JSON quotes for Python's json.loads.
    & $Python (Join-Path $Root "scripts\arena_control.py") set_mode ('{\"mode\":\"' + $Mode + '\"}') | Out-Null
    & $Python (Join-Path $Root "scripts\arena_control.py") resume | Out-Null
    $Processes += [pscustomobject]@{ Name = "worker"; Process = Start-LoggedProcess "worker" $Node @((Join-Path $Root "worker\dist\src\index.js")) (Join-Path $Root "worker") }
    Save-ProcessState
    $null = Start-SpectatorClient
    Start-SpectatorRetry (Join-Path $RunDirectory "paper.log")
    if (Wait-LocalPort $DashboardPort 20) {
        $url = "http://127.0.0.1:$DashboardPort"
        Write-Host "Live training view: $url" -ForegroundColor Cyan
        if ($env:MCAI_OPEN_DASHBOARD -ne "false") { Start-Process $url }
    } else { Write-Warning "The dashboard did not open. Training logs are in $RunDirectory." }
    Write-Host "MCAI is running. Double-click STOP_MCAI.cmd or press Ctrl+C for an emergency stop."
    Write-Host "Logs: $RunDirectory"
    while ($true) {
        Invoke-SpectatorRetry
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
