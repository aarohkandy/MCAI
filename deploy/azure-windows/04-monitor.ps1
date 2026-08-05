#Requires -Version 5.1
<#
.SYNOPSIS
    Training-health and spend monitor for the unattended MCAI Azure Windows VM.

.DESCRIPTION
    Answers the only question that matters on a pay-by-the-hour box nobody is watching:
    "is training actually progressing, or is it silently broken?"

    Three independent evidence sources are cross-checked, because any one of them can lie:
      1. runs\<run>\trainer.log  - JSONL emitted by the PPO service (rollout_progress /
         ppo_update / trainer_ready). It has no timestamps, so it tells us WHAT but not WHEN.
      2. <checkpoints>\metrics.jsonl - appended exactly once per PPO update, immediately before
         the ppo_update log line. Its mtime is therefore a trustworthy wall clock for
         "when did the last update actually land", surviving trainer restarts.
      3. The arena control socket on 127.0.0.1:8765 - live server TPS, tick latency, heap
         pressure, and how many bot pairs are actually fighting.

    A local state file keeps a rolling (time, ticks) sample ring so that throughput and data
    flow are measured over the recent past instead of over the whole run.

    Billed uptime is NOT derived from that state file. It is reconstructed from Windows' own
    boot/shutdown records in the System event log, so the spend estimate stays correct across
    Spot evictions and reboots even for the hours during which no monitor process was alive.

    -Install registers a Scheduled Task that runs a JSON pass every few minutes at boot as
    SYSTEM. Without it the sample ring has no continuity and the throughput reading only ever
    covers the minutes a human happened to be logged in.

.NOTES
    Windows-only assumptions: Get-CimInstance Win32_OperatingSystem (current boot time),
    Get-WinEvent (uptime reconstruction), and the ScheduledTasks module (-Install/-Uninstall).
    Everything else is portable PowerShell. Written for Windows PowerShell 5.1 as shipped on
    Windows Server, so no PS7-only syntax (no ?:, no ??).

    Exit codes (single-shot mode only; -Watch never exits on its own):
      0 = PASS/INFO   1 = WARN   2 = FAIL   3 = the monitor itself could not run
    Exit 3 must never be written with Write-Error: $ErrorActionPreference='Stop' turns that
    into a terminating error and PowerShell then exits 1, which is indistinguishable from WARN.

    REACHING THE DASHBOARD FROM YOUR OWN MACHINE
    --------------------------------------------
    The dashboard binds to 127.0.0.1:8788 on the VM and is never exposed publicly; the NSG
    only admits RDP/SSH. Forward the port over your existing admin session and browse to the
    tunnel end on your own loopback:

        ssh -N -L 8788:127.0.0.1:8788 azureuser@<vm-public-ip>
        # then open http://127.0.0.1:8788 in your local browser

    With RDP instead of SSH, open the browser inside the RDP session and use the same URL, or
    add a port-forwarded RemoteApp. Never change MCAI_DASHBOARD_HOST away from 127.0.0.1 and
    never open 8788/8765/8766/25565 in the network security group.

.EXAMPLE
    # Run ONCE from an elevated shell after 03-run-training.ps1 -Install.
    .\04-monitor.ps1 -Install -Root C:\MCAI

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File .\04-monitor.ps1

.EXAMPLE
    .\04-monitor.ps1 -Watch -IntervalSeconds 30

.EXAMPLE
    .\04-monitor.ps1 -Json    # one NDJSON object, for a scheduled task or a log shipper
#>
[CmdletBinding()]
param(
    # Repository root. Defaults to the grandparent of this script (deploy\azure-windows\..\..).
    [string]$Root,

    # Specific run directory. Defaults to the runs\* folder with the freshest trainer.log.
    [string]$RunDir,

    # Checkpoint directory holding latest.pt / metrics.jsonl. When omitted, MCAI_CHECKPOINT_DIR,
    # <Root>\checkpoints and every <Root>\checkpoints\<name>\ are considered and the one with the
    # freshest metrics.jsonl wins (exploiter runs write into checkpoints\exploiter-active).
    [string]$CheckpointDir,

    [int]$ArenaPort = 8765,
    [int]$DashboardPort = 8788,

    [switch]$Watch,
    [ValidateRange(5, 3600)][int]$IntervalSeconds = 60,
    [switch]$Json,

    # Append the JSON pass to this file. A Scheduled Task has nowhere to print, so this is the
    # only durable record of what the unattended passes saw.
    [string]$LogPath,
    [ValidateRange(1, 4096)][int]$MaxLogMB = 32,

    # Register / remove the Scheduled Task that keeps the ledger and the sample ring alive.
    [switch]$Install,
    [switch]$Uninstall,
    [string]$TaskName = 'MCAI-Monitor',
    [ValidateRange(1, 60)][int]$TaskIntervalMinutes = 5,

    # Verified eastus2 Standard_F16s_v2 Spot, Windows: the Server licence is inside the VM rate.
    [double]$HourlyRate = 0.6849,
    # 64 GB StandardSSD ~ $5/mo. Disks bill even while the VM is deallocated after an eviction.
    [double]$DiskMonthlyUsd = 5.0,
    [double]$BudgetUsd = 100.0,

    # No ppo_update within this many minutes is treated as a stall. Also governs how long the
    # rollout counter may sit still before "no data is reaching the trainer" becomes a failure.
    [ValidateRange(1, 1440)][int]$StallMinutes = 30,

    # Monitor state (throughput samples + uptime ledger). <Root>\.mcai\monitor-state.json.
    [string]$StatePath,
    [switch]$NoState
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$script:IsWindowsHost = [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT

# ---------------------------------------------------------------- helpers

# Fatal exits must not go through Write-Error: with $ErrorActionPreference='Stop' it throws, the
# `exit 3` after it never runs, and PowerShell exits 1 - the code that means "WARN" here.
function Exit-Monitor {
    param([string]$Message, [int]$Code = 3)
    try { $host.UI.WriteErrorLine("04-monitor: $Message") } catch { Write-Host "04-monitor: $Message" -ForegroundColor Red }
    exit $Code
}

# StrictMode makes `$obj.missing` throw, and every JSON payload here is partly optional
# (an old checkpoint has no imitation_loss, a pre-1.12 plugin has no memory_fraction).
function Get-Field {
    param([object]$Object, [string]$Name, $Default = $null)
    if ($null -eq $Object) { return $Default }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property -or $null -eq $property.Value) { return $Default }
    return $property.Value
}

function ConvertFrom-JsonLine {
    param([string]$Line)
    $trimmed = $Line.Trim()
    # trainer.log interleaves Python tracebacks with JSONL, so anything not object-shaped is noise.
    if ($trimmed.Length -lt 2 -or $trimmed[0] -ne '{') { return $null }
    try { return $trimmed | ConvertFrom-Json } catch { return $null }
}

# Every timestamp this script persists is UTC, but ConvertFrom-Json may hand it back as a
# [datetime] (whose Kind differs between Windows PowerShell 5.1 and PowerShell 7) rather than
# as the original string. Round-tripping such a value through [string] silently shifts it by
# the local UTC offset, so normalise centrally instead.
function ConvertTo-Utc {
    param($Value)
    if ($null -eq $Value) { return $null }
    if ($Value -is [datetime]) {
        if ($Value.Kind -eq [System.DateTimeKind]::Unspecified) {
            return [datetime]::SpecifyKind($Value, [System.DateTimeKind]::Utc)
        }
        return $Value.ToUniversalTime()
    }
    $parsed = [datetime]::MinValue
    $styles = [System.Globalization.DateTimeStyles]::RoundtripKind
    if ([datetime]::TryParse([string]$Value, [System.Globalization.CultureInfo]::InvariantCulture, $styles, [ref]$parsed)) {
        if ($parsed.Kind -eq [System.DateTimeKind]::Unspecified) {
            return [datetime]::SpecifyKind($parsed, [System.DateTimeKind]::Utc)
        }
        return $parsed.ToUniversalTime()
    }
    return $null
}

function Format-Duration {
    param([Nullable[double]]$Minutes)
    if ($null -eq $Minutes) { return 'n/a' }
    if ($Minutes -lt 1) { return '{0:N0}s' -f ($Minutes * 60) }
    if ($Minutes -lt 90) { return '{0:N1}m' -f $Minutes }
    if ($Minutes -lt 2880) { return '{0:N1}h' -f ($Minutes / 60) }
    return '{0:N1}d' -f ($Minutes / 1440)
}

function Format-Number {
    param($Value, [int]$Digits = 4)
    if ($null -eq $Value) { return 'n/a' }
    $number = 0.0
    if (-not [double]::TryParse([string]$Value, [ref]$number)) { return [string]$Value }
    if ([double]::IsNaN($number) -or [double]::IsInfinity($number)) { return 'NaN/Inf' }
    return $number.ToString('N' + $Digits)
}

function Test-Finite {
    param($Value)
    if ($null -eq $Value) { return $false }
    $number = 0.0
    if (-not [double]::TryParse([string]$Value, [ref]$number)) { return $false }
    return -not ([double]::IsNaN($number) -or [double]::IsInfinity($number))
}

function Get-Median {
    param([double[]]$Values)
    if ($null -eq $Values -or $Values.Count -eq 0) { return $null }
    $sorted = @($Values | Sort-Object)
    $middle = [int][math]::Floor($sorted.Count / 2)
    if ($sorted.Count % 2 -eq 1) { return [double]$sorted[$middle] }
    return ([double]$sorted[$middle - 1] + [double]$sorted[$middle]) / 2.0
}

function New-Check {
    param([string]$Name, [string]$Status, [string]$Detail)
    return [pscustomobject]@{ check = $Name; status = $Status; detail = $Detail }
}

# INFO and SKIP rank 0 on purpose: an observation that does not threaten the run must not pin
# `overall` (and therefore the exit code a supervisor branches on) to a permanent WARN.
$script:StatusRank = @{ 'PASS' = 0; 'SKIP' = 0; 'INFO' = 0; 'WARN' = 1; 'FAIL' = 2 }

function ConvertTo-ArgumentString {
    param([string[]]$Arguments)
    # Start-Process / Register-ScheduledTask take one flat string and add no quoting of their own,
    # so any path containing a space would reach the task as two arguments.
    $quoted = foreach ($argument in $Arguments) {
        if ($argument -match '[\s"]') {
            $escaped = $argument -replace '(\\*)"', '$1$1\"'
            # CommandLineToArgvW: a backslash run immediately before the closing quote escapes that
            # quote, so "C:\dir\" would swallow the next token. Double the run.
            $escaped = $escaped -replace '(\\+)$', '$1$1'
            '"' + $escaped + '"'
        } else { $argument }
    }
    return ($quoted -join ' ')
}

# ---------------------------------------------------------------- paths

if (-not $Root) { $Root = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }
if (-not (Test-Path -LiteralPath $Root)) { Exit-Monitor "repository root not found: $Root" 3 }
$Root = (Resolve-Path -LiteralPath $Root).Path

function Resolve-RunDirectory {
    if ($RunDir) {
        if (-not (Test-Path -LiteralPath $RunDir)) { throw "Run directory not found: $RunDir" }
        return (Resolve-Path -LiteralPath $RunDir).Path
    }
    $runsRoot = Join-Path $Root 'runs'
    if (-not (Test-Path -LiteralPath $runsRoot)) { return $null }
    # Freshest trainer.log wins: a resumed run reuses an old folder name but keeps writing.
    $newest = Get-ChildItem -LiteralPath $runsRoot -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { Join-Path $_.FullName 'trainer.log' } |
        Where-Object { Test-Path -LiteralPath $_ } |
        Get-Item |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if ($null -eq $newest) { return $null }
    return $newest.Directory.FullName
}

function Resolve-MetricsPath {
    <#  FRESHEST wins, not first-that-exists. 03-run-training.ps1 sends exploiter runs to
        checkpoints\exploiter-active while a stale checkpoints\metrics.jsonl from an earlier
        non-exploiter run sits in the root forever; picking the first existing candidate reported
        that dead file's age as the stall clock and failed a perfectly healthy exploiter. #>
    # An explicit -CheckpointDir is an operator override and is honoured verbatim.
    if ($CheckpointDir) { return (Join-Path $CheckpointDir 'metrics.jsonl') }
    $candidates = @()
    if ($env:MCAI_CHECKPOINT_DIR) { $candidates += (Join-Path $env:MCAI_CHECKPOINT_DIR 'metrics.jsonl') }
    $candidates += (Join-Path $Root 'checkpoints\metrics.jsonl')
    $checkpointsRoot = Join-Path $Root 'checkpoints'
    if (Test-Path -LiteralPath $checkpointsRoot) {
        foreach ($directory in @(Get-ChildItem -LiteralPath $checkpointsRoot -Directory -ErrorAction SilentlyContinue)) {
            $candidates += (Join-Path $directory.FullName 'metrics.jsonl')
        }
    }
    $best = $null
    foreach ($candidate in $candidates) {
        if (-not (Test-Path -LiteralPath $candidate)) { continue }
        $item = Get-Item -LiteralPath $candidate -ErrorAction SilentlyContinue
        if ($null -eq $item) { continue }
        if ($null -eq $best -or $item.LastWriteTimeUtc -gt $best.LastWriteTimeUtc) { $best = $item }
    }
    if ($null -eq $best) { return $null }
    return $best.FullName
}

if (-not $StatePath) { $StatePath = Join-Path $Root '.mcai\monitor-state.json' }

# ---------------------------------------------------------------- state (throughput ring + ledger floor)

function Get-BootTimeUtc {
    # Windows-only. On any failure we degrade to "no ledger", not to a wrong cost number.
    try { return (Get-CimInstance -ClassName Win32_OperatingSystem -ErrorAction Stop).LastBootUpTime.ToUniversalTime() }
    catch { return $null }
}

function New-MonitorState {
    # A brand new ledger starts at the current boot, not at "now": the bootstrap, the build and
    # the first hours of training all happened in this same billed session.
    $start = (Get-Date).ToUniversalTime()
    $boot = Get-BootTimeUtc
    if ($null -ne $boot -and $boot -lt $start) { $start = $boot }
    return [pscustomobject]@{
        schema                   = 2
        ledger_started_utc       = $start.ToString('o')
        boot_time_utc            = $null
        accumulated_billed_hours = 0.0
        session_hours            = 0.0
        last_observation_utc     = $null
        dashboard_down_passes    = 0
        samples                  = @()
    }
}

function Read-MonitorState {
    if ($NoState -or -not (Test-Path -LiteralPath $StatePath)) { return New-MonitorState }
    try {
        $loaded = Get-Content -LiteralPath $StatePath -Raw -Encoding UTF8 | ConvertFrom-Json
    } catch {
        Write-Warning "Monitor state at $StatePath is unreadable ($($_.Exception.Message)); starting a fresh ledger."
        $loaded = $null
    }
    if ($null -eq $loaded) { return New-MonitorState }
    $ledgerStarted = ConvertTo-Utc (Get-Field $loaded 'ledger_started_utc' $null)
    if ($null -eq $ledgerStarted) { $ledgerStarted = (Get-Date).ToUniversalTime() }
    $recordedBoot = ConvertTo-Utc (Get-Field $loaded 'boot_time_utc' $null)
    $lastObservation = ConvertTo-Utc (Get-Field $loaded 'last_observation_utc' $null)
    $bootText = $null
    if ($null -ne $recordedBoot) { $bootText = $recordedBoot.ToString('o') }
    $observationText = $null
    if ($null -ne $lastObservation) { $observationText = $lastObservation.ToString('o') }
    return [pscustomobject]@{
        schema                   = 2
        ledger_started_utc       = $ledgerStarted.ToString('o')
        boot_time_utc            = $bootText
        accumulated_billed_hours = [double](Get-Field $loaded 'accumulated_billed_hours' 0.0)
        session_hours            = [double](Get-Field $loaded 'session_hours' 0.0)
        last_observation_utc     = $observationText
        dashboard_down_passes    = [int](Get-Field $loaded 'dashboard_down_passes' 0)
        samples                  = @(Get-Field $loaded 'samples' @())
    }
}

function Save-MonitorState {
    param([object]$State)
    if ($NoState) { return }
    try {
        $directory = Split-Path -Parent $StatePath
        if ($directory -and -not (Test-Path -LiteralPath $directory)) {
            New-Item -ItemType Directory -Force -Path $directory | Out-Null
        }
        # Atomic-ish: a Spot eviction mid-write must not leave a truncated ledger behind.
        $temporary = "$StatePath.tmp"
        ($State | ConvertTo-Json -Depth 5) | Set-Content -LiteralPath $temporary -Encoding UTF8
        Move-Item -LiteralPath $temporary -Destination $StatePath -Force
    } catch {
        Write-Warning "Could not persist monitor state to ${StatePath}: $($_.Exception.Message)"
    }
}

function Get-BilledUptime {
    <#  Reconstruct how many hours this VM has actually been running since the ledger started,
        from Windows' own boot/shutdown records - NOT from a counter that only advances while a
        monitor process happens to be alive. Three days of unattended training followed by a Spot
        eviction and a restart must show up in full even if nobody ran the monitor in between. #>
    param([datetime]$SinceUtc, [datetime]$NowUtc, $BootTimeUtc, $LastObservationUtc)

    $result = [pscustomobject]@{ hours = $null; complete = $false; boots = 0 }
    if (-not $script:IsWindowsHost) { return $result }

    $records = New-Object System.Collections.ArrayList
    $queried = $false
    # Kernel-General 12/13 are "OS started"/"OS shutting down"; the classic EventLog 6005/6006/6008
    # pair covers hosts where the Kernel-General channel has been trimmed. Query per provider: other
    # providers also emit ids 12 and 13 into the System log.
    foreach ($spec in @(
            @{ Provider = 'Microsoft-Windows-Kernel-General'; Up = @(12); Down = @(13) },
            @{ Provider = 'EventLog'; Up = @(6005); Down = @(6006, 6008) })) {
        $found = @()
        try {
            $filter = @{ LogName = 'System'; ProviderName = $spec.Provider; Id = @($spec.Up + $spec.Down); StartTime = $SinceUtc.ToLocalTime() }
            $found = @(Get-WinEvent -FilterHashtable $filter -ErrorAction Stop)
            $queried = $true
        } catch {
            # "No events were found" is the normal answer on a VM that has never rebooted. Match on
            # the error id, not the message: the message is localised on a non-English host.
            if ($_.FullyQualifiedErrorId -like 'NoMatchingEventsFound*' -or $_.Exception.Message -match 'No events were found') { $queried = $true }
            $found = @()
        }
        foreach ($record in $found) {
            [void]$records.Add([pscustomobject]@{
                time = $record.TimeCreated.ToUniversalTime()
                up   = ($spec.Up -contains [int]$record.Id)
            })
        }
    }
    if (-not $queried) { return $result }

    $ordered = @($records | Sort-Object time)
    # The ledger is created by a monitor pass running ON the VM, so the VM was up at SinceUtc.
    $upSince = $SinceUtc
    $total = 0.0
    $boots = 0
    $previousTime = $null
    $previousClass = $null
    foreach ($item in $ordered) {
        # Both providers log the same boot (or the same shutdown) seconds apart; count it once.
        if ($null -ne $previousTime -and $item.up -eq $previousClass -and ($item.time - $previousTime).TotalSeconds -lt 60) { continue }
        $previousTime = $item.time
        $previousClass = $item.up
        if ($item.up) {
            $boots = $boots + 1
            if ($null -ne $upSince) {
                # An eviction or power loss writes no shutdown record, so the interval before this
                # boot has no end. Close it at the last moment we have evidence the VM was alive
                # rather than at the boot itself, so a long deallocation is not billed as uptime.
                $closeAt = $item.time
                if ($null -ne $LastObservationUtc -and $LastObservationUtc -gt $upSince -and $LastObservationUtc -lt $item.time) {
                    $closeAt = $LastObservationUtc.AddMinutes(10)
                    if ($closeAt -gt $item.time) { $closeAt = $item.time }
                }
                $total = $total + [math]::Max(0.0, ($closeAt - $upSince).TotalHours)
            }
            $upSince = $item.time
        } else {
            if ($null -ne $upSince) { $total = $total + [math]::Max(0.0, ($item.time - $upSince).TotalHours) }
            $upSince = $null
        }
    }
    if ($null -ne $upSince) { $total = $total + [math]::Max(0.0, ($NowUtc - $upSince).TotalHours) }

    # Truncation check: if the OS says we booted after the ledger started but the log has no boot
    # record anywhere near that moment, older records have been rotated away and the sum is a floor.
    $complete = $true
    if ($null -ne $BootTimeUtc -and $BootTimeUtc -gt $SinceUtc.AddMinutes(2)) {
        $matched = @($ordered | Where-Object { $_.up -and [math]::Abs(($_.time - $BootTimeUtc).TotalMinutes) -lt 5 })
        if ($matched.Count -eq 0) { $complete = $false }
    }

    $result.hours = $total
    $result.complete = $complete
    $result.boots = $boots
    return $result
}

# ---------------------------------------------------------------- trainer log

function Read-TrainerLog {
    param([string]$Directory)
    $result = [pscustomobject]@{
        path              = $null
        exists            = $false
        log_age_minutes   = $null
        policy_version    = $null
        total_agent_ticks = $null
        collected_ticks   = $null
        target_ticks      = $null
        device            = $null
        updates           = @()
        traceback_count   = 0
        last_error        = $null
    }
    if (-not $Directory) { return $result }
    $path = Join-Path $Directory 'trainer.log'
    if (-not (Test-Path -LiteralPath $path)) { return $result }
    $result.path = $path
    $result.exists = $true
    $result.log_age_minutes = ((Get-Date).ToUniversalTime() - (Get-Item -LiteralPath $path).LastWriteTimeUtc).TotalMinutes

    # Bounded read: trainer.log grows without limit over a multi-day run.
    $tail = @(Get-Content -LiteralPath $path -Tail 4000 -ErrorAction SilentlyContinue)
    $updates = New-Object System.Collections.ArrayList
    foreach ($line in $tail) {
        if ($line -like 'Traceback (most recent call last):*') {
            $result.traceback_count = $result.traceback_count + 1
            continue
        }
        if ($line -match '^(?<type>\w*(Error|Exception|Warning)):\s') { $result.last_error = $line.Trim() }
        $message = ConvertFrom-JsonLine $line
        if ($null -eq $message) { continue }
        # Not $event: that name collides with PowerShell's automatic event variable.
        $eventName = [string](Get-Field $message 'event' '')
        switch ($eventName) {
            'ppo_update' {
                [void]$updates.Add($message)
                $result.policy_version = Get-Field $message 'policy_version' $result.policy_version
                $result.total_agent_ticks = Get-Field $message 'total_agent_ticks' $result.total_agent_ticks
                # The buffer was just drained into this update; the rollout restarts from zero.
                $result.collected_ticks = 0
            }
            'rollout_progress' {
                $result.policy_version = Get-Field $message 'policy_version' $result.policy_version
                $result.total_agent_ticks = Get-Field $message 'total_agent_ticks' $result.total_agent_ticks
                $result.collected_ticks = Get-Field $message 'collected_agent_ticks' $null
                $result.target_ticks = Get-Field $message 'target_agent_ticks' $null
            }
            'ppo_training_started' {
                $result.total_agent_ticks = Get-Field $message 'total_agent_ticks' $result.total_agent_ticks
                $result.collected_ticks = 0
            }
            'trainer_ready' {
                $result.device = Get-Field $message 'device' $result.device
                if ($null -eq $result.policy_version) { $result.policy_version = Get-Field $message 'policy_version' $null }
            }
        }
    }
    $result.updates = @($updates | Select-Object -Last 8)
    return $result
}

function Read-MetricsFile {
    param([string]$Path)
    $result = [pscustomobject]@{
        path                = $Path
        exists              = $false
        last_update_age_min = $null
        entropy_series      = @()
        first_ticks         = $null
        last_ticks          = $null
        created_utc         = $null
        modified_utc        = $null
        latest              = $null
    }
    if (-not $Path -or -not (Test-Path -LiteralPath $Path)) { return $result }
    $result.exists = $true
    $item = Get-Item -LiteralPath $Path
    # metrics.jsonl is appended once per PPO update, so its mtime IS the last-update wall clock.
    $result.created_utc = $item.CreationTimeUtc
    $result.modified_utc = $item.LastWriteTimeUtc
    $result.last_update_age_min = ((Get-Date).ToUniversalTime() - $item.LastWriteTimeUtc).TotalMinutes

    $first = Get-Content -LiteralPath $Path -TotalCount 1 -ErrorAction SilentlyContinue
    if ($first) {
        $parsed = ConvertFrom-JsonLine ([string]$first)
        if ($null -ne $parsed) { $result.first_ticks = Get-Field $parsed 'ticks' $null }
    }
    # Bounded tail: enough updates to see a trend, cheap enough to read every minute.
    $series = New-Object System.Collections.ArrayList
    foreach ($line in @(Get-Content -LiteralPath $Path -Tail 300 -ErrorAction SilentlyContinue)) {
        $parsed = ConvertFrom-JsonLine ([string]$line)
        if ($null -eq $parsed) { continue }
        $result.latest = $parsed
        $result.last_ticks = Get-Field $parsed 'ticks' $result.last_ticks
        $entropy = Get-Field $parsed 'entropy' $null
        if (Test-Finite $entropy) { [void]$series.Add([double]$entropy) }
    }
    $result.entropy_series = @($series)
    return $result
}

# ---------------------------------------------------------------- arena control socket

function Get-ArenaStatus {
    # The plugin schedules `status` on the Bukkit main thread and waits up to 10 s for it
    # (ControlServer.dispatch). Giving up sooner would report an overloaded-but-alive server as
    # dead - and that is exactly the condition the TPS/heap checks below exist to diagnose.
    param([int]$Port, [int]$TimeoutSeconds = 15)
    $client = New-Object System.Net.Sockets.TcpClient
    $reader = $null
    $writer = $null
    try {
        $connected = $false
        try {
            $connect = $client.ConnectAsync('127.0.0.1', $Port)
            $connected = ($connect.Wait(3000) -and $client.Connected)
        } catch {
            # Wait() surfaces a connection refusal as an AggregateException; the wrapper text is noise.
            $connected = $false
        }
        if (-not $connected) {
            return [pscustomobject]@{ ok = $false; kind = 'refused'; error = "no listener on 127.0.0.1:$Port"; payload = $null }
        }
        $stream = $client.GetStream()
        $stream.ReadTimeout = 1500
        $encoding = New-Object System.Text.UTF8Encoding($false)
        $writer = New-Object System.IO.StreamWriter($stream, $encoding)
        $writer.NewLine = "`n"
        $writer.AutoFlush = $true
        $reader = New-Object System.IO.StreamReader($stream, $encoding)

        $requestId = Get-Random -Minimum 1000 -Maximum 2147483000
        $request = @{ type = 'command'; id = $requestId; command = 'status'; payload = @{} } | ConvertTo-Json -Compress
        $writer.WriteLine($request)

        # The server broadcasts async {"type":"event"} lines to EVERY client, so the first line
        # back is very often somebody else's arena_snapshot. Match on type+id, never on arrival.
        $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
        while ((Get-Date) -lt $deadline) {
            $line = $null
            try { $line = $reader.ReadLine() }
            catch [System.IO.IOException] { continue }   # ReadTimeout: keep waiting until the deadline
            if ($null -eq $line) { break }               # peer closed
            $message = ConvertFrom-JsonLine $line
            if ($null -eq $message) { continue }
            if ([string](Get-Field $message 'type' '') -ne 'response') { continue }
            if ([int](Get-Field $message 'id' (-1)) -ne $requestId) { continue }
            $ok = [bool](Get-Field $message 'ok' $false)
            return [pscustomobject]@{
                ok      = $ok
                kind    = 'answered'
                error   = [string](Get-Field $message 'error' '')
                payload = Get-Field $message 'payload' $null
            }
        }
        # Connected, but the main thread never serviced the sync task.
        return [pscustomobject]@{ ok = $false; kind = 'blocked'; error = "no response within ${TimeoutSeconds}s"; payload = $null }
    } catch {
        return [pscustomobject]@{ ok = $false; kind = 'error'; error = $_.Exception.Message; payload = $null }
    } finally {
        if ($reader) { $reader.Dispose() }
        if ($writer) { $writer.Dispose() }
        $client.Dispose()
    }
}

function Test-DashboardPort {
    param([int]$Port)
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $connect = $client.ConnectAsync('127.0.0.1', $Port)
        return ($connect.Wait(1500) -and $client.Connected)
    } catch { return $false } finally { $client.Dispose() }
}

# ---------------------------------------------------------------- one sampling pass

function Invoke-HealthPass {
    param([object]$State)

    $now = (Get-Date).ToUniversalTime()
    $runDirectory = Resolve-RunDirectory
    $trainer = Read-TrainerLog -Directory $runDirectory
    $metricsPath = Resolve-MetricsPath
    $metrics = Read-MetricsFile -Path $metricsPath
    $arena = Get-ArenaStatus -Port $ArenaPort
    $arenaPayload = $null
    if ($arena.ok) { $arenaPayload = $arena.payload }

    # ---- billed-uptime ledger -------------------------------------------------
    $ledgerStart = ConvertTo-Utc $State.ledger_started_utc
    if ($null -eq $ledgerStart) { $ledgerStart = $now; $State.ledger_started_utc = $now.ToString('o') }
    $lastObservation = ConvertTo-Utc $State.last_observation_utc
    $bootTime = Get-BootTimeUtc
    $sessionHours = $null
    if ($null -ne $bootTime) {
        $sessionHours = [math]::Max(0.0, ($now - $bootTime).TotalHours)
        $recordedBoot = ConvertTo-Utc $State.boot_time_utc
        if ($null -eq $recordedBoot) {
            $State.boot_time_utc = $bootTime.ToString('o')
        } elseif ([math]::Abs($recordedBoot.Subtract($bootTime).TotalMinutes) -gt 2) {
            # Reboot or Spot eviction+restart: bank the previous session before resetting.
            $State.accumulated_billed_hours = $State.accumulated_billed_hours + $State.session_hours
            $State.boot_time_utc = $bootTime.ToString('o')
        }
        $State.session_hours = $sessionHours
    }
    # The banked ledger is only ever a FLOOR (it advances solely while the monitor is running).
    # The event-log reconstruction is the real number; take whichever is larger so a gap in either
    # source can only over-state spend, never under-state it.
    $ledgerFloor = [double]$State.accumulated_billed_hours
    if ($null -ne $sessionHours) { $ledgerFloor = $ledgerFloor + $sessionHours }
    $uptime = Get-BilledUptime -SinceUtc $ledgerStart -NowUtc $now -BootTimeUtc $bootTime -LastObservationUtc $lastObservation
    $billedHours = $ledgerFloor
    $uptimeSource = 'monitor-ledger'
    $uptimeComplete = $false
    if ($null -ne $uptime.hours) {
        $uptimeSource = 'system-event-log'
        $uptimeComplete = $uptime.complete
        if ($uptime.hours -gt $billedHours) { $billedHours = [double]$uptime.hours }
    }
    $State.last_observation_utc = $now.ToString('o')

    # ---- throughput / data-flow sample ring -----------------------------------
    $totalTicks = $trainer.total_agent_ticks
    if ($null -eq $totalTicks -and $metrics.exists) { $totalTicks = $metrics.last_ticks }
    # total_agent_ticks only moves once per PPO update, so it is flat by design for the whole
    # rollout. collected_agent_ticks moves every few seconds while bots produce transitions and
    # resets on drain, so total+collected is a monotone counter that tracks data flow directly.
    $flowTicks = $null
    if ($null -ne $totalTicks) {
        $flowTicks = [double]$totalTicks
        if ($null -ne $trainer.collected_ticks) { $flowTicks = $flowTicks + [double]$trainer.collected_ticks }
    }

    $samples = @($State.samples)
    $recentTicksPerHour = $null
    $flowDelta = $null
    $flowPerHour = $null
    $sampleSpanMinutes = $null
    if ($null -ne $totalTicks) {
        # Compare against the oldest sample still inside a 2-hour window: long enough to be
        # stable, short enough that a stall two days ago cannot mask progress right now.
        $window = $now.AddHours(-2)
        $reference = $null
        foreach ($sample in $samples) {
            $stamp = ConvertTo-Utc (Get-Field $sample 't' $null)
            if ($null -eq $stamp -or $stamp -lt $window -or $stamp -ge $now) { continue }
            if ($null -eq $reference) {
                $referenceTicks = [double](Get-Field $sample 'ticks' 0)
                # Samples written before the flow counter existed only carry 'ticks'.
                $referenceFlow = [double](Get-Field $sample 'flow' $referenceTicks)
                $reference = [pscustomobject]@{ time = $stamp; ticks = $referenceTicks; flow = $referenceFlow }
            }
        }
        if ($null -ne $reference) {
            $span = ($now - $reference.time).TotalMinutes
            if ($span -gt 0) {
                $sampleSpanMinutes = $span
                if ($span -gt 1) { $recentTicksPerHour = ([double]$totalTicks - $reference.ticks) / ($span / 60.0) }
                if ($null -ne $flowTicks) {
                    $flowDelta = $flowTicks - $reference.flow
                    if ($span -gt 1) { $flowPerHour = $flowDelta / ($span / 60.0) }
                }
            }
        }
        $newSample = [pscustomobject]@{ t = $now.ToString('o'); ticks = [double]$totalTicks; v = $trainer.policy_version }
        if ($null -ne $flowTicks) { Add-Member -InputObject $newSample -NotePropertyName 'flow' -NotePropertyValue $flowTicks }
        $samples += , $newSample
        # Keep roughly a day of one-minute samples.
        if ($samples.Count -gt 1440) { $samples = @($samples | Select-Object -Last 1440) }
        $State.samples = $samples
    }

    # First-invocation throughput. Derived from metrics.jsonl only, where the tick delta and the
    # elapsed time span the SAME interval. (The old run-directory-age fallback divided ticks
    # accumulated over days by the age of a run folder recreated on every restart, which inflated
    # the figure by one to two orders of magnitude exactly when it was the only number shown.)
    $lifetimeTicksPerHour = $null
    if ($metrics.exists -and $null -ne $metrics.first_ticks -and $null -ne $metrics.last_ticks -and
        $null -ne $metrics.created_utc -and $null -ne $metrics.modified_utc) {
        $lifetimeHours = ($metrics.modified_utc - $metrics.created_utc).TotalHours
        $lifetimeDelta = [double]$metrics.last_ticks - [double]$metrics.first_ticks
        if ($lifetimeHours -gt 0.1 -and $lifetimeDelta -gt 0) { $lifetimeTicksPerHour = $lifetimeDelta / $lifetimeHours }
    }

    # ---- health checks --------------------------------------------------------
    $checks = New-Object System.Collections.ArrayList
    $latest = $null
    if ($trainer.updates.Count -gt 0) { $latest = $trainer.updates[-1] }
    elseif ($metrics.exists) { $latest = $metrics.latest }

    # (a) training stalled. metrics.jsonl's mtime is the exact wall clock of the last update;
    # trainer.log's mtime is a looser upper bound (it also moves on rollout_progress lines).
    $updateAge = $metrics.last_update_age_min
    if ($null -eq $updateAge) { $updateAge = $trainer.log_age_minutes }
    $everUpdated = ($metrics.exists -or $trainer.updates.Count -gt 0)
    if (-not $trainer.exists -and -not $metrics.exists) {
        [void]$checks.Add((New-Check 'trainer.progress' 'FAIL' 'No trainer.log and no metrics.jsonl under any checkpoint directory - training has never started on this VM.'))
    } elseif ($null -eq $updateAge) {
        [void]$checks.Add((New-Check 'trainer.progress' 'WARN' 'Cannot determine when the last PPO update landed.'))
    } elseif (-not $everUpdated) {
        $verdict = 'WARN'
        if ($updateAge -gt $StallMinutes) { $verdict = 'FAIL' }
        [void]$checks.Add((New-Check 'trainer.progress' $verdict ("No PPO update has completed yet; trainer.log last written {0} ago (the first rollout is slow, but not this slow past {1}m)." -f (Format-Duration $updateAge), $StallMinutes)))
    } elseif ($updateAge -gt $StallMinutes) {
        [void]$checks.Add((New-Check 'trainer.progress' 'FAIL' ("Training stalled: no ppo_update for {0} in {1} (threshold {2}m). Check the trainer and worker logs in {3}." -f (Format-Duration $updateAge), $metricsPath, $StallMinutes, $runDirectory)))
    } elseif ($updateAge -gt ($StallMinutes / 2.0)) {
        [void]$checks.Add((New-Check 'trainer.progress' 'WARN' ("Last ppo_update was {0} ago (stall threshold {1}m)." -f (Format-Duration $updateAge), $StallMinutes)))
    } else {
        [void]$checks.Add((New-Check 'trainer.progress' 'PASS' ("Last ppo_update {0} ago, policy_version {1}." -f (Format-Duration $updateAge), $trainer.policy_version)))
    }

    # (b) data actually flowing. Judged on the rollout counter, which advances continuously while
    # bots produce transitions - not on total_agent_ticks, which is a once-per-update step function
    # and therefore looks identical for "slow update" and "every bot disconnected".
    if ($null -eq $flowDelta -or $null -eq $sampleSpanMinutes) {
        $detail = 'Not enough history yet; run the monitor again in a few minutes for a throughput reading.'
        if ($null -ne $lifetimeTicksPerHour) { $detail = "{0} (lifetime average {1:N0} ticks/h)." -f $detail, $lifetimeTicksPerHour }
        [void]$checks.Add((New-Check 'data.flow' 'SKIP' $detail))
    } elseif ($flowDelta -gt 0) {
        [void]$checks.Add((New-Check 'data.flow' 'PASS' ("{0:N0} agent ticks/hour reaching the trainer over the last {1}." -f $flowPerHour, (Format-Duration $sampleSpanMinutes))))
    } elseif ($sampleSpanMinutes -ge $StallMinutes) {
        [void]$checks.Add((New-Check 'data.flow' 'FAIL' ("The rollout buffer has not grown in {0} (threshold {1}m) - no data is reaching the trainer (bots disconnected, worker dead, or arena paused)." -f (Format-Duration $sampleSpanMinutes), $StallMinutes)))
    } else {
        [void]$checks.Add((New-Check 'data.flow' 'WARN' ("Rollout collection flat over the last {0}." -f (Format-Duration $sampleSpanMinutes))))
    }

    # (c) entropy. Falling entropy is what a converging PPO policy is SUPPOSED to do, so this is
    # judged against a trailing baseline rather than the all-time peak: an append-only metrics.jsonl
    # keeps the first update of the first run forever, and comparing against it turns the entire
    # normal trajectory into a permanent FAIL. Only a fast relative drop is a real signal.
    $entropy = Get-Field $latest 'entropy' $null
    $entropySeries = @($metrics.entropy_series)
    if (-not (Test-Finite $entropy)) {
        if ($null -eq $latest) { [void]$checks.Add((New-Check 'policy.entropy' 'SKIP' 'No PPO metrics yet.')) }
        else { [void]$checks.Add((New-Check 'policy.entropy' 'FAIL' "entropy is not a finite number ($entropy) - the policy has diverged.")) }
    } else {
        $entropyValue = [double]$entropy
        $baseline = $null
        $baselineCount = 0
        $count = $entropySeries.Count
        # Baseline = median of up to 60 updates ending 20 updates back; recent = median of the last
        # 5. At a typical update cadence that puts the baseline a few hours in the past.
        if ($count -ge 25) {
            $trailEnd = $count - 20
            $trailStart = [math]::Max(0, $trailEnd - 60)
            $baselineCount = $trailEnd - $trailStart
            $baseline = Get-Median ([double[]]@($entropySeries[$trailStart..($trailEnd - 1)]))
            $recentValue = Get-Median ([double[]]@($entropySeries[($count - 5)..($count - 1)]))
            if ($null -ne $recentValue) { $entropyValue = [double]$recentValue }
        }
        if ($entropyValue -lt 0.01) {
            [void]$checks.Add((New-Check 'policy.entropy' 'FAIL' ("Entropy collapsed to {0:N4} - the policy is degenerate (it always picks the same action)." -f $entropyValue)))
        } elseif ($null -eq $baseline -or [double]$baseline -le 1e-9) {
            [void]$checks.Add((New-Check 'policy.entropy' 'PASS' ("Entropy {0:N3} ({1} updates recorded; {2} needed for a trend baseline)." -f $entropyValue, $count, 25)))
        } else {
            $ratio = $entropyValue / [double]$baseline
            if ($ratio -lt 0.15) {
                [void]$checks.Add((New-Check 'policy.entropy' 'FAIL' ("Entropy fell to {0:N3}, {1:P0} of the {2:N3} median only {3} updates ago - that is a collapse, not convergence." -f $entropyValue, $ratio, $baseline, ($baselineCount + 20))))
            } elseif ($ratio -lt 0.40) {
                [void]$checks.Add((New-Check 'policy.entropy' 'WARN' ("Entropy {0:N3} is {1:P0} of the {2:N3} median from {3} updates ago - unusually fast convergence." -f $entropyValue, $ratio, $baseline, ($baselineCount + 20))))
            } else {
                [void]$checks.Add((New-Check 'policy.entropy' 'PASS' ("Entropy {0:N3} (trailing baseline {1:N3}, {2:P0})." -f $entropyValue, $baseline, $ratio)))
            }
        }
    }

    # (d) KL divergence exploding
    $klValues = New-Object System.Collections.ArrayList
    foreach ($update in $trainer.updates) {
        $candidate = Get-Field $update 'approximate_kl' (Get-Field $update 'approx_kl' $null)
        if (Test-Finite $candidate) { [void]$klValues.Add([double]$candidate) }
    }
    $latestKl = Get-Field $latest 'approximate_kl' (Get-Field $latest 'approx_kl' $null)
    if ($klValues.Count -eq 0 -and -not (Test-Finite $latestKl)) {
        if ($null -eq $latest) { [void]$checks.Add((New-Check 'policy.kl' 'SKIP' 'No PPO metrics yet.')) }
        else { [void]$checks.Add((New-Check 'policy.kl' 'FAIL' 'approximate_kl is NaN/Inf - the update diverged.')) }
    } else {
        $recentKl = @($klValues | Select-Object -Last 5)
        $meanKl = if ($recentKl.Count -gt 0) { ($recentKl | Measure-Object -Average).Average } else { [double]$latestKl }
        $maxKl = if ($recentKl.Count -gt 0) { ($recentKl | Measure-Object -Maximum).Maximum } else { [double]$latestKl }
        # PPO targets ~0.01-0.02; sustained >0.05 means the clip range is no longer holding.
        if ($maxKl -gt 0.2) {
            [void]$checks.Add((New-Check 'policy.kl' 'FAIL' ("approximate_kl spiked to {0:N4} (mean {1:N4}) - training is diverging; consider restarting from the last good snapshot." -f $maxKl, $meanKl)))
        } elseif ($meanKl -gt 0.05) {
            [void]$checks.Add((New-Check 'policy.kl' 'WARN' ("approximate_kl averaging {0:N4} over the last {1} updates (healthy is <=0.02)." -f $meanKl, $recentKl.Count)))
        } else {
            [void]$checks.Add((New-Check 'policy.kl' 'PASS' ("approximate_kl {0:N4} (mean of last {1})." -f $meanKl, [math]::Max(1, $recentKl.Count))))
        }
    }

    # Loss sanity: NaN here poisons every future checkpoint, so it is a hard failure.
    $badLoss = @()
    foreach ($name in @('policy_loss', 'value_loss')) {
        $value = Get-Field $latest $name $null
        if ($null -ne $value -and -not (Test-Finite $value)) { $badLoss += $name }
    }
    if ($badLoss.Count -gt 0) {
        [void]$checks.Add((New-Check 'policy.loss' 'FAIL' ("Non-finite {0} in the latest update - stop training; the checkpoint is poisoned." -f ($badLoss -join ' and '))))
    } elseif ($null -ne $latest) {
        [void]$checks.Add((New-Check 'policy.loss' 'PASS' ("policy_loss {0}, value_loss {1}." -f (Format-Number (Get-Field $latest 'policy_loss' $null) 5), (Format-Number (Get-Field $latest 'value_loss' $null) 6))))
    }

    # (e/f) arena health
    if (-not $arena.ok) {
        if ($arena.kind -eq 'blocked') {
            # Connected but unanswered: the socket thread is alive and the main thread is not.
            [void]$checks.Add((New-Check 'arena.control' 'FAIL' ("Arena control on 127.0.0.1:{0} accepted the connection but the server thread did not answer the status command ({1}) - Paper's main thread is blocked or GC-thrashing. Reduce MCAI_MAX_PAIRS / MCAI_BOT_COUNT or raise MCAI_JAVA_MEMORY." -f $ArenaPort, $arena.error)))
        } else {
            [void]$checks.Add((New-Check 'arena.control' 'FAIL' ("Arena control socket 127.0.0.1:{0} unreachable: {1}. Paper is down or still starting." -f $ArenaPort, $arena.error)))
        }
    } else {
        [void]$checks.Add((New-Check 'arena.control' 'PASS' ("Arena control responded on 127.0.0.1:{0}." -f $ArenaPort)))

        if ([bool](Get-Field $arenaPayload 'paused' $false)) {
            [void]$checks.Add((New-Check 'arena.paused' 'FAIL' 'Arena manager is emergency-stopped. Send `resume` via scripts\arena_control.py to restart matches.'))
        }

        $activePairs = Get-Field $arenaPayload 'active_pairs' $null
        $maxPairs = Get-Field $arenaPayload 'max_concurrent_pairs' $null
        if ($null -eq $activePairs) {
            [void]$checks.Add((New-Check 'arena.pairs' 'WARN' 'Arena status has no active_pairs field.'))
        } elseif ([int]$activePairs -eq 0) {
            [void]$checks.Add((New-Check 'arena.pairs' 'FAIL' 'active_pairs = 0: no bots are paired. Check that the worker is running and that bots joined (worker.log in the run directory).'))
        } elseif ($null -ne $maxPairs -and [int]$maxPairs -gt 1 -and [int]$activePairs -le 1) {
            [void]$checks.Add((New-Check 'arena.pairs' 'WARN' ("Only {0} of {1} pairs active - the load controller is shedding arenas, or bots are failing to join." -f $activePairs, $maxPairs)))
        } else {
            [void]$checks.Add((New-Check 'arena.pairs' 'PASS' ("{0} of {1} pairs fighting." -f $activePairs, $maxPairs)))
        }

        $tps = Get-Field $arenaPayload 'estimated_tps' $null
        if (-not (Test-Finite $tps)) {
            [void]$checks.Add((New-Check 'arena.tps' 'SKIP' 'estimated_tps not reported yet.'))
        } elseif ([double]$tps -lt 15.0) {
            [void]$checks.Add((New-Check 'arena.tps' 'FAIL' ("Server at {0:N1} TPS (target 20). The observations the trainer is learning from are time-distorted; reduce MCAI_MAX_PAIRS or MCAI_BOT_COUNT." -f $tps)))
        } elseif ([double]$tps -lt 19.0) {
            [void]$checks.Add((New-Check 'arena.tps' 'WARN' ("Server at {0:N1} TPS (target 20)." -f $tps)))
        } else {
            [void]$checks.Add((New-Check 'arena.tps' 'PASS' ("{0:N1} TPS." -f $tps)))
        }

        # 50 ms is one tick; the worker's load controller sheds arenas past 55 ms.
        $p95 = Get-Field $arenaPayload 'p95_tick_ms' $null
        if (Test-Finite $p95) {
            if ([double]$p95 -gt 100.0) {
                [void]$checks.Add((New-Check 'arena.tick_latency' 'FAIL' ("p95 tick {0:N1} ms (one tick is 50 ms) - the server is badly overloaded." -f $p95)))
            } elseif ([double]$p95 -gt 55.0) {
                [void]$checks.Add((New-Check 'arena.tick_latency' 'WARN' ("p95 tick {0:N1} ms - above the 55 ms shed threshold." -f $p95)))
            } else {
                [void]$checks.Add((New-Check 'arena.tick_latency' 'PASS' ("p95 tick {0:N1} ms." -f $p95)))
            }
        }

        $memoryFraction = Get-Field $arenaPayload 'memory_fraction' $null
        if (Test-Finite $memoryFraction) {
            if ([double]$memoryFraction -gt 0.90) {
                [void]$checks.Add((New-Check 'arena.memory' 'FAIL' ("Paper heap {0:P0} full - GC thrash and an OOM crash are imminent; raise MCAI_JAVA_MEMORY or cut pairs." -f $memoryFraction)))
            } elseif ([double]$memoryFraction -gt 0.80) {
                [void]$checks.Add((New-Check 'arena.memory' 'WARN' ("Paper heap {0:P0} full - the load controller will start shedding arenas." -f $memoryFraction)))
            } else {
                [void]$checks.Add((New-Check 'arena.memory' 'PASS' ("Paper heap {0:P0} used." -f $memoryFraction)))
            }
        }
    }

    # Websocket close/handshake tracebacks are routine churn every time a bot reconnects, so a
    # handful of them is an INFO (rank 0) and cannot pin `overall` to WARN forever. A burst is
    # something else and does escalate.
    if ($trainer.traceback_count -gt 0) {
        $note = "Recent trainer.log tail contains $($trainer.traceback_count) Python traceback(s)."
        if ($trainer.last_error) { $note = "$note Last: $($trainer.last_error)" }
        if ($trainer.traceback_count -gt 5) {
            [void]$checks.Add((New-Check 'trainer.errors' 'WARN' "$note That is more than routine reconnect churn."))
        } else {
            [void]$checks.Add((New-Check 'trainer.errors' 'INFO' "$note Reconnect churn at this rate is normal."))
        }
    }

    # The dashboard is a convenience; training is unaffected by it being down. Only escalate once
    # it has been missing across several consecutive passes (i.e. it is not just restarting).
    $dashboardUp = Test-DashboardPort -Port $DashboardPort
    if ($dashboardUp) {
        $State.dashboard_down_passes = 0
        [void]$checks.Add((New-Check 'dashboard' 'PASS' ("Listening on 127.0.0.1:{0} (tunnel to it; never expose it)." -f $DashboardPort)))
    } else {
        $State.dashboard_down_passes = [int]$State.dashboard_down_passes + 1
        if ($State.dashboard_down_passes -ge 3) {
            [void]$checks.Add((New-Check 'dashboard' 'WARN' ("Dashboard has not been listening on 127.0.0.1:{0} for {1} consecutive passes; training itself is unaffected." -f $DashboardPort, $State.dashboard_down_passes)))
        } else {
            [void]$checks.Add((New-Check 'dashboard' 'INFO' ("Dashboard not listening on 127.0.0.1:{0}; training itself is unaffected." -f $DashboardPort)))
        }
    }

    # A give-up marker means the supervisor has stopped retrying: nothing will train again until
    # an operator intervenes, and the VM is burning Spot hours doing nothing.
    $giveUpMarker = Join-Path $Root '.mcai\azure-runner-failed.json'
    if (Test-Path -LiteralPath $giveUpMarker) {
        $reason = 'see the file'
        try {
            $marker = Get-Content -LiteralPath $giveUpMarker -Raw -Encoding UTF8 | ConvertFrom-Json
            $reason = [string](Get-Field $marker 'reason' $reason)
        } catch { }
        [void]$checks.Add((New-Check 'supervisor' 'FAIL' ("The training supervisor gave up ({0}). Nothing will restart until {1} and .mcai\stop-azure-runner are removed - deallocate the VM if you cannot fix it now." -f $reason, $giveUpMarker)))
    }

    # ---- spend ----------------------------------------------------------------
    $calendarDays = [math]::Max(0.0, ($now - $ledgerStart).TotalDays)
    $diskSpend = $calendarDays * ($DiskMonthlyUsd / 30.44)
    $computeSpend = $billedHours * $HourlyRate
    $totalSpend = $computeSpend + $diskSpend
    $diskHourly = $DiskMonthlyUsd / (30.44 * 24.0)
    $remainingHours = 0.0
    if ($HourlyRate + $diskHourly -gt 0) {
        $remainingHours = [math]::Max(0.0, ($BudgetUsd - $totalSpend) / ($HourlyRate + $diskHourly))
    }
    $budgetFraction = 0.0
    if ($BudgetUsd -gt 0) { $budgetFraction = $totalSpend / $BudgetUsd }

    # Single-quoted format strings throughout: "$" is a literal here, and "${0:N2}" in a
    # double-quoted string would parse as a drive-qualified variable reference.
    if ($budgetFraction -ge 0.95) {
        [void]$checks.Add((New-Check 'budget' 'FAIL' ('Spent about ${0:N2} of ${1:N2} ({2:P0}). Deallocate the VM now - checkpoints\latest.pt survives deallocation.' -f $totalSpend, $BudgetUsd, $budgetFraction)))
    } elseif (-not $uptimeComplete) {
        # Never PASS on an incomplete record: an under-reported ledger is the one failure mode
        # that costs real money, so an unverifiable figure is labelled a floor and warns.
        [void]$checks.Add((New-Check 'budget' 'WARN' ('At least ${0:N2} of ${1:N2} spent ({2:P0}) - this is a FLOOR, not a bill: billed uptime came from {3} and could not be verified against the System event log. Confirm in the Azure Cost Management blade.' -f $totalSpend, $BudgetUsd, $budgetFraction, $uptimeSource)))
    } elseif ($budgetFraction -ge 0.80) {
        [void]$checks.Add((New-Check 'budget' 'WARN' ('Spent about ${0:N2} of ${1:N2} ({2:P0}); roughly {3:N1}h of compute left.' -f $totalSpend, $BudgetUsd, $budgetFraction, $remainingHours)))
    } else {
        [void]$checks.Add((New-Check 'budget' 'PASS' ('Spent about ${0:N2} of ${1:N2}; roughly {2:N1}h ({3:N1} days) of compute left.' -f $totalSpend, $BudgetUsd, $remainingHours, ($remainingHours / 24.0))))
    }

    $overall = 'PASS'
    foreach ($check in $checks) {
        if ($script:StatusRank[$check.status] -gt $script:StatusRank[$overall]) { $overall = $check.status }
    }

    return [pscustomobject]@{
        timestamp_utc = $now.ToString('o')
        overall       = $overall
        run_directory = $runDirectory
        metrics_path  = $metricsPath
        trainer       = [pscustomobject]@{
            device              = $trainer.device
            policy_version      = $trainer.policy_version
            total_agent_ticks   = $totalTicks
            rollout_collected   = $trainer.collected_ticks
            rollout_target      = $trainer.target_ticks
            last_update_age_min = $updateAge
            ticks_per_hour      = $recentTicksPerHour
            ticks_per_hour_lifetime = $lifetimeTicksPerHour
            flow_ticks_per_hour = $flowPerHour
            sample_span_minutes = $sampleSpanMinutes
            traceback_count     = $trainer.traceback_count
            recent_updates      = @($trainer.updates | ForEach-Object {
                [pscustomobject]@{
                    policy_version    = Get-Field $_ 'policy_version' $null
                    total_agent_ticks = Get-Field $_ 'total_agent_ticks' $null
                    policy_loss       = Get-Field $_ 'policy_loss' $null
                    value_loss        = Get-Field $_ 'value_loss' $null
                    entropy           = Get-Field $_ 'entropy' $null
                    approximate_kl    = Get-Field $_ 'approximate_kl' $null
                    clip_fraction     = Get-Field $_ 'clip_fraction' $null
                    gradient_norm     = Get-Field $_ 'gradient_norm' $null
                }
            })
        }
        arena         = [pscustomobject]@{
            reachable          = $arena.ok
            state              = $arena.kind
            error              = $arena.error
            active_pairs       = Get-Field $arenaPayload 'active_pairs' $null
            max_concurrent_pairs = Get-Field $arenaPayload 'max_concurrent_pairs' $null
            estimated_tps      = Get-Field $arenaPayload 'estimated_tps' $null
            p95_tick_ms        = Get-Field $arenaPayload 'p95_tick_ms' $null
            memory_fraction    = Get-Field $arenaPayload 'memory_fraction' $null
            mode               = Get-Field $arenaPayload 'mode' $null
            paused             = Get-Field $arenaPayload 'paused' $null
            tick               = Get-Field $arenaPayload 'tick' $null
        }
        dashboard     = [pscustomobject]@{ port = $DashboardPort; listening = $dashboardUp }
        cost          = [pscustomobject]@{
            hourly_rate_usd     = $HourlyRate
            disk_monthly_usd    = $DiskMonthlyUsd
            budget_usd          = $BudgetUsd
            ledger_started_utc  = $ledgerStart.ToString('o')
            billed_hours        = $billedHours
            billed_hours_source = $uptimeSource
            billed_hours_verified = $uptimeComplete
            reboots_observed    = $uptime.boots
            session_hours       = $sessionHours
            compute_spend_usd   = $computeSpend
            disk_spend_usd      = $diskSpend
            total_spend_usd     = $totalSpend
            budget_fraction     = $budgetFraction
            remaining_hours     = $remainingHours
            projected_exhausted_utc = $now.AddHours($remainingHours).ToString('o')
        }
        checks        = @($checks)
    }
}

# ---------------------------------------------------------------- rendering

function Write-Report {
    param([object]$Report)

    $colors = @{ 'PASS' = 'Green'; 'WARN' = 'Yellow'; 'FAIL' = 'Red'; 'SKIP' = 'DarkGray'; 'INFO' = 'Gray' }
    $trainer = $Report.trainer
    $arena = $Report.arena
    $cost = $Report.cost

    Write-Host ''
    Write-Host ('MCAI training health  {0}  ' -f (Get-Date).ToString('yyyy-MM-dd HH:mm:ss')) -NoNewline
    Write-Host (' {0} ' -f $Report.overall) -ForegroundColor White -BackgroundColor $colors[$Report.overall]
    if ($Report.run_directory) { Write-Host ("run     : {0}" -f $Report.run_directory) -ForegroundColor DarkGray }
    if ($Report.metrics_path) { Write-Host ("metrics : {0}" -f $Report.metrics_path) -ForegroundColor DarkGray }

    Write-Host ''
    Write-Host 'TRAINER' -ForegroundColor Cyan
    Write-Host ("  policy_version    : {0}" -f $trainer.policy_version)
    Write-Host ("  total_agent_ticks : {0}" -f $(if ($null -ne $trainer.total_agent_ticks) { '{0:N0}' -f [double]$trainer.total_agent_ticks } else { 'n/a' }))
    if ($null -ne $trainer.rollout_collected -and $null -ne $trainer.rollout_target) {
        Write-Host ("  rollout buffer    : {0:N0} / {1:N0}" -f [double]$trainer.rollout_collected, [double]$trainer.rollout_target)
    }
    Write-Host ("  last ppo_update   : {0} ago" -f (Format-Duration $trainer.last_update_age_min))
    $throughput = 'measuring...'
    if ($null -ne $trainer.ticks_per_hour) {
        $throughput = '{0:N0} ticks/h (last {1})' -f $trainer.ticks_per_hour, (Format-Duration $trainer.sample_span_minutes)
    } elseif ($null -ne $trainer.ticks_per_hour_lifetime) {
        # Honest fallback: both numerator and denominator come from metrics.jsonl and span the
        # same interval. Labelled so it is never mistaken for the current rate.
        $throughput = '{0:N0} ticks/h (lifetime average of this checkpoint dir)' -f $trainer.ticks_per_hour_lifetime
    }
    Write-Host ("  throughput        : {0}" -f $throughput)
    if ($trainer.device) { Write-Host ("  device            : {0}" -f $trainer.device) -ForegroundColor DarkGray }

    $updates = @($trainer.recent_updates | Select-Object -Last 5)
    if ($updates.Count -gt 0) {
        Write-Host ''
        Write-Host 'RECENT PPO UPDATES' -ForegroundColor Cyan
        Write-Host ('  {0,5}  {1,12}  {2,10}  {3,11}  {4,8}  {5,8}  {6,8}' -f 'ver', 'ticks', 'pol_loss', 'val_loss', 'entropy', 'kl', 'clip')
        foreach ($update in $updates) {
            Write-Host ('  {0,5}  {1,12}  {2,10}  {3,11}  {4,8}  {5,8}  {6,8}' -f `
                $update.policy_version,
                $(if ($null -ne $update.total_agent_ticks) { '{0:N0}' -f [double]$update.total_agent_ticks } else { '-' }),
                (Format-Number $update.policy_loss 4),
                (Format-Number $update.value_loss 6),
                (Format-Number $update.entropy 3),
                (Format-Number $update.approximate_kl 4),
                (Format-Number $update.clip_fraction 3))
        }
    }

    Write-Host ''
    Write-Host ('ARENA (127.0.0.1:{0})' -f $ArenaPort) -ForegroundColor Cyan
    if (-not $arena.reachable) {
        Write-Host ("  unreachable ({0}): {1}" -f $arena.state, $arena.error) -ForegroundColor Red
    } else {
        Write-Host ("  pairs             : {0} / {1}   mode {2}   paused {3}" -f $arena.active_pairs, $arena.max_concurrent_pairs, $arena.mode, $arena.paused)
        Write-Host ("  tps / p95 tick    : {0} / {1} ms" -f (Format-Number $arena.estimated_tps 1), (Format-Number $arena.p95_tick_ms 1))
        Write-Host ("  paper heap        : {0}" -f $(if (Test-Finite $arena.memory_fraction) { '{0:P0}' -f [double]$arena.memory_fraction } else { 'n/a' }))
    }

    Write-Host ''
    Write-Host 'HEALTH' -ForegroundColor Cyan
    foreach ($check in $Report.checks) {
        Write-Host ('  [{0}] ' -f $check.status) -ForegroundColor $colors[$check.status] -NoNewline
        Write-Host ('{0,-20} {1}' -f $check.check, $check.detail)
    }

    Write-Host ''
    Write-Host 'SPEND' -ForegroundColor Cyan
    Write-Host ('  billed uptime     : {0:N2} h at ${1:N4}/h  (source: {2}, {3} reboot(s) since {4:yyyy-MM-dd HH:mm} UTC)' -f `
        $cost.billed_hours, $cost.hourly_rate_usd, $cost.billed_hours_source, $cost.reboots_observed, (ConvertTo-Utc $cost.ledger_started_utc))
    Write-Host ('  spent (est)       : ${0:N2} compute + ${1:N2} disk = ${2:N2} of ${3:N2}' -f $cost.compute_spend_usd, $cost.disk_spend_usd, $cost.total_spend_usd, $cost.budget_usd)
    Write-Host ('  remaining         : {0:N1} h ({1:N1} days) at the current rate' -f $cost.remaining_hours, ($cost.remaining_hours / 24.0))
    if (-not $cost.billed_hours_verified) {
        Write-Host '  (uptime could not be verified against the System event log: treat the figure as a lower bound)' -ForegroundColor Yellow
    }
    Write-Host ''
    Write-Host '  Dashboard: ssh -N -L 8788:127.0.0.1:8788 <user>@<vm-ip>  then http://127.0.0.1:8788' -ForegroundColor DarkGray
    Write-Host ''
}

function Write-JsonLog {
    param([string]$Line)
    if (-not $LogPath) { return }
    try {
        $directory = Split-Path -Parent $LogPath
        if ($directory -and -not (Test-Path -LiteralPath $directory)) {
            New-Item -ItemType Directory -Force -Path $directory | Out-Null
        }
        # One rotation only: this runs every few minutes for days on a 64 GB disk.
        if ((Test-Path -LiteralPath $LogPath) -and (Get-Item -LiteralPath $LogPath).Length -gt ($MaxLogMB * 1MB)) {
            Move-Item -LiteralPath $LogPath -Destination "$LogPath.1" -Force
        }
        Add-Content -LiteralPath $LogPath -Value $Line -Encoding UTF8
    } catch {
        Write-Warning "Could not append to ${LogPath}: $($_.Exception.Message)"
    }
}

# ---------------------------------------------------------------- scheduled task

function Install-MonitorTask {
    <#  Without this the ledger and the sample ring only advance during the seconds a human is
        running the script, so throughput reads "measuring..." forever and spend is whatever was
        banked the last time somebody logged in. Mirrors the task 03-run-training.ps1 registers. #>
    if (-not $script:IsWindowsHost) { throw 'The monitor task can only be installed on Windows.' }
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    if (-not ([Security.Principal.WindowsPrincipal]$identity).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Registering the monitor task requires an elevated PowerShell session (Run as Administrator).'
    }
    $scriptPath = $PSCommandPath
    if (-not $scriptPath -or -not (Test-Path -LiteralPath $scriptPath)) {
        throw 'Cannot register the monitor task: this script has no on-disk path. Save the repo somewhere permanent (e.g. C:\MCAI\deploy\azure-windows\) and invoke it with -File.'
    }
    $scriptPath = (Resolve-Path -LiteralPath $scriptPath).Path
    # A task stores an absolute path and runs it days later; a Run Command / CustomScriptExtension
    # temp copy is deleted the moment the extension finishes.
    foreach ($ephemeral in @($env:TEMP, $env:TMP, "$env:SystemRoot\Temp")) {
        if ($ephemeral -and $scriptPath.StartsWith($ephemeral, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to register a task for a temporary copy at $scriptPath. Copy the repo to a permanent location first."
        }
    }
    if ($scriptPath -like '*\Packages\Plugins\*') {
        throw "Refusing to register a task for the CustomScriptExtension copy at $scriptPath. Clone the repo to a permanent location (e.g. C:\MCAI) and register from there."
    }

    $hostExe = [System.Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
    $taskLog = $LogPath
    if (-not $taskLog) { $taskLog = Join-Path $Root 'runs\monitor\monitor.ndjson' }
    $arguments = ConvertTo-ArgumentString @(
        '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
        '-File', $scriptPath, '-Root', $Root, '-Json', '-LogPath', $taskLog)
    $action = New-ScheduledTaskAction -Execute $hostExe -Argument $arguments -WorkingDirectory $Root

    $startupTrigger = New-ScheduledTaskTrigger -AtStartup
    # Let the supervisor's own PT1M delay and the stack's startup elapse before the first pass.
    $startupTrigger.Delay = 'PT3M'
    $interval = New-TimeSpan -Minutes $TaskIntervalMinutes
    $start = (Get-Date).AddMinutes(1)
    try {
        $repeatTrigger = New-ScheduledTaskTrigger -Once -At $start -RepetitionInterval $interval -RepetitionDuration ([TimeSpan]::MaxValue)
    } catch {
        # [TimeSpan]::MaxValue means "indefinitely" on current builds but is rejected on some; ten
        # years outlives any plausible $100 run.
        $repeatTrigger = New-ScheduledTaskTrigger -Once -At $start -RepetitionInterval $interval -RepetitionDuration (New-TimeSpan -Days 3650)
    }

    $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
    # A pass takes seconds; a 10 minute limit guarantees a wedged socket read cannot block the next
    # one. Priority 7 (the default) keeps the monitor out of the CPU-bound trainer's way.
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 10)
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger @($startupTrigger, $repeatTrigger) `
        -Principal $principal -Settings $settings -Force `
        -Description 'MCAI training health and spend monitor (keeps the uptime ledger and throughput ring alive).' | Out-Null

    # Read the task back rather than trusting that registration implies a launchable command line.
    $registered = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $registered) { throw "Registration reported success but no task named '$TaskName' exists." }
    $registeredArguments = @($registered.Actions)[0].Arguments
    if ($registeredArguments -notmatch '(?i)-File\s+"?([^"]+04-monitor\.ps1)') {
        throw "The registered task's command line does not contain a usable -File path: $registeredArguments"
    }
    if (-not (Test-Path -LiteralPath $Matches[1])) {
        throw "The registered task points at $($Matches[1]), which does not exist."
    }

    # Smoke test the contract a supervisor branches on: a monitor that cannot run must exit 3, not
    # 1 (which means WARN). Regressions here are silent and would hide a dead monitor.
    $probeRoot = Join-Path $env:SystemDrive ('mcai-monitor-selftest-' + [guid]::NewGuid().ToString('N'))
    $probeExit = -1
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $hostExe '-NoProfile' '-NonInteractive' '-ExecutionPolicy' 'Bypass' '-File' $scriptPath '-Root' $probeRoot '-NoState' *> $null
        $probeExit = $LASTEXITCODE
    } finally { $ErrorActionPreference = $previous }
    if ($probeExit -ne 3) {
        throw "Self-test failed: '04-monitor.ps1 -Root <missing>' exited $probeExit, expected 3 (the 'monitor cannot run' code)."
    }

    Write-Host "Registered scheduled task '$TaskName': every $TaskIntervalMinutes minute(s) and at boot, as SYSTEM." -ForegroundColor Green
    Write-Host "  script : $scriptPath"
    Write-Host "  log    : $taskLog"
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "Started it now. Follow it with: Get-Content -Wait '$taskLog'"
}

function Uninstall-MonitorTask {
    if (-not $script:IsWindowsHost) { throw 'The monitor task can only be removed on Windows.' }
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $existing) { Write-Host "No scheduled task named '$TaskName'."; return }
    try { Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue } catch { }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed scheduled task '$TaskName'. The uptime ledger in $StatePath is left in place."
}

# ---------------------------------------------------------------- main

if ($Install -and $Uninstall) { Exit-Monitor 'use either -Install or -Uninstall, not both.' 3 }
if ($Install) {
    try { Install-MonitorTask; exit 0 } catch { Exit-Monitor $_.Exception.Message 3 }
}
if ($Uninstall) {
    try { Uninstall-MonitorTask; exit 0 } catch { Exit-Monitor $_.Exception.Message 3 }
}

$state = Read-MonitorState
$exitCode = 0

try {
    do {
        try {
            $report = Invoke-HealthPass -State $state
            Save-MonitorState -State $state
            $line = $report | ConvertTo-Json -Depth 8 -Compress
            Write-JsonLog $line
            if ($Json) {
                # NDJSON: one self-contained object per pass, safe to tail or ship to a log store.
                $line
            } else {
                if ($Watch) { Clear-Host }
                Write-Report -Report $report
                if ($Watch) { Write-Host ("Refreshing every {0}s. Ctrl+C to stop." -f $IntervalSeconds) -ForegroundColor DarkGray }
            }
            $exitCode = $script:StatusRank[$report.overall]
        } catch {
            # A monitor that dies on a transient file lock is worse than useless in -Watch mode.
            Write-Warning "Health pass failed: $($_.Exception.Message)"
            $exitCode = 3
            if (-not $Watch) { throw }
        }
        if ($Watch) { Start-Sleep -Seconds $IntervalSeconds }
    } while ($Watch)
} catch {
    Exit-Monitor $_.Exception.Message 3
}

exit $exitCode
