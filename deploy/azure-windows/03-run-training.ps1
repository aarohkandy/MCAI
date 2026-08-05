#Requires -Version 5.1
<#
.SYNOPSIS
    Headless, self-healing supervisor for the MCAI training stack on an Azure Windows VM.

.DESCRIPTION
    Starts and supervises the four MCAI processes (Python trainer, Paper server, Node dashboard,
    Node rollout worker) with no interactive input, and keeps them running for days across Spot
    evictions, reboots and individual process crashes. Training always resumes from
    checkpoints/latest.pt, so recycling the stack costs at most the current partial rollout.

    This is the unattended counterpart of scripts\start-windows.ps1 (which assumes a desktop
    session: it opens a browser and exits on the first failure) and mirrors the readiness ordering
    and environment of scripts\start-linux.sh / deploy\azure\entrypoint.sh.

    -Install registers a Scheduled Task that runs this script at boot as SYSTEM. A Scheduled Task
    rather than a Windows Service, because a service must be an SCM-aware executable: hosting a
    PowerShell script as one needs a third-party wrapper (nssm/srvany) that has to be downloaded,
    trusted and maintained. Task Scheduler ships with Windows Server, starts at boot with no logged-
    on user (which is what a Spot restart gives us), restarts the task if it dies, and is inspectable
    with schtasks.exe over RDP. The supervision loop below is the "service" part; the task only has
    to get it launched.

.PARAMETER Install
    Register the boot task (and start it now). Requires an elevated shell and a copy of this script
    saved at a permanent path (not a Run Command / CustomScriptExtension temp copy).

.PARAMETER Uninstall
    Stop and remove the boot task. Does not stop a running stack.

.PARAMETER Once
    Run the stack a single time and exit when it dies, instead of supervising forever. Debug aid.

.PARAMETER MaxConsecutiveFailures
    Give up after this many consecutive failures to get a stack running for ten minutes. Protects
    the compute budget from an install that can never start.

.PARAMETER ShutdownOnGiveUp
    Also power the VM off when the cap is hit. Read the note on the parameter: a guest shutdown does
    not deallocate, so pair it with an Azure auto-shutdown schedule.

.EXAMPLE
    # Unattended install. Run ONCE from an elevated shell after 02-bootstrap.ps1 completes; nothing
    # else registers the boot task, and without it a Spot eviction stops training permanently.
    .\03-run-training.ps1 -Install -RepoRoot C:\MCAI

.NOTES
    Windows-only: Scheduled Task cmdlets, taskkill.exe and the machine PATH lookup do not exist on
    other platforms. The script is written to parse cleanly everywhere so it can be linted on a Mac.

    Ports stay on loopback: 25565 (Paper), 8765 (arena control), 8766 (trainer), 8788 (dashboard).
    Reach the dashboard through an SSH/RDP tunnel; nothing here binds a public interface.

    Create the file .mcai\stop-azure-runner under the repo root to make the supervisor shut the
    stack down and exit cleanly (and stay down across reboots until the file is deleted). The
    supervisor writes that same sentinel itself when it gives up (see -MaxConsecutiveFailures),
    alongside .mcai\azure-runner-failed.json, so a known-broken install is not resurrected at every
    reboot while the VM bills by the hour. Delete both once the install is fixed.
#>
[CmdletBinding()]
param(
    [switch]$Install,
    [switch]$Uninstall,
    [switch]$Once,
    [string]$RepoRoot,
    [string]$ConfigPath,
    [string]$TaskName = 'MCAI-Training',
    # Disk budget: the VM disk is 64 GB and the run lasts days, so both the live logs and the
    # accumulated run directories are capped.
    [int]$MaxChildLogMB = 256,
    [int]$MaxSupervisorLogMB = 16,
    [int]$MaxRunsGB = 6,
    [int]$KeepRunDirectories = 6,
    # The trainer prints a rollout_progress line every rollout_steps/32 agent ticks (seconds to
    # minutes when healthy). No growth for this long means it is wedged, not slow.
    [int]$StallMinutes = 20,
    # Budget guard. A half-finished bootstrap can never start the stack, and an uncapped retry loop
    # would burn the whole $100 of Spot compute producing nothing. 12 consecutive failures is ~1 hour
    # at the 300 s backoff ceiling - long enough to ride out transient faults, short enough to notice.
    [int]$MaxConsecutiveFailures = 12,
    # Power the VM off when the cap is hit. NOTE: a guest-initiated shutdown leaves the VM in
    # "Stopped" (not "Stopped (deallocated)") and Azure keeps billing compute, so pair this with an
    # auto-shutdown schedule or a runbook that deallocates. Without it we only stop retrying.
    [switch]$ShutdownOnGiveUp
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

$Script:IsWindowsHost = [System.Environment]::OSVersion.Platform -eq [System.PlatformID]::Win32NT
$Script:RunPrefix = 'azure-'
$Script:Children = @()
$Script:CurrentRunDirectory = $null
$Script:Mutex = $null
# Set by Start-TrainingStack once every child is up; the backoff reset is measured from it so that
# twelve minutes spent in readiness waits is never mistaken for twelve minutes of training.
$Script:StackUpAt = $null
# Set when a failure cannot possibly be fixed by retrying (broken venv, bad Java, rejected EULA).
$Script:FatalReason = $null

# ---------------------------------------------------------------------------- paths and config ---

if (-not $RepoRoot) { $RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path }
elseif (-not (Test-Path $RepoRoot)) { throw "RepoRoot does not exist: $RepoRoot" }
else { $RepoRoot = (Resolve-Path $RepoRoot).Path }

$ScriptPath = $PSCommandPath
$RunsBase = Join-Path $RepoRoot 'runs'
$StateDirectory = Join-Path $RepoRoot '.mcai'
$StatePath = Join-Path $StateDirectory 'azure-runner.json'
$StopSentinel = Join-Path $StateDirectory 'stop-azure-runner'
$SupervisorLogDirectory = Join-Path $RunsBase 'supervisor'
$SupervisorLog = Join-Path $SupervisorLogDirectory 'runner.log'
$ArenaScript = Join-Path $RepoRoot 'scripts\arena_control.py'
$ConfigureRuntimeScript = Join-Path $RepoRoot 'scripts\configure_runtime.py'

New-Item -ItemType Directory -Force -Path $RunsBase, $StateDirectory, $SupervisorLogDirectory | Out-Null

function Write-RunnerLog {
    param([ValidateSet('INFO', 'WARN', 'ERROR')][string]$Level, [string]$Message)
    $line = '{0} [{1}] {2}' -f (Get-Date).ToString('yyyy-MM-dd HH:mm:ss'), $Level, $Message
    # Never let a logging failure (full disk, locked file) take down the supervisor.
    try { Add-Content -Path $SupervisorLog -Value $line -Encoding UTF8 } catch { }
    switch ($Level) {
        'ERROR' { Write-Host $line -ForegroundColor Red }
        'WARN' { Write-Host $line -ForegroundColor Yellow }
        default { Write-Host $line }
    }
}

function Sync-MachineEnvironment {
    # A task started at boot as SYSTEM inherits the machine environment, but a supervisor launched in
    # the same shell that just ran 02-bootstrap.ps1 would otherwise miss everything the bootstrap
    # wrote to the registry - both the PATH entries (node, java, uv) and the MCAI_* settings that
    # decide mode, bot count, runtime directory and which JDK Paper gets. Re-read them here; without
    # this the runner silently falls back to its defaults and trains the wrong thing convincingly.
    if (-not $Script:IsWindowsHost) { return }
    $machinePath = [Environment]::GetEnvironmentVariable('Path', 'Machine')
    if ($machinePath) {
        $seen = New-Object 'System.Collections.Generic.HashSet[string]' ([StringComparer]::OrdinalIgnoreCase)
        $merged = @()
        foreach ($part in (($env:Path + ';' + $machinePath) -split ';')) {
            if ($part -and $seen.Add($part)) { $merged += $part }
        }
        $env:Path = $merged -join ';'
    }
    # Process scope wins, so an explicit override in the current shell (or the config file loaded
    # above) is preserved; only variables this session has never seen are filled in.
    foreach ($entry in ([Environment]::GetEnvironmentVariables('Machine')).GetEnumerator()) {
        if ($entry.Key -like 'MCAI_*' -and -not [Environment]::GetEnvironmentVariable($entry.Key, 'Process')) {
            Set-Item -Path "env:$($entry.Key)" -Value $entry.Value
        }
    }
}

function Resolve-Executable {
    param([string]$Override, [string]$Command, [string[]]$Candidates, [scriptblock]$Validator)
    # An explicit override is honoured even when it fails validation, so the operator gets a precise
    # "your MCAI_JAVA_EXE is Java 17" error rather than a silent substitution behind their back.
    if ($Override -and (Test-Path $Override)) { return (Resolve-Path $Override).Path }
    $found = Get-Command $Command -ErrorAction SilentlyContinue
    if ($found -and (-not $Validator -or (& $Validator $found.Source))) { return $found.Source }
    foreach ($candidate in $Candidates) {
        # Candidates are globs (JDK/Node install roots carry a version in the path). Newest first,
        # but skip anything the validator rejects instead of returning the first match blindly.
        foreach ($match in @(Get-Item $candidate -ErrorAction SilentlyContinue | Sort-Object FullName -Descending)) {
            if (-not $Validator -or (& $Validator $match.FullName)) { return $match.FullName }
        }
    }
    # Nothing validated. Fall back to the PATH entry so Test-Prerequisites can name the real problem.
    if ($found) { return $found.Source }
    return $null
}

function Get-JavaMajorVersion {
    param([string]$JavaExe)
    if (-not $JavaExe -or -not (Test-Path -LiteralPath $JavaExe)) { return $null }
    # `java -version` writes to stderr; 2>&1 under $ErrorActionPreference='Stop' would otherwise be
    # promoted to a terminating NativeCommandError on Windows PowerShell.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $text = (& $JavaExe '-version' 2>&1 | Out-String) } catch { return $null } finally { $ErrorActionPreference = $previous }
    if ($text -notmatch 'version "(\d+)(?:\.(\d+))?') { return $null }
    # "1.8.0_412" reports 8; "11.0.22" and "17.0.10" report themselves.
    if ([int]$Matches[1] -eq 1 -and $Matches.Count -ge 3) { return [int]$Matches[2] }
    return [int]$Matches[1]
}

function Test-PaperJava {
    param([string]$JavaExe)
    # Paper 1.12.2 reflects into java.base and uses sun.misc.Unsafe: it will not boot on 17+.
    # Same 8..11 window 02-bootstrap.ps1 enforces when it picks the JDK.
    $major = Get-JavaMajorVersion $JavaExe
    return ($null -ne $major -and $major -ge 8 -and $major -le 11)
}

function ConvertTo-ArgumentString {
    param([string[]]$Arguments)
    # Start-Process joins ArgumentList with plain spaces and adds no quoting, so any path containing
    # a space ("C:\Program Files\nodejs\...") would reach the child as two arguments.
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

# Optional operator config. config.windows.ps1 is the existing repo convention; a copy next to this
# script lets the Azure bootstrap drop VM-specific settings without touching tracked files.
$ConfigCandidates = @()
if ($ConfigPath) { $ConfigCandidates += $ConfigPath }
$ConfigCandidates += (Join-Path $PSScriptRoot 'config.azure-windows.ps1')
$ConfigCandidates += (Join-Path $RepoRoot 'config.windows.ps1')
foreach ($candidate in $ConfigCandidates) {
    if ($candidate -and (Test-Path $candidate)) {
        Write-RunnerLog INFO "Loading configuration from $candidate"
        . $candidate
        break
    }
}

Sync-MachineEnvironment

$Runtime = if ($env:MCAI_RUNTIME) { $env:MCAI_RUNTIME } else { Join-Path $RepoRoot 'server-runtime' }
$Checkpoints = if ($env:MCAI_CHECKPOINT_DIR) { $env:MCAI_CHECKPOINT_DIR } else { Join-Path $RepoRoot 'checkpoints' }
$PaperJar = Join-Path $Runtime 'paper-1.12.2.jar'
$WorkerScript = Join-Path $RepoRoot 'worker\dist\src\index.js'
$DashboardScript = Join-Path $RepoRoot 'dashboard\server.mjs'
$Python = if ($env:MCAI_VENV_PYTHON) { $env:MCAI_VENV_PYTHON } else { Join-Path $RepoRoot 'trainer\.venv\Scripts\python.exe' }
$JavaMemory = if ($env:MCAI_JAVA_MEMORY) { $env:MCAI_JAVA_MEMORY } else { '3G' }
$JavaInitialMemory = if ($env:MCAI_JAVA_INITIAL_MEMORY) { $env:MCAI_JAVA_INITIAL_MEMORY } else { '1G' }
$CpuThreads = if ($env:MCAI_TORCH_THREADS) { [int]$env:MCAI_TORCH_THREADS } else { 0 }
$RolloutSteps = if ($env:MCAI_ROLLOUT_STEPS) { [int]$env:MCAI_ROLLOUT_STEPS } else { 8192 }
$DashboardPort = if ($env:MCAI_DASHBOARD_PORT) { [int]$env:MCAI_DASHBOARD_PORT } else { 8788 }
$ArenaPort = if ($env:MCAI_ARENA_PORT) { [int]$env:MCAI_ARENA_PORT } else { 8765 }
$TrainerPort = 8766
$Mode = if ($env:MCAI_MODE) { $env:MCAI_MODE.ToLowerInvariant() } else { 'sword' }

$Java = Resolve-Executable $env:MCAI_JAVA_EXE 'java.exe' @(
    'C:\Program Files\Eclipse Adoptium\jdk-*\bin\java.exe',
    'C:\Program Files\Microsoft\jdk-*\bin\java.exe',
    'C:\Program Files\Java\jdk-*\bin\java.exe') ${function:Test-PaperJava}
$Node = Resolve-Executable $env:MCAI_NODE_EXE 'node.exe' @('C:\Program Files\nodejs\node.exe')

# Loopback-only defaults, identical to scripts\start-linux.sh.
$env:MCAI_SERVER_HOST = if ($env:MCAI_SERVER_HOST) { $env:MCAI_SERVER_HOST } else { '127.0.0.1' }
$env:MCAI_SERVER_PORT = if ($env:MCAI_SERVER_PORT) { $env:MCAI_SERVER_PORT } else { '25565' }
$env:MCAI_ARENA_HOST = '127.0.0.1'
$env:MCAI_ARENA_PORT = "$ArenaPort"
$env:MCAI_TRAINER_URL = if ($env:MCAI_TRAINER_URL) { $env:MCAI_TRAINER_URL } else { "ws://127.0.0.1:$TrainerPort" }
$env:MCAI_BOT_COUNT = if ($env:MCAI_BOT_COUNT) { $env:MCAI_BOT_COUNT } else { '4' }
$env:MCAI_USERNAME_PREFIX = if ($env:MCAI_USERNAME_PREFIX) { $env:MCAI_USERNAME_PREFIX } else { 'MCAI_' }
$env:MCAI_ROLLOUT_STEPS = "$RolloutSteps"
$env:MCAI_DASHBOARD_PORT = "$DashboardPort"
$env:MCAI_CHECKPOINT_DIR = $Checkpoints
# Forced, not defaulted: a public dashboard bind would expose training telemetry and the arena
# control proxy on a VM whose only intended inbound port is RDP/SSH.
if ($env:MCAI_DASHBOARD_HOST -and $env:MCAI_DASHBOARD_HOST -ne '127.0.0.1') {
    Write-RunnerLog WARN "Ignoring MCAI_DASHBOARD_HOST=$($env:MCAI_DASHBOARD_HOST); the dashboard stays on loopback."
}
$env:MCAI_DASHBOARD_HOST = '127.0.0.1'
# The desktop launcher opens a browser on start; there is no desktop here.
$env:MCAI_OPEN_DASHBOARD = 'false'

# ------------------------------------------------------------------------------ boot task setup ---

function Register-BootTask {
    if (-not $Script:IsWindowsHost) { throw 'The boot task can only be installed on Windows.' }
    # Register-ScheduledTask with a SYSTEM principal fails with a bare Access Denied otherwise.
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    if (-not ([Security.Principal.WindowsPrincipal]$identity).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Registering the boot task requires an elevated PowerShell session (Run as Administrator).'
    }
    # The task stores an absolute -File path and runs it minutes-to-days later, so the copy we are
    # running from has to be permanent. `iex (irm ...)` leaves $PSCommandPath empty; Run Command and
    # the CustomScriptExtension materialise the script into a temp directory and delete it after.
    # Both register a task that looks fine and silently does nothing after the first eviction.
    if (-not $ScriptPath -or -not (Test-Path -LiteralPath $ScriptPath)) {
        throw 'Cannot register the boot task: this script has no on-disk path. Save the repo somewhere permanent (e.g. C:\MCAI\deploy\azure-windows\) and invoke it with -File.'
    }
    $ScriptPath = (Resolve-Path -LiteralPath $ScriptPath).Path
    foreach ($ephemeral in @($env:TEMP, $env:TMP, "$env:SystemRoot\Temp", "$env:ProgramData\Microsoft\Windows\Start Menu")) {
        if ($ephemeral -and $ScriptPath.StartsWith($ephemeral, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Refusing to register a boot task for a temporary copy at $ScriptPath. Copy the repo to a permanent location first."
        }
    }
    if ($ScriptPath -like '*\Packages\Plugins\*') {
        throw "Refusing to register a boot task for the CustomScriptExtension copy at $ScriptPath. Clone the repo to a permanent location (e.g. C:\MCAI) and register from there."
    }
    # The task runs as SYSTEM, which has no access to a mapped drive and a different profile root.
    if ($env:USERPROFILE -and $RepoRoot.StartsWith((Split-Path -Parent $env:USERPROFILE), [StringComparison]::OrdinalIgnoreCase)) {
        Write-RunnerLog WARN "RepoRoot $RepoRoot sits under a user profile; the task runs as SYSTEM and may not be able to read it. Prefer C:\MCAI."
    }
    $host_exe = [System.Diagnostics.Process]::GetCurrentProcess().MainModule.FileName
    $arguments = ConvertTo-ArgumentString @(
        '-NoProfile', '-NonInteractive', '-ExecutionPolicy', 'Bypass',
        '-File', $ScriptPath, '-RepoRoot', $RepoRoot)
    $action = New-ScheduledTaskAction -Execute $host_exe -Argument $arguments -WorkingDirectory $RepoRoot
    $trigger = New-ScheduledTaskTrigger -AtStartup
    # Let networking, the disk and any antivirus scan settle after a Spot restart before we launch
    # four processes and a JVM.
    $trigger.Delay = 'PT1M'
    # SYSTEM: runs with no logged-on user, survives RDP sign-out, and needs no stored password (a
    # named account would require one, which is exactly the interactive step we are avoiding).
    $principal = New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
    $settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -RestartCount 999 -RestartInterval (New-TimeSpan -Minutes 1) `
        -Priority 4   # default task priority (7) throttles a CPU-bound trainer; 4 is normal.
    Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal `
        -Settings $settings -Description 'MCAI self-play training supervisor (headless, resumes from checkpoints).' `
        -Force | Out-Null

    # Read the task back rather than trusting that registration implies a launchable command line.
    $registered = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $registered) { throw "Registration reported success but no task named '$TaskName' exists." }
    $registeredArguments = @($registered.Actions)[0].Arguments
    if ($registeredArguments -notmatch '(?i)-File\s+"?([^"]+03-run-training\.ps1)') {
        throw "The registered task's command line does not contain a usable -File path: $registeredArguments"
    }
    if (-not (Test-Path -LiteralPath $Matches[1])) {
        throw "The registered task points at $($Matches[1]), which does not exist."
    }
    Write-RunnerLog INFO "Registered scheduled task '$TaskName' (at startup, SYSTEM): $($registered.State). Verified -File $($Matches[1])"
    Write-RunnerLog INFO "Starting it now."
    Start-ScheduledTask -TaskName $TaskName
    Write-RunnerLog INFO "Follow the run with: Get-Content -Wait '$SupervisorLog'"
}

function Unregister-BootTask {
    if (-not $Script:IsWindowsHost) { throw 'The boot task can only be removed on Windows.' }
    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $existing) { Write-RunnerLog INFO "No scheduled task named '$TaskName'."; return }
    try { Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue } catch { }
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-RunnerLog INFO "Removed scheduled task '$TaskName'. A running stack is not stopped; create $StopSentinel for that."
}

# ------------------------------------------------------------------------------------- logging ----

function Get-LiveFileLength {
    param([string]$Path)
    # Windows updates the directory entry for an open file lazily, so Get-Item .Length can lag far
    # behind for a log that a child process is still writing. Opening the file reports the truth.
    if (-not (Test-Path $Path)) { return 0 }
    try {
        $stream = [System.IO.File]::Open($Path, 'Open', 'Read', 'ReadWrite')
        try { return $stream.Length } finally { $stream.Dispose() }
    } catch { return 0 }
}

function Invoke-SupervisorLogRotation {
    if ((Get-LiveFileLength $SupervisorLog) -lt ($MaxSupervisorLogMB * 1MB)) { return }
    for ($index = 3; $index -ge 1; $index--) {
        $older = Join-Path $SupervisorLogDirectory "runner.$index.log"
        $newer = if ($index -eq 1) { $SupervisorLog } else { Join-Path $SupervisorLogDirectory "runner.$($index - 1).log" }
        if (Test-Path $newer) { Move-Item $newer $older -Force -ErrorAction SilentlyContinue }
    }
}

function Get-DirectorySizeBytes {
    param([string]$Path)
    # Measure-Object emits NOTHING (not an object with Sum=$null) for an empty pipeline, so a run
    # directory containing no files would make `$measured.Sum` throw under Set-StrictMode. Empty run
    # directories are routine: one is created before the first child starts, and a Remove-Item that
    # loses a race with a dying java.exe can delete the files but leave the directory.
    $measured = Get-ChildItem $Path -Recurse -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum
    if (-not $measured -or $null -eq $measured.Sum) { return [int64]0 }
    return [int64]$measured.Sum
}

function Limit-RunDirectories {
    # Keep the newest run directories, then trim oldest-first until the tree fits the byte budget.
    # Only directories this script created are touched.
    $directories = @(Get-ChildItem $RunsBase -Directory -Filter "$Script:RunPrefix*" -ErrorAction SilentlyContinue |
        Sort-Object CreationTime -Descending)
    $keep = @()
    foreach ($directory in $directories) {
        if ($directory.FullName -eq $Script:CurrentRunDirectory) { $keep += $directory; continue }
        # Empty leftovers carry no evidence and would otherwise consume a "keep newest" slot.
        if (@(Get-ChildItem $directory.FullName -Force -ErrorAction SilentlyContinue).Count -eq 0) {
            Remove-Item $directory.FullName -Recurse -Force -ErrorAction SilentlyContinue
            continue
        }
        if ($keep.Count -lt $KeepRunDirectories) { $keep += $directory }
        else {
            Write-RunnerLog INFO "Pruning old run directory $($directory.Name)"
            Remove-Item $directory.FullName -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
    $budget = [int64]$MaxRunsGB * 1GB
    $sized = @($keep | Sort-Object CreationTime)
    $total = [int64]0
    foreach ($directory in $sized) { $total += Get-DirectorySizeBytes $directory.FullName }
    foreach ($directory in $sized) {
        if ($total -le $budget) { break }
        if ($directory.FullName -eq $Script:CurrentRunDirectory) { continue }
        $size = Get-DirectorySizeBytes $directory.FullName
        Write-RunnerLog WARN "Runs tree over $MaxRunsGB GB; removing $($directory.Name)"
        Remove-Item $directory.FullName -Recurse -Force -ErrorAction SilentlyContinue
        $total -= $size
    }
}

function Invoke-Housekeeping {
    # Disk maintenance must never be able to recycle a healthy stack, and it has to run on the
    # failure path too: a startup crash loop creates a run directory (and a multi-MB paper.log) per
    # cycle, which is exactly when nothing was pruning them before.
    try {
        Invoke-SupervisorLogRotation
        Limit-PaperLogs
        Limit-RunDirectories
    } catch {
        Write-RunnerLog WARN "Housekeeping failed: $($_.Exception.Message)"
    }
}

function Limit-PaperLogs {
    # Paper rotates its own console log into logs\*.log.gz daily and never deletes them.
    $logDirectory = Join-Path $Runtime 'logs'
    if (-not (Test-Path $logDirectory)) { return }
    $cutoff = (Get-Date).AddDays(-3)
    Get-ChildItem $logDirectory -File -Filter '*.log.gz' -ErrorAction SilentlyContinue |
        Where-Object { $_.LastWriteTime -lt $cutoff } |
        ForEach-Object { Remove-Item $_.FullName -Force -ErrorAction SilentlyContinue }
}

function Test-ChildLogsOverCap {
    foreach ($child in $Script:Children) {
        foreach ($path in @($child.Log, $child.ErrorLog)) {
            if ((Get-LiveFileLength $path) -gt ($MaxChildLogMB * 1MB)) {
                Write-RunnerLog WARN "$([System.IO.Path]::GetFileName($path)) exceeded $MaxChildLogMB MB; recycling the stack into a fresh run directory."
                return $true
            }
        }
    }
    return $false
}

# -------------------------------------------------------------------------- process supervision ---

function Test-LocalPortOpen {
    param([int]$Port)
    $client = [Net.Sockets.TcpClient]::new()
    try { $client.Connect('127.0.0.1', $Port); return $true } catch { return $false } finally { $client.Dispose() }
}

function Clear-ForeignListeners {
    <#  Kill anything already holding our ports before we start. Without this an orphaned trainer or
        paper (left by a hard supervisor kill - Task Scheduler calls TerminateProcess, so our finally
        block never runs, and Remove-StaleProcesses cannot help if the state file was never written)
        keeps the port open, our own freshly launched child dies instantly on the bind, Wait-LocalPort
        happily connects to the ORPHAN and reports success, and the supervisor loops forever paying
        full startup cost for zero training. #>
    if (-not $Script:IsWindowsHost) { return }
    # The dashboard is deliberately absent: failing to bind 8788 is already non-fatal, so a foreign
    # holder there must not be allowed to abort a training run.
    foreach ($port in @($TrainerPort, $ArenaPort, [int]$env:MCAI_SERVER_PORT)) {
        if (-not (Test-LocalPortOpen $port)) { continue }
        $owners = @()
        try {
            $owners = @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty OwningProcess -Unique)
        } catch { }
        if ($owners.Count -eq 0) {
            Write-RunnerLog WARN "Port $port is already in use but the owning process could not be identified."
            continue
        }
        foreach ($owningPid in $owners) {
            if ([int]$owningPid -le 4) { continue }   # 0/4 are the kernel's; never touch them.
            $name = 'unknown'
            try { $name = (Get-Process -Id ([int]$owningPid) -ErrorAction SilentlyContinue).ProcessName } catch { }
            Write-RunnerLog WARN "Port $port held by $name (PID $owningPid) from a previous run; killing it."
            try { & taskkill.exe /PID $owningPid /T /F 2>$null | Out-Null } catch { }
        }
        Start-Sleep -Seconds 2
        if (Test-LocalPortOpen $port) { throw "Port $port is still held by a foreign process; cannot start the stack." }
    }
}

function Wait-LocalPort {
    param([int]$Port, [int]$TimeoutSeconds, [string]$What)
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        # Check for a dead child BEFORE the connect: otherwise a child that exited because the port
        # was already taken still looks "ready" because something answers on it.
        foreach ($child in $Script:Children) {
            if ($child.Process.HasExited) {
                throw "$($child.Name) exited with code $($child.Process.ExitCode) while waiting for $What on port $Port."
            }
        }
        if (Test-LocalPortOpen $Port) { return $true }
        Start-Sleep -Seconds 2
    }
    return $false
}

function Start-Child {
    param([string]$Name, [string]$FilePath, [string[]]$Arguments, [string]$WorkingDirectory)
    $stdout = Join-Path $Script:CurrentRunDirectory "$Name.log"
    $stderr = Join-Path $Script:CurrentRunDirectory "$Name.error.log"
    $process = Start-Process -FilePath $FilePath -ArgumentList (ConvertTo-ArgumentString $Arguments) `
        -WorkingDirectory $WorkingDirectory -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
        -NoNewWindow -PassThru
    Write-RunnerLog INFO "Started $Name (PID $($process.Id)) -> $stdout"
    $Script:Children += [pscustomobject]@{
        Name = $Name; Process = $process; Log = $stdout; ErrorLog = $stderr
    }
    Save-RunnerState
}

function Save-RunnerState {
    # Called after EVERY child starts, not once at the end: the trainer and Paper readiness waits are
    # up to 12 minutes, and a hard supervisor kill during them would otherwise leave orphans holding
    # 8766/8765/25565 that the next Remove-StaleProcesses knows nothing about.
    $entries = @()
    foreach ($child in $Script:Children) {
        $startedAt = $null
        try { $startedAt = $child.Process.StartTime.ToString('o') } catch { }
        $entries += @{ name = $child.Name; pid = $child.Process.Id; process_started_at = $startedAt }
    }
    $payload = @{
        supervisor_pid = $PID
        started_at     = (Get-Date).ToString('o')
        run_directory  = $Script:CurrentRunDirectory
        dashboard_url  = "http://127.0.0.1:$DashboardPort"
        processes      = $entries
    }
    try { $payload | ConvertTo-Json -Depth 4 | Set-Content -Path $StatePath -Encoding UTF8 }
    catch { Write-RunnerLog WARN "Could not write $StatePath : $($_.Exception.Message)" }
}

function Remove-StaleProcesses {
    # A hard supervisor kill (task end, eviction, power loss) can leave children behind. Match the
    # recorded start time as well as the PID so a recycled PID belonging to something else survives.
    if (-not (Test-Path $StatePath)) { return }
    try { $saved = Get-Content $StatePath -Raw | ConvertFrom-Json } catch { Remove-Item $StatePath -Force -ErrorAction SilentlyContinue; return }
    foreach ($entry in @($saved.processes)) {
        $process = Get-Process -Id ([int]$entry.pid) -ErrorAction SilentlyContinue
        if (-not $process -or -not $entry.process_started_at) { continue }
        $expected = [datetime]::Parse($entry.process_started_at).ToUniversalTime()
        if ([math]::Abs(($process.StartTime.ToUniversalTime() - $expected).TotalSeconds) -lt 2) {
            Write-RunnerLog WARN "Killing orphaned $($entry.name) from a previous run (PID $($process.Id))."
            try { & taskkill.exe /PID $process.Id /T /F 2>$null | Out-Null } catch { }
        }
    }
    Remove-Item $StatePath -Force -ErrorAction SilentlyContinue
}

function Invoke-ArenaCommand {
    param([string]$Command, [string]$Payload = '{}')
    # PowerShell below 7.3 strips embedded double quotes when it builds a native command line, so
    # the JSON payload has to carry escaped quotes there; 7.3+ passes the string through verbatim.
    if ($PSVersionTable.PSVersion -lt [version]'7.3') { $Payload = $Payload -replace '"', '\"' }
    # 2>&1 on a native command raises a NativeCommandError under $ErrorActionPreference='Stop', so a
    # single DeprecationWarning on stderr would fail a command that exited 0 - and tear down a stack
    # that took 12 minutes to bring up. The exit code is the only authority here.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $output = & $Python $ArenaScript $Command $Payload --port $ArenaPort 2>&1 }
    finally { $ErrorActionPreference = $previous }
    if ($LASTEXITCODE -ne 0) { throw "arena_control $Command failed (exit $LASTEXITCODE): $(($output | Out-String).Trim())" }
}

function Stop-TrainingStack {
    if ($Script:Children.Count -eq 0) { return }
    Write-RunnerLog INFO 'Stopping the training stack.'
    # Ask the arena to end matches first so bots disconnect cleanly instead of timing out.
    try { Invoke-ArenaCommand 'stop_all' } catch { Write-RunnerLog WARN "stop_all failed (arena may already be down): $($_.Exception.Message)" }
    # Reverse dependency order: stop producing experience, then the consumers. Checkpoints are
    # written atomically, so a forced trainer kill cannot corrupt latest.pt, and the flat training
    # world is regenerated on demand.
    foreach ($name in @('worker', 'dashboard', 'paper', 'trainer')) {
        foreach ($child in @($Script:Children | Where-Object { $_.Name -eq $name })) {
            if ($child.Process.HasExited) { continue }
            try { & taskkill.exe /PID $child.Process.Id /T /F 2>$null | Out-Null } catch { }
            try { $child.Process.WaitForExit(15000) | Out-Null } catch { }
        }
    }
    $Script:Children = @()
    Remove-Item $StatePath -Force -ErrorAction SilentlyContinue
}

function Test-Prerequisites {
    $missing = @()
    $fatal = @()
    if (-not $Python -or -not (Test-Path $Python)) { $missing += "python interpreter ($Python)" }
    if (-not $Java) { $missing += 'java.exe (set MCAI_JAVA_EXE)' }
    if (-not $Node) { $missing += 'node.exe (set MCAI_NODE_EXE)' }
    foreach ($path in @($PaperJar, $WorkerScript, $DashboardScript, $ArenaScript, $ConfigureRuntimeScript)) {
        if (-not (Test-Path $path)) { $missing += $path }
    }
    # The operator accepts the Minecraft EULA during bootstrap; this script never accepts it.
    # Existence proves nothing: Paper WRITES eula.txt with eula=false the first time it is started
    # without one, then exits in two seconds - which is precisely the loop this check must catch.
    $eulaPath = Join-Path $Runtime 'eula.txt'
    if (-not (Test-Path -LiteralPath $eulaPath) -or
        ((Get-Content -LiteralPath $eulaPath -Raw -ErrorAction SilentlyContinue) -notmatch 'eula\s*=\s*true')) {
        $fatal += "$eulaPath (accept the Minecraft EULA during bootstrap: eula=true)"
    }
    if ($Mode -notin @('sword', 'crystal', 'combined')) { throw "MCAI_MODE must be sword, crystal, or combined (got '$Mode')." }
    if ($missing.Count -gt 0) { throw "Prerequisites missing - run the Azure bootstrap first: $($missing -join '; ')" }

    # Paper 1.12.2 will not boot on a JDK 17+; without this the only symptom is a 420 s readiness
    # timeout per cycle, forever.
    $javaMajor = Get-JavaMajorVersion $Java
    if ($null -eq $javaMajor) { $fatal += "could not read the Java version from $Java" }
    elseif ($javaMajor -lt 8 -or $javaMajor -gt 11) {
        $fatal += "$Java is Java $javaMajor; Paper 1.12.2 needs Java 8-11 (set MCAI_JAVA_EXE)"
    }

    # A venv whose torch wheel downloaded half-way passes every file check above and then fails 300
    # seconds at a time. Import once, loudly. Mirrors the guard in scripts\start-linux.sh.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $importOutput = & $Python '-c' 'import combat_ai' 2>&1 } finally { $ErrorActionPreference = $previous }
    if ($LASTEXITCODE -ne 0) {
        $fatal += "the trainer package does not import ($Python -c 'import combat_ai' exited $LASTEXITCODE): $(($importOutput | Out-String).Trim())"
    }

    if ($fatal.Count -gt 0) {
        # Retrying cannot fix any of these; make the supervisor give up instead of burning Spot hours.
        $Script:FatalReason = "Unrecoverable prerequisite failure: $($fatal -join '; ')"
        throw $Script:FatalReason
    }
}

function Start-TrainingStack {
    $Script:Children = @()
    $Script:StackUpAt = $null
    Clear-ForeignListeners
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $Script:CurrentRunDirectory = Join-Path $RunsBase "$Script:RunPrefix$stamp"
    New-Item -ItemType Directory -Force -Path $Script:CurrentRunDirectory, $Checkpoints | Out-Null
    $env:MCAI_RUN_DIR = $Script:CurrentRunDirectory
    Write-RunnerLog INFO "Run directory: $Script:CurrentRunDirectory (mode=$Mode bots=$($env:MCAI_BOT_COUNT))"

    # 1. Trainer. The worker's websocket target; nothing else can make progress without it.
    $checkpointDirectory = if ($env:MCAI_EXPLOITER_TARGET) { Join-Path $RepoRoot 'checkpoints\exploiter-active' } else { $Checkpoints }
    $trainerArguments = @('-m', 'combat_ai.cli', 'serve', '--host', '127.0.0.1', '--port', "$TrainerPort",
        '--checkpoints', $checkpointDirectory, '--cpu-threads', "$CpuThreads", '--rollout-steps', "$RolloutSteps")
    if ($env:MCAI_IMITATION_DATA) { $trainerArguments += @('--imitation-data', $env:MCAI_IMITATION_DATA) }
    if ($env:MCAI_INITIALIZE_FROM) {
        foreach ($initial in ($env:MCAI_INITIALIZE_FROM -split ';')) {
            if ($initial.Trim()) { $trainerArguments += @('--initialize-from', $initial.Trim()) }
        }
    }
    if ($env:MCAI_EXPLOITER_TARGET) { $trainerArguments += @('--exploiter-target', $env:MCAI_EXPLOITER_TARGET) }
    Start-Child 'trainer' $Python $trainerArguments (Join-Path $RepoRoot 'trainer')
    # Cold start loads torch and possibly a checkpoint; be generous on a shared-core Spot VM.
    if (-not (Wait-LocalPort $TrainerPort 300 'the trainer')) { throw "The trainer did not open port $TrainerPort. See trainer.log." }
    Write-RunnerLog INFO "Trainer ready on 127.0.0.1:$TrainerPort"

    # 2. Paper + the MCAIArena plugin. Its control socket is the arena readiness signal.
    Start-Child 'paper' $Java @("-Xms$JavaInitialMemory", "-Xmx$JavaMemory", '-jar', 'paper-1.12.2.jar', 'nogui') $Runtime
    if (-not (Wait-LocalPort $ArenaPort 420 'the arena control socket')) { throw "The arena did not open port $ArenaPort. See paper.log." }
    Write-RunnerLog INFO "Arena control ready on 127.0.0.1:$ArenaPort"

    # 3. Arena mode + resume. The socket can accept before the plugin's command handler is wired up,
    # so retry briefly rather than tearing the whole stack down over a race.
    $configured = $false
    for ($attempt = 1; $attempt -le 5 -and -not $configured; $attempt++) {
        try {
            Invoke-ArenaCommand 'set_mode' ('{"mode":"' + $Mode + '"}')
            Invoke-ArenaCommand 'resume'
            $configured = $true
        } catch {
            Write-RunnerLog WARN "Arena configuration attempt $attempt failed: $($_.Exception.Message)"
            Start-Sleep -Seconds 5
        }
    }
    if (-not $configured) { throw 'Could not configure the arena (set_mode/resume).' }

    # 4. Dashboard (loopback only; reach it through a tunnel). Non-fatal if it is slow to bind.
    Start-Child 'dashboard' $Node @($DashboardScript) $RepoRoot
    if (-not (Wait-LocalPort $DashboardPort 60 'the dashboard')) {
        Write-RunnerLog WARN "The dashboard did not bind port $DashboardPort; training continues."
    }

    # 5. Worker last: bots must not connect before the arena is accepting pairs.
    Start-Child 'worker' $Node @($WorkerScript) (Join-Path $RepoRoot 'worker')

    Save-RunnerState
    # Uptime is measured from here, not from the top of the loop iteration: the readiness waits above
    # can take 12 minutes on their own, and counting those as uptime would reset the backoff after
    # every failed start and disable it for exactly the slowest-to-detect faults.
    $Script:StackUpAt = Get-Date
    Write-RunnerLog INFO "Stack up. Dashboard via tunnel: http://127.0.0.1:$DashboardPort"
}

function Watch-TrainingStack {
    <# Returns the reason the stack needs recycling. Blocks until then. #>
    $trainerLog = Join-Path $Script:CurrentRunDirectory 'trainer.log'
    $lastLength = Get-LiveFileLength $trainerLog
    $lastGrowth = Get-Date
    $nextHousekeeping = (Get-Date).AddMinutes(1)
    while ($true) {
        if (Test-Path $StopSentinel) { return 'stop sentinel present' }
        foreach ($child in $Script:Children) {
            if ($child.Process.HasExited) {
                $tail = ''
                try { $tail = (Get-Content $child.ErrorLog -Tail 5 -ErrorAction SilentlyContinue) -join ' | ' } catch { }
                return "$($child.Name) exited with code $($child.Process.ExitCode). $tail"
            }
        }
        if ((Get-Date) -ge $nextHousekeeping) {
            $nextHousekeeping = (Get-Date).AddMinutes(1)
            Invoke-Housekeeping
            if (Test-ChildLogsOverCap) { return 'a child log hit the size cap' }
            $length = Get-LiveFileLength $trainerLog
            if ($length -ne $lastLength) { $lastLength = $length; $lastGrowth = Get-Date }
            elseif ($StallMinutes -gt 0 -and ((Get-Date) - $lastGrowth).TotalMinutes -ge $StallMinutes) {
                return "no trainer output for $StallMinutes minutes (stalled)"
            }
        }
        Start-Sleep -Seconds 5
    }
}

function Get-FailureDetail {
    <# Last 40 lines of whichever child log looks most like the cause, for the give-up marker. #>
    $lines = @()
    if (-not $Script:CurrentRunDirectory -or -not (Test-Path $Script:CurrentRunDirectory)) { return $lines }
    foreach ($log in @(Get-ChildItem $Script:CurrentRunDirectory -File -Filter '*.error.log' -ErrorAction SilentlyContinue |
            Sort-Object Length -Descending | Select-Object -First 1)) {
        try { $lines = @("--- $($log.Name) ---") + @(Get-Content $log.FullName -Tail 40 -ErrorAction SilentlyContinue) } catch { }
    }
    return $lines
}

function Write-GiveUpMarker {
    <# Machine-readable evidence for 04-monitor.ps1 and for anyone arriving over RDP. Without it the
       only trace of a permanently broken install is runner.log on a VM nobody is watching. #>
    param([string]$Reason, [string[]]$Detail)
    $marker = Join-Path $StateDirectory 'azure-runner-failed.json'
    try {
        @{
            reason        = $Reason
            at            = (Get-Date).ToString('o')
            repo_root     = $RepoRoot
            run_directory = $Script:CurrentRunDirectory
            detail        = @($Detail)
        } | ConvertTo-Json -Depth 4 | Set-Content -Path $marker -Encoding UTF8
        Write-RunnerLog ERROR "Wrote the give-up marker to $marker"
    } catch {
        Write-RunnerLog ERROR "Could not write the give-up marker: $($_.Exception.Message)"
    }
}

# ---------------------------------------------------------------------------------------- main ----

if ($Install -and $Uninstall) { throw 'Use either -Install or -Uninstall, not both.' }
if ($Install) { Register-BootTask; return }
if ($Uninstall) { Unregister-BootTask; return }

# One supervisor per machine: the boot task, a manual RDP launch and a Task Scheduler restart can
# otherwise overlap and fight over the same ports.
try {
    $Script:Mutex = [System.Threading.Mutex]::new($false, 'Global\MCAI-AzureWindowsRunner')
    if (-not $Script:Mutex.WaitOne(0)) {
        Write-RunnerLog WARN 'Another MCAI supervisor is already running on this machine; exiting.'
        return
    }
} catch {
    # Named mutexes are unavailable on some hosts; carry on rather than refusing to train.
    Write-RunnerLog WARN "Could not take the single-instance lock: $($_.Exception.Message)"
    $Script:Mutex = $null
}

$exitCode = 0
try {
    if (Test-Path $StopSentinel) {
        Write-RunnerLog WARN "$StopSentinel exists; not starting. Delete it to resume training."
        return
    }
    Write-RunnerLog INFO "MCAI supervisor starting. Repo=$RepoRoot python=$Python java=$Java node=$Node"
    # Log the resolved settings, not just the paths: a machine variable that failed to reach this
    # process shows up here as a silently wrong mode or bot count rather than as an error.
    Write-RunnerLog INFO ("Resolved config: mode=$Mode bots=$($env:MCAI_BOT_COUNT) runtime=$Runtime checkpoints=$Checkpoints " +
        "rollout_steps=$RolloutSteps max_pairs=$(if ($env:MCAI_MAX_PAIRS) { $env:MCAI_MAX_PAIRS } else { 'plugin default' })")
    Remove-StaleProcesses
    # A previous give-up marker is stale the moment we successfully start again.
    Remove-Item (Join-Path $StateDirectory 'azure-runner-failed.json') -Force -ErrorAction SilentlyContinue

    $failures = 0
    $backoffSeconds = 15
    $lastFailure = 'unknown'
    $lastDetail = @()
    while ($true) {
        $Script:StackUpAt = $null
        $Script:FatalReason = $null
        try {
            Test-Prerequisites
            Start-TrainingStack
            $reason = Watch-TrainingStack
            if ($reason -eq 'stop sentinel present') {
                Write-RunnerLog INFO 'Stop sentinel detected; shutting down.'
                break
            }
            $lastFailure = $reason
            Write-RunnerLog ERROR "Recycling the stack: $reason"
        } catch {
            $lastFailure = $_.Exception.Message
            $lastDetail = Get-FailureDetail
            Write-RunnerLog ERROR "Stack failed: $lastFailure"
        } finally {
            Stop-TrainingStack
            # Housekeeping belongs here, not only on the healthy path: a startup crash loop is
            # exactly the case that creates a run directory and a multi-MB paper.log every cycle.
            $Script:CurrentRunDirectory = $null
            Invoke-Housekeeping
        }

        if ($Script:FatalReason) {
            Write-RunnerLog ERROR "$Script:FatalReason - retrying cannot fix this; giving up."
            Write-GiveUpMarker $Script:FatalReason $lastDetail
            $exitCode = 2
            break
        }
        if ($Once) { Write-RunnerLog INFO '-Once was set; not restarting.'; $exitCode = 1; break }

        # Reset the backoff only once a stack has actually been UP for ten minutes. Measuring from
        # the top of the iteration would count the 12 minutes of readiness waiting a never-starting
        # stack burns, resetting the counter forever and disabling the guard entirely.
        if ($Script:StackUpAt -and ((Get-Date) - $Script:StackUpAt).TotalMinutes -ge 10) { $failures = 0 } else { $failures++ }

        if ($failures -ge $MaxConsecutiveFailures) {
            $summary = "$failures consecutive failures without a working stack; last error: $lastFailure"
            Write-RunnerLog ERROR "Giving up. $summary"
            Write-GiveUpMarker $summary $lastDetail
            $exitCode = 2
            break
        }

        $backoffSeconds = [math]::Min(300, 15 * [math]::Pow(2, [math]::Min($failures, 5)))
        Write-RunnerLog INFO "Restarting in $backoffSeconds s (consecutive fast failures: $failures/$MaxConsecutiveFailures). Training resumes from checkpoints\latest.pt."
        for ($waited = 0; $waited -lt $backoffSeconds; $waited += 5) {
            if (Test-Path $StopSentinel) { break }
            Start-Sleep -Seconds 5
        }
        if (Test-Path $StopSentinel) { Write-RunnerLog INFO 'Stop sentinel detected during backoff; shutting down.'; break }
    }

    if ($exitCode -eq 2) {
        # Leave the sentinel so the boot task does not resurrect a known-broken install on the next
        # reboot; deleting it is the operator's explicit "I fixed it" signal.
        try { Set-Content -Path $StopSentinel -Value "supervisor gave up at $(Get-Date -Format 'o')" -Encoding UTF8 } catch { }
        if ($ShutdownOnGiveUp) {
            Write-RunnerLog ERROR 'Powering the VM off in 60 s (-ShutdownOnGiveUp). NOTE: a guest shutdown leaves the VM Stopped, not Deallocated - Azure keeps billing compute until it is deallocated from the control plane.'
            try { & shutdown.exe /s /t 60 /c 'MCAI supervisor gave up; see .mcai\azure-runner-failed.json' | Out-Null } catch { }
        }
    }
} finally {
    # Runs for Ctrl-C and for a normal exit, so children are not orphaned. It does NOT run for a hard
    # kill (Task Scheduler's Stop uses TerminateProcess, and PowerShell finally blocks do not run for
    # that); Remove-StaleProcesses and Clear-ForeignListeners cover that case on the next start.
    Stop-TrainingStack
    if ($Script:Mutex) {
        try { $Script:Mutex.ReleaseMutex() } catch { }
        $Script:Mutex.Dispose()
    }
    Write-RunnerLog INFO 'Supervisor stopped.'
}
exit $exitCode
