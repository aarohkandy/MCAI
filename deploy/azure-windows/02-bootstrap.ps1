<#
.SYNOPSIS
    Unattended MCAI bootstrap for a bare Windows Server 2022 Azure VM.

.DESCRIPTION
    Installs the toolchain, clones/updates the repo, builds every component, fetches the
    (checksum-verified) Paper 1.12.2 server, and writes the runtime + arena configuration.
    Runs to completion with no prompts, no GUI and no console attached, so it is safe from a
    Custom Script Extension, a scheduled task at boot, or an RDP session.

    It is idempotent: re-running after a Spot eviction or a `git push` re-uses everything already
    installed and only redoes the cheap steps. The rebuild steps ARE destructive (npm ci, the venv,
    the plugin jar), so it refuses to run while the training stack is up - stop it first, or pass
    -StopRunningStack. It never starts training - that is the run script's job (see the README).

    Windows-only cmdlets/APIs are used throughout (registry, winget, .exe installers); the script
    parses everywhere but only runs on Windows.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File C:\mcai\MCAI\deploy\azure-windows\02-bootstrap.ps1 -AcceptEula

.EXAMPLE
    # Equivalent EULA opt-in, matching deploy/azure/entrypoint.sh and what 01-provision.ps1 prints:
    $env:MCAI_ACCEPT_EULA = 'true'
    powershell -ExecutionPolicy Bypass -File .\deploy\azure-windows\02-bootstrap.ps1
#>
[CmdletBinding()]
param(
    # Short path on purpose: npm's node_modules tree plus Maven's repo get close to Windows'
    # 260-character MAX_PATH, and long-path support is a machine-wide setting we refuse to change.
    [string]$InstallRoot = 'C:\mcai',
    [string]$RepoUrl = 'https://github.com/aarohkandy/MCAI.git',
    [string]$Branch = 'fix/verified-training-stack',
    [string]$RepoPath,
    # 0 = size from the VM's core count (~1 arena pair per 4 vCPU, 2 bots per pair).
    [int]$BotCount = 0,
    [int]$MaxPairs = 0,
    [ValidateSet('sword', 'crystal', 'combined')][string]$Mode = 'sword',
    # The operator must opt in to the Minecraft EULA explicitly; nothing here accepts it for them.
    # MCAI_ACCEPT_EULA=true in the environment is the equivalent opt-in (same gate as
    # deploy/azure/entrypoint.sh, and the form 01-provision.ps1 prints).
    [switch]$AcceptEula,
    # Must match 03-run-training.ps1's -TaskName: the preflight refuses to rebuild underneath it.
    [string]$TaskName = 'MCAI-Training',
    # Stop a running stack rather than refusing to run. Off by default: killing a training run is
    # the operator's decision, never a side effect of re-running the bootstrap.
    [switch]$StopRunningStack
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
# Invoke-WebRequest in Windows PowerShell 5.1 renders a progress bar per chunk, which makes
# 100 MB+ downloads take minutes longer and spams a non-interactive log.
$ProgressPreference = 'SilentlyContinue'
# WS2022's defaults are fine, but be explicit: several of these endpoints are TLS 1.2 only.
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12 -bor [Net.SecurityProtocolType]::Tls11

$UserAgent = 'mcai-setup'
$MavenVersion = '3.9.9'
$PaperVersion = '1.12.2'
# Paper 1.12.2 is a Java 8 era server: CraftBukkit's internals reflect into java.base and use
# sun.misc.Unsafe, which Java 16 made illegal by default, so it will NOT boot on a JDK 17+.
# 8 is the reference runtime, 11 still works; anything newer is rejected below.
$RuntimeJavaMin = 8
$RuntimeJavaMax = 11

# Files that must exist for a directory to be the MCAI repo rather than "some git checkout".
$RepoSentinels = @('trainer\pyproject.toml', 'worker\package.json', 'server-plugin\pom.xml', 'scripts\configure_runtime.py')

# All resolved in preflight (the repo path can turn out to BE $InstallRoot - see Resolve-Layout).
$SupportRoot = $null
$ToolsRoot = $null
$CacheDir = $null
$LogDir = $null
$MavenRepo = $null
$StatePath = $null
$Runtime = $null
$script:LogFile = $null
# Set when -StopRunningStack disables the boot task, so it can be put back exactly as it was.
$script:TaskDisabled = $false

# ---------------------------------------------------------------------------- logging

function Write-LogLine([string]$Text) {
    Write-Host $Text
    if ($script:LogFile) { Add-Content -LiteralPath $script:LogFile -Value $Text -Encoding UTF8 }
}

function Write-Log([string]$Message, [string]$Level = 'INFO') {
    Write-LogLine ('[{0}] [{1}] {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level, $Message)
}

function Write-Step([string]$Message) {
    Write-LogLine ''
    Write-Log ('=== {0}' -f $Message)
}

# ---------------------------------------------------------------------------- process helpers

function Invoke-Native {
    <#  Runs a native tool, streams its output into the log, and throws on a non-zero exit code. #>
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [string[]]$Arguments = @(),
        [string]$WorkingDirectory,
        [hashtable]$Environment = @{}
    )
    # An absolute path that does not exist means the installer above lied about where it put the
    # tool. Say so, instead of letting `&` fail and leaving $LASTEXITCODE unset (see below).
    if (($FilePath -match '[\\/]') -and -not (Test-Path -LiteralPath $FilePath -PathType Leaf)) {
        throw ('{0} is not an executable file; the step that was supposed to install it did not put it there.' -f $FilePath)
    }
    Write-Log ('run: {0} {1}' -f $FilePath, ($Arguments -join ' ')) 'EXEC'
    $saved = @{}
    foreach ($key in $Environment.Keys) {
        $saved[$key] = [Environment]::GetEnvironmentVariable($key, 'Process')
        [Environment]::SetEnvironmentVariable($key, $Environment[$key], 'Process')
    }
    if ($WorkingDirectory) { Push-Location -LiteralPath $WorkingDirectory }
    $code = -1
    # The full output goes to the log file, but the failure message has to carry enough of the tool's
    # own words to be actionable on its own - a bare "exit code 128" is not.
    $tail = New-Object 'System.Collections.Generic.Queue[string]'
    try {
        # git/npm/mvn write progress and warnings to stderr. Merging that into the success stream
        # while $ErrorActionPreference is 'Stop' turns ordinary chatter into a terminating
        # NativeCommandError, so relax it here and trust the exit code instead.
        $previous = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        try {
            & $FilePath @Arguments 2>&1 | ForEach-Object {
                $line = [string]$_
                Write-LogLine ('    ' + $line)
                if ($line.Trim()) {
                    $tail.Enqueue($line.Trim())
                    if ($tail.Count -gt 8) { $tail.Dequeue() | Out-Null }
                }
            }
            # If the process never launched, $LASTEXITCODE was never set and StrictMode turns reading
            # it into a terminating error that hides the real cause.
            $code = if (Test-Path variable:LASTEXITCODE) { $LASTEXITCODE } else { -1 }
        } finally { $ErrorActionPreference = $previous }
    } finally {
        if ($WorkingDirectory) { Pop-Location }
        foreach ($key in $Environment.Keys) { [Environment]::SetEnvironmentVariable($key, $saved[$key], 'Process') }
    }
    if ($code -ne 0) {
        $lastOutput = (@($tail.ToArray()) -join ' | ')
        throw ('{0} failed with exit code {1}. Command: {0} {2}. Last output: {3}. See {4}.' -f $FilePath, $code, ($Arguments -join ' '), $lastOutput, $script:LogFile)
    }
}

function Invoke-NativeCapture {
    <#  Same, but returns the tool's output instead of logging it (for version probes). #>
    param([Parameter(Mandatory = $true)][string]$FilePath, [string[]]$Arguments = @())
    if (($FilePath -match '[\\/]') -and -not (Test-Path -LiteralPath $FilePath -PathType Leaf)) {
        throw ('{0} is not an executable file; the step that was supposed to install it did not put it there.' -f $FilePath)
    }
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $output = & $FilePath @Arguments 2>&1 } finally { $ErrorActionPreference = $previous }
    $code = if (Test-Path variable:LASTEXITCODE) { $LASTEXITCODE } else { -1 }
    $text = (($output | ForEach-Object { [string]$_ }) -join "`n").Trim()
    if ($code -ne 0) { throw ('{0} {1} failed (exit {2}): {3}' -f $FilePath, ($Arguments -join ' '), $code, $text) }
    return $text
}

function Get-Field {
    <#  StrictMode makes a missing property a terminating error, and these are third-party JSON
        payloads whose shape can change. Probe instead, so we can raise a useful message. #>
    param($Object, [string]$Name)
    if ($null -eq $Object) { return $null }
    $property = $Object.PSObject.Properties[$Name]
    if ($null -eq $property) { return $null }
    return $property.Value
}

# ---------------------------------------------------------------------------- download helpers

# Every metadata call gets a hard deadline. Windows PowerShell 5.1 defaults -TimeoutSec to 0, i.e.
# wait forever, and a half-open connection (common through Azure's NAT after an eviction) would then
# park the bootstrap silently while the VM keeps billing.
$WebTimeoutSec = 120
$WebRetryDelays = @(5, 15, 45)

function Test-TransientWebFailure($ErrorRecord) {
    <#  True only for transport faults. A checksum mismatch, a 404 or a parse failure will never
        succeed on retry, and retrying them just burns paid minutes. #>
    if ($ErrorRecord.FullyQualifiedErrorId -match 'WebCmdletWebResponseException|HttpResponseException|WebException|OperationTimeout') {
        # 404/410 are answers, not faults: the artifact genuinely is not there.
        try {
            $response = $ErrorRecord.Exception.Response
            if ($response) {
                $status = [int]$response.StatusCode
                if ($status -eq 404 -or $status -eq 410) { return $false }
            }
        } catch { }
        return $true
    }
    $exception = $ErrorRecord.Exception
    while ($exception) {
        if ($exception -is [System.Net.WebException] -or $exception -is [System.IO.IOException] -or
            $exception -is [System.TimeoutException] -or
            $exception.GetType().FullName -eq 'System.Net.Http.HttpRequestException') { return $true }
        $exception = $exception.InnerException
    }
    return $false
}

function Invoke-WithRetry {
    <#  4 attempts, 5/15/45 s backoff. GitHub 403s, Apache 503s and TLS resets are routine; without
        this one of them ends provisioning and leaves an idle VM billing until a human notices. #>
    param(
        [Parameter(Mandatory = $true)][scriptblock]$Action,
        [Parameter(Mandatory = $true)][string]$Description,
        [int]$Attempts = 4
    )
    for ($attempt = 1; ; $attempt++) {
        try { return (& $Action) }
        catch {
            if ($attempt -ge $Attempts -or -not (Test-TransientWebFailure $_)) { throw }
            $wait = $WebRetryDelays[[Math]::Min($attempt - 1, $WebRetryDelays.Count - 1)]
            Write-Log ('{0} failed ({1}); retrying in {2}s ({3}/{4})' -f $Description, $_.Exception.Message, $wait, $attempt, ($Attempts - 1)) 'WARN'
            Start-Sleep -Seconds $wait
        }
    }
}

function Invoke-JsonApi([string]$Uri) {
    return Invoke-WithRetry -Description ('GET ' + $Uri) -Action {
        Invoke-RestMethod -Uri $Uri -UseBasicParsing -TimeoutSec $WebTimeoutSec -Headers @{ 'User-Agent' = $UserAgent }
    }
}

function Invoke-TextApi([string]$Uri) {
    return Invoke-WithRetry -Description ('GET ' + $Uri) -Action {
        (Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec $WebTimeoutSec -Headers @{ 'User-Agent' = $UserAgent }).Content
    }
}

function Get-RemoteFile {
    param(
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$OutFile,
        [string]$Checksum,
        [string]$Algorithm = 'SHA256'
    )
    Write-Log ('downloading {0}' -f $Uri)
    Invoke-WithRetry -Description ('download ' + $Uri) -Action {
        # Remove inside the retry: a partial file from a reset connection must never be handed to
        # Get-FileHash (or, worse, extracted) on the next attempt.
        if (Test-Path -LiteralPath $OutFile) { Remove-Item -LiteralPath $OutFile -Force }
        Invoke-WebRequest -Uri $Uri -OutFile $OutFile -UseBasicParsing -TimeoutSec 1800 `
            -Headers @{ 'User-Agent' = $UserAgent }
    } | Out-Null
    # Deliberately outside the retry: a checksum mismatch is corruption or tampering, not a transient
    # fault, and must stay fatal.
    if ($Checksum) {
        $actual = (Get-FileHash -LiteralPath $OutFile -Algorithm $Algorithm).Hash.ToLowerInvariant()
        if ($actual -ne $Checksum.ToLowerInvariant()) {
            Remove-Item -LiteralPath $OutFile -Force -ErrorAction SilentlyContinue
            throw ('{0} checksum mismatch for {1}: expected {2}, got {3}. Refusing to install.' -f $Algorithm, $Uri, $Checksum, $actual)
        }
        Write-Log ('{0} verified for {1}' -f $Algorithm, (Split-Path -Leaf $OutFile))
    } else {
        Write-Log ('no published checksum for {0}; trusting HTTPS only' -f (Split-Path -Leaf $OutFile)) 'WARN'
    }
}

function Expand-ToDirectory([string]$ZipPath, [string]$Destination) {
    # ZipFile is ~10x faster than Expand-Archive on Windows PowerShell for 100 MB+ archives.
    try { Add-Type -AssemblyName System.IO.Compression.FileSystem -ErrorAction SilentlyContinue } catch { }
    if (Test-Path -LiteralPath $Destination) { Remove-Item -LiteralPath $Destination -Recurse -Force }
    [System.IO.Compression.ZipFile]::ExtractToDirectory($ZipPath, $Destination)
}

function Install-ZipTool {
    <#  Downloads a zip, verifies it, and lands it at $Destination atomically enough that a
        half-extracted tool is never left behind for the next run to "find". #>
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Uri,
        [Parameter(Mandatory = $true)][string]$Destination,
        [string]$Checksum,
        [string]$Algorithm = 'SHA256',
        # Set when the archive wraps everything in a single top-level directory (node, maven, jdk).
        [switch]$Flatten
    )
    $zip = Join-Path $CacheDir ($Name + '.zip')
    $stage = Join-Path $CacheDir ($Name + '-stage')
    Get-RemoteFile -Uri $Uri -OutFile $zip -Checksum $Checksum -Algorithm $Algorithm
    Expand-ToDirectory $zip $stage
    $source = $stage
    if ($Flatten) {
        $entries = @(Get-ChildItem -LiteralPath $stage)
        if ($entries.Count -eq 1 -and $entries[0].PSIsContainer) { $source = $entries[0].FullName }
    }
    if (Test-Path -LiteralPath $Destination) { Remove-Item -LiteralPath $Destination -Recurse -Force }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    Move-Item -LiteralPath $source -Destination $Destination
    Remove-Item -LiteralPath $stage -Recurse -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $zip -Force -ErrorAction SilentlyContinue
    Write-Log ('installed {0} -> {1}' -f $Name, $Destination)
}

# ---------------------------------------------------------------------------- environment

function Add-ProcessPath([string]$Directory) {
    if (-not $Directory -or -not (Test-Path -LiteralPath $Directory)) { return }
    $entries = @($env:Path -split ';' | Where-Object { $_ })
    if ($entries -notcontains $Directory) { $env:Path = ($Directory + ';' + $env:Path) }
}

function Add-MachinePath {
    <#  Written through the raw registry API on purpose: [Environment]::SetEnvironmentVariable on
        'Path' rewrites the value as REG_SZ, permanently expanding %SystemRoot% and friends in the
        machine PATH. This keeps it REG_EXPAND_SZ. (Windows-only API.)

        -Prepend puts the directory first AND moves it there if it is already listed later, which is
        the only way to actually win against a pre-installed tool of the same name. #>
    param([Parameter(Mandatory = $true)][string]$Directory, [switch]$Prepend)
    $keyPath = 'SYSTEM\CurrentControlSet\Control\Session Manager\Environment'
    $key = [Microsoft.Win32.Registry]::LocalMachine.OpenSubKey($keyPath, $true)
    if ($null -eq $key) { throw "Cannot open $keyPath for writing; run this script as Administrator." }
    try {
        $current = [string]$key.GetValue('Path', '', [Microsoft.Win32.RegistryValueOptions]::DoNotExpandEnvironmentNames)
        $entries = @($current -split ';' | Where-Object { $_ })
        $updated = $null
        if ($Prepend) {
            if ($entries.Count -eq 0 -or $entries[0] -ne $Directory) {
                $updated = @($Directory) + @($entries | Where-Object { $_ -ne $Directory })
            }
        } elseif ($entries -notcontains $Directory) {
            $updated = @($entries) + $Directory
        }
        if ($updated) {
            $key.SetValue('Path', ($updated -join ';'), [Microsoft.Win32.RegistryValueKind]::ExpandString)
            Write-Log ('{0} machine PATH: {1}' -f $(if ($Prepend) { 'prepended to' } else { 'added to' }), $Directory)
        }
    } finally { $key.Close() }
    Add-ProcessPath $Directory
}

function Set-MachineVariable([string]$Name, [string]$Value) {
    [Environment]::SetEnvironmentVariable($Name, $Value, 'Machine')
    [Environment]::SetEnvironmentVariable($Name, $Value, 'Process')
    Write-Log ('{0}={1}' -f $Name, $Value)
}

function Write-TextFile([string]$Path, [string]$Content) {
    # No BOM. SnakeYAML (Bukkit's config loader) and Paper's eula reader both mis-parse a leading
    # UTF-8 BOM, and Set-Content -Encoding UTF8 on PowerShell 5.1 always emits one.
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Path) | Out-Null
    [System.IO.File]::WriteAllText($Path, $Content, (New-Object System.Text.UTF8Encoding($false)))
}

# ---------------------------------------------------------------------------- tool discovery

function Get-JavaMajor([string]$JavaExe) {
    <#  Returns 8 for "1.8.0_422", 17 for "17.0.11", or $null if the probe fails. #>
    if (-not (Test-Path -LiteralPath $JavaExe)) { return $null }
    try { $text = Invoke-NativeCapture $JavaExe @('-version') } catch { return $null }
    if ($text -notmatch 'version "(\d+)(?:\.(\d+))?') { return $null }
    $first = [int]$Matches[1]
    if ($first -eq 1 -and $Matches.Count -ge 3) { return [int]$Matches[2] }
    return $first
}

function Find-JavaCandidates {
    $candidates = New-Object System.Collections.Generic.List[string]
    if ($env:MCAI_JAVA_EXE) { $candidates.Add($env:MCAI_JAVA_EXE) }
    foreach ($pattern in @(
            (Join-Path $ToolsRoot 'jdk*\bin\java.exe'),
            "$env:ProgramFiles\Eclipse Adoptium\jdk-*\bin\java.exe",
            "$env:ProgramFiles\Microsoft\jdk-*\bin\java.exe",
            "$env:ProgramFiles\Java\jdk*\bin\java.exe")) {
        Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue |
            Sort-Object FullName -Descending | ForEach-Object { $candidates.Add($_.FullName) }
    }
    $onPath = Get-Command java.exe -ErrorAction SilentlyContinue
    if ($onPath) { $candidates.Add($onPath.Source) }
    return @($candidates | Select-Object -Unique)
}

function Resolve-Jdk {
    <#  A JDK usable for BOTH jobs: javac present (the plugin build) and a major version Paper
        1.12.2 will actually boot on. Returns $null when nothing suitable is installed. #>
    foreach ($java in Find-JavaCandidates) {
        $major = Get-JavaMajor $java
        if ($null -eq $major) { continue }
        $javaHome = Split-Path -Parent (Split-Path -Parent $java)
        $javac = Join-Path $javaHome 'bin\javac.exe'
        if ($major -lt $RuntimeJavaMin -or $major -gt $RuntimeJavaMax) {
            Write-Log ('skipping Java {0} at {1}: Paper {2} needs Java {3}-{4}' -f $major, $java, $PaperVersion, $RuntimeJavaMin, $RuntimeJavaMax) 'WARN'
            continue
        }
        if (-not (Test-Path -LiteralPath $javac)) {
            Write-Log ('skipping {0}: JRE only, no javac to build the plugin with' -f $java) 'WARN'
            continue
        }
        return [pscustomobject]@{ Home = $javaHome; Java = $java; Major = $major }
    }
    return $null
}

$script:WinGetChecked = $false
$script:WinGetPath = $null

function Get-WinGet {
    <#  Windows Server 2022 ships without App Installer, so winget is usually absent. Treat it as a
        bonus, never a requirement - every tool below has a direct-download path. #>
    if (-not $script:WinGetChecked) {
        $script:WinGetChecked = $true
        $command = Get-Command winget.exe -ErrorAction SilentlyContinue
        if ($command) { $script:WinGetPath = $command.Source }
        else { Write-Log 'winget is not available on this host; using direct downloads' 'WARN' }
    }
    return $script:WinGetPath
}

function Install-WinGetPackage([string]$Id, [string]$ExtraArgs) {
    $winget = Get-WinGet
    if (-not $winget) { return $false }
    $arguments = @('install', '--id', $Id, '--exact', '--source', 'winget', '--silent',
        '--accept-source-agreements', '--accept-package-agreements', '--disable-interactivity')
    if ($ExtraArgs) { $arguments += ($ExtraArgs -split ' ') }
    try {
        Invoke-Native -FilePath $winget -Arguments $arguments
        return $true
    } catch {
        Write-Log ('winget could not install {0} ({1}); falling back to a direct download' -f $Id, $_.Exception.Message) 'WARN'
        return $false
    }
}

# ---------------------------------------------------------------------------- installers

function Install-Git {
    $existing = Get-Command git.exe -ErrorAction SilentlyContinue
    if ($existing) { Write-Log ('git already present: {0}' -f $existing.Source); return $existing.Source }

    $portable = Join-Path $ToolsRoot 'git\cmd\git.exe'
    if (Test-Path -LiteralPath $portable) { Add-MachinePath (Split-Path -Parent $portable); return $portable }

    if (Install-WinGetPackage 'Git.Git' '--scope machine') {
        Add-ProcessPath "$env:ProgramFiles\Git\cmd"
        $installed = Get-Command git.exe -ErrorAction SilentlyContinue
        if ($installed) { return $installed.Source }
    }

    # MinGit: the redistributable Git for Windows payload. No installer, no admin, no post-install
    # reboot - it is a zip with a working git.exe, which is all the clone below needs.
    $release = Invoke-JsonApi 'https://api.github.com/repos/git-for-windows/git/releases/latest'
    $asset = @(Get-Field $release 'assets' | Where-Object { $_.name -like 'MinGit-*-64-bit.zip' -and $_.name -notlike '*busybox*' }) | Select-Object -First 1
    if (-not $asset) { throw 'No MinGit 64-bit asset in the latest git-for-windows release; install Git manually and re-run.' }
    Install-ZipTool -Name 'mingit' -Uri $asset.browser_download_url -Destination (Join-Path $ToolsRoot 'git')
    if (-not (Test-Path -LiteralPath $portable)) { throw "MinGit extracted but $portable is missing." }
    Add-MachinePath (Split-Path -Parent $portable)
    return $portable
}

function Install-Node {
    $existing = Get-Command node.exe -ErrorAction SilentlyContinue
    if ($existing) {
        $major = [int]((Invoke-NativeCapture $existing.Source @('--version')).TrimStart('v').Split('.')[0])
        if ($major -ge 20) { Write-Log ('node already present: {0} ({1})' -f $existing.Source, $major); return $existing.Source }
        Write-Log ('node {0} is too old; installing the 20 LTS line' -f $major) 'WARN'
    }
    $target = Join-Path $ToolsRoot 'node'
    $exe = Join-Path $target 'node.exe'
    if (Test-Path -LiteralPath $exe) { Add-MachinePath $target; return $exe }

    # Deliberately NOT winget: the OpenJS.NodeJS.LTS package floats to whatever the current LTS is
    # (22/24 today), while this stack is validated on 20. nodejs.org publishes SHASUMS256.txt next
    # to the archives, so the zip path is both pinned and checksum-verified.
    $index = Invoke-TextApi 'https://nodejs.org/dist/latest-v20.x/SHASUMS256.txt'
    $entry = $null
    foreach ($line in ($index -split "`n")) {
        if ($line.Trim() -match '^([0-9a-fA-F]{64})\s+\*?(node-v20\.[0-9.]+-win-x64\.zip)$') {
            $entry = [pscustomobject]@{ Sha = $Matches[1]; Name = $Matches[2] }
            break
        }
    }
    if (-not $entry) { throw 'Could not find a node-v20.x-win-x64.zip entry in nodejs.org SHASUMS256.txt.' }
    Install-ZipTool -Name 'node' -Uri ('https://nodejs.org/dist/latest-v20.x/{0}' -f $entry.Name) `
        -Checksum $entry.Sha -Destination $target -Flatten
    if (-not (Test-Path -LiteralPath $exe)) { throw "Node archive extracted but $exe is missing." }
    Add-MachinePath $target
    return $exe
}

function Install-Jdk {
    $jdk = Resolve-Jdk
    if ($jdk) { Write-Log ('JDK already present: Java {0} at {1}' -f $jdk.Major, $jdk.Home); return $jdk }

    if (Install-WinGetPackage 'EclipseAdoptium.Temurin.8.JDK') {
        $jdk = Resolve-Jdk
        if ($jdk) { return $jdk }
        Write-Log 'winget reported success but no usable JDK 8 appeared; falling back to the Adoptium archive' 'WARN'
    }

    # Temurin 8: the reference runtime for Paper 1.12.2, and new enough (javac -source/-target 1.8
    # is the plugin's own compile level) to build the plugin too, so one JDK covers both roles.
    $assets = Invoke-JsonApi 'https://api.adoptium.net/v3/assets/latest/8/hotspot?os=windows&architecture=x64&image_type=jdk'
    $package = $null
    foreach ($asset in @($assets)) {
        $binary = Get-Field $asset 'binary'
        $candidate = Get-Field $binary 'package'
        if ($candidate -and (Get-Field $candidate 'name') -like '*.zip') { $package = $candidate; break }
    }
    if (-not $package) { throw 'The Adoptium API returned no Windows x64 JDK 8 zip; install a JDK 8 manually and re-run.' }
    Install-ZipTool -Name 'jdk8' -Uri $package.link -Checksum (Get-Field $package 'checksum') `
        -Destination (Join-Path $ToolsRoot 'jdk8') -Flatten
    $jdk = Resolve-Jdk
    if (-not $jdk) { throw 'Temurin 8 was extracted but no usable JDK was detected afterwards.' }
    return $jdk
}

function Install-Maven {
    $mavenHome = Join-Path $ToolsRoot ('apache-maven-' + $MavenVersion)
    $exe = Join-Path $mavenHome 'bin\mvn.cmd'
    if (Test-Path -LiteralPath $exe) { Write-Log ('maven already present: {0}' -f $exe); return $exe }
    # archive.apache.org keeps every release forever (the dlcdn mirror drops old ones) and publishes
    # a .sha512 next to each artifact.
    $base = 'https://archive.apache.org/dist/maven/maven-3/{0}/binaries/apache-maven-{0}-bin.zip' -f $MavenVersion
    $sha = ((Invoke-TextApi ($base + '.sha512')) -split '\s+')[0].Trim()
    Install-ZipTool -Name 'maven' -Uri $base -Checksum $sha -Algorithm 'SHA512' -Destination $mavenHome -Flatten
    if (-not (Test-Path -LiteralPath $exe)) { throw "Maven archive extracted but $exe is missing." }
    return $exe
}

function Install-Uv {
    $existing = Get-Command uv.exe -ErrorAction SilentlyContinue
    if ($existing) { return $existing.Source }
    $target = Join-Path $ToolsRoot 'uv'
    $exe = Join-Path $target 'uv.exe'
    if (Test-Path -LiteralPath $exe) { Add-MachinePath $target; return $exe }
    if (Install-WinGetPackage 'astral-sh.uv') {
        $installed = Get-Command uv.exe -ErrorAction SilentlyContinue
        if ($installed) { return $installed.Source }
    }
    $release = Invoke-JsonApi 'https://api.github.com/repos/astral-sh/uv/releases/latest'
    $assets = @(Get-Field $release 'assets')
    $asset = @($assets | Where-Object { $_.name -eq 'uv-x86_64-pc-windows-msvc.zip' }) | Select-Object -First 1
    if (-not $asset) { throw 'No uv Windows x64 asset in the latest release; install Python 3.12 manually and re-run.' }
    $checksum = $null
    $shaAsset = @($assets | Where-Object { $_.name -eq ($asset.name + '.sha256') }) | Select-Object -First 1
    if ($shaAsset) {
        $checksum = ((Invoke-TextApi $shaAsset.browser_download_url) -split '\s+')[0].Trim()
    }
    Install-ZipTool -Name 'uv' -Uri $asset.browser_download_url -Checksum $checksum -Destination $target
    if (-not (Test-Path -LiteralPath $exe)) { throw "uv archive extracted but $exe is missing." }
    Add-MachinePath $target
    return $exe
}

function Test-Python312([string]$Exe) {
    if (-not $Exe -or -not (Test-Path -LiteralPath $Exe)) { return $false }
    try { $version = Invoke-NativeCapture $Exe @('-c', 'import sys; print("%d.%d" % sys.version_info[:2])') } catch { return $false }
    return ($version.Trim() -eq '3.12')
}

function Install-Python312 {
    $candidates = New-Object System.Collections.Generic.List[string]
    if ($env:MCAI_PYTHON) { $candidates.Add($env:MCAI_PYTHON) }
    foreach ($pattern in @(
            "$env:ProgramFiles\Python312\python.exe",
            "${env:ProgramFiles(x86)}\Python312\python.exe",
            "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
            (Join-Path $ToolsRoot 'python312\python.exe'))) {
        Get-ChildItem -Path $pattern -ErrorAction SilentlyContinue | ForEach-Object { $candidates.Add($_.FullName) }
    }
    foreach ($candidate in @($candidates | Select-Object -Unique)) {
        if (Test-Python312 $candidate) { Write-Log ('python 3.12 already present: {0}' -f $candidate); return $candidate }
    }

    if (Install-WinGetPackage 'Python.Python.3.12' '--scope machine') {
        foreach ($path in @("$env:ProgramFiles\Python312\python.exe", "${env:ProgramFiles(x86)}\Python312\python.exe")) {
            if (Test-Python312 $path) { return $path }
        }
        Write-Log 'winget installed Python but no 3.12 interpreter was found where expected' 'WARN'
    }

    # Fallback with no MSI and no admin: uv fetches a standalone CPython build (it verifies its own
    # download) and tells us where it landed. The interpreter is a normal CPython with venv+pip.
    $uv = Install-Uv
    Invoke-Native -FilePath $uv -Arguments @('python', 'install', '3.12')
    $found = (Invoke-NativeCapture $uv @('python', 'find', '3.12')).Trim()
    if (-not (Test-Python312 $found)) { throw "uv installed Python 3.12 but '$found' does not report 3.12." }
    Write-Log ('python 3.12 (uv-managed): {0}' -f $found)
    return $found
}

# ---------------------------------------------------------------------------- layout

function Test-McaiCheckout([string]$Path) {
    if (-not (Test-Path -LiteralPath (Join-Path $Path '.git'))) { return $false }
    foreach ($file in $RepoSentinels) {
        if (-not (Test-Path -LiteralPath (Join-Path $Path $file))) { return $false }
    }
    return $true
}

function Resolve-Layout {
    <#  01-provision.ps1 tells the operator to `git clone ... C:\MCAI`, and $InstallRoot defaults to
        C:\mcai - the SAME directory, because Windows paths are case-insensitive. Cloning
        $InstallRoot\MCAI on top of that gives two checkouts, and every edit the operator makes in
        the one they cd'd into is silently ignored. Detect it and use their checkout in place. #>
    if (-not $RepoPath) {
        if (Test-McaiCheckout $InstallRoot) {
            $script:RepoPath = $InstallRoot
            # Tools, caches and the Maven repo must not land inside the working tree: gigabytes of
            # untracked files, one `git clean -xdf` away from a full re-download.
            $script:SupportRoot = $InstallRoot.TrimEnd('\') + '-tools'
        } elseif (Test-Path -LiteralPath (Join-Path $InstallRoot '.git')) {
            throw ("$InstallRoot is a git checkout but not the MCAI repo (missing one of: " +
                ($RepoSentinels -join ', ') + '). Pass -RepoPath <your checkout>, or point -InstallRoot at a directory that is not a checkout.')
        } else {
            $script:RepoPath = Join-Path $InstallRoot 'MCAI'
        }
    }
    if (-not $SupportRoot) { $script:SupportRoot = $InstallRoot }
    $script:ToolsRoot = Join-Path $SupportRoot 'tools'
    $script:CacheDir = Join-Path $SupportRoot 'cache'
    $script:LogDir = Join-Path $SupportRoot 'logs'
    $script:MavenRepo = Join-Path $SupportRoot '.m2'
    $script:StatePath = Join-Path $SupportRoot 'bootstrap-state.json'
    $script:Runtime = Join-Path $RepoPath 'server-runtime'
}

function Get-StackProcess {
    <#  Windows-only. Matches on the image path rather than the process name, so an unrelated java or
        node elsewhere on the box is left alone. A process we cannot open reports no path. #>
    $roots = @($RepoPath, $SupportRoot, $InstallRoot) | Where-Object { $_ } | Select-Object -Unique
    $running = New-Object System.Collections.Generic.List[object]
    foreach ($process in @(Get-Process -Name 'java', 'node', 'python' -ErrorAction SilentlyContinue)) {
        $path = $null
        try { $path = $process.Path } catch { }
        if (-not $path) { continue }
        foreach ($root in $roots) {
            if ($path.StartsWith($root, [StringComparison]::OrdinalIgnoreCase)) { $running.Add($process); break }
        }
    }
    return @($running)
}

function Assert-StackStopped {
    <#  Everything below is destructive: `npm ci` wipes worker\node_modules, the venv is recreated,
        and MCAIArena.jar is replaced. A running Paper holds that jar open (Windows locks a jar loaded
        by a JVM), and a running worker is executing out of node_modules. Rebuilding underneath the
        boot task half-updates the tree, fails, and leaves a broken stack producing garbage rollouts
        on a VM that bills either way. Refuse - or stop it first, if asked to. #>
    $task = $null
    try { $task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue } catch { }
    $taskState = if ($task) { [string]$task.State } else { 'absent' }
    $processes = Get-StackProcess
    if ($taskState -ne 'Running' -and $processes.Count -eq 0) { return }

    $names = @($processes | ForEach-Object { '{0}({1})' -f $_.ProcessName, $_.Id }) -join ', '
    if (-not $names) { $names = 'none' }
    if (-not $StopRunningStack) {
        throw ("The training stack is running (task '$TaskName' state=$taskState; processes: $names). " +
            "Stop it before rebuilding - 'schtasks /End /TN $TaskName' - or re-run with -StopRunningStack. " +
            'Paper holds plugins\MCAIArena.jar open and npm ci would delete node_modules underneath the running worker.')
    }

    Write-Log ("stopping the running stack before rebuilding (task=$taskState, processes: $names)") 'WARN'
    if ($task) {
        # Disable before stopping: 03-run-training.ps1 registers the task with RestartCount 999 /
        # RestartInterval 1 minute, so Task Scheduler reads our stop as a failure and relaunches the
        # whole stack in the middle of the rebuild. Restore-BootTask puts it back.
        try {
            Disable-ScheduledTask -TaskName $TaskName -ErrorAction Stop | Out-Null
            $script:TaskDisabled = $true
        } catch { Write-Log ('could not disable {0}: {1}' -f $TaskName, $_.Exception.Message) 'WARN' }
        try { Stop-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue } catch { }
    }
    # Give the supervisor a chance to shut its own children down cleanly before killing them.
    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline -and (Get-StackProcess).Count -gt 0) { Start-Sleep -Seconds 2 }
    foreach ($process in Get-StackProcess) {
        Write-Log ('force-stopping {0} ({1})' -f $process.ProcessName, $process.Id) 'WARN'
        try { Stop-Process -Id $process.Id -Force -ErrorAction Stop }
        catch { Write-Log ('could not stop pid {0}: {1}' -f $process.Id, $_.Exception.Message) 'WARN' }
    }
    Start-Sleep -Seconds 3
    $remaining = Get-StackProcess
    if ($remaining.Count -gt 0) {
        throw ('Could not stop: ' + (@($remaining | ForEach-Object { '{0}({1})' -f $_.ProcessName, $_.Id }) -join ', ') +
            '. Kill them manually and re-run.')
    }
    Write-Log 'the training stack is stopped; restart it with 03-run-training.ps1 once this finishes.'
}

function Restore-BootTask {
    <#  Always run this, success or failure: a boot task left disabled means the next Spot eviction
        restarts the VM into a silent, idle, still-billing box. #>
    if (-not $script:TaskDisabled) { return }
    try {
        Enable-ScheduledTask -TaskName $TaskName -ErrorAction Stop | Out-Null
        $script:TaskDisabled = $false
        Write-Log ("re-enabled the boot task '{0}'; it starts the stack on the next boot. Start it now with: schtasks /Run /TN {0}" -f $TaskName)
    } catch {
        Write-Log ("could not re-enable the boot task '{0}': {1}. Re-enable it or training will NOT resume after a reboot." -f $TaskName, $_.Exception.Message) 'ERROR'
    }
}

# ---------------------------------------------------------------------------- main

try {
    Write-Step 'preflight'
    if ($env:OS -ne 'Windows_NT') { throw 'This bootstrap targets Windows Server. Use deploy/azure (Docker) on Linux.' }
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Run this script elevated (Administrator or SYSTEM); it writes machine-scope PATH/environment entries.'
    }

    Resolve-Layout
    New-Item -ItemType Directory -Force -Path $ToolsRoot, $CacheDir, $LogDir | Out-Null
    $script:LogFile = Join-Path $LogDir ('bootstrap-{0}.log' -f (Get-Date -Format 'yyyyMMdd-HHmmss'))
    # Unbounded logs are how a 64 GB disk dies three days into an unattended run: every re-run tees
    # all of npm/mvn/pip output into a new file.
    Get-ChildItem -LiteralPath $LogDir -Filter 'bootstrap-*.log' -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -Skip 10 |
        Remove-Item -Force -ErrorAction SilentlyContinue
    Write-Log ('log file: {0}' -f $script:LogFile)
    Write-Log ('repo={0} tools/cache/logs={1}' -f $RepoPath, $SupportRoot)
    if ($SupportRoot -ne $InstallRoot) {
        Write-Log ('{0} is already an MCAI checkout, so it is used in place and the toolchain lives beside it in {1}.' -f $InstallRoot, $SupportRoot) 'WARN'
    }

    Assert-StackStopped

    # Fail fast on the EULA: everything below takes ~10 minutes and the server cannot start without
    # it, so refusing up front is far kinder than refusing at the end.
    $eulaPath = Join-Path $Runtime 'eula.txt'
    $eulaAlreadyAccepted = (Test-Path -LiteralPath $eulaPath) -and ((Get-Content -LiteralPath $eulaPath -Raw) -match 'eula\s*=\s*true')
    # MCAI_ACCEPT_EULA=true is the same opt-in deploy/azure/entrypoint.sh uses and the one
    # 01-provision.ps1 prints, so honour both forms of the explicit consent.
    if (-not $AcceptEula -and $env:MCAI_ACCEPT_EULA -eq 'true') {
        $AcceptEula = $true
        Write-Log 'MCAI_ACCEPT_EULA=true is set; treating it as -AcceptEula.'
    }
    if (-not $AcceptEula -and -not $eulaAlreadyAccepted) {
        throw ('The Minecraft EULA (https://aka.ms/MinecraftEULA) has not been accepted. Read it, then re-run this script with -AcceptEula (or with MCAI_ACCEPT_EULA=true in the environment). Nothing accepts it on your behalf.')
    }

    $cores = [Environment]::ProcessorCount
    if ($MaxPairs -le 0) {
        if ($BotCount -gt 0) {
            # An explicit -BotCount has to drive the arena count, the way entrypoint.sh derives
            # MAX_PAIRS from BOTS. Sizing from vCPUs here instead would connect bots that never get
            # an arena: they tick the server and burn CPU for no rollouts.
            $MaxPairs = [Math]::Max(1, [Math]::Floor($BotCount / 2))
        } else {
            # ~1 arena pair per 4 vCPU: the same ratio deploy/azure/README.md validates (8 bots on a
            # 16 vCPU F16s_v2). Without this the plugin keeps its bundled default of 2 pairs and most
            # of the cores you are paying for sit idle.
            $MaxPairs = [Math]::Max(1, [Math]::Floor($cores / 4))
        }
    }
    if ($BotCount -le 0) { $BotCount = $MaxPairs * 2 }
    if ($BotCount -gt ($MaxPairs * 2)) {
        Write-Log ('{0} bots but only {1} arena pairs: {2} bots will idle on the server without ever being paired.' -f $BotCount, $MaxPairs, ($BotCount - $MaxPairs * 2)) 'WARN'
    }
    Write-Log ('vCPUs={0} max-concurrent-pairs={1} bots={2} mode={3}' -f $cores, $MaxPairs, $BotCount, $Mode)

    Write-Step 'toolchain'
    $git = Install-Git
    $node = Install-Node
    $npm = Join-Path (Split-Path -Parent $node) 'npm.cmd'
    if (-not (Test-Path -LiteralPath $npm)) {
        $npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
        if (-not $npmCommand) { throw 'npm.cmd was not found next to node.exe or on PATH.' }
        $npm = $npmCommand.Source
    }
    $jdk = Install-Jdk
    # Paper is launched by absolute path via MCAI_JAVA_EXE, but a java on PATH that Paper cannot
    # run is a trap for anyone who RDPs in to debug - so PREPEND: appending would leave a
    # pre-installed Java 17 winning for a bare `java`, which is exactly the trap.
    Add-MachinePath -Directory (Join-Path $jdk.Home 'bin') -Prepend
    $maven = Install-Maven
    $python = Install-Python312
    Write-Log ('git={0}' -f (Invoke-NativeCapture $git @('--version')))
    Write-Log ('node={0}' -f (Invoke-NativeCapture $node @('--version')))
    Write-Log ('java={0} (major {1})' -f (Invoke-NativeCapture $jdk.Java @('-version')).Split("`n")[0], $jdk.Major)

    Write-Step 'repository'
    # A checkout made by another account (or by SYSTEM via the Custom Script Extension) trips git's
    # "dubious ownership" guard and every git call fails. Whitelist it up front, and only once:
    # `config --add` would otherwise append a duplicate line on every re-run.
    $safeDirectory = $RepoPath.Replace('\', '/')
    $configured = ''
    try { $configured = Invoke-NativeCapture $git @('config', '--global', '--get-all', 'safe.directory') } catch { }
    if (@($configured -split "`n" | ForEach-Object { $_.Trim() }) -notcontains $safeDirectory) {
        Invoke-Native -FilePath $git -Arguments @('config', '--global', '--add', 'safe.directory', $safeDirectory)
    }
    # An unattended box must never end up at git's "Username for 'https://github.com':" prompt: with
    # a console it hangs forever, without one it dies with an opaque exit 128. GIT_TERMINAL_PROMPT=0
    # plus an empty credential.helper (which also disables Git Credential Manager's GUI flow, if
    # winget installed full Git for Windows) turns a private/moved repo into an immediate error.
    $gitEnvironment = @{ GIT_TERMINAL_PROMPT = '0'; GCM_INTERACTIVE = 'never'; GIT_ASKPASS = 'echo' }
    $gitNoAuth = @('-c', 'credential.helper=')
    try {
        if (Test-Path -LiteralPath (Join-Path $RepoPath '.git')) {
            Invoke-Native -FilePath $git -Environment $gitEnvironment -WorkingDirectory $RepoPath `
                -Arguments ($gitNoAuth + @('fetch', '--prune', 'origin', $Branch))
            Invoke-Native -FilePath $git -Environment $gitEnvironment -WorkingDirectory $RepoPath `
                -Arguments @('checkout', $Branch)
            # --ff-only on purpose: a re-run must never silently discard work someone did on the VM.
            Invoke-Native -FilePath $git -Environment $gitEnvironment -WorkingDirectory $RepoPath `
                -Arguments @('merge', '--ff-only', ('origin/' + $Branch))
        } elseif (Test-Path -LiteralPath $RepoPath) {
            throw "$RepoPath exists but is not a git checkout. Move it aside or pass -RepoPath."
        } else {
            Invoke-Native -FilePath $git -Environment $gitEnvironment `
                -Arguments ($gitNoAuth + @('clone', '--branch', $Branch, $RepoUrl, $RepoPath))
        }
    } catch {
        if ($_.Exception.Message -match '(?i)authentic|could not read Username|terminal prompts disabled|Repository not found') {
            throw ("git could not authenticate to $RepoUrl. Credential prompts are disabled on purpose (this VM is unattended). " +
                'Use a public repo, or clone it yourself with an embedded token and re-run with -RepoPath. Original error: ' + $_.Exception.Message)
        }
        throw
    }
    foreach ($required in $RepoSentinels) {
        if (-not (Test-Path -LiteralPath (Join-Path $RepoPath $required))) { throw "$RepoPath does not look like the MCAI repo (missing $required)." }
    }

    Write-Step 'python environment'
    $venv = Join-Path $RepoPath 'trainer\.venv'
    $venvPython = Join-Path $venv 'Scripts\python.exe'
    if (-not (Test-Python312 $venvPython)) {
        if (Test-Path -LiteralPath $venv) { Remove-Item -LiteralPath $venv -Recurse -Force }
        Invoke-Native -FilePath $python -Arguments @('-m', 'venv', $venv)
    }
    # CPU-only PyTorch: no GPU on an Fsv2 VM, and the CUDA wheels are several GB of disk and
    # bandwidth we are paying for. Installed first so the editable install below resolves torch
    # from what is already there.
    $pipCommon = @('-m', 'pip', 'install', '--disable-pip-version-check', '--no-input', '--no-cache-dir')
    Invoke-Native -FilePath $venvPython -Arguments ($pipCommon + @('--index-url', 'https://download.pytorch.org/whl/cpu', 'torch>=2.4,<2.8'))
    Invoke-Native -FilePath $venvPython -Arguments ($pipCommon + @('-e', ((Join-Path $RepoPath 'trainer') + '[dev]')))
    # `import combat_ai` is the real gate - without it the trainer cannot run at all. The CUDA build
    # is only a disk/bandwidth waste (there is no GPU on an Fsv2), so it is worth a warning, never a
    # failed provision after ten minutes of paid work.
    $torchProbe = Invoke-NativeCapture $venvPython @('-c', 'import combat_ai, torch; print(torch.__version__, torch.version.cuda)')
    Write-Log ('trainer imports ok: torch {0}' -f $torchProbe)
    if ($torchProbe -notmatch '\bNone\s*$') {
        Write-Log ('this is a CUDA-tagged torch build ({0}); it trains fine on CPU but wastes several GB of disk and bandwidth' -f $torchProbe) 'WARN'
    }

    Write-Step 'rollout worker'
    Invoke-Native -FilePath $npm -Arguments @('ci', '--no-audit', '--no-fund') -WorkingDirectory (Join-Path $RepoPath 'worker')
    Invoke-Native -FilePath $npm -Arguments @('run', 'build') -WorkingDirectory (Join-Path $RepoPath 'worker')
    $workerEntry = Join-Path $RepoPath 'worker\dist\src\index.js'
    if (-not (Test-Path -LiteralPath $workerEntry)) { throw "The worker build did not produce $workerEntry." }

    Write-Step 'arena plugin'
    $plugins = Join-Path $Runtime 'plugins'
    New-Item -ItemType Directory -Force -Path $plugins, $MavenRepo | Out-Null
    Invoke-Native -FilePath $maven -Environment @{ JAVA_HOME = $jdk.Home } `
        -Arguments @('-B', '-q', '-DskipTests', ('-Dmaven.repo.local=' + $MavenRepo),
        '-f', (Join-Path $RepoPath 'server-plugin\pom.xml'), 'package')
    # Match by glob so a pom <version> bump does not silently break the copy, and skip Maven's
    # pre-shade original-*.jar (it has no bundled gson and fails at runtime).
    $arenaJar = Get-ChildItem -Path (Join-Path $RepoPath 'server-plugin\target') -Filter 'mcai-arena-*.jar' -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notlike 'original-*.jar' } | Sort-Object Name | Select-Object -First 1
    if (-not $arenaJar) { throw 'No shaded mcai-arena-*.jar in server-plugin\target after the Maven build.' }
    # Stage then swap: Copy-Item -Force straight onto a jar a running JVM holds open can fail partway
    # and leave a truncated plugin behind. Staging means the live jar is replaced or left untouched,
    # never half-written - and the failure names the real cause.
    $arenaTarget = Join-Path $plugins 'MCAIArena.jar'
    $arenaStaged = $arenaTarget + '.new'
    Copy-Item -LiteralPath $arenaJar.FullName -Destination $arenaStaged -Force
    try { Move-Item -LiteralPath $arenaStaged -Destination $arenaTarget -Force }
    catch {
        Remove-Item -LiteralPath $arenaStaged -Force -ErrorAction SilentlyContinue
        throw ("Could not replace $arenaTarget ($($_.Exception.Message)). It is almost certainly held open by a running Paper server - stop the stack (schtasks /End /TN $TaskName) and re-run.")
    }
    Write-Log ('installed plugin: {0}' -f $arenaJar.Name)

    Write-Step 'paper server'
    $paperJar = Join-Path $Runtime ('paper-{0}.jar' -f $PaperVersion)
    if (Test-Path -LiteralPath $paperJar) {
        Write-Log ('paper already installed: {0}' -f $paperJar)
    } else {
        # Fetched at runtime, never baked or committed - no game binaries are redistributed.
        # PaperMC's v2 API returns 410 Gone for 1.12.2; the v3 ("fill") API still serves it and
        # publishes a sha256 we verify before trusting the jar.
        $builds = Invoke-JsonApi ('https://fill.papermc.io/v3/projects/paper/versions/{0}/builds' -f $PaperVersion)
        $build = @($builds) | Select-Object -First 1
        $download = Get-Field (Get-Field $build 'downloads') 'server:default'
        $url = Get-Field $download 'url'
        $sha = Get-Field (Get-Field $download 'checksums') 'sha256'
        if (-not $url -or -not $sha) { throw 'The PaperMC v3 API response did not contain a server:default download with a sha256.' }
        $temp = Join-Path $CacheDir 'paper-download.jar'
        Get-RemoteFile -Uri $url -OutFile $temp -Checksum $sha
        New-Item -ItemType Directory -Force -Path $Runtime | Out-Null
        Move-Item -LiteralPath $temp -Destination $paperJar -Force
        Write-Log ('installed paper build {0} ({1})' -f (Get-Field $build 'id'), (Get-Field $download 'name'))
    }

    Write-Step 'runtime configuration'
    if ($AcceptEula) { Write-TextFile $eulaPath "eula=true`n"; Write-Log 'eula.txt written (-AcceptEula was passed)' }
    # Reuse the single source of truth for server.properties / whitelist / bukkit / spigot / paper.
    # Whitelisting a few names beyond MCAI_BOT_COUNT is free (they are just offline UUIDs) and stops
    # a run started with a slightly higher bot count from having its extra bots kicked at login.
    Invoke-Native -FilePath $venvPython -WorkingDirectory $RepoPath `
        -Arguments @((Join-Path $RepoPath 'scripts\configure_runtime.py'), $Runtime,
        '--bind', '127.0.0.1', '--bots', [string]($BotCount + 4), '--prefix', 'MCAI_')

    # Mirrors deploy/azure/entrypoint.sh. Without this the plugin keeps its bundled default of
    # 2 concurrent pairs regardless of VM size.
    $pluginConfig = @"
world-name: mcai_training
control-port: 8765
max-concurrent-pairs: $MaxPairs
bot-name-prefix: MCAI_
match-timeout-seconds: 120
auto-pair-bots: true
default-mode: $Mode
arena-spacing: 96
shaping-scale: 1.0
"@
    Write-TextFile (Join-Path $plugins 'MCAIArena\config.yml') ($pluginConfig + "`n")
    Write-Log ('arena config: mode={0} max-concurrent-pairs={1} bots={2}' -f $Mode, $MaxPairs, $BotCount)

    Write-Step 'machine environment'
    # These are the handles the run script (and scripts\start-windows.ps1) read. Machine scope so a
    # scheduled task at boot, an SSH session and an RDP session all see the same stack.
    Set-MachineVariable 'MCAI_REPO' $RepoPath
    Set-MachineVariable 'MCAI_RUNTIME' $Runtime
    Set-MachineVariable 'MCAI_JAVA_EXE' $jdk.Java
    # Explicit, not PATH-dependent: 03-run-training.ps1's only fallback candidate for node is the
    # installer location (C:\Program Files\nodejs), which this ZIP install never creates. Any later
    # change to the machine PATH would otherwise leave the worker unstartable while Paper and the
    # trainer come up fine - a half-broken stack that bills without producing rollouts.
    Set-MachineVariable 'MCAI_NODE_EXE' $node
    Set-MachineVariable 'MCAI_VENV_PYTHON' $venvPython
    Set-MachineVariable 'MCAI_BOT_COUNT' ([string]$BotCount)
    Set-MachineVariable 'MCAI_MAX_PAIRS' ([string]$MaxPairs)
    Set-MachineVariable 'MCAI_MODE' $Mode
    # Loopback everywhere: only RDP/SSH is reachable on this VM and the dashboard is viewed through
    # a tunnel. Nothing here may ever bind 0.0.0.0.
    Set-MachineVariable 'MCAI_BIND_ADDRESS' '127.0.0.1'
    Set-MachineVariable 'MCAI_SERVER_HOST' '127.0.0.1'
    Set-MachineVariable 'MCAI_DASHBOARD_HOST' '127.0.0.1'
    Set-MachineVariable 'MCAI_OPEN_DASHBOARD' 'false'

    $state = [ordered]@{
        completed_at = (Get-Date).ToString('o')
        install_root = $InstallRoot
        support_root = $SupportRoot
        repo = $RepoPath
        branch = $Branch
        runtime = $Runtime
        venv_python = $venvPython
        java_exe = $jdk.Java
        java_major = $jdk.Major
        maven = $maven
        node = $node
        npm = $npm
        vcpus = $cores
        bots = $BotCount
        max_pairs = $MaxPairs
        mode = $Mode
        eula_accepted = [bool]($AcceptEula -or $eulaAlreadyAccepted)
        log = $script:LogFile
    }
    Write-TextFile $StatePath (($state | ConvertTo-Json -Depth 4) + "`n")

    Write-Step 'verification'
    foreach ($artifact in @($venvPython, $workerEntry, $paperJar, (Join-Path $plugins 'MCAIArena.jar'),
            (Join-Path $plugins 'MCAIArena\config.yml'), (Join-Path $Runtime 'server.properties'),
            (Join-Path $Runtime 'whitelist.json'), $eulaPath)) {
        if (-not (Test-Path -LiteralPath $artifact)) { throw "Bootstrap finished but $artifact is missing." }
        Write-Log ('ok: {0}' -f $artifact)
    }

    Write-Step 'bootstrap complete'
    Restore-BootTask
    Write-Log ('repo={0} branch={1}' -f $RepoPath, $Branch)
    Write-Log ('java {0} at {1} (Paper {2} requires {3}-{4})' -f $jdk.Major, $jdk.Java, $PaperVersion, $RuntimeJavaMin, $RuntimeJavaMax)
    Write-Log ('sized for {0} vCPUs: {1} pairs / {2} bots, mode {3}' -f $cores, $MaxPairs, $BotCount, $Mode)
    Write-Log 'next: start the stack with the run script in deploy\azure-windows (see its README).'
    exit 0
} catch {
    Write-Log ('BOOTSTRAP FAILED: {0}' -f $_.Exception.Message) 'ERROR'
    if ($_.ScriptStackTrace) { Write-Log $_.ScriptStackTrace 'ERROR' }
    # A failed rebuild is bad; a failed rebuild that also left the boot task disabled is worse - the
    # VM would come back from an eviction doing nothing at all while still billing.
    try { Restore-BootTask } catch { }
    Write-Log ('full log: {0}' -f $script:LogFile) 'ERROR'
    # Non-zero so a Custom Script Extension / scheduled task marks the provisioning as failed
    # instead of leaving a silently idle VM burning money.
    exit 1
}
