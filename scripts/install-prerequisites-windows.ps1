[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ($env:OS -ne "Windows_NT") {
    throw "This installer is for Windows. Run START_MCAI.cmd on the Surface."
}

function Refresh-ProcessPath {
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user = [Environment]::GetEnvironmentVariable("Path", "User")
    $extras = @(
        "$env:ProgramFiles\Git\cmd",
        "$env:ProgramFiles\nodejs",
        "$HOME\.local\bin",
        "$HOME\.cargo\bin"
    )
    $env:Path = (@($machine, $user) + $extras | Where-Object { $_ }) -join ";"
}

function Find-Java17 {
    $candidates = @()
    if ($env:MCAI_JAVA_EXE -and (Test-Path $env:MCAI_JAVA_EXE)) {
        $candidates += $env:MCAI_JAVA_EXE
    }
    $candidates += Get-ChildItem "$env:ProgramFiles\Microsoft\jdk-17*\bin\java.exe" `
        -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | ForEach-Object FullName
    $command = Get-Command java.exe -ErrorAction SilentlyContinue
    if ($command) { $candidates += $command.Source }
    foreach ($candidate in $candidates | Select-Object -Unique) {
        $line = (& $candidate -version 2>&1 | Select-Object -First 1).ToString()
        if ($line -match '"(?<major>\d+)' -and [int]$Matches.major -ge 17) { return $candidate }
    }
    return $null
}

function Install-WinGetPackage([string]$Id, [string]$Label) {
    Write-Host "Installing $Label..." -ForegroundColor Cyan
    & winget.exe install --id $Id --exact --source winget --silent `
        --accept-source-agreements --accept-package-agreements --disable-interactivity
    if ($LASTEXITCODE -ne 0) {
        throw "Windows Package Manager could not install $Label (exit $LASTEXITCODE). Open Microsoft Store, update 'App Installer', then run START_MCAI.cmd again."
    }
    Refresh-ProcessPath
}

Refresh-ProcessPath
$missing = @()
if (-not (Get-Command git.exe -ErrorAction SilentlyContinue)) { $missing += @{ Id = "Git.Git"; Label = "Git for Windows" } }
if (-not (Get-Command node.exe -ErrorAction SilentlyContinue)) { $missing += @{ Id = "OpenJS.NodeJS.LTS"; Label = "Node.js LTS" } }
if (-not (Find-Java17)) { $missing += @{ Id = "Microsoft.OpenJDK.17"; Label = "Microsoft OpenJDK 17" } }
if (-not (Get-Command uv.exe -ErrorAction SilentlyContinue)) { $missing += @{ Id = "astral-sh.uv"; Label = "uv Python manager" } }

if ($missing.Count -gt 0) {
    if (-not (Get-Command winget.exe -ErrorAction SilentlyContinue)) {
        throw "Windows Package Manager (winget) is missing. Install or update 'App Installer' from Microsoft Store, then run START_MCAI.cmd again."
    }
    if ($env:MCAI_ACCEPT_TOOL_LICENSES -ne "true") {
        Write-Host ""
        Write-Host "MCAI needs these standard development tools:" -ForegroundColor Yellow
        $missing | ForEach-Object { Write-Host "  - $($_.Label)" }
        Write-Host "Windows Package Manager will show/accept their package and source agreements."
        $answer = Read-Host "Type INSTALL to continue"
        if ($answer -cne "INSTALL") { throw "Tool installation was not accepted." }
        $env:MCAI_ACCEPT_TOOL_LICENSES = "true"
    }
    foreach ($package in $missing) { Install-WinGetPackage $package.Id $package.Label }
}

Refresh-ProcessPath
$git = Get-Command git.exe -ErrorAction SilentlyContinue
$node = Get-Command node.exe -ErrorAction SilentlyContinue
$npm = Get-Command npm.cmd -ErrorAction SilentlyContinue
$uv = Get-Command uv.exe -ErrorAction SilentlyContinue
$java = Find-Java17
if (-not $git -or -not $node -or -not $npm -or -not $uv -or -not $java) {
    throw "A prerequisite was installed but is not visible yet. Restart Windows, then run START_MCAI.cmd again."
}
$nodeMajor = [int]((& $node.Source --version).Trim().TrimStart("v").Split(".")[0])
if ($nodeMajor -lt 20) { throw "Node.js 20 or newer is required. Update Node.js LTS, then try again." }
$env:MCAI_JAVA_EXE = $java
Write-Host "Prerequisites ready: Node $(& $node.Source --version), Java 17+, Git $(& $git.Source --version), uv $(& $uv.Source --version)." -ForegroundColor Green
