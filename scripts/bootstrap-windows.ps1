$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Config = Join-Path $Root "config.windows.ps1"
if (Test-Path $Config) { . $Config }
$Runtime = if ($env:MCAI_RUNTIME) { $env:MCAI_RUNTIME } else { Join-Path $Root "server-runtime" }
$BindAddress = if ($env:MCAI_BIND_ADDRESS) { $env:MCAI_BIND_ADDRESS } else { "127.0.0.1" }
$BotCount = if ($env:MCAI_BOT_COUNT) { [int]$env:MCAI_BOT_COUNT } else { 4 }
$MavenVersion = "3.9.9"
$MavenHome = Join-Path $Root ".tools\apache-maven-$MavenVersion"

function Require-Command([string]$Name, [string]$InstallHint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing $Name. $InstallHint"
    }
}

Require-Command git "Install Git for Windows (winget install Git.Git)."
Require-Command node "Install Node.js LTS (winget install OpenJS.NodeJS.LTS)."
Require-Command npm "npm should be included with Node.js."
Require-Command java "Install Microsoft OpenJDK 17 (winget install Microsoft.OpenJDK.17)."

$NodeVersion = (& node --version).Trim()
$NodeMajor = [int](($NodeVersion -replace '^v', '').Split('.')[0])
if ($NodeMajor -lt 20) { throw "Node.js 20 or newer is required; found $(& node --version)." }

if ($env:MCAI_ACCEPT_EULA -ne "true") {
    throw "Review the Minecraft EULA, then set MCAI_ACCEPT_EULA=true in config.windows.ps1 before setup."
}

$JavaVersionText = (& java -version 2>&1 | Select-Object -First 1).ToString()
if ($JavaVersionText -notmatch '"(?<major>\d+)') { throw "Could not read the Java version." }
if ([int]$Matches.major -lt 17) { throw "Current EaglerXServer requires Java 17 or newer." }

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    $env:Path = "$HOME\.local\bin;$HOME\.cargo\bin;$env:Path"
}
Require-Command uv "Restart PowerShell after installing uv."
& uv python install 3.12

$VenvPython = Join-Path $Root "trainer\.venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Version = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($Version -ne "3.12") { throw "trainer\.venv uses Python $Version. Move it aside, then rerun." }
} else {
    & uv venv --python 3.12 (Join-Path $Root "trainer\.venv")
}
& uv pip install --python $VenvPython -e "$Root\trainer[dev,export]"

if (-not (Test-Path (Join-Path $Runtime ".git"))) {
    if (Test-Path $Runtime) { throw "$Runtime exists but is not the expected server-template checkout." }
    & git clone --depth 1 https://github.com/Eaglercraft-Templates/Eaglercraft-Server-Paper.git $Runtime
}
$Plugins = Join-Path $Runtime "plugins"
New-Item -ItemType Directory -Force -Path $Plugins, (Join-Path $Root ".tools") | Out-Null
$Release = Invoke-RestMethod https://api.github.com/repos/lax1dude/eaglerxserver/releases/latest
$Asset = $Release.assets | Where-Object { $_.name -eq "EaglerXServer.jar" } | Select-Object -First 1
if (-not $Asset) { throw "The current EaglerXServer release does not contain EaglerXServer.jar." }
Invoke-WebRequest $Asset.browser_download_url -OutFile (Join-Path $Plugins "EaglerXServer.jar")
Write-Host "Installed EaglerXServer $($Release.tag_name)."

$MavenExe = Join-Path $MavenHome "bin\mvn.cmd"
if (-not (Test-Path $MavenExe)) {
    $Archive = Join-Path $Root ".tools\maven.zip"
    Invoke-WebRequest "https://archive.apache.org/dist/maven/maven-3/$MavenVersion/binaries/apache-maven-$MavenVersion-bin.zip" -OutFile $Archive
    Expand-Archive -Path $Archive -DestinationPath (Join-Path $Root ".tools") -Force
}
& $MavenExe -q -f (Join-Path $Root "server-plugin\pom.xml") package
Copy-Item (Join-Path $Root "server-plugin\target\mcai-arena-0.1.0.jar") (Join-Path $Plugins "MCAIArena.jar") -Force
& $VenvPython (Join-Path $Root "scripts\configure_runtime.py") $Runtime --bind $BindAddress --bots $BotCount
Set-Content -Path (Join-Path $Runtime "eula.txt") -Value "eula=true" -Encoding ASCII

$Disabled = Join-Path $Runtime "plugins-disabled"
New-Item -ItemType Directory -Force -Path $Disabled | Out-Null
Get-ChildItem $Plugins -Filter "AuthMe*.jar" | ForEach-Object {
    $Destination = Join-Path $Disabled $_.Name
    if (-not (Test-Path $Destination)) { Move-Item $_.FullName $Destination }
}

Push-Location (Join-Path $Root "worker")
try {
    & npm ci
    & npm run build
    & npm test -- --run
} finally { Pop-Location }

& $VenvPython -m pytest -q (Join-Path $Root "trainer\tests")
& $VenvPython (Join-Path $Root "scripts\verify_model_parity.py")
Write-Host "Windows Surface training stack is ready. Runtime binding: $BindAddress"
Write-Host "Keep Windows Firewall private-network-only; do not port-forward ports 25565, 8765, 8766, or 8767."
