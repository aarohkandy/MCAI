$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Config = Join-Path $Root "config.windows.ps1"
if (Test-Path $Config) { . $Config }
$Runtime = if ($env:MCAI_RUNTIME) { $env:MCAI_RUNTIME } else { Join-Path $Root "server-runtime" }
$BindAddress = if ($env:MCAI_BIND_ADDRESS) { $env:MCAI_BIND_ADDRESS } else { "127.0.0.1" }
$BotCount = if ($env:MCAI_BOT_COUNT) { [int]$env:MCAI_BOT_COUNT } else { 4 }
$ArenaRadius = if ($env:MCAI_ARENA_RADIUS) { [int]$env:MCAI_ARENA_RADIUS } else { 5 }
$ArenaRadiusStages = if ($env:MCAI_ARENA_RADIUS_STAGES) { $env:MCAI_ARENA_RADIUS_STAGES } else { "5,6,7,8" }
$ArenaRadiusThresholds = if ($env:MCAI_ARENA_RADIUS_EPISODE_THRESHOLDS) { $env:MCAI_ARENA_RADIUS_EPISODE_THRESHOLDS } else { "0,64,256,1024" }
$ArenaRadiusPerformanceWindow = if ($env:MCAI_ARENA_RADIUS_PERFORMANCE_WINDOW) { [int]$env:MCAI_ARENA_RADIUS_PERFORMANCE_WINDOW } else { 32 }
$ArenaRadiusAdvanceEngagementRate = if ($env:MCAI_ARENA_RADIUS_ADVANCE_ENGAGEMENT_RATE) { [double]$env:MCAI_ARENA_RADIUS_ADVANCE_ENGAGEMENT_RATE } elseif ($env:MCAI_ARENA_RADIUS_MIN_ENGAGEMENT_RATE) { [double]$env:MCAI_ARENA_RADIUS_MIN_ENGAGEMENT_RATE } else { 0.75 }
$ArenaRadiusAdvanceNonTimeoutRate = if ($env:MCAI_ARENA_RADIUS_ADVANCE_NON_TIMEOUT_RATE) { [double]$env:MCAI_ARENA_RADIUS_ADVANCE_NON_TIMEOUT_RATE } else { 0.10 }
$ArenaRadiusRegressEngagementRate = if ($env:MCAI_ARENA_RADIUS_REGRESS_ENGAGEMENT_RATE) { [double]$env:MCAI_ARENA_RADIUS_REGRESS_ENGAGEMENT_RATE } else { 0.50 }
$ArenaRadiusRegressNonTimeoutRate = if ($env:MCAI_ARENA_RADIUS_REGRESS_NON_TIMEOUT_RATE) { [double]$env:MCAI_ARENA_RADIUS_REGRESS_NON_TIMEOUT_RATE } else { 0.05 }
$ArenaRadiusStageChangeCooldown = if ($env:MCAI_ARENA_RADIUS_STAGE_CHANGE_COOLDOWN_EPISODES) { [int]$env:MCAI_ARENA_RADIUS_STAGE_CHANGE_COOLDOWN_EPISODES } else { 32 }
$ArenaDepth = if ($env:MCAI_ARENA_DEPTH) { [int]$env:MCAI_ARENA_DEPTH } else { 12 }
$ArenaHeight = if ($env:MCAI_ARENA_HEIGHT) { [int]$env:MCAI_ARENA_HEIGHT } else { 12 }
$SpawnMinSeparation = if ($env:MCAI_SPAWN_MIN_SEPARATION) { [int]$env:MCAI_SPAWN_MIN_SEPARATION } else { 3 }
$SpawnMaxSeparation = if ($env:MCAI_SPAWN_MAX_SEPARATION) { [int]$env:MCAI_SPAWN_MAX_SEPARATION } else { 5 }
$SpawnYawJitter = if ($env:MCAI_SPAWN_YAW_JITTER_DEGREES) { [double]$env:MCAI_SPAWN_YAW_JITTER_DEGREES } else { 15.0 }
$MavenVersion = "3.9.9"
$MavenHome = Join-Path $Root ".tools\apache-maven-$MavenVersion"

& (Join-Path $PSScriptRoot "install-prerequisites-windows.ps1")

function Require-Command([string]$Name, [string]$InstallHint) {
    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Missing $Name. $InstallHint"
    }
}

Require-Command git "Install Git for Windows (winget install Git.Git)."
Require-Command node "Install Node.js LTS (winget install OpenJS.NodeJS.LTS)."
Require-Command npm "npm should be included with Node.js."
$Java = if ($env:MCAI_JAVA_EXE -and (Test-Path $env:MCAI_JAVA_EXE)) {
    $env:MCAI_JAVA_EXE
} else {
    (Get-Command java.exe -ErrorAction Stop).Source
}

$NodeVersion = (& node --version).Trim()
$NodeMajor = [int](($NodeVersion -replace '^v', '').Split('.')[0])
if ($NodeMajor -lt 20) { throw "Node.js 20 or newer is required; found $(& node --version)." }

if ($env:MCAI_ACCEPT_EULA -ne "true") {
    throw "Review the Minecraft EULA, then set MCAI_ACCEPT_EULA=true in config.windows.ps1 before setup."
}

$startInfo = New-Object System.Diagnostics.ProcessStartInfo
$startInfo.FileName = $Java
$startInfo.Arguments = "-version"
$startInfo.UseShellExecute = $false
$startInfo.RedirectStandardError = $true
$javaVersionProcess = [System.Diagnostics.Process]::Start($startInfo)
$JavaVersionText = $javaVersionProcess.StandardError.ReadLine()
$javaVersionProcess.WaitForExit()
if ($JavaVersionText -notmatch '"(?<major>\d+)') { throw "Could not read the Java version." }
if ([int]$Matches.major -lt 17) { throw "Current EaglerXServer requires Java 17 or newer." }

Require-Command uv "Run START_MCAI.cmd again so Windows Package Manager can install uv."
& uv python install 3.12

$VenvPython = Join-Path $Root "trainer\.venv\Scripts\python.exe"
if (Test-Path $VenvPython) {
    $Version = & $VenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
    if ($Version -ne "3.12") { throw "trainer\.venv uses Python $Version. Move it aside, then rerun." }
} else {
    & uv venv --python 3.12 (Join-Path $Root "trainer\.venv")
}
& uv pip install --python $VenvPython "torch>=2.4,<2.8" --index-url https://download.pytorch.org/whl/cpu
& uv pip install --python $VenvPython -e "$Root\trainer[dev,export]"
& $VenvPython -c "import torch; assert torch.version.cuda is None, 'Expected the CPU-only PyTorch build'; print('CPU PyTorch', torch.__version__)"

if (-not (Test-Path (Join-Path $Runtime ".git"))) {
    if (Test-Path $Runtime) { throw "$Runtime exists but is not the expected server-template checkout." }
    & git clone --depth 1 https://github.com/Eaglercraft-Templates/Eaglercraft-Server-Paper.git $Runtime
}
$Plugins = Join-Path $Runtime "plugins"
$Disabled = Join-Path $Runtime "plugins-disabled"
New-Item -ItemType Directory -Force -Path $Plugins, $Disabled, (Join-Path $Root ".tools") | Out-Null
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
$ArenaPluginName = "mcai-arena-0.1.0.jar"
Get-ChildItem $Plugins -File -Filter "*.jar" | Where-Object {
    $_.Name -ieq "MCAIArena.jar" -or ($_.Name -like "mcai-arena-*.jar" -and $_.Name -ine $ArenaPluginName)
} | ForEach-Object {
    Move-Item -LiteralPath $_.FullName -Destination (Join-Path $Disabled $_.Name) -Force
}
Copy-Item (Join-Path $Root "server-plugin\target\$ArenaPluginName") (Join-Path $Plugins $ArenaPluginName) -Force
& $VenvPython (Join-Path $Root "scripts\configure_runtime.py") $Runtime --bind $BindAddress --bots $BotCount
Set-Content -Path (Join-Path $Runtime "eula.txt") -Value "eula=true" -Encoding ASCII

$PluginDirectory = Join-Path $Runtime "plugins\MCAIArena"
New-Item -ItemType Directory -Force -Path $PluginDirectory | Out-Null
$Mode = if ($env:MCAI_MODE) { $env:MCAI_MODE.ToLowerInvariant() } else { "sword" }
if ($Mode -notin @("sword", "crystal", "combined", "terrain")) { throw "MCAI_MODE must be sword, crystal, combined, or terrain." }
$PluginConfig = @"
world-name: mcai_training
control-port: 8765
max-concurrent-pairs: $([math]::Max(1, [math]::Floor($BotCount / 2)))
bot-name-prefix: MCAI_
match-timeout-seconds: 35
auto-pair-bots: true
default-mode: $Mode
arena-spacing: 96
arena-radius: $ArenaRadius
arena-depth: $ArenaDepth
arena-height: $ArenaHeight
shaping-scale: 1.0
snapshot-interval-ticks: 10
curriculum:
  arena-radius-stages: [$($ArenaRadiusStages -replace '\s+', '')]
  arena-radius-episode-thresholds: [$($ArenaRadiusThresholds -replace '\s+', '')]
  arena-radius-performance-window: $ArenaRadiusPerformanceWindow
  arena-radius-advance-engagement-rate: $ArenaRadiusAdvanceEngagementRate
  arena-radius-advance-non-timeout-rate: $ArenaRadiusAdvanceNonTimeoutRate
  arena-radius-regress-engagement-rate: $ArenaRadiusRegressEngagementRate
  arena-radius-regress-non-timeout-rate: $ArenaRadiusRegressNonTimeoutRate
  arena-radius-stage-change-cooldown-episodes: $ArenaRadiusStageChangeCooldown
  spawn-min-separation: $SpawnMinSeparation
  spawn-max-separation: $SpawnMaxSeparation
  spawn-yaw-jitter-degrees: $SpawnYawJitter
reward:
  max-per-tick: 0.25
  movement-per-block: 0.004
  max-movement-delta: 0.35
  preferred-distance: 2.75
  extreme-pitch-degrees: 60.0
  extreme-pitch-grace-ticks: 20
  extreme-pitch-penalty-per-tick: -0.0001
  aim-alignment-per-potential: 0.012
  max-aim-alignment-delta: 1.0
  aim-alignment-range: 24.0
  lock-on-dot: 0.94
  lock-on-range: 5.0
  lock-on-grace-ticks: 1
  lock-on-per-tick: 0.0
  max-rewarded-lock-on-ticks: 0
  attack-aim-dot: 0.70
  attack-reach: 3.4
  attack-cooldown-ticks: 8
  valid-attack-swing: 0.004
  max-rewarded-attack-swings: 40
  missed-attack-swing: -0.002
  spam-attack-swing: -0.001
  max-penalized-attack-swings: 120
  successful-hit: 0.045
  max-rewarded-hits: 10
  damage-dealt-per-health: 0.180
  damage-taken-per-health: -0.100
  forced-totem: 0.12
  own-totem: -0.12
  policy-kill: 20.0
  policy-kill-speed-bonus: 44.0
  death-loss: -20.0
  timeout-loss: -32.0
  disengaged-loss: -32.0
  double-ko-loss: -15.0
  own-crystal-self-hit: -0.900
  own-crystal-self-damage-per-health: -0.100
  obsidian-placed: 0.015
  max-rewarded-obsidian: 4
  obsidian-combo: 0.200
  max-rewarded-obsidian-combos: 4
  tactical-mine-place-max-ticks: 120
  tactical-mine-place: 0.030
  max-rewarded-tactical-mine-place: 3
  crystal-placed: 0.040
  max-rewarded-crystal-placements: 8
  crystal-destroyed: 0.020
  max-rewarded-crystal-destructions: 8
  crystal-exploded: 0.050
  max-rewarded-crystal-explosions: 8
  crystal-combo-max-ticks: 40
  crystal-combo-damage: 0.250
  crystal-combo-pop: 0.500
  max-rewarded-crystal-combos: 6
  useful-block-mined: 0.0
  max-rewarded-mined-blocks: 0
  invalid-interaction: -0.001
  inaction-grace-ticks: 40
  inaction-penalty-per-tick: -0.0004
  max-inaction-penalty-per-tick: -0.002
  positive-reward-decay-start-ticks: 200
  positive-reward-decay-end-ticks: 700
  minimum-positive-reward-multiplier: 0.35
  fight-time-pressure-start-ticks: 20
  fight-time-pressure-per-tick: -0.001
  max-fight-time-pressure-per-tick: -0.010
"@
Set-Content -Path (Join-Path $PluginDirectory "config.yml") -Value $PluginConfig -Encoding UTF8

foreach ($DisabledPattern in @("AuthMe*.jar", "SkinsRestorer*.jar")) {
    Get-ChildItem $Plugins -File -Filter $DisabledPattern | ForEach-Object {
        # Both plugins are unnecessary on this loopback-only offline server.
        # In particular, current SkinsRestorer releases target newer Bukkit
        # events and can race EaglerXServer's first 1.12.2 browser login.
        Move-Item -LiteralPath $_.FullName -Destination (Join-Path $Disabled $_.Name) -Force
    }
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
