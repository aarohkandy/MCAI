[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$ClientHtml
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = $PSScriptRoot
$InputPath = (Resolve-Path -LiteralPath $ClientHtml).Path
if ([IO.Path]::GetExtension($InputPath) -ne ".html") {
    throw "Expected an Eaglercraft 1.12 u2 offline .html file."
}

$Injector = Join-Path $Root ".tools\EaglerForgeInjector\cli.js"
$ModBundle = Join-Path $Root "eagler-mod\build\mcai.bundle.js"
$OutputDirectory = Join-Path $Root "eagler-client"
$OutputPath = Join-Path $OutputDirectory "MCAI-Eaglercraft.html"
$SpectatorPath = Join-Path $OutputDirectory "MCAI-Spectator.html"

foreach ($RequiredPath in @($Injector, $ModBundle)) {
    if (-not (Test-Path -LiteralPath $RequiredPath)) { throw "Missing $RequiredPath" }
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null
& (Get-Command node.exe -ErrorAction Stop).Source --max-old-space-size=16384 `
    $Injector $InputPath $OutputPath /eaglerforge
if ($LASTEXITCODE -ne 0 -or -not (Test-Path -LiteralPath $OutputPath)) {
    throw "EaglerForge injection failed. The input must be an unminified, unobfuscated, unsigned Eaglercraft 1.12 u2 offline build."
}

# Make MCAI load with the client automatically. EaglerForge's loader accepts
# data URLs, so the finished client stays a single offline HTML file.
$BundleBytes = [IO.File]::ReadAllBytes($ModBundle)
$BundleUrl = "data:text/javascript;base64," + [Convert]::ToBase64String($BundleBytes)
$Html = [IO.File]::ReadAllText($OutputPath)
$OptionsMarker = '            container: "game_frame",' + [Environment]::NewLine + '            worldsDB: "worlds"'
$LocalOptions = '            container: "game_frame",' + [Environment]::NewLine +
    '            worldsDB: "mcai_worlds",' + [Environment]::NewLine +
    '            localStorageNamespace: "_mcai_eagler_1_12",' + [Environment]::NewLine +
    '            noInitialModGui: true,' + [Environment]::NewLine +
    '            joinServer: "ws://127.0.0.1:25565",' + [Environment]::NewLine +
    '            servers: [{ addr: "ws://127.0.0.1:25565", name: "MCAI Local Arena", hideAddr: false }]'
if (-not $Html.Contains($OptionsMarker)) { throw "Could not locate the Eaglercraft launch options." }
$Html = $Html.Replace($OptionsMarker, $LocalOptions)
$Html = $Html.Replace('var launchSkipCountdown = false;', 'var launchSkipCountdown = true;')
$Html = $Html.Replace('return $rt_globals.window.navigator.userActivation.hasBeenActive;', 'return true;')
$Marker = 'if (!window.eaglerMLoaderMainRun) {'
if (-not $Html.Contains($Marker)) { throw "Could not locate the injected EaglerForge mod loader." }
$Injection = $Marker + [Environment]::NewLine + '            modsArr.push("web@' + $BundleUrl + '");'
$Html = $Html.Replace($Marker, $Injection)
[IO.File]::WriteAllText($OutputPath, $Html, [Text.UTF8Encoding]::new($false))

# Build a lean spectator-only client as well. The bots and arena run server-side,
# so this version starts much faster than the injected training client.
$SpectatorHtml = [IO.File]::ReadAllText($InputPath)
$SpectatorOptions = '            container: "game_frame",' + [Environment]::NewLine +
    '            worldsDB: "mcai_spectator_worlds",' + [Environment]::NewLine +
    '            localStorageNamespace: "_mcai_spectator_1_12",' + [Environment]::NewLine +
    '            joinServer: "ws://127.0.0.1:25565",' + [Environment]::NewLine +
    '            servers: [{ addr: "ws://127.0.0.1:25565", name: "MCAI Local Arena", hideAddr: false }]'
if (-not $SpectatorHtml.Contains($OptionsMarker)) { throw "Could not locate spectator launch options." }
$SpectatorHtml = $SpectatorHtml.Replace($OptionsMarker, $SpectatorOptions)
$SpectatorHtml = $SpectatorHtml.Replace('var launchSkipCountdown = false;', 'var launchSkipCountdown = true;')
$SpectatorHtml = $SpectatorHtml.Replace('return $rt_globals.window.navigator.userActivation.hasBeenActive;', 'return true;')

# EaglerXServer's direct-Paper login injector can very occasionally lose its
# packet-handler handoff during the first browser connection after startup.
# Retry only after the client reaches its real disconnected screen; this does
# not reload a healthy session or depend on a fixed server-start delay.
$DisconnectedFunction = '    function nmcg_GuiDisconnected_updateScreen($this) {'
$DisconnectedStart = $SpectatorHtml.IndexOf($DisconnectedFunction, [StringComparison]::Ordinal)
if ($DisconnectedStart -lt 0) { throw "Could not locate the spectator disconnect screen." }
$DisconnectedCase = '        case 0:'
$DisconnectedCaseStart = $SpectatorHtml.IndexOf(
    $DisconnectedCase,
    $DisconnectedStart + $DisconnectedFunction.Length,
    [StringComparison]::Ordinal
)
if ($DisconnectedCaseStart -lt 0) { throw "Could not locate the spectator disconnect update loop." }
$ReconnectCode = [Environment]::NewLine +
    '            $this.$mcaiReconnectTicks = (($this.$mcaiReconnectTicks | 0) + 1) | 0;' + [Environment]::NewLine +
    '            if ($this.$mcaiReconnectTicks === 100) {' + [Environment]::NewLine +
    '                $rt_globals.window.location.reload();' + [Environment]::NewLine +
    '                return;' + [Environment]::NewLine +
    '            }'
$ReconnectInsertAt = $DisconnectedCaseStart + $DisconnectedCase.Length
$SpectatorHtml = $SpectatorHtml.Insert($ReconnectInsertAt, $ReconnectCode)
[IO.File]::WriteAllText($SpectatorPath, $SpectatorHtml, [Text.UTF8Encoding]::new($false))

Write-Host "Built MCAI Eaglercraft client: $OutputPath" -ForegroundColor Green
Write-Host "Built fast spectator client: $SpectatorPath" -ForegroundColor Green
Write-Host "The local launcher opens the spectator automatically."
