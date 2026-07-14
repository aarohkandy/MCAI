$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Config = Join-Path $Root "config.windows.ps1"
if (Test-Path $Config) { . $Config }
$port = if ($env:MCAI_DASHBOARD_PORT) { [int]$env:MCAI_DASHBOARD_PORT } else { 8788 }
$client = [Net.Sockets.TcpClient]::new()
try {
    $task = $client.ConnectAsync("127.0.0.1", $port)
    if (-not $task.Wait(1500) -or -not $client.Connected) { throw "offline" }
} catch {
    Write-Host "The live view is not running yet. Double-click START_MCAI.cmd first." -ForegroundColor Yellow
    exit 1
} finally { $client.Dispose() }
$url = "http://127.0.0.1:$port"
Write-Host "Opening $url"
Start-Process $url
