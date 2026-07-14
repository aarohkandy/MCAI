$ErrorActionPreference = "Continue"
Set-StrictMode -Version Latest

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$State = Join-Path $Root ".mcai\current.json"
$Python = Join-Path $Root "trainer\.venv\Scripts\python.exe"

if (Test-Path $Python) {
    try { & $Python (Join-Path $Root "scripts\arena_control.py") stop_all | Out-Null } catch { }
}

$stopped = 0
if (Test-Path $State) {
    try {
        $saved = Get-Content $State -Raw | ConvertFrom-Json
        foreach ($entry in $saved.processes) {
            $process = Get-Process -Id ([int]$entry.pid) -ErrorAction SilentlyContinue
            $matchesSavedProcess = $false
            if ($process -and $entry.process_started_at) {
                $expected = [datetime]::Parse($entry.process_started_at).ToUniversalTime()
                $matchesSavedProcess = [math]::Abs(($process.StartTime.ToUniversalTime() - $expected).TotalSeconds) -lt 2
            }
            if ($process -and $matchesSavedProcess) {
                Write-Host "Stopping $($entry.name)..."
                & taskkill.exe /PID $process.Id /T /F 2>$null | Out-Null
                $stopped++
            }
        }
    } catch { Write-Warning "Could not read the saved process list: $_" }
    Remove-Item $State -Force -ErrorAction SilentlyContinue
}
if ($stopped -gt 0) { Write-Host "MCAI stopped cleanly." -ForegroundColor Green }
else { Write-Host "MCAI was not running." }
