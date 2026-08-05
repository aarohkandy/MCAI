<#
.SYNOPSIS
    Provisions the single Azure Windows Server VM that runs the MCAI self-play training stack.

.DESCRIPTION
    Run this from YOUR machine (Windows PowerShell 5.1 / PowerShell 7) or from Azure Cloud Shell.
    It is idempotent: re-running it against an existing resource group / VM re-checks quota,
    re-applies the network lock-down, and reprints the cost + next-step summary without recreating
    anything.

    Design constraints this script exists to satisfy:
      * $100 total budget -> Spot pricing with a hard --max-price cap, cheapest workable disk.
      * Unattended, long-running training -> --eviction-policy Deallocate so the OS disk and every
        checkpoint survive an eviction; you just `az vm start` again.
      * Loopback-only game/training ports -> the NSG allows management access from ONE IP and
        denies everything else. Minecraft (25565), arena (8765), trainer (8766) and the dashboard
        (8788) are never opened; the dashboard is reached over an SSH tunnel or inside RDP.
        NOTE: Windows Server 2022 ships WITHOUT OpenSSH Server. Nothing in this stack installs it,
        so port 22 has no listener until you enable it by hand - the printed steps say how.

    This script does NOT install or start anything on the VM. The order is:
      01-provision.ps1 (here) -> 02-bootstrap.ps1 -AcceptEula (builds) -> 03-run-training.ps1
      -Install (registers the boot task; this is what actually starts training) -> 04-monitor.ps1.

.EXAMPLE
    # Interactive: prompts once for the admin password (masked, never echoed, never written to disk).
    ./01-provision.ps1

.EXAMPLE
    # Non-interactive: supply the password as a SecureString from wherever you keep secrets.
    $pw = ConvertTo-SecureString (az keyvault secret show --vault-name kv --name mcai-admin --query value -o tsv) -AsPlainText -Force
    ./01-provision.ps1 -AdminPassword $pw -AllowedSourceIp 203.0.113.42

.EXAMPLE
    # See exactly what would be created and what it would cost, without spending anything.
    ./01-provision.ps1 -WhatIf

.EXAMPLE
    # Recommended for an unattended run: also register a local 10-minute watchdog that restarts the
    # VM after a Spot eviction. Without it an eviction silently ends the run.
    ./01-provision.ps1 -EvictionWatchdog

.EXAMPLE
    # Destroy everything (the ONLY way to stop all billing). Pull your checkpoints off first.
    ./01-provision.ps1 -Teardown -Yes

.NOTES
    Windows-only assumptions: none in this script itself - it only shells out to `az`, so it runs
    on macOS/Linux PowerShell and in Cloud Shell too. The VM it creates is Windows.
#>

[CmdletBinding(SupportsShouldProcess, ConfirmImpact = 'Medium')]
param(
    [ValidatePattern('^[A-Za-z0-9._\-()]{1,90}$')]
    [string]$ResourceGroup = 'mcai-rg',

    [ValidatePattern('^[a-z0-9]+$')]
    [string]$Location = 'eastus2',

    # Windows computer names are capped at 15 characters; az derives the computer name from this.
    [ValidatePattern('^[A-Za-z][A-Za-z0-9\-]{1,13}[A-Za-z0-9]$')]
    [string]$VmName = 'mcai-train',

    # 16 vCPU comfortably drives ~4 arena pairs (8 bots). F32 buys throughput but shortens the
    # calendar window on a fixed $100 and raises eviction probability.
    [string]$Size = 'Standard_F16s_v2',

    # 'admin'/'administrator'/'root' and friends are rejected by Azure; validated below.
    [ValidatePattern('^[A-Za-z][A-Za-z0-9\-_]{1,19}$')]
    [string]$AdminUsername = 'mcaiadmin',

    # Never hardcoded, never logged, never written to a file. Omit it and you will be prompted
    # (masked) if the session is genuinely interactive; otherwise the script fails with guidance.
    [System.Security.SecureString]$AdminPassword,

    # Spot on by default. Turn it off with -Spot:$false for an evict-proof (but ~2x cost) VM.
    [bool]$Spot = $true,

    # Hard USD/hour cap. Verified eastus2 Windows F16s_v2 Spot = $0.6849/hr; PAYG is roughly 2x
    # that, so 0.90 leaves ~30% headroom for normal Spot drift while making it impossible to
    # silently slide onto full pay-as-you-go rates. -1 (Azure's "pay up to PAYG") is REFUSED:
    # on a $100 budget an unnoticed price spike is the single most expensive failure mode.
    [ValidateRange(-1.0, 100.0)]
    [double]$MaxPrice = 0.90,

    # The Windows Server 2022 marketplace image ships a 127 GiB OS disk and Azure cannot shrink it,
    # so 64 GiB is not reachable for a Windows OS disk (it is for Linux). 128 on StandardSSD is the
    # cheap end of what actually works: ~$9.60/mo vs ~$19.71/mo for the same size on Premium.
    [ValidateRange(30, 4095)]
    [int]$DiskSizeGb = 128,

    [ValidateSet('StandardSSD_LRS', 'Premium_LRS', 'Standard_LRS')]
    [string]$DiskSku = 'StandardSSD_LRS',

    # Management access source. Defaults to a lookup of the caller's current public IP.
    # CLOUD SHELL WARNING: from Cloud Shell this resolves to Cloud Shell's egress IP, not your
    # home/office IP - you would then be locked out of RDP. Pass -AllowedSourceIp explicitly there.
    [string]$AllowedSourceIp,

    # RDP alone cannot forward the dashboard port; SSH is what carries `ssh -L 8788:...`.
    # Both ports are pinned to a single /32, so opening both is still a one-address surface.
    [ValidateSet('Rdp', 'Ssh', 'Both')]
    [string]$ManagementProtocol = 'Both',

    # Azure Hybrid Benefit: strips the Windows Server licence out of the VM rate (0.6849 -> ~0.3561
    # /hr here, nearly doubling your training hours). ONLY legal if you actually own Windows Server
    # licences with active Software Assurance. Off by default.
    [switch]$HybridBenefit,

    [ValidateRange(1, 100000)]
    [double]$BudgetUsd = 100,

    # Daily DevTest-Labs auto-shutdown, 24h UTC ("0400"). OPT-IN, not the default: this VM exists to
    # run a multi-day unattended job, and a daily shutdown would stop training every 24 hours and
    # require a manual `az vm start` to resume. Set it only if you want a hard daily spend ceiling
    # more than you want continuous training. It stops COMPUTE only - disk + IP keep billing.
    [ValidatePattern('^([01]\d|2[0-3])[0-5]\d$')]
    [string]$AutoShutdownUtc,

    # Register a 10-minute watchdog ON THIS MACHINE that restarts the VM after a Spot eviction.
    # Nothing inside the VM can do this - once Azure deallocates it, every on-VM component is dead.
    [switch]$EvictionWatchdog,

    # Used only to print correct next-step commands; must match 02-bootstrap.ps1's own defaults.
    [string]$RepoUrl = 'https://github.com/aarohkandy/MCAI.git',
    [string]$Branch = 'fix/verified-training-stack',
    # 02-bootstrap.ps1 -InstallRoot; it clones the repo to <VmInstallRoot>\MCAI.
    [string]$VmInstallRoot = 'C:\mcai',

    [string]$SubscriptionId,

    # Destroy the whole resource group. Requires -Yes as well, so a stray -Teardown does nothing.
    [switch]$Teardown,
    [switch]$Yes
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$global:LASTEXITCODE = 0

$NsgName        = "$VmName-nsg"
$RuleAllowRdp   = 'allow-rdp-from-operator'
$RuleAllowSsh   = 'allow-ssh-from-operator'
$RuleDenyAll    = 'deny-all-other-inbound'
$ImageUrn       = 'MicrosoftWindowsServer:WindowsServer:2022-datacenter-azure-edition:latest'

# Verified eastus2 retail rates (2026-06 meter revision), used ONLY when the live pricing API is
# unreachable. Spot meters are re-priced monthly, so treat these as a floor, not as today's rate -
# the live lookup above is authoritative.
$FallbackPrices = @{
    WindowsSpot = 0.6849
    LinuxSpot   = 0.3561   # == Windows Spot rate under Hybrid Benefit (licence removed)
    WindowsPayg = 1.3448
}
# Approximate eastus2 USD/month by disk tier. Managed disks bill by PROVISIONED TIER, not by bytes
# used, and they bill while the VM is deallocated - this is what makes a parked VM non-free.
$DiskTierMonthly = @{
    'StandardSSD_LRS' = @{ 32 = 2.40; 64 = 4.80; 128 = 9.60; 256 = 19.20; 512 = 38.40; 1024 = 76.80 }
    'Premium_LRS'     = @{ 32 = 5.28; 64 = 10.21; 128 = 19.71; 256 = 38.02; 512 = 73.22; 1024 = 135.17 }
    'Standard_LRS'    = @{ 32 = 1.54; 64 = 3.01; 128 = 5.89; 256 = 11.33; 512 = 21.76; 1024 = 41.47 }
}
$PublicIpHourly = 0.005   # Standard SKU public IP: billed continuously, even while deallocated.

# Where 02-bootstrap.ps1 puts the checkout, and where 03/04 therefore live. Not Join-Path: this
# script also runs on macOS/Linux, where Join-Path would emit a forward slash into a Windows path.
$RepoOnVm = "$($VmInstallRoot.TrimEnd('\'))\MCAI"
$RepoOnVmPosix = $RepoOnVm.Replace('\', '/')
# raw.githubusercontent.com so the very first command on a bare Windows Server VM does not need git
# (git is what 02-bootstrap.ps1 installs). Only meaningful for a github.com origin.
$RepoRawBase = ($RepoUrl -replace '^https://github\.com/', 'https://raw.githubusercontent.com/') -replace '\.git$', ''
$BootstrapRawUrl = "$RepoRawBase/$Branch/deploy/azure-windows/02-bootstrap.ps1"

# az.cmd is a BATCH wrapper on Windows, so every argument is re-parsed by cmd.exe. PowerShell 5.1
# does not quote an argument that contains no spaces, so a password like 'Tr@in&Mc2026!' is split
# on '&' before az ever sees it - the VM is created with a different password than you typed.
$CmdMetacharacters = @('&', '|', '<', '>', '^', '%', '"', '`')

#region helpers -------------------------------------------------------------------------------

function Write-Step([string]$Message) { Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Note([string]$Message) { Write-Host "    $Message" -ForegroundColor DarkGray }

function Invoke-AzCli {
    <#
      Runs `az` with an argument ARRAY (never a composed string, so secrets never reach PSReadLine
      history or a shell parser). stderr is captured separately so az's warnings cannot corrupt the
      JSON on stdout. -Redact scrubs secret values out of any error text we surface.
    #>
    param(
        [Parameter(Mandatory)][string[]]$Arguments,
        [switch]$AllowFailure,
        [switch]$Raw,
        [string[]]$Redact = @()
    )
    # Read-only az queries (and their temp-file plumbing) must still run under -WhatIf, otherwise a
    # dry run cannot report quota or pricing and leaks temp files. Every MUTATING call in this
    # script sits behind $PSCmdlet.ShouldProcess, which still honours -WhatIf.
    $WhatIfPreference = $false
    $errFile = [System.IO.Path]::GetTempFileName()
    try {
        $stdout = & az @Arguments 2> $errFile
        $code = $LASTEXITCODE
        if ($code -ne 0) {
            if ($AllowFailure) { return $null }
            $detail = (Get-Content -LiteralPath $errFile -Raw -ErrorAction SilentlyContinue)
            $shown = ($Arguments -join ' ')
            foreach ($secret in $Redact) {
                if ($secret) { $shown = $shown.Replace($secret, '***'); $detail = "$detail".Replace($secret, '***') }
            }
            throw "az $shown failed (exit $code): $detail"
        }
        $text = ($stdout | Out-String).Trim()
        if ($Raw) { return $text }
        if (-not $text) { return $null }
        try { return $text | ConvertFrom-Json } catch { return $text }
    }
    finally { Remove-Item -LiteralPath $errFile -Force -ErrorAction SilentlyContinue }
}

function Test-JsonProperty([object]$Object, [string]$Name) {
    # StrictMode makes a missing property a terminating error, so probe before touching one.
    if ($null -eq $Object) { return $false }
    return [bool]($Object.PSObject.Properties.Name -contains $Name)
}

function Get-CallerPublicIp {
    # Several providers, because one being down should not stop a provisioning run.
    foreach ($endpoint in @('https://api.ipify.org', 'https://checkip.amazonaws.com', 'https://ifconfig.me/ip')) {
        try {
            $value = (Invoke-RestMethod -Uri $endpoint -TimeoutSec 10 -ErrorAction Stop | Out-String).Trim()
            if ($value -match '^\d{1,3}(\.\d{1,3}){3}$') { return $value }
        }
        catch { continue }
    }
    return $null
}

function Test-Interactive {
    # Cloud Shell, CI and `pwsh -NonInteractive` must never block on (or crash at) a prompt.
    if ($env:AZUREPS_NON_INTERACTIVE) { return $false }
    # UserInteractive is true even under -NonInteractive, so inspect the host's own command line.
    # PowerShell accepts any unambiguous prefix of the switch, the shortest being '-noni'.
    foreach ($arg in [Environment]::GetCommandLineArgs()) {
        if ($arg -match '^-{1,2}noni') { return $false }
    }
    if (-not [Environment]::UserInteractive) { return $false }
    try { $null = $Host.UI.RawUI.KeyAvailable } catch { return $false }
    return $true
}

function ConvertFrom-SecureStringPlain([System.Security.SecureString]$Secure) {
    $ptr = [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($Secure)
    try { return [System.Runtime.InteropServices.Marshal]::PtrToStringBSTR($ptr) }
    finally { [System.Runtime.InteropServices.Marshal]::ZeroFreeBSTR($ptr) }
}

function Test-WindowsPasswordComplexity([string]$Plain, [string]$UserName) {
    # Mirror Azure's own rules so we fail in 10ms instead of after a 60s deployment round trip,
    # PLUS the transport rule Azure cannot know about (cmd.exe re-parsing, see $CmdMetacharacters).
    if ($Plain.Length -lt 12 -or $Plain.Length -gt 123) { return 'must be 12-123 characters long' }
    $classes = 0
    if ($Plain -cmatch '[a-z]') { $classes++ }
    if ($Plain -cmatch '[A-Z]') { $classes++ }
    if ($Plain -match '\d') { $classes++ }
    if ($Plain -match '[^a-zA-Z0-9]') { $classes++ }
    if ($classes -lt 3) { return 'must contain at least 3 of: lowercase, uppercase, digit, symbol' }

    $bad = @($CmdMetacharacters | Where-Object { $Plain.Contains($_) })
    if ($bad.Count -gt 0) {
        return ("must not contain {0} - the Azure CLI launcher is a batch file (az.cmd) and cmd.exe mangles them, so the VM would end up with a password you never typed. Use other symbols (! # * - _ + = ? . : ~ etc.)" -f (($bad | ForEach-Object { "'$_'" }) -join ' '))
    }
    # Windows' own complexity policy rejects a password containing the account name; ARM surfaces
    # this only after the deployment has started.
    if ($UserName -and $UserName.Length -ge 3 -and $Plain.ToLowerInvariant().Contains($UserName.ToLowerInvariant())) {
        return "must not contain the admin username '$UserName'"
    }
    # Azure's published disallowed-password list; an exact match is rejected by ARM.
    $disallowed = @('abc@123', 'iloveyou!', 'P@$$w0rd', 'P@ssw0rd', 'P@ssword123', 'Pa$$word',
        'pass@word1', 'Password!', 'Password1', 'Password22')
    if ($disallowed -contains $Plain) { return 'is on the list of values Azure explicitly rejects' }
    return $null
}

function Resolve-ManagementCidr([string]$Value) {
    <#
      Returns a normalised IPv4 CIDR or throws. Deliberately stricter than Azure: this prefix is the
      ONLY thing standing between an internet-facing RDP endpoint and a hand-chosen password, and a
      '/0' typo (or a /8 copied out of a router UI) is indistinguishable from "open to everyone"
      once Azure normalises it. A literal blocklist alone cannot catch that, so bound the prefix.
    #>
    $trimmed = $Value.Trim()
    if (@('*', '0.0.0.0/0', '0.0.0.0', 'internet', 'any', 'internetoutbound', '::/0') -contains $trimmed.ToLowerInvariant()) {
        throw "Refusing to open management ports to the whole internet ('$Value'). An RDP endpoint on 0.0.0.0/0 is credential-stuffed within minutes. Pass a single address, e.g. -AllowedSourceIp 203.0.113.42."
    }
    $parts = $trimmed.Split('/')
    if ($parts.Count -gt 2) { throw "-AllowedSourceIp must be an IPv4 address or CIDR (e.g. 203.0.113.42 or 203.0.113.0/29). Got '$Value'." }
    $addr = $parts[0]
    if ($addr -notmatch '^\d{1,3}(\.\d{1,3}){3}$') {
        throw "-AllowedSourceIp must be an IPv4 address or CIDR (e.g. 203.0.113.42 or 203.0.113.0/29). Got '$Value'."
    }
    foreach ($octet in $addr.Split('.')) {
        if ([int]$octet -gt 255) { throw "'$addr' is not a valid IPv4 address (octet '$octet' exceeds 255)." }
    }
    $prefix = 32
    if ($parts.Count -eq 2) {
        if ($parts[1] -notmatch '^\d{1,2}$') { throw "'$Value' has an unparseable prefix length. Use e.g. 203.0.113.0/29." }
        $prefix = [int]$parts[1]
    }
    if ($prefix -lt 24 -or $prefix -gt 32) {
        throw "-AllowedSourceIp '$Value' resolves to a /$prefix. Management access must be restricted to a /24 or narrower (a /$prefix exposes $([math]::Pow(2, 32 - $prefix)) addresses). Use your own address with /32."
    }
    return "$addr/$prefix"
}

function Get-PasswordGuidance {
    return @"
No admin password supplied and this session cannot prompt for one.

Azure Windows VMs require a password for the initial admin account (unlike Linux, 'az vm create'
cannot create a Windows VM with an SSH key only). Supply one as a SecureString, e.g.:

  `$pw = Read-Host -AsSecureString 'admin password'   # typed, masked, never stored
  ./01-provision.ps1 -AdminPassword `$pw

or from a secret store:

  `$pw = ConvertTo-SecureString (az keyvault secret show --vault-name <kv> --name <secret> --query value -o tsv) -AsPlainText -Force

Do NOT put the password in a file, a script, an environment variable, or your shell history.
"@
}

function Resolve-AdminPassword([System.Security.SecureString]$Supplied) {
    if ($Supplied) { return $Supplied }
    if (-not (Test-Interactive)) { throw (Get-PasswordGuidance) }
    Write-Host ''
    Write-Host 'Set the VM admin password (12-123 chars, 3 of: lower / UPPER / digit / symbol).' -ForegroundColor Yellow
    Write-Host ("Avoid these characters - the Azure CLI is a batch file and cmd.exe mangles them: {0}" -f ($CmdMetacharacters -join ' ')) -ForegroundColor Yellow
    Write-Host 'Input is masked and is never written to disk or logged.' -ForegroundColor DarkGray
    for ($attempt = 1; $attempt -le 3; $attempt++) {
        try {
            $first = Read-Host -AsSecureString "Password for $AdminUsername"
            $again = Read-Host -AsSecureString 'Confirm'
        }
        catch {
            # Some hosts (CI runners, redirected stdin) claim to be interactive but cannot prompt.
            throw (Get-PasswordGuidance)
        }
        $p1 = ConvertFrom-SecureStringPlain $first
        $p2 = ConvertFrom-SecureStringPlain $again
        try {
            if ($p1 -ne $p2) { Write-Warning 'Passwords did not match.'; continue }
            $problem = Test-WindowsPasswordComplexity $p1 $AdminUsername
            if ($problem) { Write-Warning "Password $problem."; continue }
            return $first
        }
        finally { $p1 = $null; $p2 = $null }
    }
    throw 'Could not read a valid admin password after 3 attempts.'
}

# NOTE: the parameter is deliberately NOT called $IsWindows - that is a read-only automatic
# variable in PowerShell 6+ and binding to it throws at runtime.
function Get-RetailHourlyPrice([string]$ArmSku, [string]$Region, [bool]$SpotMeter, [bool]$WindowsMeter) {
    # Live Azure retail pricing API; anonymous, no auth. Falls back to the verified constants.
    try {
        $filter = "armRegionName eq '$Region' and armSkuName eq '$ArmSku' and priceType eq 'Consumption'"
        $uri = 'https://prices.azure.com/api/retail/prices?$filter=' + [uri]::EscapeDataString($filter)
        $response = Invoke-RestMethod -Uri $uri -TimeoutSec 15 -ErrorAction Stop
        if (-not (Test-JsonProperty $response 'Items')) { return $null }
        $now = [datetime]::UtcNow
        $candidates = @($response.Items | Where-Object {
            $_.unitOfMeasure -eq '1 Hour' -and
            $_.serviceFamily -eq 'Compute' -and
            # 'HDInsight FSv2 Series' shares the ARM SKU name at a much higher rate; exclude it.
            $_.productName -like 'Virtual Machines*' -and
            $_.meterName -notlike '*Low Priority*' -and
            (($_.skuName -like '* Spot') -eq $SpotMeter) -and
            (($_.productName -like '*Windows*') -eq $WindowsMeter) -and
            ([datetime]$_.effectiveStartDate) -le $now
        })
        if ($candidates.Count -eq 0) { return $null }
        # The API returns one row per price revision for the SAME meter, each with its own
        # effectiveStartDate. Taking the cheapest would quietly quote a superseded rate, so take
        # the most recently effective one - that is what you are actually billed today.
        $current = $candidates | Sort-Object { [datetime]$_.effectiveStartDate } -Descending | Select-Object -First 1
        return [double]$current.retailPrice
    }
    catch { return $null }
}

function Get-DiskMonthlyCost([string]$Sku, [int]$SizeGb) {
    # Managed disks round UP to the next tier, so a 130 GiB disk bills as 256 GiB.
    $tiers = $DiskTierMonthly[$Sku]
    $tier = ($tiers.Keys | Sort-Object | Where-Object { $_ -ge $SizeGb } | Select-Object -First 1)
    if ($null -eq $tier) { $tier = ($tiers.Keys | Sort-Object | Select-Object -Last 1) }
    return [double]$tiers[$tier]
}

#endregion ------------------------------------------------------------------------------------

#region preflight -----------------------------------------------------------------------------

if (-not (Get-Command az -ErrorAction SilentlyContinue)) {
    throw @'
The Azure CLI (az) is not on PATH. Either:
  * run this script in Azure Cloud Shell (https://portal.azure.com -> the >_ icon), where az is
    preinstalled and already signed in; or
  * install it: https://learn.microsoft.com/cli/azure/install-azure-cli
'@
}

# `az vm create --os-disk-delete-option / --nic-delete-option` landed in 2.30; --max-price a little
# earlier. An older az fails with a raw argparse error AFTER the password prompt and AFTER the
# resource group and NSG exist, which is the worst possible moment to discover it.
$MinAzVersion = [version]'2.30.0'
$azVersionText = $null
$azVersionJson = Invoke-AzCli -Arguments @('version', '-o', 'json') -AllowFailure   # `az version` exists since 2.7
if ($azVersionJson -and (Test-JsonProperty $azVersionJson 'azure-cli')) {
    $azVersionText = [string]$azVersionJson.'azure-cli'
}
else {
    $legacy = Invoke-AzCli -Arguments @('--version') -Raw -AllowFailure
    if ($legacy -and ($legacy -match 'azure-cli\s+(\d+\.\d+\.\d+)')) { $azVersionText = $Matches[1] }
}
if ($azVersionText -and ($azVersionText -match '^(\d+\.\d+\.\d+)')) {
    $azVersion = [version]$Matches[1]
    if ($azVersion -lt $MinAzVersion) {
        throw "Azure CLI $azVersion is too old; this script needs $MinAzVersion or newer for 'az vm create --os-disk-delete-option/--nic-delete-option/--max-price'. Upgrade with 'az upgrade' (or your package manager), or run it in Azure Cloud Shell."
    }
    Write-Note "azure-cli $azVersion"
}
else {
    Write-Warning "Could not determine the Azure CLI version; this script needs $MinAzVersion or newer. If 'az vm create' fails with 'unrecognized arguments', that is why."
}

$account = Invoke-AzCli -Arguments @('account', 'show', '-o', 'json') -AllowFailure
if (-not $account) { throw "Not signed in to Azure. Run 'az login' (or 'az login --service-principal ...') and retry." }
if ($SubscriptionId) {
    Invoke-AzCli -Arguments @('account', 'set', '--subscription', $SubscriptionId) | Out-Null
    $account = Invoke-AzCli -Arguments @('account', 'show', '-o', 'json')
}
Write-Step "Subscription: $($account.name)  ($($account.id))"

if ($env:ACC_CLOUD -or ($env:AZUREPS_HOST_ENVIRONMENT -and $env:AZUREPS_HOST_ENVIRONMENT -like '*cloud-shell*')) {
    Write-Warning 'Cloud Shell detected. An auto-detected source IP will be CLOUD SHELL''s egress IP, not yours - you would be locked out of RDP. Pass -AllowedSourceIp <your public IP>.'
}

#endregion ------------------------------------------------------------------------------------

#region teardown ------------------------------------------------------------------------------

if ($Teardown) {
    if (-not $Yes) {
        throw "Refusing to tear down without -Yes. This deletes resource group '$ResourceGroup' and EVERY checkpoint on the VM's disk. Copy checkpoints off the VM first (see 'Getting your model out' below)."
    }
    $exists = Invoke-AzCli -Arguments @('group', 'exists', '-n', $ResourceGroup) -Raw
    if ($exists -ne 'true') {
        Write-Host "Resource group '$ResourceGroup' does not exist. Nothing to tear down - ongoing cost is already `$0."
        return
    }
    Write-Warning "Deleting resource group '$ResourceGroup' - VM, OS disk, checkpoints, public IP and NSG are all destroyed."
    if ($PSCmdlet.ShouldProcess($ResourceGroup, 'az group delete')) {
        Invoke-AzCli -Arguments @('group', 'delete', '-n', $ResourceGroup, '--yes', '--no-wait') | Out-Null
        Write-Host 'Deletion started (async). Verify it finished with:' -ForegroundColor Green
        Write-Host "  az group exists -n $ResourceGroup    # 'false' means you are at `$0/hr ongoing"
    }
    return
}

#endregion ------------------------------------------------------------------------------------

#region validation ----------------------------------------------------------------------------

$reservedNames = @('administrator', 'admin', 'user', 'user1', 'test', 'user2', 'test1', 'user3',
    'admin1', '1', '123', 'a', 'actuser', 'adm', 'admin2', 'aspnet', 'backup', 'console', 'david',
    'guest', 'john', 'owner', 'root', 'server', 'sql', 'support', 'support_388945a0', 'sys',
    'test2', 'test3', 'user4', 'user5')
if ($reservedNames -contains $AdminUsername.ToLowerInvariant()) {
    throw "Azure reserves the admin username '$AdminUsername'. Pick another, e.g. -AdminUsername mcaiadmin."
}

if ($Spot -and $MaxPrice -eq -1) {
    throw @'
-MaxPrice -1 means "keep this VM running up to the full pay-as-you-go rate". On Windows F16s_v2
that is roughly 2x the Spot rate, so a capacity crunch could quietly burn the entire $100 budget in
about three days. Pick a real cap instead, e.g. -MaxPrice 0.90 (default).
Only override this if you genuinely want PAYG - in which case use -Spot:$false, which is the same
price with no eviction risk.
'@
}

if ($AllowedSourceIp) {
    $SourceCidr = Resolve-ManagementCidr $AllowedSourceIp
}
else {
    Write-Step 'Looking up your current public IP for the management firewall rule'
    $detected = Get-CallerPublicIp
    if (-not $detected) {
        throw 'Could not determine your public IP, and this script will not fall back to opening RDP to the internet. Find it (https://api.ipify.org) and pass -AllowedSourceIp <ip>.'
    }
    $SourceCidr = Resolve-ManagementCidr $detected
    Write-Note "Detected $detected. Note this is a /32 snapshot: if your ISP rotates your address you must re-run this script (or update the NSG rule) to regain access."
}

Write-Step "Checking SKU '$Size' availability in '$Location'"
$skus = Invoke-AzCli -Arguments @('vm', 'list-skus', '-l', $Location, '--size', $Size,
    '--resource-type', 'virtualMachines', '-o', 'json')
$sku = @($skus) | Where-Object { $_.name -eq $Size } | Select-Object -First 1
if (-not $sku) {
    throw "SKU '$Size' is not offered in region '$Location'. List alternatives with: az vm list-skus -l $Location --resource-type virtualMachines --query `"[?starts_with(name,'Standard_F')].name`" -o tsv"
}
if ((Test-JsonProperty $sku 'restrictions') -and @($sku.restrictions).Count -gt 0) {
    # az routinely reports Zone-scoped restrictions for compute SKUs that are perfectly deployable
    # regionally. This script never passes --zones, so only a Location restriction actually blocks
    # the deployment; treating every restriction as fatal dead-ends provisioning on a false positive.
    $restrictions = @($sku.restrictions) | Where-Object { Test-JsonProperty $_ 'type' }
    $blocking = @($restrictions | Where-Object { $_.type -eq 'Location' })
    if ($blocking.Count -gt 0) {
        $reasons = ($blocking | ForEach-Object { $_.reasonCode }) -join ', '
        throw "SKU '$Size' is restricted in '$Location' for this subscription at the LOCATION level ($reasons). Usually this means zero quota or the SKU is not enabled for your offer - request access in Portal -> Subscriptions -> Usage + quotas, or pick another -Location."
    }
    foreach ($zoneRestriction in @($restrictions | Where-Object { $_.type -eq 'Zone' })) {
        $zones = ''
        if ((Test-JsonProperty $zoneRestriction 'restrictionInfo') -and (Test-JsonProperty $zoneRestriction.restrictionInfo 'zones')) {
            $zones = (@($zoneRestriction.restrictionInfo.zones) -join ', ')
        }
        Write-Note "Zone restriction on '$Size' in '$Location' (zones: $zones) - not applicable, this deployment is regional (no --zones)."
    }
}
if (-not (Test-JsonProperty $sku 'capabilities')) { throw "Azure returned no capability list for '$Size'; cannot verify its vCPU count." }
$vCpuCapability = @($sku.capabilities) | Where-Object { $_.name -eq 'vCPUs' } | Select-Object -First 1
if (-not $vCpuCapability) { throw "Could not read the vCPU count for '$Size'." }
$RequiredVCpus = [int]$vCpuCapability.value
$SkuFamily = if (Test-JsonProperty $sku 'family') { $sku.family } else { $null }
if (-not $SkuFamily) { throw "Azure did not report a quota family for '$Size'; cannot verify quota safely." }
Write-Note "$Size = $RequiredVCpus vCPU, quota family '$SkuFamily'."

Write-Step 'Checking vCPU quota (Spot draws on a SEPARATE regional pool from the standard family quota)'
$usages = Invoke-AzCli -Arguments @('vm', 'list-usage', '-l', $Location, '-o', 'json')

function Get-QuotaHeadroom([string]$UsageName) {
    $entry = @($usages) | Where-Object { $_.name.value -eq $UsageName } | Select-Object -First 1
    if (-not $entry) { return $null }
    return [pscustomobject]@{
        Name     = $UsageName
        Current  = [int]$entry.currentValue
        Limit    = [int]$entry.limit
        Headroom = ([int]$entry.limit - [int]$entry.currentValue)
    }
}

# A Spot deployment is charged against ONE pool: 'Total Regional Spot vCPUs' (lowPriorityCores).
# The per-family and 'Total Regional vCPUs' quotas gate STANDARD-priority deployments only, and a
# fresh pay-as-you-go subscription very often has standardFSv2Family = 10 while Spot has room. Hard-
# failing on those would send the user on a quota-ticket detour for a deployment that would succeed.
$blockingChecks = if ($Spot) { @('lowPriorityCores') } else { @($SkuFamily, 'cores') }
$advisoryChecks = if ($Spot) { @($SkuFamily, 'cores') } else { @() }
$quotaFailures = @()
$quotaAdvisories = @()
foreach ($name in ($blockingChecks + $advisoryChecks)) {
    $quota = Get-QuotaHeadroom $name
    if (-not $quota) { Write-Note "Quota metric '$name' not reported by this subscription - skipping."; continue }
    Write-Note ("{0,-24} {1,4} / {2,-4} used  (headroom {3})" -f $quota.Name, $quota.Current, $quota.Limit, $quota.Headroom)
    if ($quota.Headroom -lt $RequiredVCpus) {
        if ($blockingChecks -contains $name) { $quotaFailures += $quota } else { $quotaAdvisories += $quota }
    }
}
foreach ($advisory in $quotaAdvisories) {
    Write-Warning ("Standard-priority quota '{0}' has only {1} of the {2} vCPU free. Spot does not draw on it, so this does NOT block the deployment - but -Spot:`$false would fail until you raise it." -f $advisory.Name, $advisory.Headroom, $RequiredVCpus)
}
if ($quotaFailures.Count -gt 0) {
    $lines = $quotaFailures | ForEach-Object { "  - $($_.Name): need $RequiredVCpus vCPU, only $($_.Headroom) free (limit $($_.Limit))" }
    $poolHint = if ($Spot) { "  * 'Total Regional Spot vCPUs' (lowPriorityCores) - this is the pool Spot draws on" }
                else { "  * '$SkuFamily' (standard family vCPUs) and 'Total Regional vCPUs' (cores)" }
    throw @"
Insufficient vCPU quota in '$Location':
$($lines -join "`n")

Request an increase (usually auto-approved in minutes):
  Portal -> Subscriptions -> $($account.name) -> Usage + quotas -> filter by region '$Location'
$poolHint
Or drop to a smaller SKU, e.g. -Size Standard_F8s_v2 (8 vCPU, ~half the hourly rate).
"@
}

Write-Step "Resolving image '$ImageUrn'"
$image = Invoke-AzCli -Arguments @('vm', 'image', 'show', '--urn', $ImageUrn, '-o', 'json') -AllowFailure
$imageOsDiskGb = 127
if ($image -and (Test-JsonProperty $image 'osDiskImage') -and (Test-JsonProperty $image.osDiskImage 'sizeInGb')) {
    $imageOsDiskGb = [int]$image.osDiskImage.sizeInGb
}
if ($DiskSizeGb -lt $imageOsDiskGb) {
    # Azure can grow an OS disk but never shrink one below the image's own size.
    Write-Warning "The Windows Server 2022 image needs a $imageOsDiskGb GiB OS disk; -DiskSizeGb $DiskSizeGb is not possible. Using $imageOsDiskGb GiB."
    $DiskSizeGb = $imageOsDiskGb
}

if ($DiskSku -eq 'Standard_LRS') {
    # Everything on this box shares the OS disk: Paper's chunk saves, the trainer's atomic
    # multi-hundred-MB checkpoint writes and four processes' logs.
    Write-Warning 'Standard_LRS is a spinning HDD (~500 IOPS, unbounded latency). Paper world saves contending with checkpoint writes will drop TPS and 04-monitor.ps1 will start reporting the stack as unhealthy. Over a ~6-day run it saves under $1 versus StandardSSD_LRS. Strongly prefer -DiskSku StandardSSD_LRS.'
}

#endregion ------------------------------------------------------------------------------------

#region cost ----------------------------------------------------------------------------------

Write-Step 'Pricing'
# Under Hybrid Benefit you bring your own licence, so the Linux (compute-only) meter is the rate.
$licenceIsWindows = -not $HybridBenefit.IsPresent
$live = Get-RetailHourlyPrice -ArmSku $Size -Region $Location -SpotMeter $Spot -WindowsMeter $licenceIsWindows
if ($null -ne $live) {
    $computeHourly = $live
    $priceSource = 'live Azure retail pricing API'
}
else {
    $computeHourly = if (-not $Spot) { $FallbackPrices.WindowsPayg }
                     elseif ($HybridBenefit) { $FallbackPrices.LinuxSpot }
                     else { $FallbackPrices.WindowsSpot }
    $priceSource = 'verified fallback table (pricing API unreachable)'
}

$diskMonthly = Get-DiskMonthlyCost -Sku $DiskSku -SizeGb $DiskSizeGb
$diskHourly = $diskMonthly / 730.0
$idleHourly = $diskHourly + $PublicIpHourly           # what you pay while DEALLOCATED
$runningHourly = $computeHourly + $idleHourly
$affordableHours = [math]::Floor($BudgetUsd / $runningHourly)
$affordableDays = [math]::Round($affordableHours / 24.0, 1)

Write-Note "source: $priceSource"
Write-Host ''
$computeLabel = "compute ($Size, $(if ($Spot) { 'Spot' } else { 'pay-as-you-go' }))"
$diskLabel = "OS disk ($DiskSizeGb GiB $DiskSku)"
Write-Host ("  {0,-40} `${1,8:N4} /hr" -f $computeLabel, $computeHourly)
Write-Host ("  {0,-40} `${1,8:N4} /hr  (~`${2:N2}/mo, billed even when deallocated)" -f $diskLabel, $diskHourly, $diskMonthly)
Write-Host ("  {0,-40} `${1,8:N4} /hr  (~`${2:N2}/mo, billed even when deallocated)" -f 'public IP (Standard, static)', $PublicIpHourly, ($PublicIpHourly * 730))
Write-Host ("  {0,-40} `${1,8:N4} /hr" -f 'TOTAL WHILE TRAINING', $runningHourly) -ForegroundColor Yellow
Write-Host ("  {0,-40} `${1,8:N4} /hr  (~`${2:N2}/day parked)" -f 'TOTAL WHILE DEALLOCATED', $idleHourly, ($idleHourly * 24))
Write-Host ''
Write-Host ("  `${0:N0} budget  ->  ~{1} training hours  (~{2} days continuous)" -f $BudgetUsd, $affordableHours, $affordableDays) -ForegroundColor Green
if ($Spot) {
    Write-Host ("  eviction cap: --max-price `${0:N4}/hr; above that Azure deallocates instead of overcharging." -f $MaxPrice)
    if ($MaxPrice -lt $computeHourly) {
        throw ("-MaxPrice `${0:N4} is BELOW the current Spot rate `${1:N4} for {2} in {3}. Deployment would fail (or the VM would be evicted immediately). Raise -MaxPrice above the current rate." -f $MaxPrice, $computeHourly, $Size, $Location)
    }
}
if (-not $HybridBenefit) {
    Write-Note 'Roughly half of the Windows VM rate is the Windows Server licence. If you own licences with active Software Assurance, -HybridBenefit removes it and nearly doubles your training hours.'
}
if ($DiskSku -ne 'Premium_LRS') {
    $premiumDelta = (Get-DiskMonthlyCost -Sku 'Premium_LRS' -SizeGb $DiskSizeGb) - $diskMonthly
    Write-Note ("If disk I/O turns out to be the limiter, -DiskSku Premium_LRS costs about `${0:N2} more over the whole {1}-hour window." -f (($premiumDelta / 730.0) * $affordableHours), $affordableHours)
}
Write-Host ''
# A wall-clock deadline is far more actionable than "135 hours"; a budget alert is retrospective.
$budgetDeadline = [datetime]::UtcNow.AddHours($affordableHours)
Write-Host ("BUDGET DEADLINE: running continuously from now, `${0:N0} is exhausted at ~{1} UTC. Put it in your calendar." -f $BudgetUsd, $budgetDeadline.ToString('yyyy-MM-dd HH:mm')) -ForegroundColor Yellow
Write-Host ''
Write-Host 'Create a Cost Management budget alert (version-sensitive: the command moved between az' -ForegroundColor DarkGray
Write-Host 'releases; if it errors, use Portal -> Cost Management -> Budgets -> Add):' -ForegroundColor DarkGray
Write-Host ("  az consumption budget create-with-rg --resource-group $ResourceGroup --budget-name mcai-budget ``") -ForegroundColor DarkGray
Write-Host ("    --amount $BudgetUsd --category cost --time-grain monthly --start-date {0} --end-date {1}" -f ([datetime]::UtcNow.ToString('yyyy-MM-01')), ([datetime]::UtcNow.AddMonths(6).ToString('yyyy-MM-01'))) -ForegroundColor DarkGray
Write-Host '  # Budgets ALERT, they never stop anything. Only teardown stops billing.' -ForegroundColor DarkGray
Write-Host ''

#endregion ------------------------------------------------------------------------------------

#region provision -----------------------------------------------------------------------------

$target = "$VmName in $ResourceGroup ($Location)"
if (-not $PSCmdlet.ShouldProcess($target, 'provision Azure Windows training VM')) {
    Write-Host 'WhatIf: nothing was created and nothing was billed.' -ForegroundColor Yellow
    return
}

# Probe first so that the common idempotent re-run (refreshing the firewall rule after your ISP
# hands you a new IP) never demands a password for a VM that already exists.
$vm = Invoke-AzCli -Arguments @('vm', 'show', '-g', $ResourceGroup, '-n', $VmName, '-o', 'json') -AllowFailure
$securePassword = if ($vm) { $null } else { Resolve-AdminPassword $AdminPassword }

Write-Step "Resource group '$ResourceGroup'"
if ((Invoke-AzCli -Arguments @('group', 'exists', '-n', $ResourceGroup) -Raw) -eq 'true') {
    Write-Note 'already exists.'
}
else {
    Invoke-AzCli -Arguments @('group', 'create', '-n', $ResourceGroup, '-l', $Location,
        '--tags', 'project=mcai', "budget-usd=$BudgetUsd", '-o', 'none') | Out-Null
    Write-Note 'created.'
}

# The NSG is created BEFORE the VM and handed to `az vm create --nsg`, so the VM is never briefly
# reachable through az's default "allow RDP from anywhere" rule.
Write-Step "Network security group '$NsgName'"
$nsg = Invoke-AzCli -Arguments @('network', 'nsg', 'show', '-g', $ResourceGroup, '-n', $NsgName, '-o', 'json') -AllowFailure
if (-not $nsg) {
    Invoke-AzCli -Arguments @('network', 'nsg', 'create', '-g', $ResourceGroup, '-n', $NsgName,
        '-l', $Location, '-o', 'none') | Out-Null
    Write-Note 'created.'
}
else { Write-Note 'already exists.' }

function Set-NsgRule {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][int]$Priority,
        [Parameter(Mandatory)][string]$Access,
        [Parameter(Mandatory)][string]$SourcePrefix,
        [Parameter(Mandatory)][string]$DestinationPort,
        [Parameter(Mandatory)][string]$Protocol,
        [Parameter(Mandatory)][string]$Description
    )
    $common = @(
        '-g', $ResourceGroup, '--nsg-name', $NsgName, '-n', $Name,
        '--priority', "$Priority", '--access', $Access, '--direction', 'Inbound',
        '--protocol', $Protocol, '--source-address-prefixes', $SourcePrefix,
        '--source-port-ranges', '*', '--destination-address-prefixes', '*',
        '--destination-port-ranges', $DestinationPort, '--description', $Description, '-o', 'none'
    )
    $existing = Invoke-AzCli -Arguments @('network', 'nsg', 'rule', 'show', '-g', $ResourceGroup,
        '--nsg-name', $NsgName, '-n', $Name, '-o', 'json') -AllowFailure
    $verb = if ($existing) { 'update' } else { 'create' }
    Invoke-AzCli -Arguments (@('network', 'nsg', 'rule', $verb) + $common) | Out-Null
    Write-Note "$verb rule $Name ($Access $Protocol/$DestinationPort from $SourcePrefix)"
}

# Only management ports, only from one address. Everything the training stack uses - Minecraft
# 25565, arena 8765, trainer 8766, dashboard 8788 - stays on 127.0.0.1 inside the VM and is
# deliberately absent here; the dashboard is reached with `ssh -L`, not by opening a port.
# Tradeoff: a /32 allow-list means a changing home IP locks you out until you re-run this script.
# That is the right side of the tradeoff for a box that will sit unattended for days.
if ($ManagementProtocol -in @('Rdp', 'Both')) {
    Set-NsgRule -Name $RuleAllowRdp -Priority 1000 -Access 'Allow' -SourcePrefix $SourceCidr `
        -DestinationPort '3389' -Protocol 'Tcp' -Description 'RDP from the operator address only'
}
if ($ManagementProtocol -in @('Ssh', 'Both')) {
    Set-NsgRule -Name $RuleAllowSsh -Priority 1010 -Access 'Allow' -SourcePrefix $SourceCidr `
        -DestinationPort '22' -Protocol 'Tcp' -Description 'SSH (dashboard tunnel) from the operator address only'
}
# Azure's built-in DenyAllInBound already sits at 65500; this explicit deny at 4000 additionally
# blocks anything a future rule might try to allow at a lower precedence, and makes the intent
# auditable in the portal.
Set-NsgRule -Name $RuleDenyAll -Priority 4000 -Access 'Deny' -SourcePrefix '*' `
    -DestinationPort '*' -Protocol '*' -Description 'Deny all other inbound traffic'

# Re-running with a NARROWER -ManagementProtocol must actually narrow the NSG. Without this, a rule
# created by an earlier run survives and the summary printed below ("Inbound: Rdp from x/32 only")
# would be a lie - the worst kind of security reporting.
$desiredManagementRules = @()
if ($ManagementProtocol -in @('Rdp', 'Both')) { $desiredManagementRules += $RuleAllowRdp }
if ($ManagementProtocol -in @('Ssh', 'Both')) { $desiredManagementRules += $RuleAllowSsh }
foreach ($stale in (@($RuleAllowRdp, $RuleAllowSsh) | Where-Object { $desiredManagementRules -notcontains $_ })) {
    $leftover = Invoke-AzCli -Arguments @('network', 'nsg', 'rule', 'show', '-g', $ResourceGroup,
        '--nsg-name', $NsgName, '-n', $stale, '-o', 'json') -AllowFailure
    if ($leftover) {
        Invoke-AzCli -Arguments @('network', 'nsg', 'rule', 'delete', '-g', $ResourceGroup,
            '--nsg-name', $NsgName, '-n', $stale, '-o', 'none') | Out-Null
        Write-Note "deleted stale rule $stale (not in -ManagementProtocol $ManagementProtocol)"
    }
}

Write-Step "Virtual machine '$VmName'"
if ($vm) {
    Write-Note 'already exists - skipping creation (network rules above were still re-applied).'
}
else {
    $plainPassword = ConvertFrom-SecureStringPlain $securePassword
    try {
        $problem = Test-WindowsPasswordComplexity $plainPassword $AdminUsername
        if ($problem) { throw "The supplied admin password $problem." }
        # Unavoidable with `az vm create`: ARM has no stdin path for the password, so it appears in
        # the az process command line for the ~2 minutes of the deployment (visible to a local admin
        # via Win32_Process, and captured by Event 4688 if command-line auditing is on). Rotate with
        # `az vm user update` afterwards if that box is shared or audited.
        Write-Note 'The password is passed on the az command line (ARM offers no alternative) and is visible to local admins for the duration of the deployment.'

        $createArgs = @(
            'vm', 'create',
            '-g', $ResourceGroup, '-n', $VmName, '-l', $Location,
            '--image', $ImageUrn,
            '--size', $Size,
            '--admin-username', $AdminUsername,
            '--admin-password', $plainPassword,
            '--os-disk-size-gb', "$DiskSizeGb",
            '--storage-sku', $DiskSku,
            '--os-disk-delete-option', 'Delete',
            '--nic-delete-option', 'Delete',
            # Standard SKU public IPs are always static, so the address survives an eviction +
            # restart. With a dynamic Basic IP the address would change on every Spot eviction and
            # your saved RDP/SSH target would silently break.
            '--public-ip-sku', 'Standard',
            '--public-ip-address-allocation', 'static',
            '--nsg', $NsgName,
            '--tags', 'project=mcai', 'workload=self-play-training',
            '-o', 'json'
        )
        if ($Spot) {
            # Deallocate (not Delete) is the whole reason Spot is safe here: the OS disk, the repo
            # and checkpoints/latest.pt survive an eviction and training resumes on `az vm start`.
            # Invariant culture: on a comma-decimal locale "0,90000" would be rejected by az.
            $createArgs += @('--priority', 'Spot', '--eviction-policy', 'Deallocate',
                '--max-price', $MaxPrice.ToString('0.#####', [System.Globalization.CultureInfo]::InvariantCulture))
        }
        if ($HybridBenefit) { $createArgs += @('--license-type', 'Windows_Server') }

        Write-Note 'creating (2-4 minutes)...'
        $vm = Invoke-AzCli -Arguments $createArgs -Redact @($plainPassword)
    }
    catch {
        # A create that fails partway (usually Spot allocation) can still have left an ARM-created
        # NIC and a Standard STATIC public IP behind - the IP bills at ~$0.005/hr forever, in an
        # otherwise empty resource group, which is 3.7% of a $100 budget for nothing.
        Write-Host ''
        Write-Warning "'az vm create' failed. The deployment may have left a BILLABLE public IP and NIC in '$ResourceGroup'."
        Write-Host 'Most likely causes, in order:' -ForegroundColor Yellow
        Write-Host "  * Spot capacity exhausted (OverconstrainedAllocationRequest / SkuNotAvailable / AllocationFailed)" -ForegroundColor Yellow
        Write-Host "      retry later, or: -Location westus2 | -Size Standard_F8s_v2 | -Spot:`$false (no eviction, ~2x rate)" -ForegroundColor Yellow
        Write-Host "  * -MaxPrice $MaxPrice below the live Spot rate -> raise it" -ForegroundColor Yellow
        Write-Host "  * 'unrecognized arguments' -> Azure CLI older than $MinAzVersion; run 'az upgrade'" -ForegroundColor Yellow
        Write-Host ''
        Write-Host 'Check for and clear orphaned billing resources before retrying:' -ForegroundColor Yellow
        Write-Host "  az resource list -g $ResourceGroup -o table" -ForegroundColor Yellow
        Write-Host "  ./01-provision.ps1 -ResourceGroup $ResourceGroup -Teardown -Yes    # nothing to preserve yet" -ForegroundColor Yellow
        Write-Host ''
        throw
    }
    finally {
        # Best effort: drop the plaintext reference promptly rather than leaving it for the GC.
        $plainPassword = $null
        [System.GC]::Collect()
    }
    Write-Note 'created.'
}

# `az vm create` reports the address as publicIpAddress; `az vm show -d` calls it publicIps.
$PublicIp = $null
foreach ($field in @('publicIpAddress', 'publicIps')) {
    if ((Test-JsonProperty $vm $field) -and $vm.$field) { $PublicIp = [string]$vm.$field; break }
}
if (-not $PublicIp) {
    $ipInfo = Invoke-AzCli -Arguments @('vm', 'list-ip-addresses', '-g', $ResourceGroup, '-n', $VmName, '-o', 'json') -AllowFailure
    $first = @($ipInfo) | Select-Object -First 1
    if ($first) {
        $PublicIp = @($first.virtualMachine.network.publicIpAddresses) |
            Select-Object -First 1 | ForEach-Object { $_.ipAddress }
    }
}
if (-not $PublicIp) { $PublicIp = '<public-ip-not-yet-assigned>' }

if ($AutoShutdownUtc) {
    Write-Step "Daily auto-shutdown at $AutoShutdownUtc UTC"
    Invoke-AzCli -Arguments @('vm', 'auto-shutdown', '-g', $ResourceGroup, '-n', $VmName,
        '--time', $AutoShutdownUtc, '--time-zone', 'UTC', '-o', 'none') | Out-Null
    Write-Note "Registered. This is a HARD daily compute stop: training halts at $AutoShutdownUtc UTC every day and stays halted until you run 'az vm start'."
    Write-Note ("It does NOT stop disk + public IP billing (~`${0:N2}/day while deallocated). Only teardown does." -f ($idleHourly * 24))
    Write-Note "Remove it with: az vm auto-shutdown -g $ResourceGroup -n $VmName --off"
}

# Spot eviction watchdog. Nothing INSIDE the VM can restart it - 03-run-training.ps1's boot task
# only helps once the VM boots, and 04-monitor.ps1 is dead while the VM is deallocated. So the
# restart has to be driven from outside Azure's compute, i.e. from here.
$watchdogTaskName = "MCAI-SpotRestart-$VmName"
$watchdogPowerShell = "if ((az vm get-instance-view -g $ResourceGroup -n $VmName --query ""instanceView.statuses[?starts_with(code,'PowerState')].code"" -o tsv) -match 'deallocated') { az vm start -g $ResourceGroup -n $VmName --no-wait }"
$watchdogCron = "*/10 * * * * az vm get-instance-view -g $ResourceGroup -n $VmName --query `"instanceView.statuses[?starts_with(code,'PowerState')].code`" -o tsv | grep -q deallocated && az vm start -g $ResourceGroup -n $VmName --no-wait"
if ($EvictionWatchdog) {
    Write-Step "Eviction watchdog (on THIS machine, every 10 minutes)"
    if (Get-Command Register-ScheduledTask -ErrorAction SilentlyContinue) {
        if ($PSCmdlet.ShouldProcess($watchdogTaskName, 'register local scheduled task')) {
            # -EncodedCommand sidesteps every layer of quoting between Task Scheduler, powershell.exe
            # and the JMESPath string, which is otherwise a reliable source of silent breakage.
            $encoded = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($watchdogPowerShell))
            $wdAction = New-ScheduledTaskAction -Execute 'powershell.exe' `
                -Argument "-NoProfile -NonInteractive -WindowStyle Hidden -EncodedCommand $encoded"
            # An explicit duration: -RepetitionInterval alone defaults to a 1-day repetition window
            # on some Task Scheduler builds, which would silently retire the watchdog on day two.
            $wdTrigger = New-ScheduledTaskTrigger -Once -At ([datetime]::Now.AddMinutes(2)) `
                -RepetitionInterval (New-TimeSpan -Minutes 10) -RepetitionDuration (New-TimeSpan -Days 30)
            # Runs as the interactive user only: `az` needs THIS user's token cache, which SYSTEM
            # cannot read. That means the watchdog is asleep while you are signed out.
            Register-ScheduledTask -TaskName $watchdogTaskName -Action $wdAction -Trigger $wdTrigger -Force | Out-Null
            Write-Note "Registered scheduled task '$watchdogTaskName'. It only runs while you are signed in (az needs your token cache)."
            Write-Note "Remove it with: Unregister-ScheduledTask -TaskName $watchdogTaskName -Confirm:`$false"
        }
    }
    else {
        Write-Warning 'Register-ScheduledTask is Windows-only; no watchdog was created. Add this cron entry instead (crontab -e):'
        Write-Host "  $watchdogCron"
    }
}

#endregion ------------------------------------------------------------------------------------

#region next steps ----------------------------------------------------------------------------

$dashboardPort = 8788
Write-Host ''
Write-Host '================================================================' -ForegroundColor Green
Write-Host " VM ready: $VmName   public IP: $PublicIp" -ForegroundColor Green
Write-Host " Inbound: $ManagementProtocol from $SourceCidr only. Everything else denied." -ForegroundColor Green
Write-Host '================================================================' -ForegroundColor Green
Write-Host ''
Write-Host 'NEXT COMMANDS (run these in order - nothing trains until step 4)' -ForegroundColor Cyan
Write-Host ''
Write-Host '1. RDP into the VM (this is the only way in - a bare Windows Server image has no sshd)'
Write-Host "     mstsc /v:$PublicIp                      # from Windows"
Write-Host "     open rdp://full%20address=s:${PublicIp}:3389  # from macOS (or use the Windows App)"
Write-Host "     # user: $AdminUsername   password: the one you just set"
Write-Host ''
Write-Host '2. In an ELEVATED PowerShell on the VM, download and run the bootstrap.'
Write-Host '   Do not git clone first: Windows Server has no git.exe - the bootstrap installs it and'
Write-Host "   clones the repo itself to $RepoOnVm."
Write-Host "     Invoke-WebRequest -UseBasicParsing '$BootstrapRawUrl' -OutFile C:\02-bootstrap.ps1"
Write-Host "     # 404 here means branch '$Branch' is not pushed to $RepoUrl yet - push it first."
Write-Host '     powershell -ExecutionPolicy Bypass -File C:\02-bootstrap.ps1 -AcceptEula'
Write-Host '   -AcceptEula is mandatory: the script hard-fails without it. It means you have read and'
Write-Host '   accept the Minecraft EULA (https://aka.ms/MinecraftEULA). Takes ~10-20 minutes.'
Write-Host '   It installs the toolchain, builds every component and configures the runtime.'
Write-Host '   IT DOES NOT START OR SCHEDULE TRAINING. That is step 4.'
Write-Host ''
if ($ManagementProtocol -in @('Ssh', 'Both')) {
    Write-Host "3. (optional) Enable OpenSSH Server so you can tunnel the dashboard. Port 22 is already"
    Write-Host '   allowed in the NSG but NOTHING installs sshd - run this once, elevated, on the VM:'
    Write-Host '     Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0'
    Write-Host '     Start-Service sshd; Set-Service sshd -StartupType Automatic'
    Write-Host '   Skip this and view the dashboard inside the RDP session instead (step 5);'
    Write-Host "   then re-run with -ManagementProtocol Rdp to close port 22."
    Write-Host ''
}
Write-Host '4. Register the boot task - THIS is what actually starts training, and what makes a Spot' -ForegroundColor Yellow
Write-Host '   eviction + restart resume on its own. Elevated PowerShell on the VM:' -ForegroundColor Yellow
Write-Host "     powershell -ExecutionPolicy Bypass -File $RepoOnVm\deploy\azure-windows\03-run-training.ps1 -Install" -ForegroundColor Yellow
Write-Host '   Verify it took:  schtasks /query /tn MCAI-Training' -ForegroundColor Yellow
Write-Host ''
Write-Host '5. Watch it. Health + budget, on the VM:'
Write-Host "     powershell -File $RepoOnVm\deploy\azure-windows\04-monitor.ps1 -Watch ``"
Write-Host ("       -HourlyRate {0:N4} -DiskMonthlyUsd {1:N2} -BudgetUsd {2:N0}" -f $computeHourly, ($diskMonthly + ($PublicIpHourly * 730)), $BudgetUsd)
Write-Host '     # pass those values: 04-monitor.ps1 defaults under-report this deployment (its disk'
Write-Host '     # default is 64 GB and it models no public IP), which inflates your remaining budget.'
Write-Host '   The dashboard (loopback-only, never opened in the NSG):'
if ($ManagementProtocol -in @('Ssh', 'Both')) {
    Write-Host "     ssh -N -L ${dashboardPort}:127.0.0.1:$dashboardPort $AdminUsername@$PublicIp   # needs step 3"
    Write-Host "     # then open http://127.0.0.1:$dashboardPort on your own machine"
}
Write-Host "     # or simply open http://127.0.0.1:$dashboardPort in a browser inside the RDP session"
Write-Host ''
Write-Host 'OPERATING THE BUDGET' -ForegroundColor Cyan
Write-Host "     az vm show -g $ResourceGroup -n $VmName -d --query powerState -o tsv     # is it running?"
Write-Host "     az vm start      -g $ResourceGroup -n $VmName                            # after a Spot eviction"
Write-Host "     az vm deallocate -g $ResourceGroup -n $VmName                            # pause COMPUTE billing"
Write-Host ("     # deallocated still costs ~`${0:N2}/day (disk + public IP). It is NOT free." -f ($idleHourly * 24)) -ForegroundColor Yellow
Write-Host ''
if ($Spot -and -not $EvictionWatchdog) {
    Write-Host 'SPOT EVICTION: NOTHING RESTARTS THIS VM AUTOMATICALLY' -ForegroundColor Red
    Write-Host '  When Azure evicts, the VM deallocates, training stops, and the on-VM monitor stops' -ForegroundColor Red
    Write-Host '  with it - so no alert fires. It stays down, still billing for disk + IP, until YOU' -ForegroundColor Red
    Write-Host '  run `az vm start`. On a multi-day run this is the biggest source of lost hours.' -ForegroundColor Red
    Write-Host '  Fix it by re-running this script with -EvictionWatchdog, or set it up by hand:' -ForegroundColor Red
    Write-Host "    Windows:  ./01-provision.ps1 -ResourceGroup $ResourceGroup -VmName $VmName -EvictionWatchdog"
    Write-Host '    macOS/Linux (crontab -e):'
    Write-Host "      $watchdogCron"
    Write-Host ''
}
elseif ($Spot) {
    Write-Host "SPOT EVICTION: watchdog task '$watchdogTaskName' checks every 10 min and restarts the VM." -ForegroundColor Green
    Write-Host '  It only runs while you are signed in to this machine (az needs your token cache).' -ForegroundColor Green
    Write-Host ''
}
Write-Host 'GETTING YOUR MODEL OUT (do this before teardown - teardown destroys the disk)' -ForegroundColor Cyan
Write-Host "     scp -r ${AdminUsername}@${PublicIp}:$RepoOnVmPosix/checkpoints ./checkpoints-azure   # needs step 3"
Write-Host '     # or drag the folder through the RDP session'
Write-Host ''
Write-Host 'TEARDOWN (the only path to $0/hr)' -ForegroundColor Cyan
Write-Host "     ./01-provision.ps1 -ResourceGroup $ResourceGroup -Teardown -Yes"
Write-Host "     # equivalently: az group delete -n $ResourceGroup --yes --no-wait"
Write-Host ''
Write-Host 'IF YOUR PUBLIC IP CHANGES, re-run this script to refresh the firewall rule:' -ForegroundColor DarkGray
Write-Host "     ./01-provision.ps1 -ResourceGroup $ResourceGroup -VmName $VmName    # idempotent" -ForegroundColor DarkGray
Write-Host ''

#endregion ------------------------------------------------------------------------------------
