#!/usr/bin/env bash
# Restart a Spot-evicted training VM from INSIDE Azure, with nothing running on your own machine.
#
# Why this exists: an Azure Spot VM can be evicted at any time. With --eviction-policy Deallocate the
# disk and every checkpoint survive, but *nothing restarts the VM* — Azure will not do it for you. A
# watchdog on your laptop is not good enough if the laptop is closed, asleep, or moved, which is
# exactly the case here. This installs an Azure Automation account with a managed identity and a
# schedule, so the check runs in Azure whether or not any machine of yours is on.
#
# Cost: Azure Automation includes 500 free job-minutes per month. A 5-second check every 15 minutes
# is roughly 3 minutes/month, so this is effectively free.
#
# Usage:
#   ./eviction-autorestart.sh --resource-group mcai-rg --vm mcai-train [--location eastus2]
#                             [--interval-minutes 15] [--remove]
set -euo pipefail

RESOURCE_GROUP=""
VM_NAME=""
LOCATION=""
INTERVAL_MINUTES=15
AUTOMATION_ACCOUNT="mcai-autorestart"
RUNBOOK_NAME="Start-EvictedMcaiVm"
SCHEDULE_NAME="mcai-evicted-check"
REMOVE=false

usage() { sed -n '2,18p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }
fail() { printf 'eviction-autorestart: %s\n' "$1" >&2; exit 1; }

while [ "$#" -gt 0 ]; do
  case "$1" in
    -g|--resource-group) [ "$#" -ge 2 ] || fail "$1 needs a value"; RESOURCE_GROUP="$2"; shift 2 ;;
    -n|--vm)             [ "$#" -ge 2 ] || fail "$1 needs a value"; VM_NAME="$2"; shift 2 ;;
    -l|--location)       [ "$#" -ge 2 ] || fail "$1 needs a value"; LOCATION="$2"; shift 2 ;;
    -i|--interval-minutes) [ "$#" -ge 2 ] || fail "$1 needs a value"; INTERVAL_MINUTES="$2"; shift 2 ;;
    --remove)            REMOVE=true; shift ;;
    -h|--help)           usage 0 ;;
    *) fail "unknown argument: $1 (try --help)" ;;
  esac
done

[ -n "$RESOURCE_GROUP" ] || fail "--resource-group is required"
[ -n "$VM_NAME" ] || fail "--vm is required"

# Azure Automation schedules have a one-hour minimum for HOUR frequency, but MINUTE frequency
# allows 15 minutes upward. Anything shorter is rejected by the service.
[ "$INTERVAL_MINUTES" -ge 15 ] 2>/dev/null \
  || fail "--interval-minutes must be at least 15 (Azure Automation minimum)"
command -v az >/dev/null 2>&1 || fail "the Azure CLI is required (run this in Azure Cloud Shell)"

if [ "$REMOVE" = true ]; then
  echo "Removing the auto-restart automation account (the VM itself is untouched)..."
  az automation account delete --resource-group "$RESOURCE_GROUP" \
    --name "$AUTOMATION_ACCOUNT" --yes >/dev/null 2>&1 \
    && echo "Removed $AUTOMATION_ACCOUNT." || echo "Nothing to remove."
  exit 0
fi

az extension show --name automation >/dev/null 2>&1 || {
  echo "Installing the 'automation' CLI extension..."
  az extension add --name automation --only-show-errors
}

SUBSCRIPTION="$(az account show --query id -o tsv)"
[ -n "$LOCATION" ] || LOCATION="$(az group show -n "$RESOURCE_GROUP" --query location -o tsv)"
az vm show -g "$RESOURCE_GROUP" -n "$VM_NAME" >/dev/null \
  || fail "VM '$VM_NAME' not found in resource group '$RESOURCE_GROUP'"

echo "Creating automation account '$AUTOMATION_ACCOUNT' in $LOCATION..."
az automation account create --resource-group "$RESOURCE_GROUP" --name "$AUTOMATION_ACCOUNT" \
  --location "$LOCATION" --sku Free --assign-identity >/dev/null

# The runbook authenticates as the account's managed identity, so no secret is ever stored.
PRINCIPAL_ID="$(az automation account show --resource-group "$RESOURCE_GROUP" \
  --name "$AUTOMATION_ACCOUNT" --query identity.principalId -o tsv)"
[ -n "$PRINCIPAL_ID" ] || fail "the automation account has no managed identity"

# Scope the grant to this ONE VM, not the whole subscription or even the resource group.
echo "Granting the managed identity permission to start only $VM_NAME..."
az role assignment create --assignee-object-id "$PRINCIPAL_ID" \
  --assignee-principal-type ServicePrincipal \
  --role "Virtual Machine Contributor" \
  --scope "/subscriptions/$SUBSCRIPTION/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Compute/virtualMachines/$VM_NAME" \
  >/dev/null 2>&1 || echo "  (role assignment already present)"

RUNBOOK_FILE="$(mktemp -t mcai-runbook-XXXXXX).ps1"
trap 'rm -f "$RUNBOOK_FILE"' EXIT
cat > "$RUNBOOK_FILE" <<'RUNBOOK'
# Starts the MCAI training VM if Spot eviction has deallocated it. Safe to run on any schedule:
# it is a no-op while the VM is running, so a missed or duplicated run cannot cause harm.
param(
    [Parameter(Mandatory = $true)] [string] $ResourceGroupName,
    [Parameter(Mandatory = $true)] [string] $VmName
)
$ErrorActionPreference = 'Stop'
Disable-AzContextAutosave -Scope Process | Out-Null
Connect-AzAccount -Identity | Out-Null

$status = (Get-AzVM -ResourceGroupName $ResourceGroupName -Name $VmName -Status).Statuses |
    Where-Object { $_.Code -like 'PowerState/*' } | Select-Object -First 1
$state = if ($status) { $status.Code } else { 'unknown' }
Write-Output "$VmName power state: $state"

# 'deallocated' is what a Spot eviction leaves behind. 'stopped' means someone shut it down from
# inside the guest, which is deliberate - do NOT fight an operator who stopped it on purpose.
if ($state -eq 'PowerState/deallocated') {
    Write-Output "Evicted; starting $VmName..."
    Start-AzVM -ResourceGroupName $ResourceGroupName -Name $VmName | Out-Null
    Write-Output "Start requested. Docker restart policy resumes training from checkpoints/latest.pt."
} else {
    Write-Output 'No action needed.'
}
RUNBOOK

echo "Publishing runbook '$RUNBOOK_NAME'..."
az automation runbook create --resource-group "$RESOURCE_GROUP" \
  --automation-account-name "$AUTOMATION_ACCOUNT" --name "$RUNBOOK_NAME" \
  --type PowerShell --location "$LOCATION" >/dev/null 2>&1 || true
az automation runbook replace-content --resource-group "$RESOURCE_GROUP" \
  --automation-account-name "$AUTOMATION_ACCOUNT" --name "$RUNBOOK_NAME" \
  --content "@$RUNBOOK_FILE" >/dev/null
az automation runbook publish --resource-group "$RESOURCE_GROUP" \
  --automation-account-name "$AUTOMATION_ACCOUNT" --name "$RUNBOOK_NAME" >/dev/null

echo "Scheduling a check every $INTERVAL_MINUTES minutes..."
# Start a few minutes out: Azure rejects a schedule whose first run is not comfortably in the future.
START_TIME="$(python3 -c "
import datetime
print((datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=10)).strftime('%Y-%m-%dT%H:%M:%S+00:00'))")"
az automation schedule create --resource-group "$RESOURCE_GROUP" \
  --automation-account-name "$AUTOMATION_ACCOUNT" --name "$SCHEDULE_NAME" \
  --frequency Minute --interval "$INTERVAL_MINUTES" --start-time "$START_TIME" >/dev/null 2>&1 || true
az automation job-schedule create --resource-group "$RESOURCE_GROUP" \
  --automation-account-name "$AUTOMATION_ACCOUNT" --runbook-name "$RUNBOOK_NAME" \
  --schedule-name "$SCHEDULE_NAME" \
  --parameters "ResourceGroupName=$RESOURCE_GROUP" "VmName=$VM_NAME" >/dev/null 2>&1 \
  || echo "  (schedule already linked)"

cat <<SUMMARY

Auto-restart installed.
  automation account : $AUTOMATION_ACCOUNT ($LOCATION, Free tier)
  checks every       : $INTERVAL_MINUTES minutes
  acts only when     : power state is 'deallocated' (a Spot eviction)
  permission scope   : $VM_NAME only

Your machine can be off. Worst case you lose up to $INTERVAL_MINUTES minutes of training per
eviction, and the run resumes from the last checkpoint.

Verify it works by deallocating on purpose and waiting one interval:
  az vm deallocate -g $RESOURCE_GROUP -n $VM_NAME
  az vm show -g $RESOURCE_GROUP -n $VM_NAME -d --query powerState -o tsv

Remove it later with:
  $0 --resource-group $RESOURCE_GROUP --vm $VM_NAME --remove
SUMMARY
