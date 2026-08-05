# MCAI on an Azure **Windows** VM — unattended training

Runs the whole MCAI stack (trainer + Paper server + rollout worker + dashboard) on one Windows
Server 2022 VM, headless, on loopback, with only a single admin port reachable and locked to your
own IP.

> **Read this first — Windows costs about 2× Linux.** Verified from the Azure retail pricing API
> (eastus2, `Standard_F16s_v2`, Spot): **Linux $0.3561/hr vs Windows $0.6849/hr (1.92×)** — the
> Windows Server licence is inside the VM rate. On a $100 budget that is roughly **4.9 days of
> training instead of 9.5**. Every component in this repo is platform-agnostic (Python, Node, Java),
> so the *only* thing Windows buys you is a desktop you can RDP into and familiarity. If neither
> matters to you, use [`../azure/`](../azure/) (Linux + Docker) and get double the training.
>
> Pick Windows if: you want to RDP in and watch the bots fight in a real Minecraft client, or you'd
> realistically give up on Linux when something breaks. Both are legitimate.

## The four scripts, in order

| # | Script | Runs on | Purpose |
|---|--------|---------|---------|
| 1 | `01-provision.ps1` | **your machine / Cloud Shell** | Creates the RG, NSG and Spot VM; prints next steps and cost |
| 2 | `02-bootstrap.ps1` | **the VM**, once | Installs deps, clones the repo, builds everything, fetches Paper |
| 3 | `03-run-training.ps1` | **the VM** | Supervises the stack; `-Install` registers the boot task |
| 4 | `04-monitor.ps1` | **the VM** | Training-health + spend checks; `-Install` for scheduled checks |

**Order matters.** `02` does *not* start training and `03 -Install` is the only thing that makes the
box survive a reboot. Skipping step 3 leaves a VM that bills and never trains.

## Run it

**1 — provision (your machine or Cloud Shell).** You will be prompted for an admin password; it is
never written to disk or echoed. RDP/SSH is locked to your current public IP.

```powershell
./01-provision.ps1 -ResourceGroup mcai-rg -Location eastus2 -BudgetUsd 100 -EvictionWatchdog
```

**2 — bootstrap (on the VM, via RDP).** Takes ~20 minutes. `-AcceptEula` is how you accept the
Minecraft EULA; without it the script stops.

```powershell
Invoke-WebRequest -UseBasicParsing `
  https://raw.githubusercontent.com/aarohkandy/MCAI/fix/verified-training-stack/deploy/azure-windows/02-bootstrap.ps1 `
  -OutFile C:\02-bootstrap.ps1
powershell -ExecutionPolicy Bypass -File C:\02-bootstrap.ps1 -AcceptEula
```

**3 — start training and make it survive reboots.**

```powershell
C:\mcai\MCAI\deploy\azure-windows\03-run-training.ps1 -Install
```

**4 — check it is actually learning.**

```powershell
C:\mcai\MCAI\deploy\azure-windows\04-monitor.ps1 -Watch -BudgetUsd 100 -HourlyRate 0.6849
```

## Watch the dashboard from your own machine

The dashboard is never exposed. Tunnel to it (or just open `http://127.0.0.1:8788` inside RDP):

```bash
ssh -L 8788:127.0.0.1:8788 azureuser@<public-ip>    # needs OpenSSH enabled on the VM first
```

## Known limitations (be honest with yourself about these)

- **Nothing inside Azure restarts an evicted Spot VM.** `-EvictionWatchdog` registers a task on
  *your* machine, so it only fires while you are logged in with a valid `az` token. If your laptop
  is asleep when eviction happens, training is stopped until you notice. There is no email/webhook
  alerting in any of these scripts. Mitigation: check `04-monitor.ps1` daily, or run the watchdog
  somewhere always-on.
- **Spot eviction recovery is slower on Windows** (~2–4 min boot vs ~30–60 s on Linux), so each
  eviction costs more lost time.
- **Windows Server eats ~20–30 GB** of the disk, versus ~8 GB for Linux. Size the disk accordingly;
  `03` rotates logs so a multi-day run cannot fill it.
- **None of this has been run on a real Azure VM yet.** It parses cleanly and was adversarially
  reviewed, but expect a fix round on first boot. The Linux path has the same caveat.

## Cost control

`01-provision.ps1 -Teardown -Yes` deletes the resource group. **Deallocating is not free** — the
disk keeps billing (~$5/mo for 64 GB StandardSSD, ~$19/mo for 128 GB Premium). Pull your
checkpoints off the VM *before* tearing down:

```bash
scp -r azureuser@<public-ip>:C:/mcai/MCAI/checkpoints ./checkpoints-azure
```

Set a real budget alert in the portal (Cost Management → Budgets, scope = `mcai-rg`, $100, alerts at
50/80/100%). The `az consumption budget` CLI surface changes between versions; the portal always works.
