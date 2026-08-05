# MCAI on Azure — headless self-play training

This runs the whole MCAI stack (trainer + Paper server + rollout worker + dashboard) in **one
Docker container on a single Azure VM**, on loopback, with nothing exposed to the internet. It is
built for the iterate-often loop: `git pull` on the VM → `docker compose restart` → training
resumes from the latest checkpoint with your new code.

## Why CPU, not GPU
The policy is ~296k parameters and the real bottleneck is the Minecraft simulation that generates
experience, which is CPU-bound. A GPU would sit mostly idle. So this targets a **high-core CPU VM**;
spend your budget on vCPUs (more cores → more concurrent arenas → more experience/second).

## Status — end-to-end verified locally (2026-07-29)
The full loop was run on a Mac before any cloud spend, and every stage is confirmed working:
bots connect to Paper and get paired → combat produces shaped **and** terminal rewards
(−1.0…+1.0, zero-sum) → the trainer collects rollouts → PPO updates complete and write atomic
checkpoints. Behavior cloning from recorded demonstrations was also verified end-to-end.

Fixed during that run (all would have cost cloud time to discover):
- PaperMC's v2 API now returns **410 Gone** for 1.12.2; the download uses the v3 (`fill`) API with
  sha256 verification.
- Passive-mob pathfinding in the unused default overworld force-loaded chunks until Paper's 60s
  watchdog **killed the server**. `configure_runtime.py` now disables all mob spawning, uses a FLAT
  default world, and drops `bukkit.yml` spawn limits to zero. (Server boot also went 11.4s → 4.9s.)

## Bootstrap the policy before spending on RL (do this first — it's free)
Self-play from a random policy barely fights, so the reward signal is almost nil and PPO crawls.
The scripted teacher already fights competently, so record it and behavior-clone from it:

```bash
# 1. capture demonstrations from the scripted teacher (no trainer needed)
MCAI_DEMO_OUT=runs/demos.jsonl MCAI_BOT_COUNT=2 node worker/dist/src/index.js
# 2. behavior-clone a starting policy
python -m combat_ai.cli clone runs/demos.jsonl --output checkpoints/imitation.pt
# 3. start RL from it instead of from random
python -m combat_ai.cli serve --initialize-from checkpoints/imitation.pt ...
```
Set `MCAI_FORCE_SCRIPTED=true` to keep recording the teacher even while a trainer is connected.
When you later record *your own* play with the eagler mod's `.mcai record`, it emits the same
JSONL format — just point `--imitation-data` at it.

---

## 1. Your existing Azure access (found on this Mac, 2026-07-28)

Discovered in `~/Downloads/r_sim/R_SIM/inference/azure.config.yaml` (+ macOS Keychain):

| item | value |
|---|---|
| Subscription | `8ffcf316-2929-4aaa-9374-3811655b650a` |
| Tenant | `b75a3a6e-d3be-4236-ba54-692677a38bda` |
| Service principal (client id) | `da9dda36-8b8f-4a28-9ceb-211ed3a0fcb3` |
| SP secret | macOS Keychain — service `rsim`, account `azure_sp_secret` (present; never printed) |
| SP scope | **Contributor on the `rsim-compute` resource group ONLY** |
| Region | `eastus2` |

Also present: a Foundry inference key (Keychain `rsim`/`azure_key`) — that one is for *LLM calls*
and **cannot** provision VMs. Ignore it here.

> ⚠️ **The Azure CLI is not installed on this Mac — and neither is Homebrew**, so `brew install
> azure-cli` will not work as-is. Pick one:
> - **Zero install (easiest):** run every `az` command below in **Azure Cloud Shell** —
>   <https://portal.azure.com> → the `>_` icon. The CLI is preinstalled and already signed in as you.
> - **pip:** `python3 -m pip install --user azure-cli` (then ensure `~/Library/Python/3.9/bin` is on `$PATH`).
> - **Homebrew:** install it first from <https://brew.sh>, then `brew install azure-cli`.

### ⚠️ Two blockers you must decide on before provisioning

**(a) The service principal cannot create a new resource group.** It is Contributor on
`rsim-compute` only; creating an RG needs subscription-level rights. So either:
- **Recommended —** you create a dedicated RG and grant the SP rights on it (clean separation):
  ```bash
  az login                                     # your own account, once
  az group create -n mcai-rg -l eastus2
  az role assignment create --assignee da9dda36-8b8f-4a28-9ceb-211ed3a0fcb3 \
      --role Contributor \
      --scope /subscriptions/8ffcf316-2929-4aaa-9374-3811655b650a/resourceGroups/mcai-rg
  ```
- or you simply run the provisioning commands yourself under `az login` (no SP needed).

**(b) Do NOT put MCAI inside `rsim-compute`.** R_SIM's own runbook ends with
`az group delete -n rsim-compute --yes` as its mandatory teardown step — that would **destroy your
MCAI VM, disk, and every checkpoint** along with it. Keep MCAI in its own resource group.

### What I still need from you
- **A decision on (a)/(b) above**, and confirmation to proceed (provisioning spends real money —
  I won't do it off the back of a document).
- **Nothing for SSH if you use Cloud Shell** — the commands use `--generate-ssh-keys`, which creates
  or reuses a key inside Cloud Shell. (Your Mac key `~/.ssh/winver_ed25519` is *not* visible to Cloud
  Shell, so it cannot be referenced there. Running the CLI locally instead? Swap in
  `--ssh-key-values ~/.ssh/winver_ed25519.pub` and add `-i ~/.ssh/winver_ed25519` when you ssh.)
- A one-time **quota bump** if you pick 16+ vCPUs (see step 2).
- Your explicit **Minecraft EULA acceptance** — you set `MCAI_ACCEPT_EULA=true` (step 4). I won't
  accept it for you.
- Nothing for the code — it is committed and pushed to the `fix/verified-training-stack` branch, so
  the VM just clones it.

## 2. Pick a size — and strongly consider Spot
Pay-as-you-go vs **Spot** (Spot is what your R_SIM runbook uses; 70–90% cheaper):

| SKU | vCPU / RAM | ~$/hr PAYG | ~$/hr Spot | hours in $100 (Spot) |
|-----|-----------|-----------|-----------|----------------------|
| `Standard_F8s_v2`  | 8 / 16 GB  | $0.338 | **$0.178** | ~535 |
| `Standard_F16s_v2` | 16 / 32 GB | $0.677 | **$0.356** | ~267 |
| `Standard_F32s_v2` | 32 / 64 GB | $1.353 | **$0.712** | ~133 |

Rates verified from the Azure retail pricing API (eastus2, 2026-08-04); hours assume $100 minus a
64 GB StandardSSD (~$5/mo). Spot still floats with demand — re-check before committing:
```bash
az vm list-skus -l eastus2 --size Standard_F --query "[?name=='Standard_F16s_v2'].name" -o tsv
# Live Spot price (no auth needed):
curl -s "https://prices.azure.com/api/retail/prices?\$filter=armRegionName%20eq%20'eastus2'%20and%20skuName%20eq%20'F16s%20v2%20Spot'" | head -c 400
```

**Recommendation for a $100 budget: `Standard_F16s_v2` on Spot, and cap it with `--max-price`.**
16 vCPU comfortably runs ~4 arena pairs (8 bots). F32 is tempting but at $100 the extra throughput
mostly buys you a shorter calendar window, and eviction risk rises with the larger SKU.

> **$100 reality:** F16s_v2 Spot at the verified $0.356/hr is about **267 hours ≈ 11 days** of
> continuous training. Plan for ~1.5 weeks, not a month.

**The Spot tradeoff, honestly:** Azure can **evict** the VM at any time when it needs capacity.
R_SIM tolerates this because its runs are 10–30 minutes. MCAI training runs for days, so you *will*
get evicted sometimes. Why it's still the right call here:
- Checkpoints are written atomically to the data volume and training resumes from `latest.pt`.
- With `--eviction-policy Deallocate` the disk (and all checkpoints) survives; you just `az vm start`.
- Worst case you lose the current partial rollout batch (minutes), not your training run.

If you want zero babysitting, use regular (PAYG) pricing instead and drop `--priority Spot ...`.

Default subscriptions cap vCPUs per family. **Spot draws on a SEPARATE pool** — checking only the
Fsv2 quota will mislead you. Check both:
```bash
az vm list-usage --location eastus2 -o table | grep -iE "fsv2|standardfsv2"        # regular
az vm list-usage --location eastus2 -o table | grep -i "spot"                      # Spot pool
# Portal: Subscriptions -> Usage + quotas -> request an increase for BOTH
#   "Standard FSv2 Family vCPUs" and "Total Regional Spot vCPUs" (usually auto-approved)
```

## 3. Provision the VM
```bash
# az CLI: see the install options in section 1 (Cloud Shell needs no install)
az login                      # or: az login --service-principal -u <client-id> -t <tenant> -p <secret-from-keychain>

SUBSCRIPTION=8ffcf316-2929-4aaa-9374-3811655b650a
RG=mcai-rg                    # NOT rsim-compute — see the teardown warning above
LOCATION=eastus2
VM=mcai-train
SIZE=Standard_F16s_v2

az account set --subscription "$SUBSCRIPTION"
az group create -n "$RG" -l "$LOCATION"      # needs your account, not the RG-scoped SP

# BUDGET FIRST. The `az consumption budget` CLI surface has changed repeatedly and its flags differ
# by CLI version, so set this in the PORTAL where it always works:
#   Cost Management -> Budgets -> Add -> scope = resource group mcai-rg
#   amount 100, monthly, alerts at 50% / 80% / 100% to your email.
# Do not skip it: it is the only thing that will tell you if Spot pricing spikes.

az vm create -g "$RG" -n "$VM" \
  --image Ubuntu2204 --size "$SIZE" \
  --priority Spot --max-price 0.30 --eviction-policy Deallocate \
  --admin-username azureuser \
  --generate-ssh-keys \
  --os-disk-size-gb 128 --storage-sku Premium_LRS \
  --public-ip-sku Standard

# NOTE: `az vm create` already provisions an NSG that allows inbound SSH (22) and nothing else, so
# no `az vm open-port` call is needed. Adding one would only widen the rule. The dashboard is never
# exposed; you reach it through the SSH tunnel in section 5.
```
Drop the `--priority Spot --max-price ... --eviction-policy Deallocate` line for a regular VM.
`--generate-ssh-keys` makes Cloud Shell create/reuse a key in *its own* `~/.ssh` — the key on your
Mac is not readable from Cloud Shell, which is why the local path is not referenced here.

### If Spot evicts the VM
```bash
az vm start -g mcai-rg -n mcai-train     # disk + checkpoints intact; compose auto-restarts training
```

## 4. Deploy on the VM
```bash
ssh azureuser@<public-ip>   # Cloud Shell: key already in place

# Docker + compose plugin
curl -fsSL https://get.docker.com | sudo sh
sudo usermod -aG docker $USER && newgrp docker

# Get the FIXED code (option a: your GitHub, option b: you scp'd it here)
git clone https://github.com/<you>/MCAI.git && cd MCAI     # must include the fixes

# Build + launch (first build downloads the toolchain + PyTorch; a few minutes)
cd deploy/azure
MCAI_ACCEPT_EULA=true MCAI_BOT_COUNT=8 MCAI_MODE=sword docker compose up -d --build

docker compose logs -f     # watch it come up; look for "MCAI is running"
```
Tune `MCAI_BOT_COUNT` to the box (~8 on 16 vCPU, ~16 on 32 vCPU) and start in `sword` mode — only
move to `crystal`/`combined` after sword is competent.

## 4b. Survive Spot eviction without your laptop (do this)
Azure does not restart an evicted Spot VM, and a watchdog on your own machine is useless when it is
closed. Install the Azure-side one instead — it runs in Azure, free tier, scoped to this one VM:
```bash
./deploy/azure/eviction-autorestart.sh --resource-group mcai-rg --vm mcai-train
```
Worst case you lose one check interval (default 15 min) per eviction; training resumes from
`checkpoints/latest.pt`.

## 4c. Watch the fights in a real Minecraft client
The container now publishes Minecraft on the host loopback and installs **EaglerXServer**, so the
bots train on the same server a browser client connects to. From your own machine:
```bash
./scripts/watch-fights.sh --host <public-ip>
```
Then connect a 1.12.2 client to `127.0.0.1:25565` as **AIWatcher** and run `/aiwatch arena-1 orbit`
(or `pov`), `/ainext` to cycle arenas. Rendering happens on YOUR machine, which is precisely why a
GPU-less VM is not a limitation. See [docs/SPECTATING.md](../../docs/SPECTATING.md).

## 4d. Check it is actually learning
```bash
docker compose exec mcai python -m combat_ai.cli introspect /data/checkpoints/latest.pt
```
Reports tanh saturation (gradient-dead units), GRU memory use, which observation fields drive each
decision, per-head entropy vs maximum, and behavioural probes. Run it on day one for a baseline —
it works on an untrained policy — then watch the probe signals turn positive.

## 5. Watch the dashboard (SSH tunnel — never exposed publicly)
From your machine:
```bash
ssh -L 8788:localhost:8788 azureuser@<public-ip>
# then open http://localhost:8788
```

## 6. The iterate loop
```bash
# on the VM, in the repo:
git pull
cd deploy/azure && docker compose restart      # rebuilds changed code, resumes from checkpoint
```
- **Reward / curriculum / hyperparameter changes** resume cleanly from `checkpoints/latest.pt`.
- **Model-architecture or feature-encoding changes invalidate checkpoints** — the trainer will fail
  loudly on load. Start fresh with:
  ```bash
  docker compose exec mcai rm -f /data/checkpoints/latest.pt
  docker compose restart
  ```

## 7. Cost control (this is what protects your $100)
```bash
az vm deallocate -g mcai-rg -n mcai-train   # stops COMPUTE billing (disk keeps billing, see below)
az vm start      -g mcai-rg -n mcai-train   # resume; training resumes from latest.pt
```
**Deallocate does not make it free.** The 128 GB Premium SSD keeps billing at roughly **$18–20/month**
even while the VM is off — about a fifth of your whole budget if you leave it parked for a month.

**When you are done, tear it down (this is the only way to reach $0):**
```bash
# 1. pull the trained model off the VM FIRST — deleting the group destroys the disk
docker compose -f deploy/azure/docker-compose.yml cp mcai:/data/checkpoints ./checkpoints-azure  # on the VM
scp -r azureuser@<public-ip>:~/MCAI/checkpoints-azure ./                                          # to your Mac
# 2. destroy everything in one shot
az group delete -n mcai-rg --yes --no-wait
az group show  -n mcai-rg 2>/dev/null && echo "still deleting..." || echo "gone - \$0 ongoing"
```
Cheaper alternative if you want to pause for a long time: snapshot the disk, delete the VM, and
restore later — a snapshot costs far less than a live Premium SSD.

## Data & persistence
Checkpoints, run logs, the Paper runtime, and the Maven cache live in the `mcai-data` Docker
volume on the VM's managed disk — they survive `restart`, `up --build`, and VM deallocate/start.
Deleting the VM's disk deletes them; snapshot the disk if you want backups.

## Notes
- No Minecraft/Eaglercraft binaries are baked into the image or committed; Paper is fetched at
  runtime from the official PaperMC API into the volume.
- This is **headless only**. Exposing the dashboard or a browser (Eaglercraft) client to the
  internet is deliberately out of scope here and would need firewall rules + your own legitimate
  Eaglercraft build.
