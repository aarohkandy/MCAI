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
- An **SSH key**. You already have a usable pair at `~/.ssh/winver_ed25519(.pub)` — reuse it, or
  make a dedicated one: `ssh-keygen -t ed25519 -f ~/.ssh/azure_mcai -C mcai`. You keep the private
  key; I never handle it. (Using Cloud Shell? It can generate the key for you at VM-create time.)
- A one-time **quota bump** if you pick 16+ vCPUs (see step 2).
- Your explicit **Minecraft EULA acceptance** — you set `MCAI_ACCEPT_EULA=true` (step 4). I won't
  accept it for you.
- **The fixed code on the VM.** The bug fixes are currently *local, uncommitted* changes. To get
  them onto the VM you either (a) commit + push them to your GitHub and `git clone` on the VM, or
  (b) `scp`/`rsync` this folder up. Ask me to prepare the commit — I can commit locally on a branch;
  pushing needs your GitHub auth.

## 2. Pick a size — and strongly consider Spot
Pay-as-you-go vs **Spot** (Spot is what your R_SIM runbook uses; 70–90% cheaper):

| SKU | vCPU / RAM | ~$/hr PAYG | ~$/hr Spot | hours in $500 (Spot) |
|-----|-----------|-----------|-----------|----------------------|
| `Standard_F8s_v2`  | 8 / 16 GB  | ~$0.34 | ~$0.07–0.15 | ~3,300–7,000 |
| `Standard_F16s_v2` | 16 / 32 GB | ~$0.68 | ~$0.15–0.30 | ~1,600–3,300 |
| `Standard_F32s_v2` | 32 / 64 GB | ~$1.35 | ~$0.30–0.60 | ~830–1,600 |

**Recommendation for your $500 + "make it perfect" goal: `Standard_F32s_v2` on Spot.** It doubles
experience throughput vs F16 and *still* buys far more training hours than F16 at PAYG.

**The Spot tradeoff, honestly:** Azure can **evict** the VM at any time when it needs capacity.
R_SIM tolerates this because its runs are 10–30 minutes. MCAI training runs for days, so you *will*
get evicted sometimes. Why it's still the right call here:
- Checkpoints are written atomically to the data volume and training resumes from `latest.pt`.
- With `--eviction-policy Deallocate` the disk (and all checkpoints) survives; you just `az vm start`.
- Worst case you lose the current partial rollout batch (minutes), not your training run.

If you want zero babysitting, use regular (PAYG) pricing instead and drop `--priority Spot ...`.

Default subscriptions often cap the **Fsv2 family at 10 vCPUs** per region. For F16s_v2/F32s_v2,
request an increase first:
```bash
az vm list-usage --location eastus2 -o table | grep -i fsv2   # see current limit
# Portal: Subscriptions → Usage + quotas → search "Fsv2" → request increase (usually auto-approved)
```

## 3. Provision the VM
```bash
# az CLI: see the install options in section 1 (Cloud Shell needs no install)
az login                      # or: az login --service-principal -u <client-id> -t <tenant> -p <secret-from-keychain>

SUBSCRIPTION=8ffcf316-2929-4aaa-9374-3811655b650a
RG=mcai-rg                    # NOT rsim-compute — see the teardown warning above
LOCATION=eastus2
VM=mcai-train
SIZE=Standard_F32s_v2

az account set --subscription "$SUBSCRIPTION"
az group create -n "$RG" -l "$LOCATION"      # needs your account, not the RG-scoped SP

# Budget + alert BEFORE the VM (mirrors the R_SIM guardrail).
az consumption budget create-with-rg \
  --budget-name mcai-cap --resource-group "$RG" \
  --category Cost --amount 500 --time-grain Monthly \
  --start-date "$(date +%Y-%m-01)" --end-date "2030-01-01" \
  || echo "Set a \$500 budget + alert in the portal (Cost Management → Budgets) before continuing."

az vm create -g "$RG" -n "$VM" \
  --image Ubuntu2204 --size "$SIZE" \
  --priority Spot --max-price -1 --eviction-policy Deallocate \
  --admin-username azureuser \
  --ssh-key-values ~/.ssh/winver_ed25519.pub \
  --os-disk-size-gb 128 --storage-sku Premium_LRS \
  --public-ip-sku Standard

# Lock the network down to SSH only (nothing else is exposed; the dashboard is reached by tunnel).
az vm open-port -g "$RG" -n "$VM" --port 22 --priority 100
```
Drop the `--priority Spot --max-price -1 --eviction-policy Deallocate` line for a regular VM.

### If Spot evicts the VM
```bash
az vm start -g mcai-rg -n mcai-train     # disk + checkpoints intact; compose auto-restarts training
```

## 4. Deploy on the VM
```bash
ssh -i ~/.ssh/winver_ed25519 azureuser@<public-ip>

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

## 5. Watch the dashboard (SSH tunnel — never exposed publicly)
From your machine:
```bash
ssh -i ~/.ssh/winver_ed25519 -L 8788:localhost:8788 azureuser@<public-ip>
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

## 7. Cost control (do this)
```bash
az vm deallocate -g mcai-rg -n mcai-train   # stop billing compute (keeps disk + checkpoints)
az vm start      -g mcai-rg -n mcai-train   # resume later
```

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
