# Deployment and safety

## Network boundary

The training server uses offline identities, so treat LAN isolation as mandatory:

- Bind the Paper/Eagler listener to `127.0.0.1` until LAN access is needed.
- When using a private LAN address, firewall TCP 25565 to the local subnet and preferably the invited devices only.
- Keep the Paper whitelist enabled. The setup script generates deterministic offline UUIDs for bot accounts and `AIWatcher`.
- The arena control socket always binds loopback only.
- Keep the arena control socket (8765), trainer socket (8766), and browser debug bridge (8767) bound to loopback on the Surface.
- Do not port-forward any of these services.

When you change `MCAI_BIND_ADDRESS` from loopback, open an Administrator PowerShell and add only a private-subnet rule:

```powershell
New-NetFirewallRule -DisplayName "MCAI Eagler LAN" -Direction Inbound -Action Allow `
  -Protocol TCP -LocalPort 25565 -Profile Private -RemoteAddress LocalSubnet
```

Inspect existing Java firewall rules too; remove any broad Public-profile allow rule that Windows may have created. The setup refuses to write `eula=true` until `MCAI_ACCEPT_EULA=true` is explicitly set in the local config.

## Human matches

Promote only a frozen checkpoint whose hash, model manifest, policy version, evaluation seeds, and results are recorded. Label the account as AI and tell volunteers that it has client-known structured state, off-camera tracking, and unrestricted legal timing.

Use invited players only on the server you own. Never deploy it to public third-party servers, conceal that it is a bot, or use it to probe/bypass anti-cheat.

## Emergency controls

F8 in the Eagler client releases every binding, resets the local GRU, informs the helper, and closes its socket. `.mcai off` does the same. `/aistop` on Paper terminates all matches, broadcasts `emergency_stop`, and disconnects accounts with the configured bot prefix.

## Packet audit

The MCAI adapter itself has one raw protocol use: vanilla status `6` on `block_dig`, the ordinary swap-items-in-hands action. Movement, attack, item activation, and digging use Mineflayer control APIs. Crosshair-derived obsidian/crystal interactions use Mineflayer's generic placement path with automatic looking disabled; the policy must already point at the server-reachable block. Audit logs identify each allowed operation. Any future raw route must be added to the allowlist and reviewed for reach, inventory possibility, and server-observed equivalence before use.
