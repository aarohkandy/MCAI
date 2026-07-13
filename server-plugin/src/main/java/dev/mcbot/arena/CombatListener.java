package dev.mcbot.arena;

import org.bukkit.Material;
import org.bukkit.block.Block;
import org.bukkit.entity.EnderCrystal;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.entity.Projectile;
import org.bukkit.event.EventHandler;
import org.bukkit.event.EventPriority;
import org.bukkit.event.Listener;
import org.bukkit.event.block.BlockBreakEvent;
import org.bukkit.event.block.BlockPlaceEvent;
import org.bukkit.event.entity.EntityDamageByEntityEvent;
import org.bukkit.event.entity.EntityDamageEvent;
import org.bukkit.event.entity.EntityExplodeEvent;
import org.bukkit.event.entity.EntityRegainHealthEvent;
import org.bukkit.event.entity.EntityResurrectEvent;
import org.bukkit.event.entity.PlayerDeathEvent;
import org.bukkit.event.player.PlayerInteractEvent;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.projectiles.ProjectileSource;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

public final class CombatListener implements Listener {
    private final MCAIPlugin plugin;
    private final ArenaManager manager;
    private final Map<UUID, UUID> crystalOwners = new HashMap<UUID, UUID>();

    public CombatListener(MCAIPlugin plugin, ArenaManager manager) {
        this.plugin = plugin;
        this.manager = manager;
    }

    @EventHandler(priority = EventPriority.HIGHEST, ignoreCancelled = true)
    public void protectArenas(EntityDamageByEntityEvent event) {
        if (!(event.getEntity() instanceof Player)) return;
        Player victim = (Player) event.getEntity();
        Arena victimArena = manager.arenaFor(victim);
        if (victimArena == null) return;
        Player attacker = attackingPlayer(event.getDamager());
        if (attacker != null && manager.arenaFor(attacker) != victimArena) {
            event.setCancelled(true);
        }
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void recordDamage(EntityDamageEvent event) {
        if (!(event.getEntity() instanceof Player)) return;
        Player victim = (Player) event.getEntity();
        Arena arena = manager.arenaFor(victim);
        if (arena == null) return;
        double damage = Math.max(0, event.getFinalDamage());
        CombatStats victimStats = arena.stats(victim);
        if (victimStats != null) victimStats.damageTaken += damage;
        arena.addReward(victim, -0.02 * damage);
        if (event instanceof EntityDamageByEntityEvent) {
            Player attacker = attackingPlayer(((EntityDamageByEntityEvent) event).getDamager());
            if (attacker != null && !attacker.equals(victim) && manager.arenaFor(attacker) == arena) {
                CombatStats attackerStats = arena.stats(attacker);
                if (attackerStats != null) attackerStats.damageDealt += damage;
                arena.addReward(attacker, 0.02 * damage);
            }
        }
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void recordHealing(EntityRegainHealthEvent event) {
        if (!(event.getEntity() instanceof Player)) return;
        Player player = (Player) event.getEntity();
        Arena arena = manager.arenaFor(player);
        CombatStats stats = arena == null ? null : arena.stats(player);
        if (stats != null) stats.healing += event.getAmount();
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void recordTotem(EntityResurrectEvent event) {
        if (!(event.getEntity() instanceof Player)) return;
        Player player = (Player) event.getEntity();
        Arena arena = manager.arenaFor(player);
        if (arena == null) return;
        CombatStats stats = arena.stats(player);
        if (stats != null) stats.totemPops++;
        arena.addReward(player, -0.10);
        Player opponent = arena.opponent(player);
        if (opponent != null) arena.addReward(opponent, 0.10);
    }

    @EventHandler(priority = EventPriority.HIGHEST, ignoreCancelled = true)
    public void enforceBlockPlace(BlockPlaceEvent event) {
        Arena arena = manager.arenaFor(event.getPlayer());
        if (arena == null) return;
        if (!arena.contains(event.getBlockPlaced().getLocation())) {
            invalid(arena, event.getPlayer());
            event.setCancelled(true);
            return;
        }
        arena.markTouched(event.getBlockPlaced());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void recordBlockPlace(BlockPlaceEvent event) {
        Arena arena = manager.arenaFor(event.getPlayer());
        CombatStats stats = arena == null ? null : arena.stats(event.getPlayer());
        if (stats != null) stats.blocksPlaced++;
    }

    @EventHandler(priority = EventPriority.HIGHEST, ignoreCancelled = true)
    public void enforceBlockBreak(BlockBreakEvent event) {
        Arena arena = manager.arenaFor(event.getPlayer());
        if (arena == null) return;
        if (!arena.contains(event.getBlock().getLocation()) || event.getBlock().getType() == Material.BARRIER) {
            invalid(arena, event.getPlayer());
            event.setCancelled(true);
            return;
        }
        arena.markTouched(event.getBlock());
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void recordBlockBreak(BlockBreakEvent event) {
        Arena arena = manager.arenaFor(event.getPlayer());
        CombatStats stats = arena == null ? null : arena.stats(event.getPlayer());
        if (stats != null) stats.blocksMined++;
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void recordCrystalUse(PlayerInteractEvent event) {
        if (event.getItem() == null || event.getItem().getType() != Material.END_CRYSTAL || event.getClickedBlock() == null) return;
        Player player = event.getPlayer();
        Arena arena = manager.arenaFor(player);
        if (arena == null) return;
        Block clicked = event.getClickedBlock();
        plugin.getServer().getScheduler().runTask(plugin, () -> {
            EnderCrystal closest = null;
            double distance = Double.MAX_VALUE;
            for (Entity entity : clicked.getWorld().getNearbyEntities(clicked.getLocation().add(0.5, 1.5, 0.5), 2, 3, 2)) {
                if (entity instanceof EnderCrystal && entity.getTicksLived() < 10) {
                    double candidate = entity.getLocation().distanceSquared(clicked.getLocation());
                    if (candidate < distance) { closest = (EnderCrystal) entity; distance = candidate; }
                }
            }
            if (closest != null) {
                crystalOwners.put(closest.getUniqueId(), player.getUniqueId());
                CombatStats stats = arena.stats(player);
                if (stats != null) stats.crystalsPlaced++;
            }
        });
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void recordCrystalBreak(EntityDamageByEntityEvent event) {
        if (!(event.getEntity() instanceof EnderCrystal)) return;
        UUID crystalId = event.getEntity().getUniqueId();
        plugin.getServer().getScheduler().runTaskLater(plugin, () -> crystalOwners.remove(crystalId), 20L);
        Player attacker = attackingPlayer(event.getDamager());
        if (attacker == null) return;
        crystalOwners.put(crystalId, attacker.getUniqueId());
        Arena arena = manager.arenaFor(attacker);
        CombatStats stats = arena == null ? null : arena.stats(attacker);
        if (stats != null) stats.crystalsDestroyed++;
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void recordExplosionBlocks(EntityExplodeEvent event) {
        if (event.getEntity() instanceof EnderCrystal) {
            UUID crystalId = event.getEntity().getUniqueId();
            plugin.getServer().getScheduler().runTaskLater(plugin, () -> crystalOwners.remove(crystalId), 20L);
        }
        Arena arena = null;
        for (Arena candidate : manager.activeArenas()) {
            if (candidate.contains(event.getLocation())) { arena = candidate; break; }
        }
        if (arena == null) return;
        for (Block block : event.blockList()) arena.markTouched(block);
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onDeath(PlayerDeathEvent event) {
        Arena arena = manager.arenaFor(event.getEntity());
        if (arena == null) return;
        event.setKeepInventory(true);
        event.getDrops().clear();
        event.setDroppedExp(0);
        arena.finish(arena.opponent(event.getEntity()), false, "death");
    }

    @EventHandler
    public void onQuit(PlayerQuitEvent event) {
        Arena arena = manager.arenaFor(event.getPlayer());
        if (arena != null) arena.finish(arena.opponent(event.getPlayer()), false, "disconnect");
    }

    private Player attackingPlayer(Entity damager) {
        if (damager instanceof Player) return (Player) damager;
        if (damager instanceof Projectile) {
            ProjectileSource shooter = ((Projectile) damager).getShooter();
            return shooter instanceof Player ? (Player) shooter : null;
        }
        if (damager instanceof EnderCrystal) {
            UUID owner = crystalOwners.get(damager.getUniqueId());
            return owner == null ? null : plugin.getServer().getPlayer(owner);
        }
        return null;
    }

    private void invalid(Arena arena, Player player) {
        CombatStats stats = arena.stats(player);
        if (stats != null) stats.invalidInteractions++;
        arena.addReward(player, -0.001);
    }
}
