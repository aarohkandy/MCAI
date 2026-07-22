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
import org.bukkit.event.player.PlayerAnimationEvent;
import org.bukkit.event.player.PlayerAnimationType;
import org.bukkit.event.player.PlayerQuitEvent;
import org.bukkit.projectiles.ProjectileSource;
import org.bukkit.event.block.Action;

import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

public final class CombatListener implements Listener {
    private final MCAIPlugin plugin;
    private final ArenaManager manager;
    private final Map<UUID, CrystalAttribution> crystalAttributions =
            new HashMap<UUID, CrystalAttribution>();
    private final Map<String, BlockAttribution> blockInteractions =
            new HashMap<String, BlockAttribution>();

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
        if (event.getDamager() instanceof EnderCrystal) {
            CrystalAttribution attribution = crystalAttributions.get(
                    event.getDamager().getUniqueId());
            if (attribution != null
                    && !victimArena.getEpisodeId().equals(attribution.episodeId)) {
                event.setCancelled(true);
                return;
            }
        }
        Player attacker = attackingPlayer(event.getDamager(), victimArena);
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
        double damage = rewardableDamage(event.getFinalDamage(), victim.getHealth(), absorption(victim));
        RewardShaper shaper = arena.rewardShaper();
        CombatStats victimStats = arena.stats(victim);
        if (victimStats != null) victimStats.damageTaken += damage;
        ExecutionSource attributedCrystalSource = event instanceof EntityDamageByEntityEvent
                ? attributedCrystalSource(arena,
                ((EntityDamageByEntityEvent) event).getDamager()) : null;
        // This check happens before opponent validation because a crystal also
        // damages its detonator. Teacher/safety self-damage stays visible in
        // Minecraft and stats but must never become a delayed PPO penalty.
        boolean rewardEligible = damageRewardEligible(attributedCrystalSource);
        if (event instanceof EntityDamageByEntityEvent) {
            EntityDamageByEntityEvent damageByEntity = (EntityDamageByEntityEvent) event;
            Player attacker = attackingPlayer(damageByEntity.getDamager(), arena);
            if (attacker != null && manager.arenaFor(attacker) == arena) {
                boolean crystalDamage = damageByEntity.getDamager() instanceof EnderCrystal;
                ExecutionSource source = executionSourceForDamage(
                        arena, attacker, damageByEntity.getDamager());
                rewardEligible = damageRewardEligible(source);
                UUID crystalId = crystalDamage ? damageByEntity.getDamager().getUniqueId() : null;
                if (victim.equals(arena.opponent(attacker))) {
                    arena.recordOpponentDamage(attacker, victim, damage, source, crystalDamage, crystalId);
                    if (rewardEligible) {
                        arena.addReward(attacker, shaper.damageDealt(damage), RewardReason.DAMAGE_DEALT);
                    }
                    if (damage > 0 && damageByEntity.getDamager() instanceof Player) {
                        arena.recordSuccessfulHit(attacker, victim, source);
                    }
                } else if (victim.equals(attacker)) {
                    arena.recordSelfDamage(victim, damage, source, crystalDamage, crystalId);
                }
            }
        }
        if (rewardEligible) arena.addReward(victim, shaper.damageTaken(damage), RewardReason.DAMAGE_TAKEN);
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void recordAttackSwing(PlayerAnimationEvent event) {
        if (event.getAnimationType() != PlayerAnimationType.ARM_SWING) return;
        Arena arena = manager.arenaFor(event.getPlayer());
        if (arena != null && arena.getMode().usesGenericArmSwingShaping()) {
            arena.recordAttackSwing(event.getPlayer(), arena.executionSource(event.getPlayer()));
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
        boolean forcedByOpponent = arena.lastDamageWasAssignedOpponent(player);
        ExecutionSource source = arena.recordTotemAttribution(player);
        boolean rewardEligible = source == null || source.isAutonomous();
        RewardShaper shaper = arena.rewardShaper();
        if (rewardEligible) arena.addReward(player, shaper.ownTotem(), RewardReason.TOTEM);
        Player opponent = arena.opponent(player);
        if (opponent != null && forcedByOpponent) {
            CombatStats opponentStats = arena.stats(opponent);
            if (opponentStats != null) opponentStats.totemPopsForced++;
            if (rewardEligible) arena.addReward(opponent, shaper.forcedTotem(), RewardReason.TOTEM);
        }
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
        if (arena != null) arena.recordBlockPlacement(event.getPlayer(), event.getBlockPlaced(),
                arena.executionSource(event.getPlayer()));
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
        if (arena == null) return;
        ExecutionSource source = arena.executionSource(event.getPlayer());
        BlockAttribution attribution = blockInteractions.remove(blockKey(event.getPlayer(), event.getBlock()));
        if (attribution != null && arena.getEpisodeId().equals(attribution.episodeId)
                && manager.currentTick() - attribution.tick <= 200L) {
            source = attribution.source;
        }
        arena.recordBlockMined(event.getPlayer(), event.getBlock(), event.getBlock().getType(), source);
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void recordBlockIntent(PlayerInteractEvent event) {
        if (event.getAction() != Action.LEFT_CLICK_BLOCK || event.getClickedBlock() == null) return;
        Arena arena = manager.arenaFor(event.getPlayer());
        if (arena == null) return;
        String key = blockKey(event.getPlayer(), event.getClickedBlock());
        ExecutionSource source = arena.executionSource(event.getPlayer());
        if (source == ExecutionSource.POLICY) {
            blockInteractions.remove(key);
            return;
        }
        blockInteractions.put(key, new BlockAttribution(
                arena.getEpisodeId(), source, manager.currentTick()));
        if (blockInteractions.size() > 512) {
            final long oldestAllowedTick = manager.currentTick() - 200L;
            blockInteractions.entrySet().removeIf(entry -> entry.getValue().tick < oldestAllowedTick);
        }
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void recordCrystalUse(PlayerInteractEvent event) {
        if (event.getItem() == null || event.getItem().getType() != Material.END_CRYSTAL || event.getClickedBlock() == null) return;
        Player player = event.getPlayer();
        Arena arena = manager.arenaFor(player);
        if (arena == null) return;
        Block clicked = event.getClickedBlock();
        boolean onObsidian = clicked.getType() == Material.OBSIDIAN;
        String crystalBaseKey = crystalBaseKey(clicked);
        ExecutionSource source = arena.executionSource(player);
        String expectedEpisodeId = arena.getEpisodeId();
        long placementTick = manager.currentTick();
        plugin.getServer().getScheduler().runTask(plugin, () -> {
            EnderCrystal closest = null;
            double distance = Double.MAX_VALUE;
            org.bukkit.Location expected = clicked.getLocation().add(0.5, 1.0, 0.5);
            for (Entity entity : clicked.getWorld().getNearbyEntities(clicked.getLocation().add(0.5, 1.5, 0.5), 2, 3, 2)) {
                if (entity instanceof EnderCrystal && entity.getTicksLived() < 10
                        && !crystalAttributions.containsKey(entity.getUniqueId())) {
                    double candidate = entity.getLocation().distanceSquared(expected);
                    if (candidate < distance) { closest = (EnderCrystal) entity; distance = candidate; }
                }
            }
            if (closest != null && arena.isActive() && arena.contains(closest.getLocation())
                    && expectedEpisodeId.equals(arena.getEpisodeId()) && distance <= 2.25
                    && !crystalAttributions.containsKey(closest.getUniqueId())) {
                UUID crystalId = closest.getUniqueId();
                crystalAttributions.put(crystalId, new CrystalAttribution(
                        expectedEpisodeId, player.getUniqueId(), source, placementTick,
                        crystalBaseKey));
                arena.recordCrystalPlacement(player, crystalId, onObsidian, crystalBaseKey, source);
                pruneCrystalAttributions();
            }
        });
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void recordCrystalBreak(EntityDamageByEntityEvent event) {
        if (!(event.getEntity() instanceof EnderCrystal)) return;
        UUID crystalId = event.getEntity().getUniqueId();
        plugin.getServer().getScheduler().runTaskLater(plugin, () -> crystalAttributions.remove(crystalId), 20L);
        Player attacker = attackingPlayer(event.getDamager());
        if (attacker == null) return;
        Arena arena = manager.arenaFor(attacker);
        if (arena != null) {
            ExecutionSource source = arena.executionSource(attacker);
            CrystalAttribution attribution = crystalAttributions.get(crystalId);
            if (attribution == null || !arena.getEpisodeId().equals(attribution.episodeId)) {
                attribution = new CrystalAttribution(arena.getEpisodeId(), null, null, -1L,
                        crystalBaseKey(event.getEntity()));
                crystalAttributions.put(crystalId, attribution);
            }
            attribution.recordDetonation(attacker.getUniqueId(), source, manager.currentTick());
            arena.recordCrystalDestruction(attacker, crystalId, attribution.baseKey, source);
        }
    }

    @EventHandler(priority = EventPriority.MONITOR, ignoreCancelled = true)
    public void recordExplosionBlocks(EntityExplodeEvent event) {
        UUID crystalId = null;
        UUID crystalDetonator = null;
        if (event.getEntity() instanceof EnderCrystal) {
            final UUID explodingCrystalId = event.getEntity().getUniqueId();
            crystalId = explodingCrystalId;
            CrystalAttribution attribution = crystalAttributions.get(crystalId);
            crystalDetonator = attribution == null ? null : attribution.detonatorId;
            plugin.getServer().getScheduler().runTaskLater(plugin,
                    () -> crystalAttributions.remove(explodingCrystalId), 20L);
        }
        Arena arena = null;
        for (Arena candidate : manager.activeArenas()) {
            if (candidate.contains(event.getLocation())) { arena = candidate; break; }
        }
        if (arena == null) return;
        if (crystalId != null && crystalDetonator != null) {
            Player detonator = plugin.getServer().getPlayer(crystalDetonator);
            CrystalAttribution attribution = crystalAttributions.get(crystalId);
            if (detonator != null && attribution != null
                    && arena.getEpisodeId().equals(attribution.episodeId)
                    && manager.arenaFor(detonator) == arena) {
                arena.recordCrystalExplosion(detonator, crystalId, attribution.baseKey,
                        attribution.detonationSource);
            }
        }
        for (Block block : event.blockList()) {
            arena.forgetPolicyObsidianBase(block);
            arena.markTouched(block);
        }
    }

    @EventHandler(priority = EventPriority.MONITOR)
    public void onDeath(PlayerDeathEvent event) {
        Arena arena = manager.arenaFor(event.getEntity());
        if (arena == null) return;
        event.setKeepInventory(true);
        event.getDrops().clear();
        event.setDroppedExp(0);
        arena.observeDeath(event.getEntity());
    }

    @EventHandler
    public void onQuit(PlayerQuitEvent event) {
        Arena arena = manager.arenaFor(event.getPlayer());
        if (arena != null) arena.finish(arena.opponent(event.getPlayer()), false, "disconnect");
    }

    private Player attackingPlayer(Entity damager) {
        return attackingPlayer(damager, null);
    }

    private Player attackingPlayer(Entity damager, Arena expectedArena) {
        if (damager instanceof Player) return (Player) damager;
        if (damager instanceof Projectile) {
            ProjectileSource shooter = ((Projectile) damager).getShooter();
            return shooter instanceof Player ? (Player) shooter : null;
        }
        if (damager instanceof EnderCrystal) {
            CrystalAttribution attribution = crystalAttributions.get(damager.getUniqueId());
            return attribution == null || attribution.detonatorId == null
                    || (expectedArena != null
                    && !expectedArena.getEpisodeId().equals(attribution.episodeId))
                    ? null : plugin.getServer().getPlayer(attribution.detonatorId);
        }
        return null;
    }

    private ExecutionSource executionSourceForDamage(Arena arena, Player attacker, Entity damager) {
        ExecutionSource crystalSource = attributedCrystalSource(arena, damager);
        if (crystalSource != null) return crystalSource;
        return arena.executionSource(attacker);
    }

    private ExecutionSource attributedCrystalSource(Arena arena, Entity damager) {
        if (!(damager instanceof EnderCrystal)) return null;
        CrystalAttribution attribution = crystalAttributions.get(damager.getUniqueId());
        return attribution != null && arena.getEpisodeId().equals(attribution.episodeId)
                ? attribution.detonationSource : null;
    }

    static boolean damageRewardEligible(ExecutionSource attributedSource) {
        return attributedSource == null || attributedSource.isAutonomous();
    }

    static double rewardableDamage(double finalDamage, double health, double absorption) {
        if (!Double.isFinite(finalDamage) || !Double.isFinite(health)
                || !Double.isFinite(absorption)) return 0.0;
        double removableHealth = Math.max(0.0, health) + Math.max(0.0, absorption);
        return Math.min(Math.max(0.0, finalDamage), removableHealth);
    }

    private static double absorption(Player player) {
        try {
            Object value = player.getClass().getMethod("getAbsorptionAmount").invoke(player);
            return value instanceof Number ? ((Number) value).doubleValue() : 0.0;
        } catch (ReflectiveOperationException ignored) {
            return 0.0;
        }
    }

    private static String blockKey(Player player, Block block) {
        return player.getUniqueId() + ":" + block.getWorld().getName() + ":"
                + block.getX() + ":" + block.getY() + ":" + block.getZ();
    }

    private static String crystalBaseKey(Block block) {
        return block.getWorld().getName() + ':' + block.getX() + ':'
                + block.getY() + ':' + block.getZ();
    }

    private static String crystalBaseKey(Entity crystal) {
        org.bukkit.Location location = crystal.getLocation();
        return location.getWorld().getName() + ':' + location.getBlockX() + ':'
                + (location.getBlockY() - 1) + ':' + location.getBlockZ();
    }

    private void invalid(Arena arena, Player player) {
        CombatStats stats = arena.stats(player);
        if (stats != null) stats.invalidInteractions++;
        arena.addReward(player, arena.rewardShaper().invalidInteraction(), RewardReason.INVALID);
    }

    private void pruneCrystalAttributions() {
        if (crystalAttributions.size() <= 512) return;
        final long oldestAllowedTick = manager.currentTick() - 800L;
        crystalAttributions.entrySet().removeIf(entry -> {
            CrystalAttribution value = entry.getValue();
            long mostRecentTick = Math.max(value.placementTick, value.detonationTick);
            return mostRecentTick < oldestAllowedTick;
        });
    }

    private static final class CrystalAttribution {
        private final String episodeId;
        private final UUID placerId;
        private final ExecutionSource placementSource;
        private final long placementTick;
        private final String baseKey;
        private UUID detonatorId;
        private ExecutionSource detonationSource;
        private long detonationTick = -1L;

        private CrystalAttribution(String episodeId, UUID placerId,
                                   ExecutionSource placementSource, long placementTick,
                                   String baseKey) {
            this.episodeId = episodeId;
            this.placerId = placerId;
            this.placementSource = placementSource;
            this.placementTick = placementTick;
            this.baseKey = baseKey;
        }

        private void recordDetonation(UUID playerId, ExecutionSource source, long tick) {
            if (detonationTick >= 0L) return;
            detonatorId = playerId;
            detonationSource = source;
            detonationTick = tick;
        }
    }

    private static final class BlockAttribution {
        private final String episodeId;
        private final ExecutionSource source;
        private final long tick;

        private BlockAttribution(String episodeId, ExecutionSource source, long tick) {
            this.episodeId = episodeId;
            this.source = source;
            this.tick = tick;
        }
    }
}
