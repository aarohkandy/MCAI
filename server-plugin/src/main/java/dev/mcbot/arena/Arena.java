package dev.mcbot.arena;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.bukkit.GameMode;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.block.Block;
import org.bukkit.enchantments.Enchantment;
import org.bukkit.entity.Entity;
import org.bukkit.entity.EnderCrystal;
import org.bukkit.entity.Player;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.PlayerInventory;
import org.bukkit.potion.PotionEffect;
import org.bukkit.util.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.UUID;

public final class Arena {
    private static final int FLOOR_Y = 63;
    private static final long KILL_CREDIT_TICKS = 20L;
    private static final long DEATH_SETTLEMENT_TICKS = 2L;
    private final ArenaManager manager;
    private final String id;
    private final Location center;
    private int outerRadius;
    private final int maximumOuterRadius;
    private final int floorDepth;
    private final int arenaHeight;
    private final int configuredSpawnMinSeparation;
    private final int configuredSpawnMaxSeparation;
    private int spawnMinSeparation;
    private int spawnMaxSeparation;
    private final double spawnYawJitter;
    private final Set<BlockKey> touchedBlocks = new HashSet<BlockKey>();
    private final Set<UUID> observedDeaths = new HashSet<UUID>();
    private final Map<UUID, CombatStats> stats = new HashMap<UUID, CombatStats>();
    private final Map<UUID, ExecutionMarker> executionMarkers = new HashMap<UUID, ExecutionMarker>();
    private final Map<UUID, DamageAttribution> lastDamageByVictim = new HashMap<UUID, DamageAttribution>();
    private final Map<String, PolicyObsidianPlacement> policyObsidianBases =
            new HashMap<String, PolicyObsidianPlacement>();
    private final Map<String, PolicyTerrainClearance> policyTerrainClearances =
            new HashMap<String, PolicyTerrainClearance>();
    private final CrystalChainTracker crystalChains;
    private RewardProfile rewardProfile;
    private Player first;
    private Player second;
    private long seed;
    private String episodeId;
    private long startedTick;
    private long deadlineTick;
    private int arenaSize = 21;
    private ArenaMode mode = ArenaMode.COMBINED;
    private String lane = "combined";
    private boolean active;
    private boolean basePrepared;
    private int curriculumStage = 1;
    private int curriculumStageCount = 1;
    private long curriculumCompletedEpisodes;
    private boolean curriculumEligible;
    private long lastAutonomousDamageTick;
    private long deathResolutionTick = -1L;
    private String terminalReason = "none";
    private Player terminalWinner;
    private boolean policyOwnedTerminalKill;

    public Arena(ArenaManager manager, String id, Location center, int outerRadius, int floorDepth, int arenaHeight,
                 int spawnMinSeparation, int spawnMaxSeparation, double spawnYawJitter) {
        this(manager, id, center, outerRadius, outerRadius, floorDepth, arenaHeight,
                spawnMinSeparation, spawnMaxSeparation, spawnYawJitter);
    }

    Arena(ArenaManager manager, String id, Location center, int outerRadius, int maximumOuterRadius,
          int floorDepth, int arenaHeight, int spawnMinSeparation, int spawnMaxSeparation,
          double spawnYawJitter) {
        this.manager = manager;
        this.rewardProfile = manager.getRewardProfile();
        this.crystalChains = new CrystalChainTracker(
                rewardProfile.shaper().config().crystalComboMaxTicks);
        this.id = id;
        this.center = center;
        this.maximumOuterRadius = Math.max(outerRadius, maximumOuterRadius);
        this.floorDepth = floorDepth;
        this.arenaHeight = arenaHeight;
        this.configuredSpawnMinSeparation = spawnMinSeparation;
        this.configuredSpawnMaxSeparation = spawnMaxSeparation;
        this.spawnYawJitter = spawnYawJitter;
        configureRadius(outerRadius);
    }

    public void start(Player first, Player second, long seed, ArenaMode mode, long tick, int timeoutSeconds,
                      int actionDelay, int observationDelay) {
        start(first, second, seed, mode, tick, timeoutSeconds, actionDelay, observationDelay,
                mode.name().toLowerCase(Locale.ROOT), outerRadius, 1, 1, 0L, true);
    }

    void start(Player first, Player second, long seed, ArenaMode mode, long tick, int timeoutSeconds,
               int actionDelay, int observationDelay, String lane, int requestedRadius, int curriculumStage,
               int curriculumStageCount, long curriculumCompletedEpisodes, boolean curriculumEligible) {
        this.first = first;
        this.second = second;
        this.seed = seed;
        this.mode = mode;
        // Hot reward changes are generation settings, never mid-trajectory
        // mutations. Every reward calculation in this episode uses this exact
        // immutable profile until finish().
        this.rewardProfile = manager.getRewardProfile();
        this.lane = lane;
        this.curriculumStage = Math.max(1, curriculumStage);
        this.curriculumStageCount = Math.max(this.curriculumStage, curriculumStageCount);
        this.curriculumCompletedEpisodes = Math.max(0L, curriculumCompletedEpisodes);
        this.curriculumEligible = curriculumEligible;
        this.episodeId = UUID.randomUUID().toString();
        crystalChains.reset(this.episodeId);
        this.startedTick = tick;
        this.deadlineTick = tick + timeoutSeconds * 20L;
        this.lastAutonomousDamageTick = tick;
        this.deathResolutionTick = -1L;
        this.terminalReason = "none";
        this.terminalWinner = null;
        this.policyOwnedTerminalKill = false;
        stats.clear();
        executionMarkers.clear();
        lastDamageByVictim.clear();
        policyObsidianBases.clear();
        policyTerrainClearances.clear();
        observedDeaths.clear();
        stats.put(first.getUniqueId(), new CombatStats(first.getUniqueId()));
        stats.put(second.getUniqueId(), new CombatStats(second.getUniqueId()));
        stats(first).markAutonomousAction(tick);
        stats(second).markAutonomousAction(tick);
        // Restore the previous episode against its original geometry before a
        // newly unlocked stage expands the shell. Radius then stays fixed for
        // the complete lifetime of this match.
        resetTouchedBlocks();
        configureRadius(requestedRadius);
        prepareBase();
        clearEntities();
        Random random = new Random(seed);
        int size = layoutSize(random, mode);
        arenaSize = size;
        Location[] spawns = null;
        for (ArenaGeometry.PreparationStep step : ArenaGeometry.preparationSteps(
                mode.hasTerrainLayout(), mode.hasCrystalLayout())) {
            switch (step) {
                case GENERATE_LAYOUT:
                    generateLayout(random, size);
                    break;
                case CHOOSE_SPAWNS:
                    spawns = spawnLocations(random);
                    break;
                case CLEAR_SPAWN_LANE:
                    clearSpawnLane(spawns[0], spawns[1]);
                    break;
                case CLEAR_SPAWNS:
                    clearSpawn(spawns[0]);
                    clearSpawn(spawns[1]);
                    break;
                case GUARANTEE_CRYSTAL_PADS:
                    placeReachableCrystalPads(spawns);
                    break;
                default:
                    throw new IllegalStateException("unknown arena preparation step: " + step);
            }
        }
        if (spawns == null) throw new IllegalStateException("arena preparation did not choose fighter spawns");
        KitCurriculum.KitSpec episodeKit = KitCurriculum.forEpisode(
                mode, this.curriculumStage, this.curriculumStageCount);
        int obsidianSupply = ArenaGeometry.obsidianSupply(seed);
        preparePlayer(first, spawns[0], episodeKit, obsidianSupply);
        preparePlayer(second, spawns[1], episodeKit, obsidianSupply);
        active = true;
        manager.matchStarted(this, first, actionDelay, observationDelay);
        manager.matchStarted(this, second, actionDelay, observationDelay);
    }

    public void tick(long currentTick) {
        if (!active) return;
        if (deathResolutionTick >= 0L) {
            if (currentTick < deathResolutionTick) return;
            boolean firstDead = observedDeaths.contains(first.getUniqueId()) || fighterDead(first);
            boolean secondDead = observedDeaths.contains(second.getUniqueId()) || fighterDead(second);
            deathResolutionTick = -1L;
            if (firstDead || secondDead) {
                if (firstDead && secondDead) finish(null, false, "double_ko");
                else finish(firstDead ? second : first, false, "death");
                return;
            }
        }
        applyDenseRewards();
        flushRewards();
        if (!first.isOnline() || !second.isOnline()) {
            Player winner = first.isOnline() ? first : (second.isOnline() ? second : null);
            finish(winner, false, "disconnect");
        } else if (fighterDead(first) || fighterDead(second)) {
            if (fighterDead(first)) observeDeath(first);
            if (fighterDead(second)) observeDeath(second);
        } else if (currentTick - lastAutonomousDamageTick >= 200L && fighterDistance() > 6.0) {
            finish(null, true, "disengaged");
        } else if (currentTick >= deadlineTick) {
            // A timeout is a failed fight for both policies. Health-point
            // tiebreaks rewarded passive chip damage and then retreating.
            finish(null, true, "timeout");
        }
    }

    public void finish(Player winner, boolean truncated, String reason) {
        if (!active) return;
        flushRewards();
        active = false;
        terminalReason = reason;
        terminalWinner = winner;
        String winningSource = winner == null ? "none"
                : damageTerminalSource(opponent(winner));
        boolean policyKill = "death".equals(reason) && "policy".equals(winningSource)
                && policyOwnedKill(winner);
        policyOwnedTerminalKill = policyKill;
        RewardConfig rewardConfig = rewardShaper().config();
        long elapsedTicks = Math.max(0L, manager.currentTick() - startedTick);
        long totalTicks = Math.max(1L, deadlineTick - startedTick);
        double winnerReward = winnerTerminalReward(reason, policyKill,
                rewardConfig.policyKill, rewardConfig.policyKillSpeedBonus,
                elapsedTicks, totalTicks);
        double firstOutcome = terminalOutcome(reason,
                winner != null, winner != null && winner.equals(first), winnerReward, rewardConfig);
        double secondOutcome = terminalOutcome(reason,
                winner != null, winner != null && winner.equals(second), winnerReward, rewardConfig);
        if (policyKill) stats(winner).policyKills++;
        stats(first).terminalReward = firstOutcome;
        stats(second).terminalReward = secondOutcome;
        manager.matchEnded(this, first, firstOutcome, truncated, reason);
        manager.matchEnded(this, second, secondOutcome, truncated, reason);
        manager.curriculumEpisodeCompleted(this, curriculumEligible, reason);
        for (Player player : players()) {
            if (player.isOnline()) {
                player.setVelocity(new Vector(0, 0, 0));
                player.getInventory().clear();
            }
        }
        manager.recycle(this, Arrays.asList(first, second));
    }

    /** Defers death resolution so both halves of one explosion can settle before scoring. */
    public void observeDeath(Player player) {
        if (!active || player == null || stats(player) == null) return;
        observedDeaths.add(player.getUniqueId());
        if (deathResolutionTick < 0L) {
            deathResolutionTick = manager.currentTick() + DEATH_SETTLEMENT_TICKS;
        }
    }

    static boolean policyKillRewardEligible(ExecutionSource source, boolean assignedAttacker,
                                            long attributionAgeTicks, boolean winnerSurvived) {
        return source != null && source.isAutonomous() && assignedAttacker
                && attributionAgeTicks >= 0L && attributionAgeTicks <= KILL_CREDIT_TICKS
                && winnerSurvived;
    }

    static double winnerTerminalReward(String reason, boolean policyOwnedKill,
                                       double policyKillReward, double speedBonus,
                                       long elapsedTicks, long totalTicks) {
        if (!"death".equals(reason) || !policyOwnedKill) return 0.0;
        double remainingFraction = 1.0 - Math.max(0.0, Math.min(1.0,
                elapsedTicks / (double) Math.max(1L, totalTicks)));
        // Concentrate the bonus in the opening and middle of the fight. A
        // last-second kill should not look almost as good as an immediate one.
        double speedFactor = remainingFraction * remainingFraction;
        return Math.max(1.0, policyKillReward) + Math.max(0.0, speedBonus) * speedFactor;
    }

    static double terminalOutcome(String reason, boolean hasWinner, boolean isWinner,
                                  double winnerReward, RewardConfig config) {
        if ("double_ko".equals(reason)) return config.doubleKoLoss;
        if ("timeout".equals(reason)) return config.timeoutLoss;
        if ("disengaged".equals(reason)) return config.disengagedLoss;
        if ("death".equals(reason)) return isWinner && hasWinner
                ? winnerReward : config.deathLoss;
        if (!hasWinner) return -1.0;
        return isWinner ? 1.0 : config.deathLoss;
    }

    public void addReward(Player player, double reward, RewardReason reason) {
        if (player == null) return;
        CombatStats value = stats.get(player.getUniqueId());
        if (value != null && reward != 0) {
            double paced = reward > 0
                    ? reward * rewardShaper().positiveRewardMultiplier(
                    Math.max(0L, manager.currentTick() - startedTick))
                    : reward;
            value.addPending(reason, paced * manager.getShapingScale());
        }
    }

    /**
     * Curriculum-only mechanic credit.  The first stage teaches that an
     * operation is possible, later retention stages keep only a weak hint,
     * and the final combined/terrain league stage is driven by combat
     * advantage and terminal outcomes alone.
     */
    private void addMechanicReward(Player player, double reward, RewardReason reason) {
        addReward(player, reward * mechanicRewardMultiplier(
                curriculumStage, curriculumStageCount, mode), reason);
    }

    static double mechanicRewardMultiplier(int stage, int stageCount, ArenaMode mode) {
        int boundedStage = Math.max(1, stage);
        int boundedCount = Math.max(boundedStage, stageCount);
        if (boundedStage <= 1) return 1.0;
        if (boundedStage >= boundedCount
                && (mode == ArenaMode.COMBINED || mode == ArenaMode.TERRAIN)) {
            return 0.0;
        }
        return 0.1;
    }

    public void recordBlockPlacement(Player player, Block block, ExecutionSource source) {
        CombatStats value = stats(player);
        if (value == null) return;
        value.blocksPlaced++;
        value.events(source).blocksPlaced++;
        String key = rewardBlockKey(block);
        value.rememberPlacedBlock(key);
        PolicyTerrainClearance clearance = policyTerrainClearances.remove(key);
        if (block.getType() == Material.OBSIDIAN) {
            value.obsidianPlaced++;
            value.events(source).obsidianPlaced++;
            if (source != null && source.isAutonomous()) {
                policyObsidianBases.put(key, new PolicyObsidianPlacement(
                        player.getUniqueId(), manager.currentTick()));
            } else {
                // A rebuilt teacher/safety block must not inherit stale policy
                // ownership from an earlier block at the same coordinate.
                policyObsidianBases.remove(key);
            }
            boolean useful = isUsefulTacticalObsidianPlacement(player, block);
            if (useful && source != null && source.isAutonomous()) {
                value.markAutonomousAction(manager.currentTick());
            }
            // Teacher placement remains visible in its attributed counters, but
            // claimAutonomousObsidianReward below keeps it out of PPO reward.
            if (useful) {
                value.tacticalObsidianPlaced++;
                value.events(source).tacticalObsidianPlaced++;
            }
            RewardConfig config = rewardShaper().config();
            if (claimAutonomousObsidianReward(
                    value, key, config.maxRewardedObsidian, source, useful)) {
                addMechanicReward(player, rewardShaper().obsidianPlaced(), RewardReason.OBSIDIAN);
            }
            if (claimAutonomousTacticalMinePlace(
                    value, key, clearance == null ? null : clearance.ownerId,
                    clearance == null ? -1L : clearance.tick, player.getUniqueId(),
                    manager.currentTick(), config.tacticalMinePlaceMaxTicks,
                    config.maxRewardedTacticalMinePlace, source, useful)) {
                value.tacticalMinePlaceSequences++;
                value.events(source).tacticalMinePlaceSequences++;
                addMechanicReward(player, rewardShaper().tacticalMinePlace(),
                        RewardReason.TACTICAL_MINE_PLACE);
            }
        }
    }

    public void recordBlockMined(Player player, Block block, Material originalMaterial, ExecutionSource source) {
        CombatStats value = stats(player);
        if (value == null) return;
        value.blocksMined++;
        value.events(source).blocksMined++;
        String key = rewardBlockKey(block);
        policyObsidianBases.remove(key);
        policyTerrainClearances.remove(key);
        // A raw block break is deliberately neutral. Credit arrives only if
        // this policy-owned natural-stone clearance is followed by a useful
        // obsidian base at the exact same combat site.
        if (originalMaterial == Material.STONE && source != null && source.isAutonomous()
                && !value.wasPlacedBlock(key)
                && isUsefulTacticalSite(player, block, Material.STONE)) {
            value.markAutonomousAction(manager.currentTick());
            policyTerrainClearances.put(key, new PolicyTerrainClearance(
                    player.getUniqueId(), manager.currentTick()));
        }
    }

    static boolean claimAutonomousObsidianReward(CombatStats stats, String blockKey,
                                                  int cap, ExecutionSource source,
                                                  boolean usefulTacticalBase) {
        return source != null && source.isAutonomous()
                && usefulTacticalBase
                && stats.claimObsidian(blockKey, cap);
    }

    static boolean claimAutonomousMiningReward(CombatStats stats, String blockKey,
                                                int cap, ExecutionSource source) {
        // Retained as a compatibility seam for older callers. Generic mining
        // is no longer a rewarded event.
        return false;
    }

    static boolean claimAutonomousTacticalMinePlace(
            CombatStats stats, String blockKey, UUID clearanceOwner, long clearanceTick,
            UUID builderId, long currentTick, int maxAgeTicks, int cap,
            ExecutionSource source, boolean usefulTacticalBase) {
        long age = currentTick - clearanceTick;
        return stats != null && source != null && source.isAutonomous()
                && usefulTacticalBase && clearanceOwner != null
                && clearanceOwner.equals(builderId) && clearanceTick >= 0L
                && age >= 0L && age <= Math.max(1, maxAgeTicks)
                && stats.claimTacticalMinePlace(blockKey, cap);
    }

    public void recordCrystalPlacement(Player player, UUID crystalId, boolean onObsidian,
                                       String baseKey, ExecutionSource source) {
        CombatStats value = stats(player);
        if (value == null) return;
        value.crystalsPlaced++;
        value.events(source).crystalsPlaced++;
        PolicyObsidianPlacement builtBase = policyObsidianBases.get(baseKey);
        boolean policyBuiltBase = builtBase != null && source != null && source.isAutonomous()
                && builtBase.ownerId.equals(player.getUniqueId())
                && builtBase.tick <= manager.currentTick();
        CrystalChainTracker.Chain chain = crystalChains.recordPlacement(
                crystalId, player.getUniqueId(), source, manager.currentTick(), onObsidian,
                baseKey, policyBuiltBase);
        if (chain != null && onObsidian && source.isAutonomous()) {
            value.policyCrystalChainsStarted++;
            value.markAutonomousAction(manager.currentTick());
        }
        RewardConfig config = rewardShaper().config();
        if (source.isAutonomous() && onObsidian
                && value.claimCrystalPlacement(crystalId, baseKey,
                config.maxRewardedCrystalPlacements)) {
            addMechanicReward(player, rewardShaper().crystalPlaced(), RewardReason.CRYSTAL_PLACEMENT);
        }
    }

    public void recordCrystalDestruction(Player player, UUID crystalId, String baseKey,
                                         ExecutionSource source) {
        CombatStats value = stats(player);
        if (value == null) return;
        value.crystalsDestroyed++;
        value.events(source).crystalsDestroyed++;
        CrystalChainTracker.Chain chain = crystalChains.recordDetonation(
                crystalId, player.getUniqueId(), source, manager.currentTick());
        if (chain != null && chain.claimDetonationCount()) {
            value.policyCrystalChainsDetonated++;
        }
        if (source != null && source.isAutonomous()) {
            value.markAutonomousAction(manager.currentTick());
        }
        RewardConfig config = rewardShaper().config();
        if (source.isAutonomous()
                && value.claimCrystalDestruction(crystalId, baseKey,
                config.maxRewardedCrystalDestructions)) {
            addMechanicReward(player, rewardShaper().crystalDestroyed(), RewardReason.CRYSTAL_DESTRUCTION);
        }
    }

    public void recordCrystalExplosion(Player player, UUID crystalId, String baseKey,
                                       ExecutionSource source) {
        CombatStats value = stats(player);
        if (value == null) return;
        value.crystalsExploded++;
        value.events(source).crystalsExploded++;
        if (source != null && source.isAutonomous()) {
            value.markAutonomousAction(manager.currentTick());
        }
        RewardConfig config = rewardShaper().config();
        if (source.isAutonomous()
                && value.claimCrystalExplosion(crystalId, baseKey,
                config.maxRewardedCrystalExplosions)) {
            addMechanicReward(player, rewardShaper().crystalExploded(), RewardReason.CRYSTAL_EXPLOSION);
        }
    }

    /** Records an arm swing only against this fighter's explicitly assigned opponent. */
    public void recordAttackSwing(Player player, ExecutionSource source) {
        if (!active) return;
        CombatStats value = stats(player);
        Player assignedOpponent = opponent(player);
        if (value == null || assignedOpponent == null || !assignedOpponent.isOnline()
                || manager.arenaFor(assignedOpponent) != this) return;

        value.attackSwings++;
        RewardShaper shaper = rewardShaper();
        RewardShaper.AttackResult result = shaper.evaluateAttack(
                gazeDot(player, assignedOpponent), playerDistance(player, assignedOpponent),
                player.hasLineOfSight(assignedOpponent),
                value.ticksSinceRewardedAttack(manager.currentTick()));
        if (result == RewardShaper.AttackResult.VALID) {
            value.validAttackSwings++;
            if (source != null && source.isAutonomous()) {
                value.markAutonomousAction(manager.currentTick());
            }
            if (value.claimAttackReward(manager.currentTick(), shaper.config().maxRewardedAttackSwings)) {
                addReward(player, shaper.attackSwing(result), RewardReason.ATTACK);
            }
        } else if (result == RewardShaper.AttackResult.SPAM) {
            value.spamAttackSwings++;
            if (value.claimAttackPenalty(shaper.config().maxPenalizedAttackSwings)) {
                addReward(player, shaper.attackSwing(result), RewardReason.ATTACK_SPAM);
            }
        } else {
            value.missedAttackSwings++;
            if (value.claimAttackPenalty(shaper.config().maxPenalizedAttackSwings)) {
                addReward(player, shaper.attackSwing(result), RewardReason.ATTACK_MISS);
            }
        }
    }

    /** A direct melee hit is valid only when victim is the attacker's assigned opponent. */
    public void recordSuccessfulHit(Player attacker, Player victim, ExecutionSource source) {
        if (!active || !victim.equals(opponent(attacker)) || manager.arenaFor(victim) != this) return;
        CombatStats value = stats(attacker);
        if (value == null) return;
        value.hitsLanded++;
        CombatStats.AttributedEvents attributed = value.events(source);
        attributed.hitsLanded++;
        if (attributed.firstHitTick < 0L) {
            attributed.firstHitTick = Math.max(0L, manager.currentTick() - startedTick);
        }
        if (source != null && source.isAutonomous()) {
            value.markAutonomousAction(manager.currentTick());
        }
        RewardShaper shaper = rewardShaper();
        if (source.isAutonomous() && value.claimHitReward(shaper.config().maxRewardedHits)) {
            addReward(attacker, shaper.successfulHit(), RewardReason.HIT);
        }
    }

    public CombatStats stats(Player player) {
        return stats.get(player.getUniqueId());
    }

    public void recordOpponentDamage(Player attacker, Player victim, double damage,
                                     ExecutionSource source, boolean crystalDamage, UUID crystalId) {
        if (!active || damage <= 0 || !victim.equals(opponent(attacker))
                || manager.arenaFor(victim) != this) return;
        CombatStats attackerStats = stats(attacker);
        if (attackerStats == null) return;
        attackerStats.damageDealt += damage;
        CombatStats.AttributedEvents events = attackerStats.events(source);
        events.damageDealt += damage;
        if (events.firstDamageTick < 0L) {
            events.firstDamageTick = Math.max(0L, manager.currentTick() - startedTick);
        }
        if (crystalDamage) {
            events.crystalDamageEvents++;
            events.crystalDamageDealt += damage;
        } else {
            events.directDamageDealt += damage;
        }
        lastDamageByVictim.put(victim.getUniqueId(), new DamageAttribution(
                attacker.getUniqueId(), source, crystalDamage, crystalId, manager.currentTick()));
        if (crystalDamage && crystalId != null) {
            CrystalChainTracker.Chain chain = crystalChains.recordOpponentDamage(
                    crystalId, attacker.getUniqueId(), source, manager.currentTick());
            if (chain != null && chain.claimDamageCount()) {
                attackerStats.policyCrystalChainsDamaging++;
                RewardConfig config = rewardShaper().config();
                if (attackerStats.claimCrystalCombo(config.maxRewardedCrystalCombos)
                        && chain.claimComboReward()) {
                    addMechanicReward(attacker, rewardShaper().crystalComboDamage(),
                            RewardReason.CRYSTAL_COMBO);
                }
            }
            if (chain != null && chain.claimPolicyBuiltDamageCount()) {
                attackerStats.policyBuiltCrystalChainsDamaging++;
                attackerStats.events(source).policyBuiltCrystalChainsDamaging++;
                RewardConfig config = rewardShaper().config();
                if (attackerStats.claimObsidianCombo(
                        chain.baseKey(), config.maxRewardedObsidianCombos)) {
                    addMechanicReward(attacker, rewardShaper().obsidianCombo(),
                            RewardReason.OBSIDIAN_COMBO);
                }
            }
        }
        if (source.isAutonomous()) lastAutonomousDamageTick = manager.currentTick();
        if (source.isAutonomous()) attackerStats.markAutonomousAction(manager.currentTick());
    }

    /** Records causal ownership for self-inflicted projectiles/crystals without opponent credit. */
    public void recordSelfDamage(Player player, double damage, ExecutionSource source,
                                 boolean crystalDamage, UUID crystalId) {
        if (!active || player == null || damage <= 0 || stats(player) == null) return;
        stats(player).selfDamage += damage;
        lastDamageByVictim.put(player.getUniqueId(), new DamageAttribution(
                player.getUniqueId(), source, crystalDamage, crystalId, manager.currentTick()));
        if (crystalDamage && source != null && source.isAutonomous()) {
            addReward(player, rewardShaper().ownCrystalSelfDamage(damage),
                    RewardReason.SELF_DAMAGE);
            if (stats(player).claimOwnCrystalSelfHit(crystalId)) {
                addReward(player, rewardShaper().ownCrystalSelfHit(),
                        RewardReason.SELF_DAMAGE);
            }
        }
    }

    public boolean lastDamageWasAssignedOpponent(Player victim) {
        DamageAttribution attribution = recentDamageAttribution(victim, 5L);
        Player assignedOpponent = opponent(victim);
        return attribution != null && assignedOpponent != null
                && attribution.attackerId.equals(assignedOpponent.getUniqueId());
    }

    public ExecutionSource recordTotemAttribution(Player victim) {
        DamageAttribution attribution = recentDamageAttribution(victim, 5L);
        if (attribution == null) return null;
        Player assignedOpponent = opponent(victim);
        if (assignedOpponent == null
                || !attribution.attackerId.equals(assignedOpponent.getUniqueId())) {
            return attribution.source;
        }
        CombatStats attackerStats = stats.get(attribution.attackerId);
        if (attackerStats == null) return attribution.source;
        CombatStats.AttributedEvents events = attackerStats.events(attribution.source);
        events.totemsForced++;
        if (attribution.crystalDamage) events.crystalTotemsForced++;
        if (attribution.crystalDamage && attribution.crystalId != null) {
            CrystalChainTracker.Chain chain = crystalChains.recordOpponentPop(
                    attribution.crystalId, attribution.attackerId,
                    attribution.source, manager.currentTick());
            if (chain != null && chain.claimPopCount()) {
                attackerStats.policyCrystalChainsPopping++;
                if (chain.claimPopReward()) {
                    addMechanicReward(player(attribution.attackerId),
                            rewardShaper().crystalComboPop(),
                            RewardReason.CRYSTAL_COMBO_POP);
                }
            }
        }
        return attribution.source;
    }

    private DamageAttribution recentDamageAttribution(Player victim, long maximumAgeTicks) {
        if (victim == null) return null;
        DamageAttribution attribution = lastDamageByVictim.get(victim.getUniqueId());
        long age = attribution == null ? -1L : manager.currentTick() - attribution.tick;
        return attribution != null && age >= 0L && age <= maximumAgeTicks
                ? attribution : null;
    }

    private boolean policyOwnedKill(Player winner) {
        if (winner == null) return false;
        Player victim = opponent(winner);
        DamageAttribution attribution = recentDamageAttribution(victim, KILL_CREDIT_TICKS);
        boolean assignedAttacker = attribution != null
                && attribution.attackerId.equals(winner.getUniqueId());
        boolean winnerSurvived = winner.isOnline() && !fighterDead(winner);
        long age = attribution == null ? -1L : manager.currentTick() - attribution.tick;
        return policyKillRewardEligible(attribution == null ? null : attribution.source,
                assignedAttacker, age, winnerSurvived);
    }

    private String damageTerminalSource(Player victim) {
        DamageAttribution attribution = recentDamageAttribution(victim, KILL_CREDIT_TICKS);
        if (attribution == null) return "environment";
        Player assignedOpponent = opponent(victim);
        boolean selfCaused = victim != null
                && attribution.attackerId.equals(victim.getUniqueId());
        boolean assignedOpponentCaused = assignedOpponent != null
                && attribution.attackerId.equals(assignedOpponent.getUniqueId());
        return terminalSourceLabel(attribution.source, selfCaused, assignedOpponentCaused);
    }

    static String terminalSourceLabel(ExecutionSource source, boolean selfCaused,
                                      boolean assignedOpponentCaused) {
        if (selfCaused) {
            if (source == ExecutionSource.POLICY) return "self";
            return source == null ? "environment" : source.wireName();
        }
        if (assignedOpponentCaused) {
            return source == null ? "environment" : source.wireName();
        }
        return "environment";
    }

    private static boolean fighterDead(Player player) {
        return player == null || player.isDead() || player.getHealth() <= 0.0;
    }

    public ExecutionSource executionSource(Player player) {
        ExecutionMarker marker = executionMarkers.get(player.getUniqueId());
        if (marker == null) return ExecutionSource.POLICY;
        if (manager.currentTick() > marker.untilTick) {
            executionMarkers.remove(player.getUniqueId());
            return ExecutionSource.POLICY;
        }
        return marker.source;
    }

    public boolean markExecutionSource(Player player, String expectedEpisodeId,
                                       ExecutionSource source, int durationTicks) {
        if (!active || player == null || stats(player) == null
                || !episodeId.equals(expectedEpisodeId)) return false;
        if (source == ExecutionSource.POLICY) {
            executionMarkers.remove(player.getUniqueId());
        } else {
            int duration = Math.max(1, Math.min(4, durationTicks));
            executionMarkers.put(player.getUniqueId(), new ExecutionMarker(
                    source, manager.currentTick() + duration - 1L));
        }
        return true;
    }

    public Player opponent(Player player) {
        if (first != null && first.equals(player)) return second;
        if (second != null && second.equals(player)) return first;
        return null;
    }

    private Player player(UUID playerId) {
        if (first != null && first.getUniqueId().equals(playerId)) return first;
        if (second != null && second.getUniqueId().equals(playerId)) return second;
        return null;
    }

    public boolean contains(Location location) {
        return location != null && location.getWorld().equals(center.getWorld())
                && Math.abs(location.getBlockX() - center.getBlockX()) <= outerRadius
                && Math.abs(location.getBlockZ() - center.getBlockZ()) <= outerRadius
                && location.getY() >= bottomBarrierY() && location.getY() <= FLOOR_Y + arenaHeight;
    }

    public void markTouched(Block block) {
        if (contains(block.getLocation())) touchedBlocks.add(new BlockKey(block.getX(), block.getY(), block.getZ()));
    }

    /** Removes causal base ownership when mining or an explosion destroys it. */
    public void forgetPolicyObsidianBase(Block block) {
        if (block != null) policyObsidianBases.remove(rewardBlockKey(block));
    }

    public Collection<Player> players() {
        List<Player> result = new ArrayList<Player>();
        if (first != null) result.add(first);
        if (second != null) result.add(second);
        return result;
    }

    public String getId() { return id; }
    public Location getCenter() { return center.clone(); }
    public int getOuterRadius() { return outerRadius; }
    public int getCurriculumStage() { return curriculumStage; }
    public int getCurriculumStageCount() { return curriculumStageCount; }
    public long getCurriculumCompletedEpisodes() { return curriculumCompletedEpisodes; }
    public long getSeed() { return seed; }
    public String getEpisodeId() { return episodeId; }
    public long getStartedTick() { return startedTick; }
    public ArenaMode getMode() { return mode; }
    public String getLane() { return lane; }
    public RewardProfile getRewardProfile() { return rewardProfile; }
    RewardShaper rewardShaper() { return rewardProfile.shaper(); }
    public boolean isActive() { return active; }
    public String getTerminalSource(Player player) {
        if (player == null) return "none";
        if ("double_ko".equals(terminalReason)) return damageTerminalSource(player);
        if (!"death".equals(terminalReason) || terminalWinner == null) return "none";
        Player victim = player.equals(terminalWinner) ? opponent(player) : player;
        return damageTerminalSource(victim);
    }
    public boolean isPolicyOwnedTerminalKill() { return policyOwnedTerminalKill; }

    public JsonObject statsJson(Player player) {
        CombatStats value = stats(player);
        JsonObject json = new JsonObject();
        if (value == null) return json;
        json.addProperty("damage_dealt", value.damageDealt);
        json.addProperty("damage_taken", value.damageTaken);
        json.addProperty("self_damage", value.selfDamage);
        json.addProperty("policy_kills", value.policyKills);
        json.addProperty("healing", value.healing);
        json.addProperty("totem_pops", value.totemPops);
        json.addProperty("totem_pops_forced", value.totemPopsForced);
        json.addProperty("obsidian_placed", value.obsidianPlaced);
        json.addProperty("tactical_obsidian_placed", value.tacticalObsidianPlaced);
        json.addProperty("tactical_mine_place_sequences", value.tacticalMinePlaceSequences);
        json.addProperty("policy_built_crystal_chains_damaging",
                value.policyBuiltCrystalChainsDamaging);
        json.addProperty("rewarded_obsidian_combos", value.rewardedObsidianCombos);
        json.addProperty("crystals_placed", value.crystalsPlaced);
        json.addProperty("crystals_destroyed", value.crystalsDestroyed);
        json.addProperty("crystals_exploded", value.crystalsExploded);
        json.addProperty("policy_crystal_chains_started", value.policyCrystalChainsStarted);
        json.addProperty("policy_crystal_chains_detonated", value.policyCrystalChainsDetonated);
        json.addProperty("policy_crystal_chains_damaging", value.policyCrystalChainsDamaging);
        json.addProperty("policy_crystal_chains_popping", value.policyCrystalChainsPopping);
        json.addProperty("rewarded_crystal_combos", value.rewardedCrystalCombos);
        json.addProperty("policy_crystal_chain_damage_rate", value.policyCrystalChainsStarted == 0
                ? 0.0 : value.policyCrystalChainsDamaging / (double) value.policyCrystalChainsStarted);
        json.addProperty("blocks_placed", value.blocksPlaced);
        json.addProperty("blocks_mined", value.blocksMined);
        json.addProperty("invalid_interactions", value.invalidInteractions);
        json.addProperty("approach_ticks", value.approachTicks);
        json.addProperty("extreme_pitch_ticks", value.extremePitchTicks);
        json.addProperty("aim_alignment_ticks", value.aimAlignmentTicks);
        json.addProperty("lock_on_ticks", value.lockOnTicks);
        json.addProperty("rewarded_lock_on_ticks", value.rewardedLockOnTicks);
        json.addProperty("attack_swings", value.attackSwings);
        json.addProperty("valid_attack_swings", value.validAttackSwings);
        json.addProperty("rewarded_attack_swings", value.rewardedAttackSwings);
        json.addProperty("missed_attack_swings", value.missedAttackSwings);
        json.addProperty("spam_attack_swings", value.spamAttackSwings);
        json.addProperty("penalized_attack_swings", value.penalizedAttackSwings);
        json.addProperty("hits_landed", value.hitsLanded);
        json.addProperty("rewarded_hits", value.rewardedHits);
        json.addProperty("inaction_penalty_ticks", value.inactionPenaltyTicks);
        json.addProperty("ticks_since_autonomous_action",
                Math.max(0L, manager.currentTick() - value.lastAutonomousActionTick));
        json.addProperty("shaping_reward", value.shapingRewardTotal);
        json.addProperty("shaping_points", value.shapingRewardTotal * 100.0);
        json.addProperty("terminal_points", value.terminalReward * 100.0);
        json.addProperty("total_points", (value.shapingRewardTotal + value.terminalReward) * 100.0);
        JsonObject execution = new JsonObject();
        for (ExecutionSource source : ExecutionSource.values()) {
            CombatStats.AttributedEvents events = value.existingEvents(source);
            JsonObject attributed = new JsonObject();
            attributed.addProperty("damage_dealt", events.damageDealt);
            attributed.addProperty("hits_landed", events.hitsLanded);
            attributed.addProperty("crystals_placed", events.crystalsPlaced);
            attributed.addProperty("crystals_destroyed", events.crystalsDestroyed);
            attributed.addProperty("crystals_exploded", events.crystalsExploded);
            attributed.addProperty("crystal_damage_events", events.crystalDamageEvents);
            attributed.addProperty("totems_forced", events.totemsForced);
            attributed.addProperty("crystal_totems_forced", events.crystalTotemsForced);
            attributed.addProperty("blocks_placed", events.blocksPlaced);
            attributed.addProperty("blocks_mined", events.blocksMined);
            attributed.addProperty("obsidian_placed", events.obsidianPlaced);
            attributed.addProperty("tactical_obsidian_placed", events.tacticalObsidianPlaced);
            attributed.addProperty("tactical_mine_place_sequences",
                    events.tacticalMinePlaceSequences);
            attributed.addProperty("policy_built_crystal_chains_damaging",
                    events.policyBuiltCrystalChainsDamaging);
            attributed.addProperty("first_hit_tick", events.firstHitTick);
            attributed.addProperty("first_damage_tick", events.firstDamageTick);
            execution.add(source.wireName(), attributed);
        }
        json.add("execution", execution);
        JsonObject breakdown = new JsonObject();
        for (RewardReason reason : RewardReason.values()) {
            breakdown.addProperty(reason.name().toLowerCase(Locale.ROOT), value.rewardTotal(reason) * 100.0);
        }
        json.add("point_breakdown", breakdown);
        return json;
    }

    public double points(Player player) {
        CombatStats value = stats(player);
        return value == null ? 0 : (value.shapingRewardTotal + value.terminalReward) * 100.0;
    }

    boolean autonomousEngagement() {
        return autonomousEngagementFor(mode, stats.values());
    }

    static boolean autonomousEngagementFor(ArenaMode selectedMode,
                                            Collection<CombatStats> combatStats) {
        int policyHits = 0;
        double policyDirectDamage = 0;
        boolean policyCrystalChain = false;
        for (CombatStats value : combatStats) {
            policyCrystalChain = policyCrystalChain
                    || value.policyCrystalChainsDamaging > 0
                    || value.policyCrystalChainsPopping > 0;
            CombatStats.AttributedEvents policy = value.existingEvents(ExecutionSource.POLICY);
            policyHits += policy.hitsLanded;
            policyDirectDamage += policy.directDamageDealt;
        }
        if (selectedMode == ArenaMode.CRYSTAL) return policyCrystalChain;
        return policyHits >= 3 || policyDirectDamage >= 4.0
                || (selectedMode.hasCrystalLayout() && policyCrystalChain);
    }

    public JsonObject snapshotJson(long currentTick) {
        JsonObject json = new JsonObject();
        json.addProperty("episode_id", episodeId);
        json.addProperty("arena_seed", seed);
        json.addProperty("mode", mode.name().toLowerCase());
        json.addProperty("lane", lane);
        json.addProperty("arena_size", ArenaGeometry.playableInteriorDiameter(outerRadius));
        json.addProperty("layout_size", arenaSize);
        json.addProperty("arena_radius", outerRadius);
        json.addProperty("arena_depth", floorDepth);
        json.addProperty("arena_height", arenaHeight);
        json.addProperty("spawn_min_separation", spawnMinSeparation);
        json.addProperty("spawn_max_separation", spawnMaxSeparation);
        json.addProperty("spawn_yaw_jitter_degrees", spawnYawJitter);
        json.addProperty("curriculum_stage", curriculumStage);
        json.addProperty("curriculum_stage_count", curriculumStageCount);
        json.addProperty("curriculum_completed_episodes", curriculumCompletedEpisodes);
        json.add("reward_profile", rewardProfile.toJson());
        json.addProperty("mechanic_reward_multiplier", mechanicRewardMultiplier(
                curriculumStage, curriculumStageCount, mode));
        json.addProperty("autonomous_engagement", autonomousEngagement());
        json.addProperty("valid_crystal_pads", countValidCrystalPads());
        int policyChainsStarted = 0;
        int policyChainsDetonated = 0;
        int policyChainsDamaging = 0;
        int policyChainsPopping = 0;
        for (CombatStats value : stats.values()) {
            policyChainsStarted += value.policyCrystalChainsStarted;
            policyChainsDetonated += value.policyCrystalChainsDetonated;
            policyChainsDamaging += value.policyCrystalChainsDamaging;
            policyChainsPopping += value.policyCrystalChainsPopping;
        }
        json.addProperty("policy_crystal_chains_started", policyChainsStarted);
        json.addProperty("policy_crystal_chains_detonated", policyChainsDetonated);
        json.addProperty("policy_crystal_chains_damaging", policyChainsDamaging);
        json.addProperty("policy_crystal_chains_popping", policyChainsPopping);
        json.addProperty("policy_crystal_chain_damage_rate", policyChainsStarted == 0
                ? 0.0 : policyChainsDamaging / (double) policyChainsStarted);
        json.addProperty("elapsed_ticks", Math.max(0, currentTick - startedTick));
        json.addProperty("remaining_ticks", Math.max(0, deadlineTick - currentTick));

        JsonArray fighters = new JsonArray();
        for (Player player : players()) {
            JsonObject fighter = new JsonObject();
            Location location = player.getLocation();
            Vector velocity = player.getVelocity();
            fighter.addProperty("agent_id", manager.agentIdFor(player));
            fighter.addProperty("name", player.getName());
            fighter.addProperty("x", location.getX() - center.getX());
            fighter.addProperty("y", location.getY() - FLOOR_Y);
            fighter.addProperty("z", location.getZ() - center.getZ());
            fighter.addProperty("yaw", location.getYaw());
            fighter.addProperty("pitch", location.getPitch());
            fighter.addProperty("vx", velocity.getX());
            fighter.addProperty("vy", velocity.getY());
            fighter.addProperty("vz", velocity.getZ());
            fighter.addProperty("health", Math.max(0, player.getHealth()));
            fighter.addProperty("absorption", Math.max(0, absorption(player)));
            fighter.addProperty("food", player.getFoodLevel());
            fighter.addProperty("grounded", player.isOnGround());
            fighter.add("stats", statsJson(player));
            fighters.add(fighter);
        }
        json.add("fighters", fighters);

        JsonArray entities = new JsonArray();
        for (Entity entity : center.getWorld().getEntities()) {
            if (!(entity instanceof EnderCrystal) || !contains(entity.getLocation())) continue;
            Location location = entity.getLocation();
            JsonObject value = new JsonObject();
            value.addProperty("type", "crystal");
            value.addProperty("x", location.getX() - center.getX());
            value.addProperty("y", location.getY() - FLOOR_Y);
            value.addProperty("z", location.getZ() - center.getZ());
            CrystalChainTracker.Chain chain = crystalChains.chain(entity.getUniqueId());
            if (chain != null) {
                value.addProperty("sequence_id", chain.sequenceId());
                value.addProperty("placer_id", chain.placerId().toString());
                value.addProperty("placement_source", chain.placementSource().wireName());
                value.addProperty("placement_tick", chain.placementTick());
                value.addProperty("legal_base", chain.legalBase());
                if (chain.detonatorId() != null) {
                    value.addProperty("detonator_id", chain.detonatorId().toString());
                    value.addProperty("detonation_source", chain.detonationSource().wireName());
                    value.addProperty("detonation_tick", chain.detonationTick());
                }
            }
            entities.add(value);
        }
        json.add("entities", entities);

        JsonArray chainSequences = new JsonArray();
        for (CrystalChainTracker.Chain chain : crystalChains.chains()) {
            JsonObject value = new JsonObject();
            value.addProperty("sequence_id", chain.sequenceId());
            value.addProperty("crystal_id", chain.crystalId().toString());
            value.addProperty("placer_id", chain.placerId().toString());
            value.addProperty("placement_source", chain.placementSource().wireName());
            value.addProperty("placement_tick", chain.placementTick());
            value.addProperty("legal_base", chain.legalBase());
            value.addProperty("autonomous_sequence", chain.isAutonomousSequence());
            value.addProperty("opponent_damaged", chain.opponentDamaged());
            value.addProperty("opponent_popped", chain.opponentPopped());
            if (chain.detonatorId() != null) {
                value.addProperty("detonator_id", chain.detonatorId().toString());
                value.addProperty("detonation_source", chain.detonationSource().wireName());
                value.addProperty("detonation_tick", chain.detonationTick());
            }
            chainSequences.add(value);
            if (chainSequences.size() >= 128) break;
        }
        json.add("crystal_chain_sequences", chainSequences);

        JsonArray blocks = new JsonArray();
        for (BlockKey key : touchedBlocks) {
            if (key.y <= FLOOR_Y) continue;
            Block block = center.getWorld().getBlockAt(key.x, key.y, key.z);
            if (block.getType() == Material.AIR || block.getType() == Material.BARRIER) continue;
            JsonObject value = new JsonObject();
            value.addProperty("x", key.x - center.getBlockX());
            value.addProperty("y", key.y - FLOOR_Y);
            value.addProperty("z", key.z - center.getBlockZ());
            value.addProperty("type", block.getType().name().toLowerCase());
            blocks.add(value);
            if (blocks.size() >= 256) break;
        }
        json.add("blocks", blocks);
        return json;
    }

    /** Counts presently placeable obsidian bases, including entity clearance. */
    private int countValidCrystalPads() {
        if (!mode.hasCrystalLayout()) return 0;
        List<Location> occupyingEntities = new ArrayList<Location>();
        for (Entity entity : center.getWorld().getEntities()) {
            if ((entity instanceof Player || entity instanceof EnderCrystal)
                    && contains(entity.getLocation())) {
                occupyingEntities.add(entity.getLocation());
            }
        }
        int count = 0;
        for (int x = center.getBlockX() - outerRadius + 1;
             x <= center.getBlockX() + outerRadius - 1; x++) {
            for (int z = center.getBlockZ() - outerRadius + 1;
                 z <= center.getBlockZ() + outerRadius - 1; z++) {
                for (int y = bottomBarrierY() + 1; y <= FLOOR_Y + arenaHeight - 2; y++) {
                    Block base = center.getWorld().getBlockAt(x, y, z);
                    if (base.getType() != Material.OBSIDIAN
                            || center.getWorld().getBlockAt(x, y + 1, z).getType() != Material.AIR
                            || center.getWorld().getBlockAt(x, y + 2, z).getType() != Material.AIR
                            || crystalPadOccupied(x, y, z, occupyingEntities)) continue;
                    count++;
                }
            }
        }
        return count;
    }

    private boolean crystalPadOccupied(int x, int y, int z, List<Location> entities) {
        double centerX = x + 0.5;
        double centerZ = z + 0.5;
        for (Location location : entities) {
            if (Math.abs(location.getX() - centerX) < 1.0
                    && Math.abs(location.getZ() - centerZ) < 1.0
                    && location.getY() > y && location.getY() < y + 3.0) return true;
        }
        return false;
    }

    private void flushRewards() {
        for (Player player : players()) {
            CombatStats value = stats(player);
            if (value == null || Math.abs(value.pendingReward) < 1e-12) continue;
            double clipped = rewardShaper().clipTick(value.pendingReward);
            value.pendingReward = 0;
            value.recordDelivered(clipped);
            manager.stepFeedback(this, player, clipped);
        }
    }

    private void applyDenseRewards() {
        RewardShaper shaper = rewardShaper();
        for (Player player : players()) {
            CombatStats value = stats(player);
            Player opponent = opponent(player);
            if (value == null || opponent == null || !player.isOnline() || !opponent.isOnline()) continue;
            Location current = player.getLocation();
            Location target = opponent.getLocation();
            if (!current.getWorld().equals(target.getWorld())) continue;

            if (value.hasPreviousPosition()) {
                double previousDistance = distance(value.previousX(), value.previousY(), value.previousZ(),
                        target.getX(), target.getY(), target.getZ());
                double currentDistance = distance(current.getX(), current.getY(), current.getZ(),
                        target.getX(), target.getY(), target.getZ());
                double approachReward = shaper.approach(previousDistance, currentDistance);
                if (approachReward > 0) {
                    value.approachTicks++;
                    if (executionSource(player).isAutonomous()) {
                        value.markAutonomousAction(manager.currentTick());
                    }
                }
                addReward(player, approachReward, RewardReason.MOVEMENT);
            }
            value.rememberPosition(current.getX(), current.getY(), current.getZ());

            if (shaper.isExtremePitch(current.getPitch())) {
                value.extremePitchStreak++;
                value.extremePitchTicks++;
            } else {
                value.extremePitchStreak = 0;
            }
            addReward(player, shaper.posture(current.getPitch(), value.extremePitchStreak), RewardReason.POSTURE);

            // The only gaze target used here is arena.opponent(player). Spectators
            // and fighters in parallel arenas are never queried by this reward path.
            double opponentDistance = playerDistance(player, opponent);
            double gazeDot = gazeDot(player, opponent);
            boolean clearLineOfSight = player.hasLineOfSight(opponent);
            double aimPotential = shaper.aimPotential(gazeDot, opponentDistance, clearLineOfSight);
            if (value.hasPreviousAimPotential()) {
                double alignmentReward = shaper.aimAlignment(value.previousAimPotential(), aimPotential);
                if (alignmentReward > 0) value.aimAlignmentTicks++;
                addReward(player, alignmentReward, RewardReason.AIM_ALIGNMENT);
            }
            value.rememberAimPotential(aimPotential);

            if (shaper.isLockedOn(gazeDot, opponentDistance, clearLineOfSight)) {
                value.lockOnTicks++;
                value.lockOnStreak++;
                RewardConfig config = shaper.config();
                if (value.lockOnStreak > config.lockOnGraceTicks
                        && value.claimLockOnTick(config.maxRewardedLockOnTicks)) {
                    addReward(player, shaper.lockOn(), RewardReason.LOCK_ON);
                }
            } else {
                value.lockOnStreak = 0;
            }

            // Only useful autonomous progress resets this clock. A stationary
            // fighter, random camera motion, invalid clicks, and attack spam all
            // accumulate an increasingly expensive negative learning signal.
            if (executionSource(player).isAutonomous()) {
                double inaction = shaper.inaction(
                        Math.max(0L, manager.currentTick() - value.lastAutonomousActionTick));
                if (inaction < 0) {
                    value.inactionPenaltyTicks++;
                    addReward(player, inaction, RewardReason.INACTION);
                }
                double fightTimePressure = shaper.fightTimePressure(
                        Math.max(0L, manager.currentTick() - startedTick),
                        Math.max(1L, deadlineTick - startedTick));
                if (fightTimePressure < 0) {
                    addReward(player, fightTimePressure, RewardReason.FIGHT_TIME);
                }
            }
        }
    }

    private static double playerDistance(Player player, Player opponent) {
        Location current = player.getLocation();
        Location target = opponent.getLocation();
        return current.getWorld().equals(target.getWorld()) ? current.distance(target) : Double.POSITIVE_INFINITY;
    }

    private static double gazeDot(Player player, Player opponent) {
        Location eye = player.getEyeLocation();
        Location torso = opponent.getLocation().add(0, 1.0, 0);
        if (!eye.getWorld().equals(torso.getWorld())) return -1;
        Vector toOpponent = torso.toVector().subtract(eye.toVector());
        if (toOpponent.lengthSquared() < 1.0e-12) return 1;
        return eye.getDirection().normalize().dot(toOpponent.normalize());
    }

    private static double distance(double ax, double ay, double az, double bx, double by, double bz) {
        double dx = ax - bx;
        double dy = ay - by;
        double dz = az - bz;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
    }

    private static String rewardBlockKey(Block block) {
        return block.getWorld().getName() + ':' + block.getX() + ':' + block.getY() + ':' + block.getZ();
    }

    private boolean isUsefulTacticalObsidianPlacement(Player builder, Block block) {
        return isUsefulTacticalSite(builder, block, Material.OBSIDIAN);
    }

    private boolean isUsefulTacticalSite(Player builder, Block block, Material requiredMaterial) {
        if (!mode.hasCrystalLayout() || builder == null || block == null
                || block.getType() != requiredMaterial || block.getY() != FLOOR_Y + 1) return false;
        Player assignedOpponent = opponent(builder);
        if (assignedOpponent == null || !assignedOpponent.isOnline()
                || manager.arenaFor(assignedOpponent) != this
                || !builder.getWorld().equals(center.getWorld())
                || !assignedOpponent.getWorld().equals(center.getWorld())) return false;
        World world = block.getWorld();
        if (world.getBlockAt(block.getX(), block.getY() + 1, block.getZ()).getType() != Material.AIR
                || world.getBlockAt(block.getX(), block.getY() + 2, block.getZ()).getType() != Material.AIR) {
            return false;
        }
        Location builderLocation = builder.getLocation();
        Location opponentLocation = assignedOpponent.getLocation();
        return ArenaGeometry.usefulTacticalObsidianGeometry(
                outerRadius, block.getX() - center.getBlockX(),
                block.getZ() - center.getBlockZ(),
                builderLocation.getX() - center.getBlockX(),
                builderLocation.getZ() - center.getBlockZ(),
                opponentLocation.getX() - center.getBlockX(),
                opponentLocation.getZ() - center.getBlockZ());
    }

    private void prepareBase() {
        if (basePrepared) return;
        World world = center.getWorld();
        // Radius 20 was the previous local default. Clear that bounded legacy
        // shell so an existing training world visibly becomes the new 15x15
        // curriculum platform instead of retaining an unreachable outer ring.
        int cleanupRadius = Math.max(20, maximumOuterRadius);
        for (int x = -cleanupRadius; x <= cleanupRadius; x++) {
            for (int z = -cleanupRadius; z <= cleanupRadius; z++) {
                int absoluteX = center.getBlockX() + x;
                int absoluteZ = center.getBlockZ() + z;
                boolean outsideArena = Math.abs(x) > outerRadius || Math.abs(z) > outerRadius;
                boolean boundary = Math.abs(x) == outerRadius || Math.abs(z) == outerRadius;
                for (int y = bottomBarrierY(); y <= FLOOR_Y + arenaHeight; y++) {
                    Material material;
                    if (outsideArena) material = Material.AIR;
                    else if (boundary || y == bottomBarrierY()) material = Material.BARRIER;
                    else if (y <= FLOOR_Y) material = Material.STONE;
                    else material = Material.AIR;
                    world.getBlockAt(absoluteX, y, absoluteZ).setType(material, false);
                }
            }
        }
        basePrepared = true;
    }

    private void resetTouchedBlocks() {
        World world = center.getWorld();
        for (BlockKey key : touchedBlocks) {
            world.getBlockAt(key.x, key.y, key.z).setType(baseMaterial(key), false);
        }
        touchedBlocks.clear();
    }

    private Material baseMaterial(BlockKey key) {
        int relativeX = key.x - center.getBlockX();
        int relativeZ = key.z - center.getBlockZ();
        if (Math.abs(relativeX) > outerRadius || Math.abs(relativeZ) > outerRadius
                || key.y < bottomBarrierY() || key.y > FLOOR_Y + arenaHeight) return Material.AIR;
        if (Math.abs(relativeX) == outerRadius || Math.abs(relativeZ) == outerRadius
                || key.y == bottomBarrierY()) return Material.BARRIER;
        return key.y <= FLOOR_Y ? Material.STONE : Material.AIR;
    }

    private int bottomBarrierY() {
        return FLOOR_Y - floorDepth;
    }

    private void configureRadius(int requestedRadius) {
        int selectedRadius = Math.max(ArenaGeometry.MINIMUM_RADIUS,
                Math.min(maximumOuterRadius, requestedRadius));
        if (outerRadius != selectedRadius) {
            outerRadius = selectedRadius;
            basePrepared = false;
        }
        spawnMinSeparation = ArenaGeometry.clampSpawnMinimum(outerRadius, configuredSpawnMinSeparation);
        spawnMaxSeparation = ArenaGeometry.clampSpawnMaximum(
                outerRadius, spawnMinSeparation, configuredSpawnMaxSeparation);
    }

    private int layoutSize(Random random, ArenaMode selectedMode) {
        int maximum = outerRadius * 2 - 3;
        // Keep the close-combat curriculum's useful layout compact even if a
        // larger outer shell is selected later for digging experiments.
        return Math.min(21, maximum);
    }

    private void clearEntities() {
        int cleanupRadius = Math.max(20, maximumOuterRadius);
        for (Entity entity : center.getWorld().getEntities()) {
            Location location = entity.getLocation();
            boolean inLegacyShell = Math.abs(location.getBlockX() - center.getBlockX()) <= cleanupRadius
                    && Math.abs(location.getBlockZ() - center.getBlockZ()) <= cleanupRadius
                    && location.getY() >= bottomBarrierY() && location.getY() <= FLOOR_Y + arenaHeight;
            if (!(entity instanceof Player) && inLegacyShell) entity.remove();
        }
    }

    private void generateLayout(Random random, int size) {
        int half = size / 2;
        int holes = random.nextInt(2);
        for (int index = 0; index < holes; index++) {
            setLayoutBlock(randomOffset(random, half), FLOOR_Y, randomOffset(random, half), Material.AIR);
        }
        int structures = random.nextInt(4);
        for (int index = 0; index < structures; index++) {
            int x = randomOffset(random, half);
            int z = randomOffset(random, half);
            int height = 1 + random.nextInt(2);
            if (random.nextBoolean()) {
                for (int y = 1; y <= height; y++) setLayoutBlock(x, FLOOR_Y + y, z, Material.STONE);
            } else {
                int length = 2 + random.nextInt(2);
                boolean alongX = random.nextBoolean();
                for (int offset = 0; offset < length; offset++) {
                    for (int y = 1; y <= height; y++) {
                        setLayoutBlock(x + (alongX ? offset : 0), FLOOR_Y + y,
                                z + (alongX ? 0 : offset), Material.STONE);
                    }
                }
            }
        }
        // Random pads add terrain variety. The episode-seeded reachable bases
        // are a separate final pass after spawn/lane clearing, so they cannot vanish.
        int obsidian = random.nextInt(7);
        for (int index = 0; index < obsidian; index++) {
            placeObsidianPad(randomOffset(random, half), randomOffset(random, half));
        }
    }

    private void placeReachableCrystalPads(Location[] spawns) {
        int firstX = spawns[0].getBlockX() - center.getBlockX();
        int firstZ = spawns[0].getBlockZ() - center.getBlockZ();
        int secondX = spawns[1].getBlockX() - center.getBlockX();
        int secondZ = spawns[1].getBlockZ() - center.getBlockZ();
        for (int[] pad : ArenaGeometry.reachableCrystalPadOffsets(
                outerRadius, firstX, firstZ, secondX, secondZ, seed)) {
            placeObsidianPad(pad[0], pad[1]);
        }
    }

    private double fighterDistance() {
        if (first == null || second == null || !first.isOnline() || !second.isOnline()) {
            return Double.POSITIVE_INFINITY;
        }
        Location firstLocation = first.getLocation();
        Location secondLocation = second.getLocation();
        if (!firstLocation.getWorld().equals(secondLocation.getWorld())) return Double.POSITIVE_INFINITY;
        return firstLocation.distance(secondLocation);
    }

    private void placeObsidianPad(int relativeX, int relativeZ) {
        // A floor-level base is safe beneath a spawn and matches 1.12 crystal
        // clearance: the two blocks immediately above must both be empty.
        setLayoutBlock(relativeX, FLOOR_Y, relativeZ, Material.OBSIDIAN);
        setLayoutBlock(relativeX, FLOOR_Y + 1, relativeZ, Material.AIR);
        setLayoutBlock(relativeX, FLOOR_Y + 2, relativeZ, Material.AIR);
    }

    private int randomOffset(Random random, int half) {
        // Radius 3 intentionally produces a compact three-block layout. There
        // is only one safe interior offset in that case.
        int span = half * 2 - 3;
        return span <= 0 ? 0 : random.nextInt(span) - half + 2;
    }

    private void setLayoutBlock(int relativeX, int y, int relativeZ, Material material) {
        Block block = center.getWorld().getBlockAt(center.getBlockX() + relativeX, y, center.getBlockZ() + relativeZ);
        touchedBlocks.add(new BlockKey(block.getX(), block.getY(), block.getZ()));
        block.setType(material, false);
    }

    private Location[] spawnLocations(Random random) {
        // ArenaManager already clamps this interval to the usable interior.
        // Do not derive another cap from layoutSize: sword rounds have no
        // generated layout, and doing so collapsed the 7x7 curriculum's 2..5
        // configured range to a constant two-block spawn.
        int separation = spawnMinSeparation
                + random.nextInt(spawnMaxSeparation - spawnMinSeparation + 1);
        double angle = random.nextDouble() * Math.PI * 2;
        double dx = Math.cos(angle) * separation / 2.0;
        double dz = Math.sin(angle) * separation / 2.0;
        Location a = center.clone().add(dx, FLOOR_Y + 1.01 - center.getY(), dz);
        Location b = center.clone().add(-dx, FLOOR_Y + 1.01 - center.getY(), -dz);
        a.setYaw(yawToward(a, b) + yawJitter(random));
        b.setYaw(yawToward(b, a) + yawJitter(random));
        a.setPitch(0.0F);
        b.setPitch(0.0F);
        return new Location[]{a, b};
    }

    private float yawToward(Location from, Location to) {
        double deltaX = to.getX() - from.getX();
        double deltaZ = to.getZ() - from.getZ();
        return (float) Math.toDegrees(Math.atan2(-deltaX, deltaZ));
    }

    private float yawJitter(Random random) {
        return (float) ((random.nextDouble() * 2.0 - 1.0) * spawnYawJitter);
    }

    private void clearSpawn(Location spawn) {
        int relativeX = spawn.getBlockX() - center.getBlockX();
        int relativeZ = spawn.getBlockZ() - center.getBlockZ();
        setLayoutBlock(relativeX, FLOOR_Y, relativeZ, Material.STONE);
        setLayoutBlock(relativeX, FLOOR_Y + 1, relativeZ, Material.AIR);
        setLayoutBlock(relativeX, FLOOR_Y + 2, relativeZ, Material.AIR);
    }

    private void clearSpawnLane(Location firstSpawn, Location secondSpawn) {
        double deltaX = secondSpawn.getX() - firstSpawn.getX();
        double deltaZ = secondSpawn.getZ() - firstSpawn.getZ();
        int samples = Math.max(1, (int) Math.ceil(Math.max(Math.abs(deltaX), Math.abs(deltaZ)) * 4.0));
        for (int index = 0; index <= samples; index++) {
            double fraction = index / (double) samples;
            int absoluteX = (int) Math.floor(firstSpawn.getX() + deltaX * fraction);
            int absoluteZ = (int) Math.floor(firstSpawn.getZ() + deltaZ * fraction);
            int relativeX = absoluteX - center.getBlockX();
            int relativeZ = absoluteZ - center.getBlockZ();
            setLayoutBlock(relativeX, FLOOR_Y, relativeZ, Material.STONE);
            setLayoutBlock(relativeX, FLOOR_Y + 1, relativeZ, Material.AIR);
            setLayoutBlock(relativeX, FLOOR_Y + 2, relativeZ, Material.AIR);
        }
    }

    private void preparePlayer(Player player, Location spawn,
                               KitCurriculum.KitSpec kit, int obsidianSupply) {
        player.setGameMode(GameMode.SURVIVAL);
        PlayerInventory inventory = player.getInventory();
        inventory.clear();
        inventory.setArmorContents(new ItemStack[4]);
        inventory.setItemInOffHand(null);
        for (PotionEffect effect : player.getActivePotionEffects()) player.removePotionEffect(effect.getType());
        player.setHealth(player.getMaxHealth());
        setAbsorption(player, 0.0);
        player.setFoodLevel(20);
        player.setSaturation(5);
        player.setFireTicks(0);
        player.setFallDistance(0);
        player.setVelocity(new Vector(0, 0, 0));
        equipKit(inventory, mode, obsidianSupply, kit);
        player.teleport(spawn);
    }

    private void equipKit(PlayerInventory inventory, ArenaMode selectedMode,
                          int obsidianSupply, KitCurriculum.KitSpec kit) {
        ItemStack sword = new ItemStack(Material.DIAMOND_SWORD);
        sword.addUnsafeEnchantment(Enchantment.DAMAGE_ALL, 5);
        inventory.setItem(0, sword);
        if (selectedMode != ArenaMode.SWORD) {
            ItemStack pickaxe = new ItemStack(Material.DIAMOND_PICKAXE);
            pickaxe.addUnsafeEnchantment(Enchantment.DIG_SPEED, 5);
            inventory.setItem(1, pickaxe);
            inventory.setItem(2, new ItemStack(Material.OBSIDIAN,
                    Math.max(16, Math.min(64, obsidianSupply))));
            inventory.setItem(3, new ItemStack(Material.END_CRYSTAL, 64));
            if (kit.goldenApples() > 0) {
                inventory.setItem(4, new ItemStack(
                        Material.GOLDEN_APPLE, kit.goldenApples(), (short) 0));
            }
            for (int spareIndex = 0; spareIndex < kit.spareTotems(); spareIndex++) {
                inventory.setItem(5 + spareIndex, new ItemStack(Material.TOTEM, 1));
            }
            if (kit.hasOffhandTotem()) {
                inventory.setItemInOffHand(new ItemStack(Material.TOTEM, 1));
            }
        }
        int protectionLevel = kit.protectionLevel();
        inventory.setHelmet(armor(Material.DIAMOND_HELMET, protectionLevel));
        inventory.setChestplate(armor(Material.DIAMOND_CHESTPLATE, protectionLevel));
        inventory.setLeggings(armor(Material.DIAMOND_LEGGINGS, protectionLevel));
        inventory.setBoots(armor(Material.DIAMOND_BOOTS, protectionLevel));
        inventory.setHeldItemSlot(0);
    }

    private ItemStack armor(Material material, int protectionLevel) {
        ItemStack item = new ItemStack(material);
        item.addUnsafeEnchantment(Enchantment.PROTECTION_ENVIRONMENTAL, protectionLevel);
        item.addUnsafeEnchantment(Enchantment.DURABILITY, 3);
        return item;
    }

    private double absorption(Player player) {
        try {
            Object result = player.getClass().getMethod("getAbsorptionAmount").invoke(player);
            return result instanceof Number ? ((Number) result).doubleValue() : 0.0;
        } catch (ReflectiveOperationException ignored) {
            return 0.0;
        }
    }

    private void setAbsorption(Player player, double amount) {
        try {
            player.getClass().getMethod("setAbsorptionAmount", double.class).invoke(player, amount);
        } catch (ReflectiveOperationException ignored) {
            // Bukkit 1.12 does not expose this method; removing potion effects is the portable reset path.
        }
    }

    private static final class BlockKey {
        private final int x;
        private final int y;
        private final int z;

        private BlockKey(int x, int y, int z) { this.x = x; this.y = y; this.z = z; }

        @Override
        public boolean equals(Object other) {
            if (!(other instanceof BlockKey)) return false;
            BlockKey key = (BlockKey) other;
            return x == key.x && y == key.y && z == key.z;
        }

        @Override
        public int hashCode() { return (x * 31 + y) * 31 + z; }
    }

    private static final class ExecutionMarker {
        private final ExecutionSource source;
        private final long untilTick;

        private ExecutionMarker(ExecutionSource source, long untilTick) {
            this.source = source;
            this.untilTick = untilTick;
        }
    }

    private static final class DamageAttribution {
        private final UUID attackerId;
        private final ExecutionSource source;
        private final boolean crystalDamage;
        private final UUID crystalId;
        private final long tick;

        private DamageAttribution(UUID attackerId, ExecutionSource source,
                                  boolean crystalDamage, UUID crystalId, long tick) {
            this.attackerId = attackerId;
            this.source = source;
            this.crystalDamage = crystalDamage;
            this.crystalId = crystalId;
            this.tick = tick;
        }
    }

    private static final class PolicyObsidianPlacement {
        private final UUID ownerId;
        private final long tick;

        private PolicyObsidianPlacement(UUID ownerId, long tick) {
            this.ownerId = ownerId;
            this.tick = tick;
        }
    }

    private static final class PolicyTerrainClearance {
        private final UUID ownerId;
        private final long tick;

        private PolicyTerrainClearance(UUID ownerId, long tick) {
            this.ownerId = ownerId;
            this.tick = tick;
        }
    }
}
