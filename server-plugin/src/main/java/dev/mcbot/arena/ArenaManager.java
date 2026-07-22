package dev.mcbot.arena;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.entity.Player;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.Deque;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;

public final class ArenaManager {
    static final int DEFAULT_SNAPSHOT_INTERVAL_TICKS = 10;

    private final MCAIPlugin plugin;
    private final List<Arena> arenas = new ArrayList<Arena>();
    private final Map<UUID, Arena> byPlayer = new HashMap<UUID, Arena>();
    private final Map<String, String> agentByUsername = new HashMap<String, String>();
    private final Map<UUID, Long> cooldownUntil = new HashMap<UUID, Long>();
    private final Deque<Double> tickIntervalsMs = new ArrayDeque<Double>();
    private final Random seedSource = new Random(7);
    private ControlServer control;
    private ArenaMode defaultMode;
    private int maxConcurrentPairs;
    private final int timeoutSeconds;
    private final String botPrefix;
    private final boolean autoPair;
    private final RewardConfig baseRewardConfig;
    private RewardProfile rewardProfile;
    private final List<TrainingLane> trainingLanes = new ArrayList<TrainingLane>();
    private final Map<String, TrainingLane> laneById = new LinkedHashMap<String, TrainingLane>();
    private final Map<Arena, TrainingLane> laneByArena = new HashMap<Arena, TrainingLane>();
    private final int snapshotIntervalTicks;
    private double shapingScale;
    private long tick;
    private long previousTickNanos;
    private boolean paused;

    public ArenaManager(MCAIPlugin plugin, World world) {
        this.plugin = plugin;
        this.maxConcurrentPairs = plugin.getConfig().getInt("max-concurrent-pairs", 4);
        this.timeoutSeconds = plugin.getConfig().getInt("match-timeout-seconds", 35);
        this.botPrefix = plugin.getConfig().getString("bot-name-prefix", "MCAI_");
        this.autoPair = plugin.getConfig().getBoolean("auto-pair-bots", true);
        this.defaultMode = ArenaMode.parse(plugin.getConfig().getString("default-mode", "combined"));
        this.shapingScale = plugin.getConfig().getDouble("shaping-scale", 1.0);
        this.snapshotIntervalTicks = Math.max(1, plugin.getConfig().getInt(
                "snapshot-interval-ticks", DEFAULT_SNAPSHOT_INTERVAL_TICKS));
        RewardConfig defaults = RewardConfig.defaults();
        this.baseRewardConfig = new RewardConfig(
                plugin.getConfig().getDouble("reward.max-per-tick", defaults.maxPerTick),
                plugin.getConfig().getDouble("reward.movement-per-block", defaults.movementPerBlock),
                plugin.getConfig().getDouble("reward.max-movement-delta", defaults.maxMovementDelta),
                plugin.getConfig().getDouble("reward.preferred-distance", defaults.preferredDistance),
                plugin.getConfig().getDouble("reward.extreme-pitch-degrees", defaults.extremePitchDegrees),
                plugin.getConfig().getInt("reward.extreme-pitch-grace-ticks", defaults.extremePitchGraceTicks),
                plugin.getConfig().getDouble("reward.extreme-pitch-penalty-per-tick", defaults.extremePitchPenaltyPerTick),
                plugin.getConfig().getDouble("reward.aim-alignment-per-potential", defaults.aimAlignmentPerPotential),
                plugin.getConfig().getDouble("reward.max-aim-alignment-delta", defaults.maxAimAlignmentDelta),
                plugin.getConfig().getDouble("reward.aim-alignment-range", defaults.aimAlignmentRange),
                plugin.getConfig().getDouble("reward.lock-on-dot", defaults.lockOnDot),
                plugin.getConfig().getDouble("reward.lock-on-range", defaults.lockOnRange),
                plugin.getConfig().getInt("reward.lock-on-grace-ticks", defaults.lockOnGraceTicks),
                plugin.getConfig().getDouble("reward.lock-on-per-tick", defaults.lockOnPerTick),
                plugin.getConfig().getInt("reward.max-rewarded-lock-on-ticks", defaults.maxRewardedLockOnTicks),
                plugin.getConfig().getDouble("reward.attack-aim-dot", defaults.attackAimDot),
                plugin.getConfig().getDouble("reward.attack-reach", defaults.attackReach),
                plugin.getConfig().getInt("reward.attack-cooldown-ticks", defaults.attackCooldownTicks),
                plugin.getConfig().getDouble("reward.valid-attack-swing", defaults.validAttackSwing),
                plugin.getConfig().getInt("reward.max-rewarded-attack-swings", defaults.maxRewardedAttackSwings),
                plugin.getConfig().getDouble("reward.missed-attack-swing", defaults.missedAttackSwing),
                plugin.getConfig().getDouble("reward.spam-attack-swing", defaults.spamAttackSwing),
                plugin.getConfig().getInt("reward.max-penalized-attack-swings", defaults.maxPenalizedAttackSwings),
                plugin.getConfig().getDouble("reward.successful-hit", defaults.successfulHit),
                plugin.getConfig().getInt("reward.max-rewarded-hits", defaults.maxRewardedHits),
                plugin.getConfig().getDouble("reward.damage-dealt-per-health", defaults.damageDealtPerHealth),
                plugin.getConfig().getDouble("reward.damage-taken-per-health", defaults.damageTakenPerHealth),
                plugin.getConfig().getDouble("reward.forced-totem", defaults.forcedTotem),
                plugin.getConfig().getDouble("reward.own-totem", defaults.ownTotem),
                plugin.getConfig().getDouble("reward.policy-kill", defaults.policyKill),
                plugin.getConfig().getDouble("reward.policy-kill-speed-bonus", defaults.policyKillSpeedBonus),
                plugin.getConfig().getDouble("reward.death-loss", defaults.deathLoss),
                plugin.getConfig().getDouble("reward.timeout-loss", defaults.timeoutLoss),
                plugin.getConfig().getDouble("reward.disengaged-loss", defaults.disengagedLoss),
                plugin.getConfig().getDouble("reward.double-ko-loss", defaults.doubleKoLoss),
                plugin.getConfig().getDouble("reward.own-crystal-self-hit", defaults.ownCrystalSelfHit),
                plugin.getConfig().getDouble("reward.own-crystal-self-damage-per-health",
                        defaults.ownCrystalSelfDamagePerHealth),
                plugin.getConfig().getDouble("reward.obsidian-placed", defaults.obsidianPlaced),
                plugin.getConfig().getInt("reward.max-rewarded-obsidian", defaults.maxRewardedObsidian),
                plugin.getConfig().getDouble("reward.obsidian-combo", defaults.obsidianCombo),
                plugin.getConfig().getInt("reward.max-rewarded-obsidian-combos", defaults.maxRewardedObsidianCombos),
                plugin.getConfig().getInt("reward.tactical-mine-place-max-ticks", defaults.tacticalMinePlaceMaxTicks),
                plugin.getConfig().getDouble("reward.tactical-mine-place", defaults.tacticalMinePlace),
                plugin.getConfig().getInt("reward.max-rewarded-tactical-mine-place", defaults.maxRewardedTacticalMinePlace),
                plugin.getConfig().getDouble("reward.crystal-placed", defaults.crystalPlaced),
                plugin.getConfig().getInt("reward.max-rewarded-crystal-placements", defaults.maxRewardedCrystalPlacements),
                plugin.getConfig().getDouble("reward.crystal-destroyed", defaults.crystalDestroyed),
                plugin.getConfig().getInt("reward.max-rewarded-crystal-destructions", defaults.maxRewardedCrystalDestructions),
                plugin.getConfig().getDouble("reward.crystal-exploded", defaults.crystalExploded),
                plugin.getConfig().getInt("reward.max-rewarded-crystal-explosions", defaults.maxRewardedCrystalExplosions),
                plugin.getConfig().getInt("reward.crystal-combo-max-ticks", defaults.crystalComboMaxTicks),
                plugin.getConfig().getDouble("reward.crystal-combo-damage", defaults.crystalComboDamage),
                plugin.getConfig().getDouble("reward.crystal-combo-pop", defaults.crystalComboPop),
                plugin.getConfig().getInt("reward.max-rewarded-crystal-combos", defaults.maxRewardedCrystalCombos),
                plugin.getConfig().getDouble("reward.useful-block-mined", defaults.usefulBlockMined),
                plugin.getConfig().getInt("reward.max-rewarded-mined-blocks", defaults.maxRewardedMinedBlocks),
                plugin.getConfig().getDouble("reward.invalid-interaction", defaults.invalidInteraction),
                plugin.getConfig().getInt("reward.inaction-grace-ticks", defaults.inactionGraceTicks),
                plugin.getConfig().getDouble("reward.inaction-penalty-per-tick", defaults.inactionPenaltyPerTick),
                plugin.getConfig().getDouble("reward.max-inaction-penalty-per-tick", defaults.maxInactionPenaltyPerTick),
                plugin.getConfig().getInt("reward.positive-reward-decay-start-ticks", defaults.positiveRewardDecayStartTicks),
                plugin.getConfig().getInt("reward.positive-reward-decay-end-ticks", defaults.positiveRewardDecayEndTicks),
                plugin.getConfig().getDouble("reward.minimum-positive-reward-multiplier",
                        defaults.minimumPositiveRewardMultiplier),
                plugin.getConfig().getInt("reward.fight-time-pressure-start-ticks",
                        defaults.fightTimePressureStartTicks),
                plugin.getConfig().getDouble("reward.fight-time-pressure-per-tick",
                        defaults.fightTimePressurePerTick),
                plugin.getConfig().getDouble("reward.max-fight-time-pressure-per-tick",
                        defaults.maxFightTimePressurePerTick));
        this.rewardProfile = RewardProfile.initial(baseRewardConfig);
        int spacing = plugin.getConfig().getInt("arena-spacing", 96);
        int fallbackArenaRadius = plugin.getConfig().getInt("arena-radius", 5);
        List<Integer> progressiveRadii = configuredOrDefault(
                plugin.getConfig().getIntegerList("curriculum.arena-radius-stages"),
                Arrays.asList(5, 6, 7, 8));
        List<Integer> progressiveThresholds = configuredOrDefault(
                plugin.getConfig().getIntegerList("curriculum.arena-radius-episode-thresholds"),
                Arrays.asList(0, 64, 256, 1024));
        int performanceWindow = plugin.getConfig().getInt(
                "curriculum.arena-radius-performance-window", 32);
        double advanceEngagementRate = plugin.getConfig().getDouble(
                "curriculum.arena-radius-advance-engagement-rate", 0.75);
        double advanceNonTimeoutRate = plugin.getConfig().getDouble(
                "curriculum.arena-radius-advance-non-timeout-rate", 0.10);
        double regressEngagementRate = plugin.getConfig().getDouble(
                "curriculum.arena-radius-regress-engagement-rate", 0.50);
        double regressNonTimeoutRate = plugin.getConfig().getDouble(
                "curriculum.arena-radius-regress-non-timeout-rate", 0.05);
        int stageChangeCooldown = plugin.getConfig().getInt(
                "curriculum.arena-radius-stage-change-cooldown-episodes", 32);

        addLane(new TrainingLane("sword_retention", ArenaMode.SWORD,
                curriculum(spacing, 5, Arrays.asList(5), Arrays.asList(0), performanceWindow,
                        advanceEngagementRate, advanceNonTimeoutRate, regressEngagementRate,
                        regressNonTimeoutRate, stageChangeCooldown)));
        addLane(new TrainingLane("crystal_retention", ArenaMode.CRYSTAL,
                curriculum(spacing, 5, Arrays.asList(5, 6), Arrays.asList(0, 64), performanceWindow,
                        advanceEngagementRate, advanceNonTimeoutRate, regressEngagementRate,
                        regressNonTimeoutRate, stageChangeCooldown)));
        addLane(new TrainingLane("combined", ArenaMode.COMBINED,
                curriculum(spacing, fallbackArenaRadius, progressiveRadii, progressiveThresholds,
                        performanceWindow, advanceEngagementRate, advanceNonTimeoutRate,
                        regressEngagementRate, regressNonTimeoutRate, stageChangeCooldown)));
        addLane(new TrainingLane("terrain", ArenaMode.TERRAIN,
                curriculum(spacing, fallbackArenaRadius, progressiveRadii, progressiveThresholds,
                        performanceWindow, advanceEngagementRate, advanceNonTimeoutRate,
                        regressEngagementRate, regressNonTimeoutRate, stageChangeCooldown)));

        int arenaRadius = 5;
        int maximumArenaRadius = 5;
        for (TrainingLane lane : trainingLanes) {
            maximumArenaRadius = Math.max(maximumArenaRadius, lane.curriculum().maximumRadius());
        }
        int arenaDepth = Math.max(12, Math.min(32, plugin.getConfig().getInt("arena-depth", 12)));
        int arenaHeight = Math.max(8, Math.min(24, plugin.getConfig().getInt("arena-height", 12)));
        int spawnMinSeparation = plugin.getConfig().getInt("curriculum.spawn-min-separation", 2);
        int spawnMaxSeparation = plugin.getConfig().getInt("curriculum.spawn-max-separation", 5);
        double spawnYawJitter = Math.max(0.0, Math.min(90.0,
                plugin.getConfig().getDouble("curriculum.spawn-yaw-jitter-degrees", 15.0)));
        for (int index = 0; index < 16; index++) {
            int x = (index % 4) * spacing;
            int z = (index / 4) * spacing;
            Arena arena = new Arena(this, "arena-" + (index + 1), new Location(world, x, 64, z),
                    arenaRadius, maximumArenaRadius, arenaDepth, arenaHeight, spawnMinSeparation, spawnMaxSeparation,
                    spawnYawJitter);
            arenas.add(arena);
            laneByArena.put(arena, trainingLanes.get(laneIndexForArena(index, trainingLanes.size())));
        }
    }

    public void setControl(ControlServer control) {
        this.control = control;
    }

    public void tick() {
        tick++;
        long now = System.nanoTime();
        if (previousTickNanos != 0) {
            tickIntervalsMs.addLast((now - previousTickNanos) / 1_000_000.0);
            while (tickIntervalsMs.size() > 1200) tickIntervalsMs.removeFirst();
        }
        previousTickNanos = now;
        for (Arena arena : arenas) arena.tick(tick);
        if (autoPair) pairWaitingBots();
        if (isSnapshotTick(tick, snapshotIntervalTicks)) {
            for (Arena arena : activeArenas()) emit("arena_snapshot", arena, null, arena.snapshotJson(tick));
        }
    }

    static boolean isSnapshotTick(long tick, int intervalTicks) {
        return intervalTicks > 0 && tick % intervalTicks == 0;
    }

    static int laneIndexForArena(int zeroBasedArenaIndex, int laneCount) {
        if (zeroBasedArenaIndex < 0 || laneCount < 1) throw new IllegalArgumentException("invalid lane assignment");
        return zeroBasedArenaIndex % laneCount;
    }

    public Arena arenaFor(Player player) {
        return player == null ? null : byPlayer.get(player.getUniqueId());
    }

    public Arena byId(String id) {
        for (Arena arena : arenas) if (arena.getId().equalsIgnoreCase(id)) return arena;
        return null;
    }

    String agentIdFor(Player player) {
        return agentByUsername.getOrDefault(player.getName().toLowerCase(), player.getName());
    }

    public List<Arena> activeArenas() {
        List<Arena> active = new ArrayList<Arena>();
        for (Arena arena : arenas) if (arena.isActive()) active.add(arena);
        return active;
    }

    public double getShapingScale() {
        return shapingScale;
    }

    public RewardShaper getRewardShaper() {
        return rewardProfile.shaper();
    }

    public RewardProfile getRewardProfile() {
        return rewardProfile;
    }

    long currentTick() {
        return tick;
    }

    public void recycle(Arena arena, Collection<Player> players) {
        for (Player player : players) {
            byPlayer.remove(player.getUniqueId());
            cooldownUntil.put(player.getUniqueId(), tick + 20);
        }
    }

    public void matchStarted(Arena arena, Player player, int actionDelay, int observationDelay) {
        JsonObject payload = basePayload(arena);
        payload.addProperty("action_delay_ticks", actionDelay);
        payload.addProperty("observation_delay_ticks", observationDelay);
        Player opponent = arena.opponent(player);
        if (opponent != null) payload.addProperty("opponent_username", opponent.getName());
        emit("match_started", arena, player, payload);
    }

    public void stepFeedback(Arena arena, Player player, double reward) {
        JsonObject payload = basePayload(arena);
        payload.addProperty("reward", reward);
        payload.add("stats", arena.statsJson(player));
        emit("step_feedback", arena, player, payload);
    }

    public void matchEnded(Arena arena, Player player, double reward, boolean truncated, String reason) {
        JsonObject payload = basePayload(arena);
        payload.addProperty("reward", reward);
        payload.addProperty("truncated", truncated);
        payload.addProperty("reason", reason);
        payload.addProperty("outcome", reward > 0 ? "win" : (reward < -0.05 ? "loss" : "draw"));
        payload.addProperty("terminal_source", arena.getTerminalSource(player));
        payload.addProperty("policy_owned_kill", arena.isPolicyOwnedTerminalKill());
        payload.addProperty("match_ticks", tick - arena.getStartedTick());
        payload.add("stats", arena.statsJson(player));
        emit("match_ended", arena, player, payload);
    }

    void curriculumEpisodeCompleted(Arena arena, boolean eligible, String reason) {
        if (!eligible || !("death".equals(reason) || "timeout".equals(reason)
                || "disengaged".equals(reason))) return;
        TrainingLane lane = laneById.get(arena.getLane());
        if (lane == null) return;
        ArenaSizeCurriculum curriculum = lane.curriculum();
        int previousStage = curriculum.stageNumber();
        ArenaSizeCurriculum.StageChange change = curriculum.recordCompletedEpisode(
                arena.autonomousEngagement(), "death".equals(reason));
        if (change == ArenaSizeCurriculum.StageChange.NONE) return;

        JsonObject payload = new JsonObject();
        payload.addProperty("lane", lane.id());
        payload.addProperty("change", change.name().toLowerCase());
        payload.addProperty("previous_stage", previousStage);
        payload.addProperty("curriculum_stage", curriculum.stageNumber());
        payload.addProperty("curriculum_stage_count", curriculum.stageCount());
        payload.addProperty("arena_radius", curriculum.currentRadius());
        payload.addProperty("arena_size", ArenaGeometry.playableInteriorDiameter(
                curriculum.currentRadius()));
        payload.addProperty("completed_episodes", curriculum.completedEpisodes());
        payload.addProperty("recent_engagement_rate", curriculum.recentEngagementRate());
        payload.addProperty("recent_non_timeout_rate", curriculum.recentNonTimeoutRate());
        emit("curriculum_stage_changed", arena, null, payload);
    }

    public JsonObject handleCommand(String command, JsonObject payload) {
        if ("ping".equals(command)) {
            JsonObject result = new JsonObject();
            result.addProperty("pong", true);
            return result;
        }
        if ("status".equals(command)) return status();
        if ("register_agent".equals(command)) {
            String username = string(payload, "username", "");
            String agentId = string(payload, "agent_id", username);
            if (username.isEmpty()) throw new IllegalArgumentException("username is required");
            agentByUsername.put(username.toLowerCase(), agentId);

            // Mineflayer begins connecting as soon as BotAgent is constructed, so
            // the arena can pair a bot before the worker's control socket finishes
            // registering it. Replay the assignment when that race occurs; without
            // this, the bot never learns its episode or assigned opponent.
            Player registered = Bukkit.getPlayerExact(username);
            Arena activeArena = arenaFor(registered);
            if (registered != null && activeArena != null && activeArena.isActive()) {
                int[] delays = delaysFor(activeArena.getSeed(), activeArena.getMode(),
                        activeArena.getCurriculumStage());
                matchStarted(activeArena, registered, delays[0], delays[1]);
            }
            JsonObject result = new JsonObject();
            result.addProperty("registered", username);
            return result;
        }
        if ("mark_execution_source".equals(command)) {
            String username = string(payload, "username", "");
            String episodeId = string(payload, "episode_id", "");
            ExecutionSource source = ExecutionSource.parse(string(payload, "source", ""));
            if (username.isEmpty() || episodeId.isEmpty()) {
                throw new IllegalArgumentException("username and episode_id are required");
            }
            Player player = Bukkit.getPlayerExact(username);
            Arena arena = arenaFor(player);
            int durationTicks = Math.max(1, Math.min(4, integer(payload, "duration_ticks", 4)));
            boolean marked = arena != null && arena.markExecutionSource(
                    player, episodeId, source, durationTicks);
            JsonObject result = new JsonObject();
            result.addProperty("marked", marked);
            result.addProperty("source", source.wireName());
            result.addProperty("duration_ticks", source == ExecutionSource.POLICY ? 0 : durationTicks);
            return result;
        }
        if ("set_max_pairs".equals(command)) {
            maxConcurrentPairs = Math.max(1, Math.min(arenas.size(), integer(payload, "pairs", maxConcurrentPairs)));
            JsonObject result = new JsonObject();
            result.addProperty("max_concurrent_pairs", maxConcurrentPairs);
            return result;
        }
        if ("set_mode".equals(command)) {
            defaultMode = ArenaMode.parse(string(payload, "mode", defaultMode.name()));
            JsonObject result = new JsonObject();
            result.addProperty("mode", defaultMode.name().toLowerCase());
            return result;
        }
        if ("set_shaping_scale".equals(command)) {
            shapingScale = Math.max(0, Math.min(1, number(payload, "scale", shapingScale)));
            JsonObject result = new JsonObject();
            result.addProperty("shaping_scale", shapingScale);
            return result;
        }
        if ("set_reward_multipliers".equals(command)) {
            long previousVersion = rewardProfile.version();
            RewardProfile updated = updateRewardProfile(
                    rewardProfile, baseRewardConfig, tick, payload);
            boolean idempotent = updated == rewardProfile;
            rewardProfile = updated;
            JsonObject result = rewardProfileAcknowledgement(idempotent);
            if (rewardProfile.version() != previousVersion) {
                JsonObject eventPayload = rewardProfile.toJson();
                eventPayload.addProperty("applies_to", "new_episodes");
                eventPayload.addProperty("active_episodes_retained", activeArenas().size());
                emit("reward_profile_changed", null, null, eventPayload);
            }
            return result;
        }
        if ("sample_evaluation".equals(command)) {
            ArenaMode mode = ArenaMode.parse(string(payload, "mode", defaultMode.name()));
            TrainingLane lane = laneForMode(mode);
            int stage = lane.curriculum().stageNumber();
            long seed = nextEvaluationSeed(mode, stage);
            int[] delays = delaysFor(seed, mode, stage);
            JsonObject result = new JsonObject();
            result.addProperty("mode", mode.name().toLowerCase());
            result.addProperty("arena_seed", seed);
            result.addProperty("action_delay_ticks", delays[0]);
            result.addProperty("observation_delay_ticks", delays[1]);
            result.addProperty("held_out", true);
            return result;
        }
        if ("resume".equals(command)) {
            paused = false;
            JsonObject result = new JsonObject();
            result.addProperty("paused", false);
            return result;
        }
        if ("start_match".equals(command)) {
            if (paused) throw new IllegalStateException("arena manager is emergency-stopped; send resume first");
            Player first = Bukkit.getPlayerExact(string(payload, "player_a", ""));
            Player second = Bukkit.getPlayerExact(string(payload, "player_b", ""));
            if (first == null || second == null) throw new IllegalArgumentException("both named players must be online");
            ArenaMode mode = ArenaMode.parse(string(payload, "mode", defaultMode.name()));
            TrainingLane lane = laneForMode(mode);
            int stage = lane.curriculum().stageNumber();
            boolean evaluation = payload.has("evaluation") && payload.get("evaluation").getAsBoolean();
            long seed = payload.has("seed") ? payload.get("seed").getAsLong()
                    : (evaluation ? nextEvaluationSeed(mode, stage) : nextTrainingSeed(mode, stage));
            int[] delays = delaysFor(seed, mode, stage);
            if (!evaluation && SeedSplit.isHeldOut(seed, delays[0], delays[1])) {
                throw new IllegalArgumentException("seed/delay tuple is permanently held out; set evaluation=true");
            }
            Arena arena = start(first, second, seed, mode, evaluation);
            JsonObject result = new JsonObject();
            result.addProperty("arena_id", arena.getId());
            result.addProperty("episode_id", arena.getEpisodeId());
            result.addProperty("arena_seed", seed);
            result.addProperty("action_delay_ticks", delays[0]);
            result.addProperty("observation_delay_ticks", delays[1]);
            result.addProperty("held_out", SeedSplit.isHeldOut(seed, delays[0], delays[1]));
            result.addProperty("arena_radius", arena.getOuterRadius());
            result.addProperty("curriculum_stage", arena.getCurriculumStage());
            result.addProperty("curriculum_stage_count", arena.getCurriculumStageCount());
            result.addProperty("lane", arena.getLane());
            return result;
        }
        if ("stop_all".equals(command)) {
            stopAll("control_stop");
            JsonObject result = new JsonObject();
            result.addProperty("stopped", true);
            return result;
        }
        throw new IllegalArgumentException("unknown command: " + command);
    }

    public void stopAll(String reason) {
        paused = true;
        for (Arena arena : new ArrayList<Arena>(activeArenas())) arena.finish(null, true, reason);
        JsonObject payload = new JsonObject();
        payload.addProperty("reason", reason);
        emit("emergency_stop", null, null, payload);
    }

    public JsonObject status() {
        JsonObject result = new JsonObject();
        result.addProperty("tick", tick);
        result.addProperty("active_pairs", activeArenas().size());
        result.addProperty("max_concurrent_pairs", maxConcurrentPairs);
        result.addProperty("estimated_tps", estimatedTps());
        result.addProperty("p95_tick_ms", percentile95());
        Runtime runtime = Runtime.getRuntime();
        long used = runtime.totalMemory() - runtime.freeMemory();
        result.addProperty("memory_used_bytes", used);
        result.addProperty("memory_max_bytes", runtime.maxMemory());
        result.addProperty("memory_fraction", runtime.maxMemory() == 0 ? 0 : (double) used / runtime.maxMemory());
        result.addProperty("mode", defaultMode.name().toLowerCase());
        result.addProperty("paused", paused);
        result.add("reward_profile", rewardProfile.toJson());
        JsonArray activeRewardProfiles = new JsonArray();
        for (Arena arena : activeArenas()) {
            JsonObject activeProfile = new JsonObject();
            activeProfile.addProperty("arena_id", arena.getId());
            activeProfile.addProperty("episode_id", arena.getEpisodeId());
            activeProfile.addProperty("generation", arena.getRewardProfile().generation());
            activeProfile.addProperty("version", arena.getRewardProfile().version());
            activeRewardProfiles.add(activeProfile);
        }
        result.add("active_reward_profiles", activeRewardProfiles);
        ArenaSizeCurriculum summary = laneById.get("combined").curriculum();
        result.addProperty("arena_radius", summary.currentRadius());
        result.addProperty("arena_size", ArenaGeometry.playableInteriorDiameter(
                summary.currentRadius()));
        result.addProperty("curriculum_stage", summary.stageNumber());
        result.addProperty("curriculum_stage_count", summary.stageCount());
        result.addProperty("curriculum_completed_episodes", summary.completedEpisodes());
        result.addProperty("curriculum_recent_engagement_rate", summary.recentEngagementRate());
        result.addProperty("curriculum_recent_non_timeout_rate", summary.recentNonTimeoutRate());
        JsonArray radiusStages = new JsonArray();
        for (int radius : summary.radii()) radiusStages.add(radius);
        result.add("curriculum_radius_stages", radiusStages);
        JsonArray episodeThresholds = new JsonArray();
        for (int threshold : summary.episodeThresholds()) episodeThresholds.add(threshold);
        result.add("curriculum_episode_thresholds", episodeThresholds);
        JsonArray lanes = new JsonArray();
        for (TrainingLane lane : trainingLanes) {
            ArenaSizeCurriculum curriculum = lane.curriculum();
            JsonObject laneStatus = new JsonObject();
            laneStatus.addProperty("lane", lane.id());
            laneStatus.addProperty("mode", lane.mode().name().toLowerCase());
            laneStatus.addProperty("arena_radius", curriculum.currentRadius());
            laneStatus.addProperty("curriculum_stage", curriculum.stageNumber());
            laneStatus.addProperty("curriculum_stage_count", curriculum.stageCount());
            laneStatus.addProperty("completed_episodes", curriculum.completedEpisodes());
            laneStatus.addProperty("recent_engagement_rate", curriculum.recentEngagementRate());
            laneStatus.addProperty("recent_non_timeout_rate", curriculum.recentNonTimeoutRate());
            lanes.add(laneStatus);
        }
        result.add("training_lanes", lanes);
        JsonArray active = new JsonArray();
        for (Arena arena : activeArenas()) active.add(arena.getId());
        result.add("arenas", active);
        return result;
    }

    private void pairWaitingBots() {
        if (paused) return;
        if (activeArenas().size() >= maxConcurrentPairs) return;
        List<Player> waiting = new ArrayList<Player>();
        for (Player player : Bukkit.getOnlinePlayers()) {
            if (!player.getName().startsWith(botPrefix) || player.isDead() || byPlayer.containsKey(player.getUniqueId())) continue;
            if (cooldownUntil.getOrDefault(player.getUniqueId(), 0L) > tick) continue;
            waiting.add(player);
        }
        // Rotate opponents between episodes instead of permanently pairing
        // adjacent usernames (001/002, 003/004, ...). seedSource is fixed for
        // reproducible local runs while still producing a new order each pass.
        Collections.shuffle(waiting, seedSource);
        while (waiting.size() >= 2 && activeArenas().size() < maxConcurrentPairs) {
            startAuto(waiting.remove(0), waiting.remove(0));
        }
    }

    private Arena start(Player first, Player second, long seed, ArenaMode mode, boolean evaluation) {
        Arena selected = firstAvailableArena();
        if (selected == null) throw new IllegalStateException("no arena is available");
        TrainingLane lane = laneForMode(mode);
        return startSelected(selected, first, second, seed, mode, lane, evaluation);
    }

    private Arena startAuto(Player first, Player second) {
        Arena selected = firstAvailableArena();
        if (selected == null) throw new IllegalStateException("no arena is available");
        TrainingLane lane = laneByArena.get(selected);
        ArenaMode mode = lane.mode();
        return startSelected(selected, first, second,
                nextTrainingSeed(mode, lane.curriculum().stageNumber()), mode, lane, false);
    }

    private Arena startSelected(Arena selected, Player first, Player second, long seed,
                                ArenaMode mode, TrainingLane lane, boolean evaluation) {
        if (first.equals(second)) throw new IllegalArgumentException("a fighter cannot fight itself in one client");
        if (byPlayer.containsKey(first.getUniqueId()) || byPlayer.containsKey(second.getUniqueId())) {
            throw new IllegalStateException("a fighter is already in a match");
        }
        byPlayer.put(first.getUniqueId(), selected);
        byPlayer.put(second.getUniqueId(), selected);
        ArenaSizeCurriculum curriculum = lane.curriculum();
        int[] delays = delaysFor(seed, mode, curriculum.stageNumber());
        if (!evaluation && SeedSplit.isHeldOut(seed, delays[0], delays[1])) {
            byPlayer.remove(first.getUniqueId());
            byPlayer.remove(second.getUniqueId());
            throw new IllegalArgumentException("held-out tuple cannot enter training");
        }
        selected.start(first, second, seed, mode, tick, timeoutSeconds, delays[0], delays[1],
                lane.id(), curriculum.currentRadius(), curriculum.stageNumber(),
                curriculum.stageCount(), curriculum.completedEpisodes(), !evaluation);
        return selected;
    }

    private Arena firstAvailableArena() {
        for (Arena arena : arenas) if (!arena.isActive()) return arena;
        return null;
    }

    private JsonObject basePayload(Arena arena) {
        JsonObject payload = new JsonObject();
        payload.addProperty("episode_id", arena.getEpisodeId());
        payload.addProperty("arena_seed", arena.getSeed());
        payload.addProperty("mode", arena.getMode().name().toLowerCase());
        payload.addProperty("lane", arena.getLane());
        payload.addProperty("arena_radius", arena.getOuterRadius());
        payload.addProperty("arena_size", ArenaGeometry.playableInteriorDiameter(arena.getOuterRadius()));
        payload.addProperty("curriculum_stage", arena.getCurriculumStage());
        payload.addProperty("curriculum_stage_count", arena.getCurriculumStageCount());
        payload.addProperty("curriculum_completed_episodes", arena.getCurriculumCompletedEpisodes());
        payload.add("reward_profile", arena.getRewardProfile().toJson());
        return payload;
    }

    private void emit(String eventName, Arena arena, Player player, JsonObject payload) {
        if (control == null) return;
        JsonObject event = new JsonObject();
        event.addProperty("type", "event");
        event.addProperty("event", eventName);
        if (arena != null) event.addProperty("arena_id", arena.getId());
        if (player != null) event.addProperty("agent_id", agentIdFor(player));
        event.add("payload", payload);
        control.broadcast(event);
    }

    private long nextSeed() {
        return seedSource.nextInt(Integer.MAX_VALUE);
    }

    private long nextTrainingSeed(ArenaMode mode, int curriculumStage) {
        while (true) {
            long seed = nextSeed();
            int[] delays = delaysFor(seed, mode, curriculumStage);
            if (!SeedSplit.isHeldOut(seed, delays[0], delays[1])) return seed;
        }
    }

    private long nextEvaluationSeed(ArenaMode mode, int curriculumStage) {
        while (true) {
            long seed = nextSeed();
            int[] delays = delaysFor(seed, mode, curriculumStage);
            if (SeedSplit.isHeldOut(seed, delays[0], delays[1])) return seed;
        }
    }

    private int[] delaysFor(long seed, ArenaMode mode, int curriculumStage) {
        Random delayRandom = new Random(seed ^ 0x4D4341495F444C59L);
        int bound = delayBoundFor(mode, curriculumStage);
        return new int[]{delayRandom.nextInt(bound), delayRandom.nextInt(bound)};
    }

    static int delayBoundFor(ArenaMode mode, int curriculumStage) {
        // Stage one is for acquiring the combat skill, so every lane sees at
        // most one tick of action and observation latency.  Higher stages add
        // delay progressively after autonomous engagement is already stable;
        // asking a new combined/terrain policy to react to 5+5 stale ticks
        // made otherwise informed actions look random.
        return Math.min(6, Math.max(2, Math.max(1, curriculumStage) * 2));
    }

    private void addLane(TrainingLane lane) {
        trainingLanes.add(lane);
        laneById.put(lane.id(), lane);
    }

    private TrainingLane laneForMode(ArenaMode mode) {
        if (mode == ArenaMode.SWORD) return laneById.get("sword_retention");
        if (mode == ArenaMode.CRYSTAL) return laneById.get("crystal_retention");
        if (mode == ArenaMode.TERRAIN) return laneById.get("terrain");
        return laneById.get("combined");
    }

    private static ArenaSizeCurriculum curriculum(int spacing, int fallbackRadius,
                                                   List<Integer> radii, List<Integer> thresholds,
                                                   int performanceWindow,
                                                   double advanceEngagementRate,
                                                   double advanceNonTimeoutRate,
                                                   double regressEngagementRate,
                                                   double regressNonTimeoutRate,
                                                   int stageChangeCooldown) {
        return ArenaSizeCurriculum.create(spacing, fallbackRadius, radii, thresholds,
                performanceWindow, advanceEngagementRate, advanceNonTimeoutRate,
                regressEngagementRate, regressNonTimeoutRate, stageChangeCooldown);
    }

    private static List<Integer> configuredOrDefault(List<Integer> configured,
                                                     List<Integer> defaults) {
        return configured == null || configured.isEmpty() ? defaults : configured;
    }

    private double estimatedTps() {
        if (tickIntervalsMs.isEmpty()) return 20.0;
        double sum = 0;
        for (double value : tickIntervalsMs) sum += value;
        return Math.min(20.0, 1000.0 / (sum / tickIntervalsMs.size()));
    }

    private double percentile95() {
        if (tickIntervalsMs.isEmpty()) return 50.0;
        List<Double> sorted = new ArrayList<Double>(tickIntervalsMs);
        Collections.sort(sorted);
        return sorted.get(Math.min(sorted.size() - 1, (int) Math.floor(sorted.size() * 0.95)));
    }

    static RewardProfile updateRewardProfile(RewardProfile current, RewardConfig baseConfig,
                                              long tick, JsonObject payload) {
        if (payload == null) throw new IllegalArgumentException("reward profile payload is required");
        List<String> permitted = Arrays.asList("generation", "reason", "multipliers");
        for (String key : payload.keySet()) {
            if (!permitted.contains(key)) throw new IllegalArgumentException(
                    "unknown reward profile field: " + key);
        }
        if (!payload.has("generation") || !payload.get("generation").isJsonPrimitive()
                || !payload.getAsJsonPrimitive("generation").isNumber()) {
            throw new IllegalArgumentException("reward generation must be a non-negative integer");
        }
        long generation;
        try {
            generation = payload.getAsJsonPrimitive("generation").getAsBigDecimal()
                    .toBigIntegerExact().longValueExact();
        } catch (ArithmeticException | NumberFormatException error) {
            throw new IllegalArgumentException("reward generation must be a non-negative integer");
        }
        if (generation < 0L) throw new IllegalArgumentException(
                "reward generation must be a non-negative integer");
        if (!payload.has("reason") || !payload.get("reason").isJsonPrimitive()
                || !payload.getAsJsonPrimitive("reason").isString()) {
            throw new IllegalArgumentException("reward profile reason is required");
        }
        if (!payload.has("multipliers") || !payload.get("multipliers").isJsonObject()) {
            throw new IllegalArgumentException("multipliers must be an object");
        }
        RewardMultipliers multipliers = RewardMultipliers.fromJson(
                payload.getAsJsonObject("multipliers"));
        return current.next(generation, tick,
                payload.get("reason").getAsString(), multipliers, baseConfig);
    }

    private JsonObject rewardProfileAcknowledgement(boolean idempotent) {
        JsonObject result = new JsonObject();
        result.addProperty("accepted", true);
        result.addProperty("idempotent", idempotent);
        result.addProperty("applies_to", "new_episodes");
        result.addProperty("active_episodes_retained", activeArenas().size());
        result.add("reward_profile", rewardProfile.toJson());
        return result;
    }

    private static String string(JsonObject object, String key, String fallback) {
        return object != null && object.has(key) ? object.get(key).getAsString() : fallback;
    }

    private static int integer(JsonObject object, String key, int fallback) {
        return object != null && object.has(key) ? object.get(key).getAsInt() : fallback;
    }

    private static double number(JsonObject object, String key, double fallback) {
        return object != null && object.has(key) ? object.get(key).getAsDouble() : fallback;
    }
}
