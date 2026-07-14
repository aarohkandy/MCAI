package dev.mcbot.arena;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import org.bukkit.Bukkit;
import org.bukkit.Location;
import org.bukkit.World;
import org.bukkit.entity.Player;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.UUID;

public final class ArenaManager {
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
    private double shapingScale;
    private long tick;
    private long previousTickNanos;
    private boolean paused;

    public ArenaManager(MCAIPlugin plugin, World world) {
        this.plugin = plugin;
        this.maxConcurrentPairs = plugin.getConfig().getInt("max-concurrent-pairs", 2);
        this.timeoutSeconds = plugin.getConfig().getInt("match-timeout-seconds", 120);
        this.botPrefix = plugin.getConfig().getString("bot-name-prefix", "MCAI_");
        this.autoPair = plugin.getConfig().getBoolean("auto-pair-bots", true);
        this.defaultMode = ArenaMode.parse(plugin.getConfig().getString("default-mode", "combined"));
        this.shapingScale = plugin.getConfig().getDouble("shaping-scale", 1.0);
        int spacing = plugin.getConfig().getInt("arena-spacing", 96);
        for (int index = 0; index < 16; index++) {
            int x = (index % 4) * spacing;
            int z = (index / 4) * spacing;
            arenas.add(new Arena(this, "arena-" + (index + 1), new Location(world, x, 64, z)));
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
        if (tick % 2 == 0) {
            for (Arena arena : activeArenas()) emit("arena_snapshot", arena, null, arena.snapshotJson(tick));
        }
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
        payload.addProperty("match_ticks", tick - arena.getStartedTick());
        payload.add("stats", arena.statsJson(player));
        emit("match_ended", arena, player, payload);
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
            JsonObject result = new JsonObject();
            result.addProperty("registered", username);
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
        if ("sample_evaluation".equals(command)) {
            ArenaMode mode = ArenaMode.parse(string(payload, "mode", defaultMode.name()));
            long seed = nextEvaluationSeed(mode);
            int[] delays = delaysFor(seed, mode);
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
            boolean evaluation = payload.has("evaluation") && payload.get("evaluation").getAsBoolean();
            long seed = payload.has("seed") ? payload.get("seed").getAsLong()
                    : (evaluation ? nextEvaluationSeed(mode) : nextTrainingSeed(mode));
            int[] delays = delaysFor(seed, mode);
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
        Collections.sort(waiting, Comparator.comparing(Player::getName));
        while (waiting.size() >= 2 && activeArenas().size() < maxConcurrentPairs) {
            start(waiting.remove(0), waiting.remove(0), nextTrainingSeed(defaultMode), defaultMode, false);
        }
    }

    private Arena start(Player first, Player second, long seed, ArenaMode mode, boolean evaluation) {
        if (first.equals(second)) throw new IllegalArgumentException("a fighter cannot fight itself in one client");
        if (byPlayer.containsKey(first.getUniqueId()) || byPlayer.containsKey(second.getUniqueId())) {
            throw new IllegalStateException("a fighter is already in a match");
        }
        Arena selected = null;
        for (Arena arena : arenas) if (!arena.isActive()) { selected = arena; break; }
        if (selected == null) throw new IllegalStateException("no arena is available");
        byPlayer.put(first.getUniqueId(), selected);
        byPlayer.put(second.getUniqueId(), selected);
        int[] delays = delaysFor(seed, mode);
        if (!evaluation && SeedSplit.isHeldOut(seed, delays[0], delays[1])) {
            byPlayer.remove(first.getUniqueId());
            byPlayer.remove(second.getUniqueId());
            throw new IllegalArgumentException("held-out tuple cannot enter training");
        }
        selected.start(first, second, seed, mode, tick, timeoutSeconds, delays[0], delays[1]);
        return selected;
    }

    private JsonObject basePayload(Arena arena) {
        JsonObject payload = new JsonObject();
        payload.addProperty("episode_id", arena.getEpisodeId());
        payload.addProperty("arena_seed", arena.getSeed());
        payload.addProperty("mode", arena.getMode().name().toLowerCase());
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

    private long nextTrainingSeed(ArenaMode mode) {
        while (true) {
            long seed = nextSeed();
            int[] delays = delaysFor(seed, mode);
            if (!SeedSplit.isHeldOut(seed, delays[0], delays[1])) return seed;
        }
    }

    private long nextEvaluationSeed(ArenaMode mode) {
        while (true) {
            long seed = nextSeed();
            int[] delays = delaysFor(seed, mode);
            if (SeedSplit.isHeldOut(seed, delays[0], delays[1])) return seed;
        }
    }

    private int[] delaysFor(long seed, ArenaMode mode) {
        Random delayRandom = new Random(seed ^ 0x4D4341495F444C59L);
        int bound = mode == ArenaMode.SWORD ? 3 : 6;
        return new int[]{delayRandom.nextInt(bound), delayRandom.nextInt(bound)};
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
