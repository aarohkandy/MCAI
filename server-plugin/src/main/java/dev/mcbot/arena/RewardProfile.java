package dev.mcbot.arena;

import com.google.gson.JsonObject;

/** Immutable, episode-snapshotted reward settings. */
public final class RewardProfile {
    private final long generation;
    private final long version;
    private final long appliedTick;
    private final String reason;
    private final RewardMultipliers multipliers;
    private final RewardShaper shaper;

    private RewardProfile(long generation, long version, long appliedTick, String reason,
                          RewardMultipliers multipliers, RewardShaper shaper) {
        this.generation = generation;
        this.version = version;
        this.appliedTick = appliedTick;
        this.reason = reason;
        this.multipliers = multipliers;
        this.shaper = shaper;
    }

    public static RewardProfile initial(RewardConfig baseConfig) {
        RewardMultipliers identity = RewardMultipliers.identity();
        return new RewardProfile(0L, 0L, 0L, "server_defaults", identity,
                new RewardShaper(baseConfig.withMultipliers(identity)));
    }

    public RewardProfile next(long requestedGeneration, long tick, String requestedReason,
                              RewardMultipliers requestedMultipliers, RewardConfig baseConfig) {
        if (requestedGeneration < generation) {
            throw new IllegalArgumentException("stale reward generation " + requestedGeneration
                    + "; current generation is " + generation);
        }
        String normalizedReason = normalizeReason(requestedReason);
        if (requestedGeneration == generation) {
            if (!reason.equals(normalizedReason) || !multipliers.equals(requestedMultipliers)) {
                throw new IllegalArgumentException(
                        "reward generation already exists with different settings: " + generation);
            }
            return this;
        }
        return new RewardProfile(requestedGeneration, version + 1L, Math.max(0L, tick),
                normalizedReason, requestedMultipliers,
                new RewardShaper(baseConfig.withMultipliers(requestedMultipliers)));
    }

    public long generation() { return generation; }
    public long version() { return version; }
    public RewardMultipliers multipliers() { return multipliers; }
    public RewardShaper shaper() { return shaper; }

    public JsonObject toJson() {
        JsonObject result = new JsonObject();
        result.addProperty("generation", generation);
        result.addProperty("version", version);
        result.addProperty("applied_tick", appliedTick);
        result.addProperty("reason", reason);
        result.add("multipliers", multipliers.toJson());
        JsonObject bounds = new JsonObject();
        bounds.addProperty("minimum", RewardMultipliers.MINIMUM);
        bounds.addProperty("maximum", RewardMultipliers.MAXIMUM);
        result.add("bounds", bounds);
        return result;
    }

    static String normalizeReason(String value) {
        String reason = value == null ? "" : value.trim();
        if (reason.isEmpty()) throw new IllegalArgumentException("reward profile reason is required");
        if (reason.length() > 512) throw new IllegalArgumentException(
                "reward profile reason must be at most 512 characters");
        return reason;
    }
}
