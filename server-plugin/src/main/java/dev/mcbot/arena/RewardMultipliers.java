package dev.mcbot.arena;

import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Set;

/**
 * Small, bounded set of reward-group multipliers controlled by the adaptive
 * trainer. Values are deliberately coarse grained so the controller cannot
 * accidentally change legality, reward caps, or arena mechanics at runtime.
 */
public final class RewardMultipliers {
    public static final double MINIMUM = 0.25;
    public static final double MAXIMUM = 4.0;

    private static final Set<String> KEYS = new HashSet<String>(Arrays.asList(
            "damage", "crystal", "terminal_speed", "activity", "building"));

    public final double damage;
    public final double crystal;
    public final double terminalSpeed;
    public final double activity;
    public final double building;

    public RewardMultipliers(double damage, double crystal, double terminalSpeed,
                             double activity, double building) {
        this.damage = bounded("damage", damage);
        this.crystal = bounded("crystal", crystal);
        this.terminalSpeed = bounded("terminal_speed", terminalSpeed);
        this.activity = bounded("activity", activity);
        this.building = bounded("building", building);
    }

    public static RewardMultipliers identity() {
        return new RewardMultipliers(1.0, 1.0, 1.0, 1.0, 1.0);
    }

    /** Strict field validation with finite values clamped to the safety envelope. */
    public static RewardMultipliers fromJson(JsonObject object) {
        if (object == null) throw new IllegalArgumentException("multipliers must be an object");
        for (String key : object.keySet()) {
            if (!KEYS.contains(key)) throw new IllegalArgumentException(
                    "unknown reward multiplier: " + key);
        }
        for (String key : KEYS) {
            if (!object.has(key)) throw new IllegalArgumentException(
                    "missing reward multiplier: " + key);
        }
        return new RewardMultipliers(
                numeric(object, "damage"),
                numeric(object, "crystal"),
                numeric(object, "terminal_speed"),
                numeric(object, "activity"),
                numeric(object, "building"));
    }

    public JsonObject toJson() {
        JsonObject result = new JsonObject();
        result.addProperty("damage", damage);
        result.addProperty("crystal", crystal);
        result.addProperty("terminal_speed", terminalSpeed);
        result.addProperty("activity", activity);
        result.addProperty("building", building);
        return result;
    }

    @Override
    public boolean equals(Object other) {
        if (this == other) return true;
        if (!(other instanceof RewardMultipliers)) return false;
        RewardMultipliers value = (RewardMultipliers) other;
        return Double.compare(damage, value.damage) == 0
                && Double.compare(crystal, value.crystal) == 0
                && Double.compare(terminalSpeed, value.terminalSpeed) == 0
                && Double.compare(activity, value.activity) == 0
                && Double.compare(building, value.building) == 0;
    }

    @Override
    public int hashCode() {
        long result = Double.doubleToLongBits(damage);
        result = 31 * result + Double.doubleToLongBits(crystal);
        result = 31 * result + Double.doubleToLongBits(terminalSpeed);
        result = 31 * result + Double.doubleToLongBits(activity);
        result = 31 * result + Double.doubleToLongBits(building);
        return (int) (result ^ (result >>> 32));
    }

    private static double numeric(JsonObject object, String key) {
        JsonElement element = object.get(key);
        if (element == null || !element.isJsonPrimitive()
                || !element.getAsJsonPrimitive().isNumber()) {
            throw new IllegalArgumentException("reward multiplier must be numeric: " + key);
        }
        return element.getAsDouble();
    }

    private static double bounded(String name, double value) {
        if (Double.isNaN(value) || Double.isInfinite(value)) {
            throw new IllegalArgumentException("reward multiplier must be finite: " + name);
        }
        return Math.max(MINIMUM, Math.min(MAXIMUM, value));
    }
}
