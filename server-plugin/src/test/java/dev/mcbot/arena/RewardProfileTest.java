package dev.mcbot.arena;

import com.google.gson.JsonObject;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotSame;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertThrows;

final class RewardProfileTest {
    private static final double EPSILON = 1.0e-12;

    @Test
    void parsesExactGroupsAndClampsFiniteValuesToSafetyEnvelope() {
        JsonObject values = multipliers(0.01, 9.0, 2.0, 1.5, 0.5);

        RewardMultipliers parsed = RewardMultipliers.fromJson(values);

        assertEquals(0.25, parsed.damage, EPSILON);
        assertEquals(4.0, parsed.crystal, EPSILON);
        assertEquals(2.0, parsed.terminalSpeed, EPSILON);
        assertEquals(1.5, parsed.activity, EPSILON);
        assertEquals(0.5, parsed.building, EPSILON);
    }

    @Test
    void rejectsUnknownMissingNonNumericAndNonFiniteValues() {
        JsonObject unknown = multipliers(1, 1, 1, 1, 1);
        unknown.addProperty("kill", 2.0);
        assertThrows(IllegalArgumentException.class,
                () -> RewardMultipliers.fromJson(unknown));

        JsonObject missing = multipliers(1, 1, 1, 1, 1);
        missing.remove("building");
        assertThrows(IllegalArgumentException.class,
                () -> RewardMultipliers.fromJson(missing));

        JsonObject text = multipliers(1, 1, 1, 1, 1);
        text.addProperty("damage", "lots");
        assertThrows(IllegalArgumentException.class,
                () -> RewardMultipliers.fromJson(text));

        JsonObject nonFinite = multipliers(1, 1, 1, 1, 1);
        nonFinite.addProperty("crystal", Double.NaN);
        assertThrows(IllegalArgumentException.class,
                () -> RewardMultipliers.fromJson(nonFinite));
    }

    @Test
    void scalesOnlyApprovedRewardFamiliesAndKeepsCoreObjectiveFixed() {
        RewardConfig base = RewardConfig.defaults();
        RewardConfig scaled = base.withMultipliers(
                new RewardMultipliers(2.0, 3.0, 4.0, 0.5, 0.25));

        // Damage is symmetric, including hits and both sides of a totem pop.
        assertEquals(base.successfulHit * 2.0, scaled.successfulHit, EPSILON);
        assertEquals(base.damageDealtPerHealth * 2.0, scaled.damageDealtPerHealth, EPSILON);
        assertEquals(base.damageTakenPerHealth * 2.0, scaled.damageTakenPerHealth, EPSILON);
        assertEquals(base.forcedTotem * 2.0, scaled.forcedTotem, EPSILON);
        assertEquals(base.ownTotem * 2.0, scaled.ownTotem, EPSILON);

        // Crystal positives and self-crystal negatives move together.
        assertEquals(base.crystalPlaced * 3.0, scaled.crystalPlaced, EPSILON);
        assertEquals(base.crystalComboDamage * 3.0, scaled.crystalComboDamage, EPSILON);
        assertEquals(base.ownCrystalSelfHit * 3.0, scaled.ownCrystalSelfHit, EPSILON);
        assertEquals(base.ownCrystalSelfDamagePerHealth * 3.0,
                scaled.ownCrystalSelfDamagePerHealth, EPSILON);

        // Speed changes urgency, but never changes the base win/loss objective.
        assertEquals(base.policyKill, scaled.policyKill, EPSILON);
        assertEquals(base.deathLoss, scaled.deathLoss, EPSILON);
        assertEquals(base.doubleKoLoss, scaled.doubleKoLoss, EPSILON);
        assertEquals(base.policyKillSpeedBonus * 4.0, scaled.policyKillSpeedBonus, EPSILON);
        assertEquals(base.timeoutLoss * 4.0, scaled.timeoutLoss, EPSILON);
        assertEquals(base.disengagedLoss * 4.0, scaled.disengagedLoss, EPSILON);
        assertEquals(base.fightTimePressurePerTick * 4.0,
                scaled.fightTimePressurePerTick, EPSILON);

        assertEquals(base.movementPerBlock * 0.5, scaled.movementPerBlock, EPSILON);
        assertEquals(base.validAttackSwing * 0.5, scaled.validAttackSwing, EPSILON);
        assertEquals(base.inactionPenaltyPerTick * 0.5,
                scaled.inactionPenaltyPerTick, EPSILON);
        assertEquals(base.obsidianPlaced * 0.25, scaled.obsidianPlaced, EPSILON);
        assertEquals(base.tacticalMinePlace * 0.25, scaled.tacticalMinePlace, EPSILON);
        assertEquals(base.maxPerTick, scaled.maxPerTick, EPSILON);
    }

    @Test
    void generationsAreMonotonicAndExactDuplicatesAreIdempotent() {
        RewardConfig base = RewardConfig.defaults();
        RewardProfile initial = RewardProfile.initial(base);
        RewardMultipliers values = new RewardMultipliers(1.2, 1.4, 1.6, 0.8, 1.1);
        RewardProfile generationSeven = initial.next(7, 200, "metrics window 7", values, base);

        assertEquals(7L, generationSeven.generation());
        assertEquals(1L, generationSeven.version());
        assertNotSame(initial.shaper(), generationSeven.shaper());
        assertSame(generationSeven, generationSeven.next(
                7, 300, "metrics window 7", values, base));
        assertThrows(IllegalArgumentException.class, () -> generationSeven.next(
                6, 300, "metrics window 6", values, base));
        assertThrows(IllegalArgumentException.class, () -> generationSeven.next(
                7, 300, "changed", values, base));
        assertThrows(IllegalArgumentException.class, () -> generationSeven.next(
                7, 300, "metrics window 7", RewardMultipliers.identity(), base));
    }

    @Test
    void managerCommandParserIsStrictAndTelemetryIsComplete() {
        RewardConfig base = RewardConfig.defaults();
        RewardProfile current = RewardProfile.initial(base);
        JsonObject payload = payload(3, "low crystal conversion",
                multipliers(1.25, 2.0, 1.5, 0.75, 1.1));

        RewardProfile updated = ArenaManager.updateRewardProfile(current, base, 456, payload);
        JsonObject telemetry = updated.toJson();

        assertEquals(3L, telemetry.get("generation").getAsLong());
        assertEquals(1L, telemetry.get("version").getAsLong());
        assertEquals(456L, telemetry.get("applied_tick").getAsLong());
        assertEquals("low crystal conversion", telemetry.get("reason").getAsString());
        assertEquals(2.0, telemetry.getAsJsonObject("multipliers")
                .get("crystal").getAsDouble(), EPSILON);
        assertEquals(0.25, telemetry.getAsJsonObject("bounds")
                .get("minimum").getAsDouble(), EPSILON);

        assertSame(updated, ArenaManager.updateRewardProfile(updated, base, 999, payload));

        JsonObject extra = payload.deepCopy();
        extra.addProperty("force", true);
        assertThrows(IllegalArgumentException.class,
                () -> ArenaManager.updateRewardProfile(updated, base, 999, extra));

        JsonObject fractional = payload.deepCopy();
        fractional.addProperty("generation", 3.5);
        assertThrows(IllegalArgumentException.class,
                () -> ArenaManager.updateRewardProfile(updated, base, 999, fractional));
    }

    private static JsonObject payload(long generation, String reason, JsonObject multipliers) {
        JsonObject payload = new JsonObject();
        payload.addProperty("generation", generation);
        payload.addProperty("reason", reason);
        payload.add("multipliers", multipliers);
        return payload;
    }

    private static JsonObject multipliers(double damage, double crystal, double terminalSpeed,
                                          double activity, double building) {
        JsonObject values = new JsonObject();
        values.addProperty("damage", damage);
        values.addProperty("crystal", crystal);
        values.addProperty("terminal_speed", terminalSpeed);
        values.addProperty("activity", activity);
        values.addProperty("building", building);
        return values;
    }
}
