package dev.mcbot.arena;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class CrystalEngagementTest {
    @Test
    void crystalRetentionIgnoresSwordHitsAndUnchainedCrystalDamage() {
        CombatStats stats = new CombatStats(UUID.randomUUID());
        CombatStats.AttributedEvents policy = stats.events(ExecutionSource.POLICY);
        policy.hitsLanded = 10;
        policy.damageDealt = 20.0;
        policy.crystalDamageEvents = 3;

        assertFalse(Arena.autonomousEngagementFor(ArenaMode.CRYSTAL, Arrays.asList(stats)));
        assertTrue(Arena.autonomousEngagementFor(ArenaMode.COMBINED, Arrays.asList(stats)));
    }

    @Test
    void combinedAndTerrainDoNotAcceptUnchainedPolicyCrystalDamage() {
        CombatStats stats = new CombatStats(UUID.randomUUID());
        CombatStats.AttributedEvents policy = stats.events(ExecutionSource.POLICY);
        policy.damageDealt = 20.0;
        policy.crystalDamageDealt = 20.0;

        assertFalse(Arena.autonomousEngagementFor(ArenaMode.COMBINED, Arrays.asList(stats)));
        assertFalse(Arena.autonomousEngagementFor(ArenaMode.TERRAIN, Arrays.asList(stats)));

        policy.directDamageDealt = 4.0;
        assertTrue(Arena.autonomousEngagementFor(ArenaMode.COMBINED, Arrays.asList(stats)));
    }

    @Test
    void crystalRetentionRequiresAuthoritativeDamagingOrPoppingChain() {
        CombatStats stats = new CombatStats(UUID.randomUUID());
        stats.policyCrystalChainsDetonated = 2;
        assertFalse(Arena.autonomousEngagementFor(ArenaMode.CRYSTAL, Arrays.asList(stats)));

        stats.policyCrystalChainsDamaging = 1;
        assertTrue(Arena.autonomousEngagementFor(ArenaMode.CRYSTAL, Arrays.asList(stats)));
    }
}
