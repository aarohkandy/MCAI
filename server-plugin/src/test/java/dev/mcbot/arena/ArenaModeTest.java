package dev.mcbot.arena;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

final class ArenaModeTest {
    @Test
    void parsesModesAndFallsBackSafely() {
        assertEquals(ArenaMode.SWORD, ArenaMode.parse("sword"));
        assertEquals(ArenaMode.COMBINED, ArenaMode.parse("unknown"));
        assertEquals(ArenaMode.COMBINED, ArenaMode.parse(null));
    }

    @Test
    void statsResetClearsEveryCounter() {
        CombatStats stats = new CombatStats(java.util.UUID.randomUUID());
        stats.damageDealt = 10;
        stats.totemPops = 2;
        stats.pendingReward = 1;
        stats.reset();
        assertEquals(0, stats.damageDealt);
        assertEquals(0, stats.totemPops);
        assertEquals(0, stats.pendingReward);
    }
}
