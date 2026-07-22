package dev.mcbot.arena;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

final class ArenaModeTest {
    @Test
    void parsesModesAndFallsBackSafely() {
        assertEquals(ArenaMode.SWORD, ArenaMode.parse("sword"));
        assertEquals(ArenaMode.CRYSTAL, ArenaMode.parse("crystal"));
        assertEquals(ArenaMode.COMBINED, ArenaMode.parse("combined"));
        assertEquals(ArenaMode.TERRAIN, ArenaMode.parse("terrain"));
        assertEquals(ArenaMode.COMBINED, ArenaMode.parse("unknown"));
        assertEquals(ArenaMode.COMBINED, ArenaMode.parse(null));
    }

    @Test
    void genericArmSwingShapingIsSwordOnly() {
        assertEquals(true, ArenaMode.SWORD.usesGenericArmSwingShaping());
        assertEquals(false, ArenaMode.CRYSTAL.usesGenericArmSwingShaping());
        assertEquals(false, ArenaMode.COMBINED.usesGenericArmSwingShaping());
        assertEquals(false, ArenaMode.SCRIPTED.usesGenericArmSwingShaping());
        assertEquals(false, ArenaMode.TERRAIN.usesGenericArmSwingShaping());
        assertEquals(true, ArenaMode.TERRAIN.hasCrystalLayout());
        assertEquals(true, ArenaMode.TERRAIN.hasTerrainLayout());
    }

    @Test
    void statsResetClearsEveryCounter() {
        CombatStats stats = new CombatStats(java.util.UUID.randomUUID());
        stats.damageDealt = 10;
        stats.totemPops = 2;
        stats.pendingReward = 1;
        stats.events(ExecutionSource.TEACHER_SWORD).hitsLanded = 3;
        stats.reset();
        assertEquals(0, stats.damageDealt);
        assertEquals(0, stats.totemPops);
        assertEquals(0, stats.pendingReward);
        assertEquals(0, stats.existingEvents(ExecutionSource.TEACHER_SWORD).hitsLanded);
    }
}
