package dev.mcbot.arena;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertEquals;

final class ArenaManagerTest {
    @Test
    void snapshotsEveryTenTicksByDefault() {
        int interval = ArenaManager.DEFAULT_SNAPSHOT_INTERVAL_TICKS;

        assertFalse(ArenaManager.isSnapshotTick(2, interval));
        assertFalse(ArenaManager.isSnapshotTick(9, interval));
        assertTrue(ArenaManager.isSnapshotTick(10, interval));
        assertTrue(ArenaManager.isSnapshotTick(20, interval));
    }

    @Test
    void supportsConfiguredSnapshotIntervals() {
        assertFalse(ArenaManager.isSnapshotTick(4, 5));
        assertTrue(ArenaManager.isSnapshotTick(5, 5));
        assertTrue(ArenaManager.isSnapshotTick(15, 5));
        assertFalse(ArenaManager.isSnapshotTick(10, 0));
    }

    @Test
    void firstFourPhysicalArenasAlwaysRepresentAllFourLanes() {
        assertEquals(0, ArenaManager.laneIndexForArena(0, 4));
        assertEquals(1, ArenaManager.laneIndexForArena(1, 4));
        assertEquals(2, ArenaManager.laneIndexForArena(2, 4));
        assertEquals(3, ArenaManager.laneIndexForArena(3, 4));
        assertEquals(0, ArenaManager.laneIndexForArena(4, 4));
    }

    @Test
    void crystalRetentionLatencyStartsAtZeroToOneThenExpands() {
        assertEquals(2, ArenaManager.delayBoundFor(ArenaMode.CRYSTAL, 1));
        assertEquals(4, ArenaManager.delayBoundFor(ArenaMode.CRYSTAL, 2));
        assertEquals(2, ArenaManager.delayBoundFor(ArenaMode.SWORD, 1));
        assertEquals(2, ArenaManager.delayBoundFor(ArenaMode.COMBINED, 1));
        assertEquals(6, ArenaManager.delayBoundFor(ArenaMode.TERRAIN, 4));
    }
}
