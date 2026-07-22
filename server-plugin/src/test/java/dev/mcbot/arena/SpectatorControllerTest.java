package dev.mcbot.arena;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class SpectatorControllerTest {
    @Test
    void selectsFirstArenaWhenSpectatorHasNoCurrentFight() {
        assertEquals("arena-1", SpectatorController.selectNextArenaId(
                Arrays.asList("arena-1", "arena-2"), null));
    }

    @Test
    void selectsGenuinelyDifferentArenaAndWraps() {
        assertEquals("arena-2", SpectatorController.selectNextArenaId(
                Arrays.asList("arena-1", "arena-2", "arena-3"), "arena-1"));
        assertEquals("arena-1", SpectatorController.selectNextArenaId(
                Arrays.asList("arena-1", "arena-2", "arena-3"), "ARENA-3"));
    }

    @Test
    void fallsBackToFirstActiveArenaWhenPreviousFightEnded() {
        assertEquals("arena-2", SpectatorController.selectNextArenaId(
                Arrays.asList("arena-2", "arena-4"), "arena-1"));
    }

    @Test
    void reportsEmptyAndSingleArenaSelectionsPrecisely() {
        assertNull(SpectatorController.selectNextArenaId(Collections.<String>emptyList(), "arena-1"));
        assertEquals("arena-1", SpectatorController.selectNextArenaId(
                Collections.singletonList("arena-1"), "arena-1"));
    }

    @Test
    void keepsSamePhysicalPlatformDuringNormalEpisodeRecycleGap() {
        assertEquals("arena-1", SpectatorController.selectArenaForContinuity(
                Arrays.asList("arena-2", "arena-3"), "arena-1",
                SpectatorController.ARENA_RESTART_GRACE_TICKS - 1));
    }

    @Test
    void followsReactivatedArenaWithoutJumpingAndEventuallyFailsOver() {
        assertEquals("arena-1", SpectatorController.selectArenaForContinuity(
                Arrays.asList("arena-1", "arena-2"), "arena-1",
                SpectatorController.ARENA_RESTART_GRACE_TICKS));
        assertEquals("arena-2", SpectatorController.selectArenaForContinuity(
                Arrays.asList("arena-2", "arena-3"), "arena-1",
                SpectatorController.ARENA_RESTART_GRACE_TICKS));
    }

    @Test
    void holdsLastPlatformWhenAllMatchesAreBetweenEpisodes() {
        assertEquals("arena-1", SpectatorController.selectArenaForContinuity(
                Collections.<String>emptyList(), "arena-1",
                SpectatorController.ARENA_RESTART_GRACE_TICKS + 100));
    }

    @Test
    void expandsOrbitToFrameSevenBySevenArena() {
        assertEquals(18.0, SpectatorController.orbitRadius(4, 0.0), 1.0e-9);
        assertEquals(10.8, SpectatorController.orbitHeight(4, 0.0), 1.0e-9);
    }

    @Test
    void orbitFramingRemainsBoundedForLargeSpreads() {
        assertEquals(24.0, SpectatorController.orbitRadius(4, 100.0), 1.0e-9);
        assertEquals(15.0, SpectatorController.orbitHeight(4, 100.0), 1.0e-9);
    }

    @Test
    void updatesOrbitAtFiveHertz() {
        assertFalse(SpectatorController.shouldUpdateOrbit(1));
        assertFalse(SpectatorController.shouldUpdateOrbit(2));
        assertFalse(SpectatorController.shouldUpdateOrbit(3));
        assertTrue(SpectatorController.shouldUpdateOrbit(4));
    }

    @Test
    void throttlingPreservesOrbitAngularSpeed() {
        assertEquals(0.3, SpectatorController.orbitAngle(20), 1.0e-9);
        assertEquals(0.06,
                SpectatorController.orbitAngle(20) - SpectatorController.orbitAngle(16),
                1.0e-9);
    }
}
