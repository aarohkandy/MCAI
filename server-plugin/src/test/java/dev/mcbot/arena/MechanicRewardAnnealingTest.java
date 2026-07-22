package dev.mcbot.arena;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

final class MechanicRewardAnnealingTest {
    @Test
    void bootstrapRetentionAndFinalLeagueStagesAreDistinct() {
        assertEquals(1.0, Arena.mechanicRewardMultiplier(1, 4, ArenaMode.COMBINED));
        assertEquals(0.1, Arena.mechanicRewardMultiplier(2, 4, ArenaMode.COMBINED));
        assertEquals(0.0, Arena.mechanicRewardMultiplier(4, 4, ArenaMode.COMBINED));
        assertEquals(0.0, Arena.mechanicRewardMultiplier(4, 4, ArenaMode.TERRAIN));
        assertEquals(0.1, Arena.mechanicRewardMultiplier(2, 2, ArenaMode.CRYSTAL));
    }
}
