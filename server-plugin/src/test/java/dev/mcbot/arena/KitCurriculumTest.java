package dev.mcbot.arena;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertSame;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class KitCurriculumTest {
    @Test
    void combinedAndTerrainUseTheFourProgressiveTiers() {
        int[] protection = {2, 2, 3, 4};
        int[] apples = {0, 0, 4, 16};
        int[] spareTotems = {0, 0, 1, 4};

        for (ArenaMode mode : new ArenaMode[]{ArenaMode.COMBINED, ArenaMode.TERRAIN}) {
            for (int stage = 1; stage <= 4; stage++) {
                KitCurriculum.KitSpec kit = KitCurriculum.forEpisode(mode, stage, 4);
                assertEquals(protection[stage - 1], kit.protectionLevel(), mode + " stage " + stage);
                assertEquals(apples[stage - 1], kit.goldenApples(), mode + " stage " + stage);
                assertEquals(spareTotems[stage - 1], kit.spareTotems(), mode + " stage " + stage);
                assertTrue(kit.hasOffhandTotem(), mode + " stage " + stage);
            }
        }
    }

    @Test
    void swordAndCrystalRetentionKitsNeverChangeWithCurriculumStage() {
        for (int stage = 1; stage <= 4; stage++) {
            KitCurriculum.KitSpec sword = KitCurriculum.forEpisode(ArenaMode.SWORD, stage, 4);
            assertEquals(4, sword.protectionLevel());
            assertEquals(0, sword.goldenApples());
            assertEquals(0, sword.spareTotems());
            assertFalse(sword.hasOffhandTotem());

            KitCurriculum.KitSpec crystal = KitCurriculum.forEpisode(ArenaMode.CRYSTAL, stage, 4);
            assertEquals(2, crystal.protectionLevel());
            assertEquals(0, crystal.goldenApples());
            assertEquals(0, crystal.spareTotems());
            assertTrue(crystal.hasOffhandTotem());
        }
    }

    @Test
    void finalAndSingleStageMatchesPreserveTheHistoricalFullKit() {
        for (ArenaMode mode : new ArenaMode[]{ArenaMode.COMBINED, ArenaMode.TERRAIN}) {
            assertFullKit(KitCurriculum.forEpisode(mode, 4, 4));
            assertFullKit(KitCurriculum.forEpisode(mode, 1, 1));
        }
        assertFullKit(KitCurriculum.forEpisode(ArenaMode.SCRIPTED, 1, 4));
    }

    @Test
    void stagesClampAndBothFightersResolveTheSameImmutableSpec() {
        KitCurriculum.KitSpec beforeFirst = KitCurriculum.forEpisode(ArenaMode.COMBINED, -5, 4);
        KitCurriculum.KitSpec firstFighter = KitCurriculum.forEpisode(ArenaMode.COMBINED, 2, 4);
        KitCurriculum.KitSpec secondFighter = KitCurriculum.forEpisode(ArenaMode.COMBINED, 2, 4);
        KitCurriculum.KitSpec afterLast = KitCurriculum.forEpisode(ArenaMode.COMBINED, 99, 4);

        assertEquals(2, beforeFirst.protectionLevel());
        assertSame(firstFighter, secondFighter);
        assertEquals(2, firstFighter.protectionLevel());
        assertEquals(0, firstFighter.goldenApples());
        assertEquals(0, firstFighter.spareTotems());
        assertFullKit(afterLast);
    }

    @Test
    void stageTwoChangesSpaceWithoutChangingStageOneSurvivability() {
        for (ArenaMode mode : new ArenaMode[]{ArenaMode.COMBINED, ArenaMode.TERRAIN}) {
            KitCurriculum.KitSpec first = KitCurriculum.forEpisode(mode, 1, 4);
            KitCurriculum.KitSpec second = KitCurriculum.forEpisode(mode, 2, 4);

            assertEquals(first.protectionLevel(), second.protectionLevel());
            assertEquals(first.goldenApples(), second.goldenApples());
            assertEquals(first.spareTotems(), second.spareTotems());
            assertEquals(first.hasOffhandTotem(), second.hasOffhandTotem());
        }
    }

    @Test
    void arbitraryStageCountsRemainMonotonicAndEndAtFullStrength() {
        int previousProtection = 0;
        int previousApples = -1;
        int previousTotems = -1;
        for (int stage = 1; stage <= 7; stage++) {
            KitCurriculum.KitSpec kit = KitCurriculum.forEpisode(ArenaMode.TERRAIN, stage, 7);
            assertTrue(kit.protectionLevel() >= previousProtection);
            assertTrue(kit.goldenApples() >= previousApples);
            assertTrue(kit.spareTotems() >= previousTotems);
            previousProtection = kit.protectionLevel();
            previousApples = kit.goldenApples();
            previousTotems = kit.spareTotems();
        }
        assertFullKit(KitCurriculum.forEpisode(ArenaMode.TERRAIN, 7, 7));
    }

    private static void assertFullKit(KitCurriculum.KitSpec kit) {
        assertEquals(4, kit.protectionLevel());
        assertEquals(16, kit.goldenApples());
        assertEquals(4, kit.spareTotems());
        assertTrue(kit.hasOffhandTotem());
    }
}
