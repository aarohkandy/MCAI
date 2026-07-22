package dev.mcbot.arena;

import org.junit.jupiter.api.Test;

import java.util.Arrays;
import java.util.Collections;

import static org.junit.jupiter.api.Assertions.assertEquals;

final class ArenaSizeCurriculumTest {
    @Test
    void advancesOnlyAtApprovedThresholdWithFullAutonomousWindow() {
        ArenaSizeCurriculum curriculum = progressive();

        for (int episode = 1; episode < 64; episode++) {
            assertEquals(ArenaSizeCurriculum.StageChange.NONE,
                    curriculum.recordCompletedEpisode(true, episode % 8 == 0));
        }
        assertEquals(5, curriculum.currentRadius());
        assertEquals(ArenaSizeCurriculum.StageChange.ADVANCED,
                curriculum.recordCompletedEpisode(true, true));
        assertEquals(6, curriculum.currentRadius());

        for (int episode = 65; episode < 256; episode++) {
            assertEquals(ArenaSizeCurriculum.StageChange.NONE,
                    curriculum.recordCompletedEpisode(true, episode % 8 == 0));
        }
        assertEquals(ArenaSizeCurriculum.StageChange.ADVANCED,
                curriculum.recordCompletedEpisode(true, true));
        assertEquals(7, curriculum.currentRadius());
    }

    @Test
    void poorFreshWindowRegressesAndOneExtraWindowPreventsImmediateBounce() {
        ArenaSizeCurriculum curriculum = progressive();
        for (int episode = 1; episode < 64; episode++) {
            curriculum.recordCompletedEpisode(true, episode % 8 == 0);
        }
        curriculum.recordCompletedEpisode(true, true);
        assertEquals(6, curriculum.currentRadius());

        for (int episode = 0; episode < 31; episode++) {
            assertEquals(ArenaSizeCurriculum.StageChange.NONE,
                    curriculum.recordCompletedEpisode(false, false));
        }
        assertEquals(ArenaSizeCurriculum.StageChange.REGRESSED,
                curriculum.recordCompletedEpisode(false, false));
        assertEquals(5, curriculum.currentRadius());

        for (int episode = 0; episode < 32; episode++) {
            assertEquals(ArenaSizeCurriculum.StageChange.NONE,
                    curriculum.recordCompletedEpisode(true, true));
        }
        assertEquals(5, curriculum.currentRadius());

        for (int episode = 0; episode < 31; episode++) {
            assertEquals(ArenaSizeCurriculum.StageChange.NONE,
                    curriculum.recordCompletedEpisode(true, true));
        }
        assertEquals(ArenaSizeCurriculum.StageChange.ADVANCED,
                curriculum.recordCompletedEpisode(true, true));
        assertEquals(6, curriculum.currentRadius());
    }

    @Test
    void timeoutOnlyRoundsCannotAdvanceEvenWhenTheyEngage() {
        ArenaSizeCurriculum curriculum = progressive();
        for (int episode = 0; episode < 200; episode++) {
            assertEquals(ArenaSizeCurriculum.StageChange.NONE,
                    curriculum.recordCompletedEpisode(true, false));
        }
        assertEquals(5, curriculum.currentRadius());
        assertEquals(1.0, curriculum.recentEngagementRate(), 1.0e-9);
        assertEquals(0.0, curriculum.recentNonTimeoutRate(), 1.0e-9);
    }

    @Test
    void stageTwoCliffRegressesAfterExactlyOneFreshWindow() {
        ArenaSizeCurriculum curriculum = progressive();
        for (int episode = 1; episode < 64; episode++) {
            curriculum.recordCompletedEpisode(true, episode % 8 == 0);
        }
        assertEquals(ArenaSizeCurriculum.StageChange.ADVANCED,
                curriculum.recordCompletedEpisode(true, true));

        // Engagement alone must not hide a terminal-rate collapse.  The
        // 32-episode cooldown and fresh performance window expire together,
        // so there is no additional delay to remove from production code.
        for (int episode = 0; episode < 31; episode++) {
            assertEquals(ArenaSizeCurriculum.StageChange.NONE,
                    curriculum.recordCompletedEpisode(true, false));
        }
        assertEquals(ArenaSizeCurriculum.StageChange.REGRESSED,
                curriculum.recordCompletedEpisode(true, false));
        assertEquals(5, curriculum.currentRadius());
    }

    @Test
    void retryUsesTheLatestWindowAndEventuallyReadvances() {
        ArenaSizeCurriculum curriculum = progressive();
        for (int episode = 1; episode < 64; episode++) {
            curriculum.recordCompletedEpisode(true, episode % 8 == 0);
        }
        curriculum.recordCompletedEpisode(true, true);
        for (int episode = 0; episode < 32; episode++) {
            curriculum.recordCompletedEpisode(true, false);
        }
        assertEquals(5, curriculum.currentRadius());
        assertEquals(160, curriculum.nextAdvanceEpisode());

        // An initially good window cannot be banked: by the time retry is
        // eligible, the latest full window is poor and promotion is denied.
        for (int episode = 0; episode < 32; episode++) {
            curriculum.recordCompletedEpisode(true, true);
        }
        for (int episode = 0; episode < 32; episode++) {
            assertEquals(ArenaSizeCurriculum.StageChange.NONE,
                    curriculum.recordCompletedEpisode(true, false));
        }
        assertEquals(5, curriculum.currentRadius());

        // Four good endings are enough for the configured 10% gate once they
        // enter the latest 32-match window; the preceding good window was not
        // banked across the subsequent poor one.
        for (int episode = 0; episode < 3; episode++) {
            assertEquals(ArenaSizeCurriculum.StageChange.NONE,
                    curriculum.recordCompletedEpisode(true, true));
        }
        assertEquals(ArenaSizeCurriculum.StageChange.ADVANCED,
                curriculum.recordCompletedEpisode(true, true));
        assertEquals(6, curriculum.currentRadius());
    }

    @Test
    void absentStagesPreserveLegacyFixedArenaRadius() {
        ArenaSizeCurriculum curriculum = ArenaSizeCurriculum.create(96, 7,
                Collections.<Integer>emptyList(), Collections.<Integer>emptyList(),
                32, 0.75, 0.10, 0.50, 0.05, 32);

        for (int episode = 0; episode < 100; episode++) {
            curriculum.recordCompletedEpisode(true, true);
        }
        assertEquals(Collections.singletonList(7), curriculum.radii());
        assertEquals(Collections.singletonList(0), curriculum.episodeThresholds());
        assertEquals(7, curriculum.currentRadius());
        assertEquals(1, curriculum.stageCount());
    }

    @Test
    void malformedStagesAreClampedAndKeptStrictlyProgressive() {
        ArenaSizeCurriculum curriculum = ArenaSizeCurriculum.create(20, 5,
                Arrays.asList(1, 5, 4, 99), Arrays.asList(99, -5, 2, 2),
                1, 0.0, 0.0, 0.0, 0.0, 0);

        assertEquals(Arrays.asList(2, 5, 6), curriculum.radii());
        assertEquals(Arrays.asList(0, 1, 2), curriculum.episodeThresholds());
        assertEquals(6, curriculum.maximumRadius());
    }

    private static ArenaSizeCurriculum progressive() {
        return ArenaSizeCurriculum.create(96, 5,
                Arrays.asList(5, 6, 7, 8), Arrays.asList(0, 64, 256, 1024),
                32, 0.75, 0.10, 0.50, 0.05, 32);
    }
}
