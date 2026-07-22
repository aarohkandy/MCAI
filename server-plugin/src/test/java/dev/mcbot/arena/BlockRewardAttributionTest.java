package dev.mcbot.arena;

import org.junit.jupiter.api.Test;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class BlockRewardAttributionTest {
    @Test
    void teacherPlacementIsVisibleButCannotClaimObsidianReward() {
        CombatStats stats = new CombatStats(UUID.randomUUID());
        String block = "world:1:63:1";

        assertFalse(Arena.claimAutonomousObsidianReward(
                stats, block, 4, ExecutionSource.TEACHER_BLOCK, true));
        // The teacher path did not consume the UUID/key claim or the cap.
        assertTrue(Arena.claimAutonomousObsidianReward(
                stats, block, 4, ExecutionSource.POLICY, true));
    }

    @Test
    void policyPlacementMustPassTacticalGeometryBeforeItCanClaimReward() {
        CombatStats stats = new CombatStats(UUID.randomUUID());
        String block = "world:1:64:1";

        assertFalse(Arena.claimAutonomousObsidianReward(
                stats, block, 4, ExecutionSource.POLICY, false));
        // A rejected generic placement does not consume the useful-placement key.
        assertTrue(Arena.claimAutonomousObsidianReward(
                stats, block, 4, ExecutionSource.POLICY, true));
        assertFalse(Arena.claimAutonomousObsidianReward(
                stats, block, 4, ExecutionSource.POLICY, true));
    }

    @Test
    void usefulPlacementAndBuiltBaseComboClaimsHaveIndependentCapsAndKeys() {
        CombatStats stats = new CombatStats(UUID.randomUUID());
        for (int index = 0; index < 4; index++) {
            String base = "world:" + index + ":64:0";
            assertTrue(Arena.claimAutonomousObsidianReward(
                    stats, base, 4, ExecutionSource.POLICY, true));
            assertTrue(stats.claimObsidianCombo(base, 4));
            assertFalse(stats.claimObsidianCombo(base, 4));
        }
        assertFalse(Arena.claimAutonomousObsidianReward(
                stats, "world:9:64:0", 4, ExecutionSource.POLICY, true));
        assertFalse(stats.claimObsidianCombo("world:9:64:0", 4));
        assertTrue(stats.rewardedObsidianCombos == 4);
    }

    @Test
    void genericMiningIsVisibleButNeverClaimsReward() {
        CombatStats stats = new CombatStats(UUID.randomUUID());
        String block = "world:2:62:2";

        assertFalse(Arena.claimAutonomousMiningReward(
                stats, block, 32, ExecutionSource.TEACHER_BLOCK));
        assertFalse(Arena.claimAutonomousMiningReward(
                stats, block, 32, ExecutionSource.SAFETY));
        assertFalse(Arena.claimAutonomousMiningReward(
                stats, block, 32, ExecutionSource.POLICY));
    }

    @Test
    void policyCannotFarmMiningRewardFromAnyPreviouslyPlacedBlock() {
        CombatStats stats = new CombatStats(UUID.randomUUID());
        String block = "world:3:64:3";
        stats.rememberPlacedBlock(block);

        assertFalse(Arena.claimAutonomousMiningReward(
                stats, block, 32, ExecutionSource.POLICY));
    }

    @Test
    void exactPolicyMineThenUsefulPlaceClaimsOnceWithinWindow() {
        CombatStats stats = new CombatStats(UUID.randomUUID());
        UUID builder = stats.playerId;
        String block = "world:3:64:3";

        assertTrue(Arena.claimAutonomousTacticalMinePlace(
                stats, block, builder, 100L, builder, 160L,
                120, 3, ExecutionSource.POLICY, true));
        assertFalse(Arena.claimAutonomousTacticalMinePlace(
                stats, block, builder, 100L, builder, 161L,
                120, 3, ExecutionSource.POLICY, true));
    }

    @Test
    void minePlaceRejectsTeacherWrongOwnerStaleAndNonTacticalSequences() {
        UUID builder = UUID.randomUUID();
        UUID other = UUID.randomUUID();
        String block = "world:4:64:4";

        assertFalse(Arena.claimAutonomousTacticalMinePlace(
                new CombatStats(builder), block, builder, 100L, builder, 110L,
                120, 3, ExecutionSource.TEACHER_BLOCK, true));
        assertFalse(Arena.claimAutonomousTacticalMinePlace(
                new CombatStats(builder), block, other, 100L, builder, 110L,
                120, 3, ExecutionSource.POLICY, true));
        assertFalse(Arena.claimAutonomousTacticalMinePlace(
                new CombatStats(builder), block, builder, 100L, builder, 221L,
                120, 3, ExecutionSource.POLICY, true));
        assertFalse(Arena.claimAutonomousTacticalMinePlace(
                new CombatStats(builder), block, builder, 100L, builder, 110L,
                120, 3, ExecutionSource.POLICY, false));
    }
}
