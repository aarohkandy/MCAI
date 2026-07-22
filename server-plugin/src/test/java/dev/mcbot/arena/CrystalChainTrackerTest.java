package dev.mcbot.arena;

import org.junit.jupiter.api.Test;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class CrystalChainTrackerTest {
    @Test
    void acceptsSamePolicyFighterPlaceDetonateAndDamageWithinWindow() {
        CrystalChainTracker tracker = new CrystalChainTracker(40);
        UUID fighter = UUID.randomUUID();
        UUID crystal = UUID.randomUUID();
        tracker.reset("episode-a");

        tracker.recordPlacement(crystal, fighter, ExecutionSource.POLICY, 100, true);
        CrystalChainTracker.Chain chain = tracker.recordDetonation(
                crystal, fighter, ExecutionSource.POLICY, 140);
        assertNotNull(chain);
        assertTrue(chain.claimDetonationCount());
        assertFalse(chain.claimDetonationCount());

        tracker.recordOpponentDamage(crystal, fighter, ExecutionSource.POLICY, 140);
        assertTrue(chain.claimDamageCount());
        assertTrue(chain.claimComboReward());
        assertFalse(chain.claimComboReward());
        assertTrue(chain.sequenceId().startsWith("episode-a:"));
    }

    @Test
    void rejectsTeacherMixedOpponentAndLateChains() {
        UUID placer = UUID.randomUUID();
        UUID opponent = UUID.randomUUID();

        assertRejected(placer, placer, ExecutionSource.TEACHER_CRYSTAL, 10);
        assertRejected(placer, opponent, ExecutionSource.POLICY, 10);
        assertRejected(placer, placer, ExecutionSource.POLICY, 41);
    }

    @Test
    void popBonusRequiresARewardedDamagingChain() {
        CrystalChainTracker tracker = new CrystalChainTracker(40);
        UUID fighter = UUID.randomUUID();
        UUID crystal = UUID.randomUUID();
        tracker.reset("episode-pop");
        tracker.recordPlacement(crystal, fighter, ExecutionSource.POLICY, 1, true);
        CrystalChainTracker.Chain chain = tracker.recordDetonation(
                crystal, fighter, ExecutionSource.POLICY, 2);
        tracker.recordOpponentPop(crystal, fighter, ExecutionSource.POLICY, 2);

        assertTrue(chain.claimPopCount());
        assertFalse(chain.claimPopReward());
        tracker.recordOpponentDamage(crystal, fighter, ExecutionSource.POLICY, 2);
        assertTrue(chain.claimComboReward());
        assertTrue(chain.claimPopReward());
    }

    @Test
    void exactPolicyBuiltBaseGetsOneCreditOnlyAfterAutonomousOpponentDamage() {
        CrystalChainTracker tracker = new CrystalChainTracker(40);
        UUID fighter = UUID.randomUUID();
        UUID crystal = UUID.randomUUID();
        tracker.reset("episode-built-base");
        CrystalChainTracker.Chain chain = tracker.recordPlacement(
                crystal, fighter, ExecutionSource.POLICY, 10, true,
                "world:2:64:1", true);

        assertFalse(chain.claimPolicyBuiltDamageCount(), "placement alone is not enough");
        tracker.recordDetonation(crystal, fighter, ExecutionSource.POLICY, 20);
        assertFalse(chain.claimPolicyBuiltDamageCount(), "detonation alone is not enough");
        tracker.recordOpponentDamage(crystal, fighter, ExecutionSource.POLICY, 20);
        assertTrue(chain.claimPolicyBuiltDamageCount());
        assertFalse(chain.claimPolicyBuiltDamageCount());
        assertTrue(chain.policyBuiltBase());
        assertTrue("world:2:64:1".equals(chain.baseKey()));
    }

    @Test
    void generatedTeacherMixedAndLateBasesCannotGetBuiltBaseDamageCredit() {
        UUID fighter = UUID.randomUUID();
        UUID opponent = UUID.randomUUID();
        assertBuiltBaseRejected(fighter, fighter, ExecutionSource.POLICY, 10, false);
        assertBuiltBaseRejected(fighter, fighter, ExecutionSource.TEACHER_CRYSTAL, 10, true);
        assertBuiltBaseRejected(fighter, opponent, ExecutionSource.POLICY, 10, true);
        assertBuiltBaseRejected(fighter, fighter, ExecutionSource.POLICY, 41, true);
    }

    private static void assertRejected(UUID placer, UUID detonator,
                                       ExecutionSource detonationSource, long delay) {
        CrystalChainTracker tracker = new CrystalChainTracker(40);
        UUID crystal = UUID.randomUUID();
        tracker.reset("episode-reject");
        tracker.recordPlacement(crystal, placer, ExecutionSource.POLICY, 10, true);
        CrystalChainTracker.Chain chain = tracker.recordDetonation(
                crystal, detonator, detonationSource, 10 + delay);
        tracker.recordOpponentDamage(crystal, detonator, detonationSource, 10 + delay);

        assertFalse(chain.isAutonomousSequence());
        assertFalse(chain.claimDetonationCount());
        assertFalse(chain.claimDamageCount());
        assertFalse(chain.claimComboReward());
    }

    private static void assertBuiltBaseRejected(UUID placer, UUID detonator,
                                                ExecutionSource detonationSource,
                                                long delay, boolean policyBuiltBase) {
        CrystalChainTracker tracker = new CrystalChainTracker(40);
        UUID crystal = UUID.randomUUID();
        tracker.reset("episode-built-reject");
        CrystalChainTracker.Chain chain = tracker.recordPlacement(
                crystal, placer, ExecutionSource.POLICY, 10, true,
                "world:3:64:1", policyBuiltBase);
        tracker.recordDetonation(crystal, detonator, detonationSource, 10 + delay);
        tracker.recordOpponentDamage(crystal, detonator, detonationSource, 10 + delay);
        assertFalse(chain.claimPolicyBuiltDamageCount());
    }
}
