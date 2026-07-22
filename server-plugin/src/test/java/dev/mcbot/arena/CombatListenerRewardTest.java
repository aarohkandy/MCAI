package dev.mcbot.arena;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class CombatListenerRewardTest {
    @Test
    void attributedTeacherAndSafetyCrystalDamageCannotProduceRewards() {
        assertFalse(CombatListener.damageRewardEligible(ExecutionSource.TEACHER_CRYSTAL));
        assertFalse(CombatListener.damageRewardEligible(ExecutionSource.TEACHER_SWORD));
        assertFalse(CombatListener.damageRewardEligible(ExecutionSource.TEACHER_BLOCK));
        assertFalse(CombatListener.damageRewardEligible(ExecutionSource.SAFETY));
    }

    @Test
    void policyAndUnattributedEnvironmentalDamageRetainExistingRewardBehavior() {
        assertTrue(CombatListener.damageRewardEligible(ExecutionSource.POLICY));
        assertTrue(CombatListener.damageRewardEligible(null));
    }

    @Test
    void lethalOverkillRewardsOnlyActuallyRemovableHealth() {
        assertEquals(7.0, CombatListener.rewardableDamage(40.0, 5.0, 2.0), 1.0e-12);
        assertEquals(5.0, CombatListener.rewardableDamage(5.0, 20.0, 0.0), 1.0e-12);
        assertEquals(0.0, CombatListener.rewardableDamage(-3.0, 20.0, 0.0), 1.0e-12);
        assertEquals(0.0,
                CombatListener.rewardableDamage(Double.NaN, 20.0, 0.0), 1.0e-12);
    }

    @Test
    void hugeKillRewardRequiresFreshPolicyDamageAndASurvivingAssignedAttacker() {
        assertTrue(Arena.policyKillRewardEligible(ExecutionSource.POLICY, true, 0L, true));
        assertTrue(Arena.policyKillRewardEligible(ExecutionSource.POLICY, true, 20L, true));
        assertFalse(Arena.policyKillRewardEligible(ExecutionSource.POLICY, true, 21L, true));
        assertFalse(Arena.policyKillRewardEligible(ExecutionSource.POLICY, false, 0L, true));
        assertFalse(Arena.policyKillRewardEligible(ExecutionSource.POLICY, true, 0L, false));
        assertFalse(Arena.policyKillRewardEligible(ExecutionSource.TEACHER_SWORD, true, 0L, true));
        assertFalse(Arena.policyKillRewardEligible(ExecutionSource.TEACHER_CRYSTAL, true, 0L, true));
        assertFalse(Arena.policyKillRewardEligible(ExecutionSource.SAFETY, true, 0L, true));
    }

    @Test
    void onlyPolicyOwnedDeathGetsSpeedScaledFloodAndFailuresAreStrongLosses() {
        RewardConfig config = RewardConfig.defaults();
        assertEquals(64.0, Arena.winnerTerminalReward(
                "death", true, 20.0, 44.0, 0L, 700L), 1.0e-12);
        assertEquals(31.0, Arena.winnerTerminalReward(
                "death", true, 20.0, 44.0, 350L, 700L), 1.0e-12);
        assertEquals(20.0, Arena.winnerTerminalReward(
                "death", true, 20.0, 44.0, 700L, 700L), 1.0e-12);
        assertEquals(0.0, Arena.winnerTerminalReward(
                "death", false, 20.0, 44.0, 0L, 700L), 1.0e-12);
        assertEquals(0.0, Arena.winnerTerminalReward(
                "timeout", false, 20.0, 44.0, 0L, 700L), 1.0e-12);

        assertEquals(-32.0, Arena.terminalOutcome(
                "timeout", false, false, 0.0, config), 1.0e-12);
        assertEquals(-32.0, Arena.terminalOutcome(
                "timeout", true, true, 30.0, config), 1.0e-12,
                "timeout health leaders must still lose");
        assertEquals(-32.0, Arena.terminalOutcome(
                "disengaged", false, false, 0.0, config), 1.0e-12);
        assertEquals(-15.0, Arena.terminalOutcome(
                "double_ko", false, false, 0.0, config), 1.0e-12);
        assertEquals(25.0, Arena.terminalOutcome(
                "death", true, true, 25.0, config), 1.0e-12);
        assertEquals(-20.0, Arena.terminalOutcome(
                "death", true, false, 25.0, config), 1.0e-12);
        assertEquals(0.0, Arena.terminalOutcome(
                "death", true, true, 0.0, config), 1.0e-12,
                "teacher and safety kills cannot earn winner credit");
        assertTrue(config.timeoutLoss < config.deathLoss,
                "stalling must be worse than losing a decisive fight");
    }

    @Test
    void doubleKoSourcesDistinguishPolicySelfHarmFromTeacherAndSafetyControl() {
        assertEquals("self", Arena.terminalSourceLabel(ExecutionSource.POLICY, true, false));
        assertEquals("teacher_crystal",
                Arena.terminalSourceLabel(ExecutionSource.TEACHER_CRYSTAL, true, false));
        assertEquals("teacher_sword",
                Arena.terminalSourceLabel(ExecutionSource.TEACHER_SWORD, true, false));
        assertEquals("safety", Arena.terminalSourceLabel(ExecutionSource.SAFETY, true, false));
        assertEquals("policy", Arena.terminalSourceLabel(ExecutionSource.POLICY, false, true));
        assertEquals("environment", Arena.terminalSourceLabel(null, true, false));
        assertEquals("environment", Arena.terminalSourceLabel(ExecutionSource.POLICY, false, false));
    }
}
