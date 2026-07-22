package dev.mcbot.arena;

import org.junit.jupiter.api.Test;

import java.util.UUID;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class RewardShaperTest {
    private final RewardShaper shaper = new RewardShaper(RewardConfig.defaults());

    @Test
    void approachIsSignedPotentialAndCannotBeOscillationFarmed() {
        double closer = shaper.approach(8.0, 7.75);
        double farther = shaper.approach(7.75, 8.0);
        assertEquals(0.001, closer, 1.0e-12);
        assertEquals(-closer, farther, 1.0e-12);
        assertEquals(0.0, closer + farther, 1.0e-12);
    }

    @Test
    void approachStopsPressuringInsideUsefulCombatRangeAndCapsTeleports() {
        assertEquals(0.0, shaper.approach(2.5, 2.0), 1.0e-12);
        assertEquals(0.35 * 0.004, shaper.approach(30.0, 3.0), 1.0e-12);
        assertEquals(-0.35 * 0.004, shaper.approach(3.0, 30.0), 1.0e-12);
    }

    @Test
    void postureOnlyPenalizesSustainedExtremePitch() {
        assertEquals(0.0, shaper.posture(80.0F, 20), 1.0e-12);
        assertEquals(-0.0001, shaper.posture(80.0F, 21), 1.0e-12);
        assertEquals(0.0, shaper.posture(45.0F, 200), 1.0e-12);
        assertEquals(-0.0001, shaper.posture(-80.0F, 21), 1.0e-12);
    }

    @Test
    void aggregateRewardIsCappedPerServerTick() {
        assertEquals(0.25, shaper.clipTick(4.0), 1.0e-12);
        assertEquals(-0.25, shaper.clipTick(-4.0), 1.0e-12);
        assertEquals(0.04, shaper.clipTick(0.04), 1.0e-12);
    }

    @Test
    void inactionHasGraceThenEscalatesToABoundedPenalty() {
        assertEquals(0.0, shaper.inaction(40), 1.0e-12);
        assertEquals(-0.00041, shaper.inaction(41), 1.0e-12);
        assertEquals(-0.0008, shaper.inaction(80), 1.0e-12);
        assertEquals(-0.002, shaper.inaction(240), 1.0e-12);
        assertEquals(-0.002, shaper.inaction(10_000), 1.0e-12);
    }

    @Test
    void positiveActionCreditDecaysAsMatchDragsOn() {
        assertEquals(1.0, shaper.positiveRewardMultiplier(200), 1.0e-12);
        assertEquals(0.675, shaper.positiveRewardMultiplier(450), 1.0e-12);
        assertEquals(0.35, shaper.positiveRewardMultiplier(700), 1.0e-12);
        assertEquals(0.35, shaper.positiveRewardMultiplier(10_000), 1.0e-12);
    }

    @Test
    void fightTimePressureStartsAfterOpeningAndEscalatesToDeadline() {
        assertEquals(0.0, shaper.fightTimePressure(20, 700), 1.0e-12);
        assertEquals(-0.001013235294117647, shaper.fightTimePressure(21, 700), 1.0e-12);
        assertEquals(-0.0055, shaper.fightTimePressure(360, 700), 1.0e-12);
        assertEquals(-0.010, shaper.fightTimePressure(700, 700), 1.0e-12);
        assertEquals(-0.010, shaper.fightTimePressure(10_000, 700), 1.0e-12);
    }

    @Test
    void cleanOpponentDamageIsAggressivelyPositive() {
        assertEquals(1.80, shaper.damageDealt(10.0), 1.0e-12);
        assertEquals(-1.00, shaper.damageTaken(10.0), 1.0e-12);
        assertEquals(0.80,
                shaper.damageDealt(10.0) + shaper.damageTaken(10.0), 1.0e-12);
    }

    @Test
    void completePolicyBuiltCrystalEqualDamageChainsStayNegativeAcrossBothRewardLayers() {
        double mechanicCredit = shaper.crystalPlaced() + shaper.crystalDestroyed()
                + shaper.crystalExploded() + shaper.crystalComboDamage();
        assertEquals(0.360, mechanicCredit, 1.0e-12);
        double builtBaseCredit = shaper.obsidianPlaced() + shaper.obsidianCombo();
        assertEquals(0.215, builtBaseCredit, 1.0e-12);
        double oneHealthTrade = mechanicCredit + builtBaseCredit + shaper.damageDealt(1.0)
                + shaper.damageTaken(1.0) + shaper.ownCrystalSelfDamage(1.0)
                + shaper.ownCrystalSelfHit();
        double tenHealthTrade = mechanicCredit + builtBaseCredit + shaper.damageDealt(10.0)
                + shaper.damageTaken(10.0) + shaper.ownCrystalSelfDamage(10.0)
                + shaper.ownCrystalSelfHit();
        assertEquals(-0.345, oneHealthTrade, 1.0e-12);
        assertEquals(-0.525, tenHealthTrade, 1.0e-12);
        // Trainer-side verified crystal/setup/built-base bonuses total another
        // at most 0.044, including the exact mine/place sequence. Even
        // with both layers, an equal own-crystal trade must remain negative.
        assertTrue(oneHealthTrade + 0.044 < 0);
        assertTrue(oneHealthTrade < 0 && tenHealthTrade < 0);
    }

    @Test
    void damagePointBreakdownKeepsEnemyAndSelfEconomicsSeparate() {
        CombatStats stats = new CombatStats(UUID.randomUUID());
        stats.addPending(RewardReason.DAMAGE_DEALT, shaper.damageDealt(4.0));
        stats.addPending(RewardReason.DAMAGE_TAKEN, shaper.damageTaken(4.0));
        stats.addPending(RewardReason.SELF_DAMAGE,
                shaper.ownCrystalSelfHit() + shaper.ownCrystalSelfDamage(4.0));
        assertEquals(0.72, stats.rewardTotal(RewardReason.DAMAGE_DEALT), 1.0e-12);
        assertEquals(-0.40, stats.rewardTotal(RewardReason.DAMAGE_TAKEN), 1.0e-12);
        assertEquals(-1.300, stats.rewardTotal(RewardReason.SELF_DAMAGE), 1.0e-12);
    }

    @Test
    void aimAlignmentIsSignedOpponentPotentialAndCannotBeLookAwayFarmed() {
        double away = shaper.aimPotential(-1.0, 8.0, true);
        double locked = shaper.aimPotential(1.0, 8.0, true);
        double acquire = shaper.aimAlignment(away, locked);
        double lose = shaper.aimAlignment(locked, away);
        assertEquals(0.012, acquire, 1.0e-12);
        assertEquals(-acquire, lose, 1.0e-12);
        assertEquals(0.0, acquire + lose, 1.0e-12);
        assertEquals(0.0, shaper.aimPotential(1.0, 25.0, true), 1.0e-12);
        assertEquals(0.0, shaper.aimPotential(1.0, 8.0, false), 1.0e-12);
    }

    @Test
    void lockOnRequiresTightGazeUsefulRangeAndLineOfSight() {
        assertTrue(shaper.isLockedOn(0.95, 4.0, true));
        assertFalse(shaper.isLockedOn(0.93, 4.0, true));
        assertFalse(shaper.isLockedOn(0.99, 6.0, true));
        assertFalse(shaper.isLockedOn(0.99, 4.0, false));
        assertEquals(0.0, shaper.lockOn(), 1.0e-12);
    }

    @Test
    void attackClicksAreValidOnlyOnTargetInReachAndRateLimited() {
        assertEquals(RewardShaper.AttackResult.VALID,
                shaper.evaluateAttack(0.95, 3.0, true, Long.MAX_VALUE));
        assertEquals(RewardShaper.AttackResult.SPAM,
                shaper.evaluateAttack(0.95, 3.0, true, 7));
        assertEquals(RewardShaper.AttackResult.MISS,
                shaper.evaluateAttack(0.69, 3.0, true, Long.MAX_VALUE));
        assertEquals(RewardShaper.AttackResult.MISS,
                shaper.evaluateAttack(0.99, 3.5, true, Long.MAX_VALUE));
        assertEquals(RewardShaper.AttackResult.MISS,
                shaper.evaluateAttack(0.99, 3.0, false, Long.MAX_VALUE));
        assertEquals(0.004, shaper.attackSwing(RewardShaper.AttackResult.VALID), 1.0e-12);
        assertEquals(-0.002, shaper.attackSwing(RewardShaper.AttackResult.MISS), 1.0e-12);
        assertEquals(-0.001, shaper.attackSwing(RewardShaper.AttackResult.SPAM), 1.0e-12);
    }

    @Test
    void newAttackCurriculumPositiveCapsStayBelowAWin() {
        RewardConfig config = shaper.config();
        double maximum = config.aimAlignmentPerPotential
                + config.lockOnPerTick * config.maxRewardedLockOnTicks
                + config.validAttackSwing * config.maxRewardedAttackSwings
                + config.successfulHit * config.maxRewardedHits;
        assertTrue(maximum < 1.0, "kill reward must dominate all new positive curriculum rewards");
        assertEquals(0.622, maximum, 1.0e-12);
        double worstMissPenalty = Math.abs(config.missedAttackSwing) * config.maxPenalizedAttackSwings;
        assertTrue(worstMissPenalty < 1.0, "loss reward must dominate capped click penalties");
        assertEquals(0.24, worstMissPenalty, 1.0e-12);
    }

    @Test
    void oneTimeClaimsRejectDuplicateAndPlacedThenMinedFarming() {
        CombatStats stats = new CombatStats(UUID.randomUUID());
        assertTrue(stats.claimObsidian("world:1:64:1", 2));
        assertFalse(stats.claimObsidian("world:1:64:1", 2));
        stats.rememberPlacedBlock("world:1:64:1");
        assertFalse(stats.claimMinedBlock("world:1:64:1", 32));
        assertTrue(stats.claimMinedBlock("world:2:63:1", 32));

        UUID crystal = UUID.randomUUID();
        assertTrue(stats.claimCrystalPlacement(crystal, "world:3:63:1", 12));
        assertFalse(stats.claimCrystalPlacement(crystal, "world:4:63:1", 12));
        assertFalse(stats.claimCrystalPlacement(UUID.randomUUID(), "world:3:63:1", 12));
        assertTrue(stats.claimCrystalPlacement(UUID.randomUUID(), "world:4:63:1", 12));

        UUID destroyedCrystal = UUID.randomUUID();
        assertTrue(stats.claimCrystalDestruction(destroyedCrystal, "world:3:63:1", 12));
        assertFalse(stats.claimCrystalDestruction(destroyedCrystal, "world:4:63:1", 12));
        assertFalse(stats.claimCrystalDestruction(
                UUID.randomUUID(), "world:3:63:1", 12));
        assertTrue(stats.claimCrystalDestruction(
                UUID.randomUUID(), "world:4:63:1", 12));

        UUID explodedCrystal = UUID.randomUUID();
        assertTrue(stats.claimCrystalExplosion(explodedCrystal, "world:3:63:1", 12));
        assertFalse(stats.claimCrystalExplosion(explodedCrystal, "world:4:63:1", 12));
        assertFalse(stats.claimCrystalExplosion(
                UUID.randomUUID(), "world:3:63:1", 12));
        assertTrue(stats.claimCrystalExplosion(
                UUID.randomUUID(), "world:4:63:1", 12));

        UUID selfHitCrystal = UUID.randomUUID();
        assertTrue(stats.claimOwnCrystalSelfHit(selfHitCrystal));
        assertFalse(stats.claimOwnCrystalSelfHit(selfHitCrystal));
        assertFalse(stats.claimOwnCrystalSelfHit(null));

        assertTrue(stats.claimLockOnTick(1));
        assertFalse(stats.claimLockOnTick(1));
        assertEquals(Long.MAX_VALUE, stats.ticksSinceRewardedAttack(10));
        assertTrue(stats.claimAttackReward(10, 1));
        assertFalse(stats.claimAttackReward(20, 1));
        assertEquals(8, stats.ticksSinceRewardedAttack(18));
        assertTrue(stats.claimHitReward(1));
        assertFalse(stats.claimHitReward(1));
        assertTrue(stats.claimAttackPenalty(1));
        assertFalse(stats.claimAttackPenalty(1));
    }

    @Test
    void crystalMechanicClaimsStopAtTwelveUniqueBasesPerEpisode() {
        CombatStats stats = new CombatStats(UUID.randomUUID());
        for (int index = 0; index < 12; index++) {
            String base = "world:" + index + ":63:0";
            assertTrue(stats.claimCrystalPlacement(UUID.randomUUID(), base, 12));
            assertTrue(stats.claimCrystalDestruction(UUID.randomUUID(), base, 12));
            assertTrue(stats.claimCrystalExplosion(UUID.randomUUID(), base, 12));
        }
        assertFalse(stats.claimCrystalPlacement(UUID.randomUUID(), "world:20:63:0", 12));
        assertFalse(stats.claimCrystalDestruction(UUID.randomUUID(), "world:20:63:0", 12));
        assertFalse(stats.claimCrystalExplosion(UUID.randomUUID(), "world:20:63:0", 12));
    }
}
