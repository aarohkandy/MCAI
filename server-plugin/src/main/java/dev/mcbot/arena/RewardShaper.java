package dev.mcbot.arena;

/** Pure, bounded dense-reward calculations. */
public final class RewardShaper {
    public enum AttackResult { VALID, MISS, SPAM }

    private final RewardConfig config;

    public RewardShaper(RewardConfig config) {
        this.config = config;
    }

    public RewardConfig config() {
        return config;
    }

    /**
     * Potential-based approach reward. Moving away pays back moving closer, so an
     * agent cannot farm points by oscillating. No pressure is added inside melee range.
     */
    public double approach(double previousDistance, double currentDistance) {
        if (!finite(previousDistance) || !finite(currentDistance)) return 0;
        double previousPotential = -Math.max(config.preferredDistance, previousDistance);
        double currentPotential = -Math.max(config.preferredDistance, currentDistance);
        double progress = clamp(currentPotential - previousPotential,
                -config.maxMovementDelta, config.maxMovementDelta);
        return progress * config.movementPerBlock;
    }

    /** Only sustained extreme camera pitch is penalized; ordinary combat aiming is free. */
    public double posture(float pitchDegrees, int extremePitchStreak) {
        return Math.abs(pitchDegrees) >= config.extremePitchDegrees
                && extremePitchStreak > config.extremePitchGraceTicks
                ? config.extremePitchPenaltyPerTick : 0;
    }

    public boolean isExtremePitch(float pitchDegrees) {
        return Math.abs(pitchDegrees) >= config.extremePitchDegrees;
    }

    /**
     * A bounded [0, 1] potential for looking at the assigned opponent. Returning
     * zero outside visibility/range makes acquiring and losing the same target
     * pay equal and opposite rewards.
     */
    public double aimPotential(double gazeDot, double distance, boolean clearLineOfSight) {
        if (!finite(gazeDot) || !finite(distance) || distance < 0
                || distance > config.aimAlignmentRange || !clearLineOfSight) return 0;
        return (clamp(gazeDot, -1, 1) + 1.0) * 0.5;
    }

    /** Signed potential delta; repeatedly looking away and back cannot create net reward. */
    public double aimAlignment(double previousPotential, double currentPotential) {
        if (!finite(previousPotential) || !finite(currentPotential)) return 0;
        return clamp(currentPotential - previousPotential,
                -config.maxAimAlignmentDelta, config.maxAimAlignmentDelta)
                * config.aimAlignmentPerPotential;
    }

    public boolean isLockedOn(double gazeDot, double distance, boolean clearLineOfSight) {
        return finite(gazeDot) && finite(distance) && distance >= 0
                && distance <= config.lockOnRange && clearLineOfSight
                && gazeDot >= config.lockOnDot;
    }

    public double lockOn() { return config.lockOnPerTick; }

    /** Pure classification for an arm swing directed at the assigned opponent. */
    public AttackResult evaluateAttack(double gazeDot, double distance, boolean clearLineOfSight,
                                       long ticksSinceRewardedAttack) {
        if (!finite(gazeDot) || !finite(distance) || distance < 0
                || distance > config.attackReach || !clearLineOfSight
                || gazeDot < config.attackAimDot) return AttackResult.MISS;
        return ticksSinceRewardedAttack < config.attackCooldownTicks
                ? AttackResult.SPAM : AttackResult.VALID;
    }

    public double attackSwing(AttackResult result) {
        if (result == AttackResult.VALID) return config.validAttackSwing;
        if (result == AttackResult.SPAM) return config.spamAttackSwing;
        return config.missedAttackSwing;
    }

    public double successfulHit() { return config.successfulHit; }

    public double damageDealt(double health) { return Math.max(0, health) * config.damageDealtPerHealth; }
    public double damageTaken(double health) { return Math.max(0, health) * config.damageTakenPerHealth; }
    public double forcedTotem() { return config.forcedTotem; }
    public double ownTotem() { return config.ownTotem; }
    public double ownCrystalSelfHit() { return config.ownCrystalSelfHit; }
    public double ownCrystalSelfDamage(double health) {
        return Math.max(0, health) * config.ownCrystalSelfDamagePerHealth;
    }
    public double obsidianPlaced() { return config.obsidianPlaced; }
    public double obsidianCombo() { return config.obsidianCombo; }
    public double tacticalMinePlace() { return config.tacticalMinePlace; }
    public double crystalPlaced() { return config.crystalPlaced; }
    public double crystalDestroyed() { return config.crystalDestroyed; }
    public double crystalExploded() { return config.crystalExploded; }
    public double crystalComboDamage() { return config.crystalComboDamage; }
    public double crystalComboPop() { return config.crystalComboPop; }
    public double usefulBlockMined() { return config.usefulBlockMined; }
    public double invalidInteraction() { return config.invalidInteraction; }

    /** Escalating per-tick cost after a short window without useful autonomous action. */
    public double inaction(long ticksSinceAction) {
        if (ticksSinceAction <= config.inactionGraceTicks) return 0;
        long overdue = ticksSinceAction - config.inactionGraceTicks;
        double escalation = 1.0 + Math.min(4.0, overdue / 40.0);
        return Math.max(config.maxInactionPenaltyPerTick,
                config.inactionPenaltyPerTick * escalation);
    }

    /** Makes early progress more valuable without weakening penalties. */
    public double positiveRewardMultiplier(long elapsedTicks) {
        if (elapsedTicks <= config.positiveRewardDecayStartTicks) return 1.0;
        if (elapsedTicks >= config.positiveRewardDecayEndTicks) {
            return config.minimumPositiveRewardMultiplier;
        }
        double progress = (elapsedTicks - config.positiveRewardDecayStartTicks)
                / (double) (config.positiveRewardDecayEndTicks
                - config.positiveRewardDecayStartTicks);
        return 1.0 - progress * (1.0 - config.minimumPositiveRewardMultiplier);
    }

    /**
     * A universal opportunity cost for letting a fight run long. It starts
     * after the opening engagement window and ramps smoothly to a bounded
     * penalty at the deadline. Because both fighters pay it and it is always
     * negative, it cannot be farmed by oscillation, teachers, or self-damage.
     */
    public double fightTimePressure(long elapsedTicks, long totalTicks) {
        if (elapsedTicks <= config.fightTimePressureStartTicks || totalTicks <= 0) return 0;
        long pressureWindow = Math.max(1L, totalTicks - config.fightTimePressureStartTicks);
        double progress = clamp((elapsedTicks - config.fightTimePressureStartTicks)
                / (double) pressureWindow, 0.0, 1.0);
        return config.fightTimePressurePerTick
                + progress * (config.maxFightTimePressurePerTick
                - config.fightTimePressurePerTick);
    }

    public boolean belowCap(int alreadyRewarded, int cap) {
        return alreadyRewarded >= 0 && alreadyRewarded < cap;
    }

    public double clipTick(double reward) {
        return clamp(reward, -config.maxPerTick, config.maxPerTick);
    }

    private static boolean finite(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }

    private static double clamp(double value, double minimum, double maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }
}
