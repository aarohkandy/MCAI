package dev.mcbot.arena;

/** Immutable reward weights. Kept independent of Bukkit so reward logic is easy to test. */
public final class RewardConfig {
    public final double maxPerTick;
    public final double movementPerBlock;
    public final double maxMovementDelta;
    public final double preferredDistance;
    public final double extremePitchDegrees;
    public final int extremePitchGraceTicks;
    public final double extremePitchPenaltyPerTick;
    public final double aimAlignmentPerPotential;
    public final double maxAimAlignmentDelta;
    public final double aimAlignmentRange;
    public final double lockOnDot;
    public final double lockOnRange;
    public final int lockOnGraceTicks;
    public final double lockOnPerTick;
    public final int maxRewardedLockOnTicks;
    public final double attackAimDot;
    public final double attackReach;
    public final int attackCooldownTicks;
    public final double validAttackSwing;
    public final int maxRewardedAttackSwings;
    public final double missedAttackSwing;
    public final double spamAttackSwing;
    public final int maxPenalizedAttackSwings;
    public final double successfulHit;
    public final int maxRewardedHits;
    public final double damageDealtPerHealth;
    public final double damageTakenPerHealth;
    public final double forcedTotem;
    public final double ownTotem;
    public final double policyKill;
    public final double policyKillSpeedBonus;
    public final double deathLoss;
    public final double timeoutLoss;
    public final double disengagedLoss;
    public final double doubleKoLoss;
    public final double ownCrystalSelfHit;
    public final double ownCrystalSelfDamagePerHealth;
    public final double obsidianPlaced;
    public final int maxRewardedObsidian;
    public final double obsidianCombo;
    public final int maxRewardedObsidianCombos;
    public final int tacticalMinePlaceMaxTicks;
    public final double tacticalMinePlace;
    public final int maxRewardedTacticalMinePlace;
    public final double crystalPlaced;
    public final int maxRewardedCrystalPlacements;
    public final double crystalDestroyed;
    public final int maxRewardedCrystalDestructions;
    public final double crystalExploded;
    public final int maxRewardedCrystalExplosions;
    public final int crystalComboMaxTicks;
    public final double crystalComboDamage;
    public final double crystalComboPop;
    public final int maxRewardedCrystalCombos;
    public final double usefulBlockMined;
    public final int maxRewardedMinedBlocks;
    public final double invalidInteraction;
    public final int inactionGraceTicks;
    public final double inactionPenaltyPerTick;
    public final double maxInactionPenaltyPerTick;
    public final int positiveRewardDecayStartTicks;
    public final int positiveRewardDecayEndTicks;
    public final double minimumPositiveRewardMultiplier;
    public final int fightTimePressureStartTicks;
    public final double fightTimePressurePerTick;
    public final double maxFightTimePressurePerTick;

    public RewardConfig(double maxPerTick,
                        double movementPerBlock,
                        double maxMovementDelta,
                        double preferredDistance,
                        double extremePitchDegrees,
                        int extremePitchGraceTicks,
                        double extremePitchPenaltyPerTick,
                        double aimAlignmentPerPotential,
                        double maxAimAlignmentDelta,
                        double aimAlignmentRange,
                        double lockOnDot,
                        double lockOnRange,
                        int lockOnGraceTicks,
                        double lockOnPerTick,
                        int maxRewardedLockOnTicks,
                        double attackAimDot,
                        double attackReach,
                        int attackCooldownTicks,
                        double validAttackSwing,
                        int maxRewardedAttackSwings,
                        double missedAttackSwing,
                        double spamAttackSwing,
                        int maxPenalizedAttackSwings,
                        double successfulHit,
                        int maxRewardedHits,
                        double damageDealtPerHealth,
                        double damageTakenPerHealth,
                        double forcedTotem,
                        double ownTotem,
                        double policyKill,
                        double policyKillSpeedBonus,
                        double deathLoss,
                        double timeoutLoss,
                        double disengagedLoss,
                        double doubleKoLoss,
                        double ownCrystalSelfHit,
                        double ownCrystalSelfDamagePerHealth,
                        double obsidianPlaced,
                        int maxRewardedObsidian,
                        double obsidianCombo,
                        int maxRewardedObsidianCombos,
                        int tacticalMinePlaceMaxTicks,
                        double tacticalMinePlace,
                        int maxRewardedTacticalMinePlace,
                        double crystalPlaced,
                        int maxRewardedCrystalPlacements,
                        double crystalDestroyed,
                        int maxRewardedCrystalDestructions,
                        double crystalExploded,
                        int maxRewardedCrystalExplosions,
                        int crystalComboMaxTicks,
                        double crystalComboDamage,
                        double crystalComboPop,
                        int maxRewardedCrystalCombos,
                        double usefulBlockMined,
                        int maxRewardedMinedBlocks,
                        double invalidInteraction,
                        int inactionGraceTicks,
                        double inactionPenaltyPerTick,
                        double maxInactionPenaltyPerTick,
                        int positiveRewardDecayStartTicks,
                        int positiveRewardDecayEndTicks,
                        double minimumPositiveRewardMultiplier,
                        int fightTimePressureStartTicks,
                        double fightTimePressurePerTick,
                        double maxFightTimePressurePerTick) {
        this.maxPerTick = Math.max(0.001, maxPerTick);
        this.movementPerBlock = Math.max(0, movementPerBlock);
        this.maxMovementDelta = Math.max(0, maxMovementDelta);
        this.preferredDistance = Math.max(0, preferredDistance);
        this.extremePitchDegrees = Math.max(0, Math.min(90, extremePitchDegrees));
        this.extremePitchGraceTicks = Math.max(0, extremePitchGraceTicks);
        this.extremePitchPenaltyPerTick = Math.min(0, extremePitchPenaltyPerTick);
        this.aimAlignmentPerPotential = Math.max(0, aimAlignmentPerPotential);
        this.maxAimAlignmentDelta = Math.max(0, Math.min(1, maxAimAlignmentDelta));
        this.aimAlignmentRange = Math.max(0, aimAlignmentRange);
        this.lockOnDot = Math.max(-1, Math.min(1, lockOnDot));
        this.lockOnRange = Math.max(0, lockOnRange);
        this.lockOnGraceTicks = Math.max(0, lockOnGraceTicks);
        this.lockOnPerTick = Math.max(0, lockOnPerTick);
        this.maxRewardedLockOnTicks = Math.max(0, maxRewardedLockOnTicks);
        this.attackAimDot = Math.max(-1, Math.min(1, attackAimDot));
        this.attackReach = Math.max(0, attackReach);
        this.attackCooldownTicks = Math.max(1, attackCooldownTicks);
        this.validAttackSwing = Math.max(0, validAttackSwing);
        this.maxRewardedAttackSwings = Math.max(0, maxRewardedAttackSwings);
        this.missedAttackSwing = Math.min(0, missedAttackSwing);
        this.spamAttackSwing = Math.min(0, spamAttackSwing);
        this.maxPenalizedAttackSwings = Math.max(0, maxPenalizedAttackSwings);
        this.successfulHit = Math.max(0, successfulHit);
        this.maxRewardedHits = Math.max(0, maxRewardedHits);
        this.damageDealtPerHealth = Math.max(0, damageDealtPerHealth);
        this.damageTakenPerHealth = Math.min(0, damageTakenPerHealth);
        this.forcedTotem = Math.max(0, forcedTotem);
        this.ownTotem = Math.min(0, ownTotem);
        this.policyKill = Math.max(1.0, policyKill);
        this.policyKillSpeedBonus = Math.max(0.0, policyKillSpeedBonus);
        this.deathLoss = Math.min(-1.0, deathLoss);
        this.timeoutLoss = Math.min(-1.0, timeoutLoss);
        this.disengagedLoss = Math.min(-1.0, disengagedLoss);
        this.doubleKoLoss = Math.min(-1.0, doubleKoLoss);
        this.ownCrystalSelfHit = Math.min(0, ownCrystalSelfHit);
        this.ownCrystalSelfDamagePerHealth = Math.min(0, ownCrystalSelfDamagePerHealth);
        this.obsidianPlaced = Math.max(0, obsidianPlaced);
        this.maxRewardedObsidian = Math.max(0, maxRewardedObsidian);
        this.obsidianCombo = Math.max(0, obsidianCombo);
        this.maxRewardedObsidianCombos = Math.max(0, maxRewardedObsidianCombos);
        this.tacticalMinePlaceMaxTicks = Math.max(1, tacticalMinePlaceMaxTicks);
        this.tacticalMinePlace = Math.max(0, tacticalMinePlace);
        this.maxRewardedTacticalMinePlace = Math.max(0, maxRewardedTacticalMinePlace);
        this.crystalPlaced = Math.max(0, crystalPlaced);
        this.maxRewardedCrystalPlacements = Math.max(0, maxRewardedCrystalPlacements);
        this.crystalDestroyed = Math.max(0, crystalDestroyed);
        this.maxRewardedCrystalDestructions = Math.max(0, maxRewardedCrystalDestructions);
        this.crystalExploded = Math.max(0, crystalExploded);
        this.maxRewardedCrystalExplosions = Math.max(0, maxRewardedCrystalExplosions);
        this.crystalComboMaxTicks = Math.max(1, crystalComboMaxTicks);
        this.crystalComboDamage = Math.max(0, crystalComboDamage);
        this.crystalComboPop = Math.max(0, crystalComboPop);
        this.maxRewardedCrystalCombos = Math.max(0, maxRewardedCrystalCombos);
        this.usefulBlockMined = Math.max(0, usefulBlockMined);
        this.maxRewardedMinedBlocks = Math.max(0, maxRewardedMinedBlocks);
        this.invalidInteraction = Math.min(0, invalidInteraction);
        this.inactionGraceTicks = Math.max(0, inactionGraceTicks);
        this.inactionPenaltyPerTick = Math.min(0, inactionPenaltyPerTick);
        this.maxInactionPenaltyPerTick = Math.min(this.inactionPenaltyPerTick,
                maxInactionPenaltyPerTick);
        this.positiveRewardDecayStartTicks = Math.max(0, positiveRewardDecayStartTicks);
        this.positiveRewardDecayEndTicks = Math.max(
                this.positiveRewardDecayStartTicks + 1, positiveRewardDecayEndTicks);
        this.minimumPositiveRewardMultiplier = Math.max(0, Math.min(1,
                minimumPositiveRewardMultiplier));
        this.fightTimePressureStartTicks = Math.max(0, fightTimePressureStartTicks);
        this.fightTimePressurePerTick = Math.min(0, fightTimePressurePerTick);
        this.maxFightTimePressurePerTick = Math.min(this.fightTimePressurePerTick,
                maxFightTimePressurePerTick);
    }

    /**
     * Builds an immutable episode configuration from the server defaults and a
     * bounded adaptive profile. Structural limits and the base kill/death
     * objective never change at runtime.
     */
    public RewardConfig withMultipliers(RewardMultipliers values) {
        RewardMultipliers multipliers = values == null ? RewardMultipliers.identity() : values;
        return new RewardConfig(
                maxPerTick,
                movementPerBlock * multipliers.activity,
                maxMovementDelta,
                preferredDistance,
                extremePitchDegrees,
                extremePitchGraceTicks,
                extremePitchPenaltyPerTick * multipliers.activity,
                aimAlignmentPerPotential * multipliers.activity,
                maxAimAlignmentDelta,
                aimAlignmentRange,
                lockOnDot,
                lockOnRange,
                lockOnGraceTicks,
                lockOnPerTick * multipliers.activity,
                maxRewardedLockOnTicks,
                attackAimDot,
                attackReach,
                attackCooldownTicks,
                validAttackSwing * multipliers.activity,
                maxRewardedAttackSwings,
                missedAttackSwing * multipliers.activity,
                spamAttackSwing * multipliers.activity,
                maxPenalizedAttackSwings,
                successfulHit * multipliers.damage,
                maxRewardedHits,
                damageDealtPerHealth * multipliers.damage,
                damageTakenPerHealth * multipliers.damage,
                forcedTotem * multipliers.damage,
                ownTotem * multipliers.damage,
                policyKill,
                policyKillSpeedBonus * multipliers.terminalSpeed,
                deathLoss,
                timeoutLoss * multipliers.terminalSpeed,
                disengagedLoss * multipliers.terminalSpeed,
                doubleKoLoss,
                ownCrystalSelfHit * multipliers.crystal,
                ownCrystalSelfDamagePerHealth * multipliers.crystal,
                obsidianPlaced * multipliers.building,
                maxRewardedObsidian,
                obsidianCombo * multipliers.building,
                maxRewardedObsidianCombos,
                tacticalMinePlaceMaxTicks,
                tacticalMinePlace * multipliers.building,
                maxRewardedTacticalMinePlace,
                crystalPlaced * multipliers.crystal,
                maxRewardedCrystalPlacements,
                crystalDestroyed * multipliers.crystal,
                maxRewardedCrystalDestructions,
                crystalExploded * multipliers.crystal,
                maxRewardedCrystalExplosions,
                crystalComboMaxTicks,
                crystalComboDamage * multipliers.crystal,
                crystalComboPop * multipliers.crystal,
                maxRewardedCrystalCombos,
                usefulBlockMined * multipliers.building,
                maxRewardedMinedBlocks,
                invalidInteraction * multipliers.activity,
                inactionGraceTicks,
                inactionPenaltyPerTick * multipliers.activity,
                maxInactionPenaltyPerTick * multipliers.activity,
                positiveRewardDecayStartTicks,
                positiveRewardDecayEndTicks,
                minimumPositiveRewardMultiplier,
                fightTimePressureStartTicks,
                fightTimePressurePerTick * multipliers.terminalSpeed,
                maxFightTimePressurePerTick * multipliers.terminalSpeed);
    }

    public static RewardConfig defaults() {
        return new RewardConfig(
                0.25, 0.004, 0.35, 2.75,
                60.0, 20, -0.0001,
                0.012, 1.0, 24.0,
                0.94, 5.0, 1, 0.0, 0,
                0.70, 3.4, 8, 0.004, 40,
                -0.002, -0.001, 120, 0.045, 10,
                0.180, -0.100, 0.12, -0.12,
                20.0, 44.0, -20.0, -32.0, -32.0, -15.0, -0.900, -0.100,
                0.015, 4,
                0.200, 4,
                120, 0.030, 3,
                0.040, 8,
                0.020, 8,
                0.050, 8,
                40, 0.250, 0.500, 6,
                0.0, 0,
                -0.001,
                40, -0.0004, -0.002,
                200, 700, 0.35,
                20, -0.001, -0.010);
    }
}
