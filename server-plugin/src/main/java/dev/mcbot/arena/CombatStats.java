package dev.mcbot.arena;

import java.util.EnumMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;
import java.util.UUID;

public final class CombatStats {
    public final UUID playerId;
    public double damageDealt;
    public double damageTaken;
    public double selfDamage;
    public int policyKills;
    public double healing;
    public int totemPops;
    public int totemPopsForced;
    public int obsidianPlaced;
    public int tacticalObsidianPlaced;
    public int tacticalMinePlaceSequences;
    public int policyBuiltCrystalChainsDamaging;
    public int rewardedObsidianCombos;
    public int crystalsPlaced;
    public int crystalsDestroyed;
    public int crystalsExploded;
    public int policyCrystalChainsStarted;
    public int policyCrystalChainsDetonated;
    public int policyCrystalChainsDamaging;
    public int policyCrystalChainsPopping;
    public int rewardedCrystalCombos;
    public int blocksPlaced;
    public int blocksMined;
    public int invalidInteractions;
    public long approachTicks;
    public long extremePitchTicks;
    public int extremePitchStreak;
    public long aimAlignmentTicks;
    public int lockOnTicks;
    public int rewardedLockOnTicks;
    public int lockOnStreak;
    public int attackSwings;
    public int validAttackSwings;
    public int missedAttackSwings;
    public int spamAttackSwings;
    public int penalizedAttackSwings;
    public int rewardedAttackSwings;
    public int hitsLanded;
    public int rewardedHits;
    public double pendingReward;
    public double shapingRewardTotal;
    public double terminalReward;
    public long lastAutonomousActionTick;
    public long inactionPenaltyTicks;

    private final Map<RewardReason, Double> rewardTotals = new EnumMap<RewardReason, Double>(RewardReason.class);
    private final Map<ExecutionSource, AttributedEvents> attributedEvents =
            new EnumMap<ExecutionSource, AttributedEvents>(ExecutionSource.class);
    private final Set<String> placedBlockKeys = new HashSet<String>();
    private final Set<String> rewardedObsidianKeys = new HashSet<String>();
    private final Set<String> rewardedObsidianComboBases = new HashSet<String>();
    private final Set<String> rewardedTacticalMinePlaceKeys = new HashSet<String>();
    private final Set<String> rewardedMinedKeys = new HashSet<String>();
    private final Set<UUID> rewardedCrystalPlacements = new HashSet<UUID>();
    private final Set<String> rewardedCrystalPlacementBases = new HashSet<String>();
    private final Set<UUID> rewardedCrystalDestructions = new HashSet<UUID>();
    private final Set<String> rewardedCrystalDestructionBases = new HashSet<String>();
    private final Set<UUID> rewardedCrystalExplosions = new HashSet<UUID>();
    private final Set<UUID> penalizedOwnCrystalHits = new HashSet<UUID>();
    private final Set<String> rewardedCrystalExplosionBases = new HashSet<String>();
    private boolean hasPreviousPosition;
    private double previousX;
    private double previousY;
    private double previousZ;
    private boolean hasPreviousAimPotential;
    private double previousAimPotential;
    private long lastRewardedAttackTick = Long.MIN_VALUE;

    public CombatStats(UUID playerId) {
        this.playerId = playerId;
    }

    public void reset() {
        damageDealt = 0;
        damageTaken = 0;
        selfDamage = 0;
        policyKills = 0;
        healing = 0;
        totemPops = 0;
        totemPopsForced = 0;
        obsidianPlaced = 0;
        tacticalObsidianPlaced = 0;
        tacticalMinePlaceSequences = 0;
        policyBuiltCrystalChainsDamaging = 0;
        rewardedObsidianCombos = 0;
        crystalsPlaced = 0;
        crystalsDestroyed = 0;
        crystalsExploded = 0;
        policyCrystalChainsStarted = 0;
        policyCrystalChainsDetonated = 0;
        policyCrystalChainsDamaging = 0;
        policyCrystalChainsPopping = 0;
        rewardedCrystalCombos = 0;
        blocksPlaced = 0;
        blocksMined = 0;
        invalidInteractions = 0;
        approachTicks = 0;
        extremePitchTicks = 0;
        extremePitchStreak = 0;
        aimAlignmentTicks = 0;
        lockOnTicks = 0;
        rewardedLockOnTicks = 0;
        lockOnStreak = 0;
        attackSwings = 0;
        validAttackSwings = 0;
        missedAttackSwings = 0;
        spamAttackSwings = 0;
        penalizedAttackSwings = 0;
        rewardedAttackSwings = 0;
        hitsLanded = 0;
        rewardedHits = 0;
        pendingReward = 0;
        shapingRewardTotal = 0;
        terminalReward = 0;
        lastAutonomousActionTick = 0;
        inactionPenaltyTicks = 0;
        rewardTotals.clear();
        attributedEvents.clear();
        placedBlockKeys.clear();
        rewardedObsidianKeys.clear();
        rewardedObsidianComboBases.clear();
        rewardedTacticalMinePlaceKeys.clear();
        rewardedMinedKeys.clear();
        rewardedCrystalPlacements.clear();
        rewardedCrystalPlacementBases.clear();
        rewardedCrystalDestructions.clear();
        rewardedCrystalDestructionBases.clear();
        rewardedCrystalExplosions.clear();
        rewardedCrystalExplosionBases.clear();
        penalizedOwnCrystalHits.clear();
        hasPreviousPosition = false;
        previousX = previousY = previousZ = 0;
        hasPreviousAimPotential = false;
        previousAimPotential = 0;
        lastRewardedAttackTick = Long.MIN_VALUE;
    }

    void addPending(RewardReason reason, double reward) {
        pendingReward += reward;
        rewardTotals.put(reason, rewardTotals.containsKey(reason) ? rewardTotals.get(reason) + reward : reward);
    }

    void recordDelivered(double reward) {
        shapingRewardTotal += reward;
    }

    void markAutonomousAction(long tick) {
        lastAutonomousActionTick = Math.max(lastAutonomousActionTick, tick);
    }

    double rewardTotal(RewardReason reason) {
        Double value = rewardTotals.get(reason);
        return value == null ? 0 : value;
    }

    boolean claimObsidian(String blockKey, int cap) {
        placedBlockKeys.add(blockKey);
        return rewardedObsidianKeys.size() < cap && rewardedObsidianKeys.add(blockKey);
    }

    boolean claimObsidianCombo(String baseKey, int cap) {
        if (baseKey == null || baseKey.isEmpty()
                || rewardedObsidianComboBases.size() >= cap) return false;
        boolean claimed = rewardedObsidianComboBases.add(baseKey);
        if (claimed) rewardedObsidianCombos++;
        return claimed;
    }

    void rememberPlacedBlock(String blockKey) {
        placedBlockKeys.add(blockKey);
    }

    boolean wasPlacedBlock(String blockKey) {
        return placedBlockKeys.contains(blockKey);
    }

    boolean claimTacticalMinePlace(String blockKey, int cap) {
        return blockKey != null && !blockKey.isEmpty()
                && rewardedTacticalMinePlaceKeys.size() < cap
                && rewardedTacticalMinePlaceKeys.add(blockKey);
    }

    boolean claimMinedBlock(String blockKey, int cap) {
        return !placedBlockKeys.contains(blockKey)
                && rewardedMinedKeys.size() < cap
                && rewardedMinedKeys.add(blockKey);
    }

    boolean claimCrystalPlacement(UUID crystalId, String baseKey, int cap) {
        if (crystalId == null || baseKey == null || baseKey.isEmpty()
                || !rewardedCrystalPlacements.add(crystalId)) return false;
        return rewardedCrystalPlacementBases.size() < cap
                && rewardedCrystalPlacementBases.add(baseKey);
    }

    boolean claimCrystalDestruction(UUID crystalId, String baseKey, int cap) {
        if (crystalId == null || baseKey == null || baseKey.isEmpty()
                || !rewardedCrystalDestructions.add(crystalId)) return false;
        return rewardedCrystalDestructionBases.size() < cap
                && rewardedCrystalDestructionBases.add(baseKey);
    }

    boolean claimCrystalExplosion(UUID crystalId, String baseKey, int cap) {
        if (crystalId == null || baseKey == null || baseKey.isEmpty()
                || !rewardedCrystalExplosions.add(crystalId)) return false;
        return rewardedCrystalExplosionBases.size() < cap
                && rewardedCrystalExplosionBases.add(baseKey);
    }

    boolean claimCrystalCombo(int cap) {
        if (rewardedCrystalCombos >= cap) return false;
        rewardedCrystalCombos++;
        return true;
    }

    boolean claimOwnCrystalSelfHit(UUID crystalId) {
        return crystalId != null && penalizedOwnCrystalHits.add(crystalId);
    }

    boolean claimLockOnTick(int cap) {
        if (rewardedLockOnTicks >= cap) return false;
        rewardedLockOnTicks++;
        return true;
    }

    long ticksSinceRewardedAttack(long tick) {
        return lastRewardedAttackTick == Long.MIN_VALUE
                ? Long.MAX_VALUE : Math.max(0, tick - lastRewardedAttackTick);
    }

    boolean claimAttackReward(long tick, int cap) {
        if (rewardedAttackSwings >= cap) return false;
        rewardedAttackSwings++;
        lastRewardedAttackTick = tick;
        return true;
    }

    boolean claimHitReward(int cap) {
        if (rewardedHits >= cap) return false;
        rewardedHits++;
        return true;
    }

    boolean claimAttackPenalty(int cap) {
        if (penalizedAttackSwings >= cap) return false;
        penalizedAttackSwings++;
        return true;
    }

    boolean hasPreviousAimPotential() { return hasPreviousAimPotential; }
    double previousAimPotential() { return previousAimPotential; }

    void rememberAimPotential(double potential) {
        previousAimPotential = potential;
        hasPreviousAimPotential = true;
    }

    boolean hasPreviousPosition() { return hasPreviousPosition; }
    double previousX() { return previousX; }
    double previousY() { return previousY; }
    double previousZ() { return previousZ; }

    void rememberPosition(double x, double y, double z) {
        previousX = x;
        previousY = y;
        previousZ = z;
        hasPreviousPosition = true;
    }

    AttributedEvents events(ExecutionSource source) {
        AttributedEvents value = attributedEvents.get(source);
        if (value == null) {
            value = new AttributedEvents();
            attributedEvents.put(source, value);
        }
        return value;
    }

    AttributedEvents existingEvents(ExecutionSource source) {
        AttributedEvents value = attributedEvents.get(source);
        return value == null ? AttributedEvents.EMPTY : value;
    }

    static final class AttributedEvents {
        private static final AttributedEvents EMPTY = new AttributedEvents();
        double damageDealt;
        double directDamageDealt;
        double crystalDamageDealt;
        int hitsLanded;
        int crystalsPlaced;
        int crystalsDestroyed;
        int crystalsExploded;
        int crystalDamageEvents;
        int totemsForced;
        int crystalTotemsForced;
        int blocksPlaced;
        int blocksMined;
        int obsidianPlaced;
        int tacticalObsidianPlaced;
        int tacticalMinePlaceSequences;
        int policyBuiltCrystalChainsDamaging;
        long firstHitTick = -1L;
        long firstDamageTick = -1L;
    }
}
