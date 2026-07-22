package dev.mcbot.arena;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * Episode-local attribution for a crystal's complete place -> detonate -> damage chain.
 * A chain is autonomous only when the same fighter performed both control actions
 * with policy-owned input on a legal base within the configured time window.
 */
final class CrystalChainTracker {
    private final int maximumPlacementToDetonationTicks;
    private final Map<UUID, Chain> chains = new HashMap<UUID, Chain>();
    private String episodeId = "";

    CrystalChainTracker(int maximumPlacementToDetonationTicks) {
        this.maximumPlacementToDetonationTicks = Math.max(1, maximumPlacementToDetonationTicks);
    }

    void reset(String episodeId) {
        this.episodeId = episodeId == null ? "" : episodeId;
        chains.clear();
    }

    Chain recordPlacement(UUID crystalId, UUID placerId, ExecutionSource source,
                          long tick, boolean legalBase) {
        return recordPlacement(crystalId, placerId, source, tick, legalBase, "", false);
    }

    Chain recordPlacement(UUID crystalId, UUID placerId, ExecutionSource source,
                          long tick, boolean legalBase, String baseKey,
                          boolean policyBuiltBase) {
        if (crystalId == null || placerId == null || source == null) return null;
        Chain chain = new Chain(episodeId + ":" + crystalId, crystalId, placerId,
                source, tick, legalBase, baseKey, policyBuiltBase,
                maximumPlacementToDetonationTicks);
        chains.put(crystalId, chain);
        return chain;
    }

    Chain recordDetonation(UUID crystalId, UUID detonatorId, ExecutionSource source, long tick) {
        Chain chain = chains.get(crystalId);
        if (chain == null || detonatorId == null || source == null) return chain;
        chain.recordDetonation(detonatorId, source, tick);
        return chain;
    }

    Chain recordOpponentDamage(UUID crystalId, UUID attackerId, ExecutionSource source, long tick) {
        Chain chain = chains.get(crystalId);
        if (chain != null) chain.recordOpponentDamage(attackerId, source, tick);
        return chain;
    }

    Chain recordOpponentPop(UUID crystalId, UUID attackerId, ExecutionSource source, long tick) {
        Chain chain = chains.get(crystalId);
        if (chain != null) chain.recordOpponentPop(attackerId, source, tick);
        return chain;
    }

    Chain chain(UUID crystalId) {
        return chains.get(crystalId);
    }

    Collection<Chain> chains() {
        return new ArrayList<Chain>(chains.values());
    }

    static final class Chain {
        private final String sequenceId;
        private final UUID crystalId;
        private final UUID placerId;
        private final ExecutionSource placementSource;
        private final long placementTick;
        private final boolean legalBase;
        private final String baseKey;
        private final boolean policyBuiltBase;
        private final int maximumPlacementToDetonationTicks;
        private UUID detonatorId;
        private ExecutionSource detonationSource;
        private long detonationTick = -1L;
        private boolean detonationCounted;
        private boolean opponentDamaged;
        private boolean damageCounted;
        private boolean opponentPopped;
        private boolean popCounted;
        private boolean policyBuiltDamageCounted;
        private boolean comboRewardClaimed;
        private boolean popRewardClaimed;

        private Chain(String sequenceId, UUID crystalId, UUID placerId,
                      ExecutionSource placementSource, long placementTick,
                      boolean legalBase, String baseKey, boolean policyBuiltBase,
                      int maximumPlacementToDetonationTicks) {
            this.sequenceId = sequenceId;
            this.crystalId = crystalId;
            this.placerId = placerId;
            this.placementSource = placementSource;
            this.placementTick = placementTick;
            this.legalBase = legalBase;
            this.baseKey = baseKey == null ? "" : baseKey;
            this.policyBuiltBase = policyBuiltBase;
            this.maximumPlacementToDetonationTicks = maximumPlacementToDetonationTicks;
        }

        private void recordDetonation(UUID playerId, ExecutionSource source, long tick) {
            if (detonationTick >= 0L || tick < placementTick) return;
            detonatorId = playerId;
            detonationSource = source;
            detonationTick = tick;
        }

        private void recordOpponentDamage(UUID playerId, ExecutionSource source, long tick) {
            if (isAutonomousActionByDetonator(playerId, source, tick)) opponentDamaged = true;
        }

        private void recordOpponentPop(UUID playerId, ExecutionSource source, long tick) {
            if (isAutonomousActionByDetonator(playerId, source, tick)) opponentPopped = true;
        }

        private boolean isAutonomousActionByDetonator(UUID playerId, ExecutionSource source, long tick) {
            return isAutonomousSequence() && detonatorId.equals(playerId)
                    && source == ExecutionSource.POLICY && tick >= detonationTick;
        }

        boolean isAutonomousSequence() {
            return legalBase
                    && placementSource == ExecutionSource.POLICY
                    && detonationSource == ExecutionSource.POLICY
                    && placerId.equals(detonatorId)
                    && detonationTick >= placementTick
                    && detonationTick - placementTick <= maximumPlacementToDetonationTicks;
        }

        boolean claimDetonationCount() {
            if (!isAutonomousSequence() || detonationCounted) return false;
            detonationCounted = true;
            return true;
        }

        boolean claimDamageCount() {
            if (!opponentDamaged || damageCounted) return false;
            damageCounted = true;
            return true;
        }

        boolean claimPolicyBuiltDamageCount() {
            if (!opponentDamaged || !isAutonomousSequence() || !policyBuiltBase
                    || baseKey.isEmpty() || policyBuiltDamageCounted) return false;
            policyBuiltDamageCounted = true;
            return true;
        }

        boolean claimPopCount() {
            if (!opponentPopped || popCounted) return false;
            popCounted = true;
            return true;
        }

        boolean claimComboReward() {
            if (!opponentDamaged || comboRewardClaimed) return false;
            comboRewardClaimed = true;
            return true;
        }

        boolean claimPopReward() {
            if (!opponentPopped || !comboRewardClaimed || popRewardClaimed) return false;
            popRewardClaimed = true;
            return true;
        }

        String sequenceId() { return sequenceId; }
        UUID crystalId() { return crystalId; }
        UUID placerId() { return placerId; }
        ExecutionSource placementSource() { return placementSource; }
        long placementTick() { return placementTick; }
        boolean legalBase() { return legalBase; }
        String baseKey() { return baseKey; }
        boolean policyBuiltBase() { return policyBuiltBase; }
        UUID detonatorId() { return detonatorId; }
        ExecutionSource detonationSource() { return detonationSource; }
        long detonationTick() { return detonationTick; }
        boolean opponentDamaged() { return opponentDamaged; }
        boolean opponentPopped() { return opponentPopped; }
    }
}
