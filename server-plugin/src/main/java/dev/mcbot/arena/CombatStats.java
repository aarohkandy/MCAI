package dev.mcbot.arena;

import java.util.UUID;

public final class CombatStats {
    public final UUID playerId;
    public double damageDealt;
    public double damageTaken;
    public double healing;
    public int totemPops;
    public int crystalsPlaced;
    public int crystalsDestroyed;
    public int blocksPlaced;
    public int blocksMined;
    public int invalidInteractions;
    public double pendingReward;

    public CombatStats(UUID playerId) {
        this.playerId = playerId;
    }

    public void reset() {
        damageDealt = 0;
        damageTaken = 0;
        healing = 0;
        totemPops = 0;
        crystalsPlaced = 0;
        crystalsDestroyed = 0;
        blocksPlaced = 0;
        blocksMined = 0;
        invalidInteractions = 0;
        pendingReward = 0;
    }
}
