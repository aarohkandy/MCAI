package dev.mcbot.arena;

/**
 * Pure, deterministic defensive-kit curriculum.
 *
 * <p>The four tiers are deliberately independent of arena randomness.  An
 * episode resolves one immutable {@link KitSpec}, and that same object is
 * applied to both fighters so layout seeds can never produce unequal gear.</p>
 */
final class KitCurriculum {
    private static final KitSpec SWORD_RETENTION = new KitSpec(4, 0, 0, false);
    private static final KitSpec CRYSTAL_RETENTION = new KitSpec(2, 0, 0, true);
    private static final KitSpec FULL_COMBAT = new KitSpec(4, 16, 4, true);

    private static final KitSpec[] PROGRESSIVE_COMBAT = new KitSpec[]{
            new KitSpec(2, 0, 0, true),
            // Stage two expands the arena only.  Keeping survivability fixed
            // isolates the spatial skill instead of making the first radius
            // increase simultaneously require several extra kill actions.
            new KitSpec(2, 0, 0, true),
            new KitSpec(3, 4, 1, true),
            FULL_COMBAT
    };

    private KitCurriculum() {}

    static KitSpec forEpisode(ArenaMode mode, int stage, int stageCount) {
        if (mode == ArenaMode.SWORD) return SWORD_RETENTION;
        if (mode == ArenaMode.CRYSTAL) return CRYSTAL_RETENTION;
        if (mode != ArenaMode.COMBINED && mode != ArenaMode.TERRAIN) return FULL_COMBAT;

        // A one-stage/non-curriculum match preserves the historical full kit.
        // Multi-stage combined and terrain lanes map their stages across four
        // fixed tiers, with the final stage always receiving the full kit.
        int boundedCount = Math.max(1, stageCount);
        if (boundedCount == 1) return FULL_COMBAT;
        int boundedStage = Math.max(1, Math.min(stage, boundedCount));
        double progress = (boundedStage - 1.0) / (boundedCount - 1.0);
        int tier = (int) Math.round(progress * (PROGRESSIVE_COMBAT.length - 1));
        return PROGRESSIVE_COMBAT[tier];
    }

    static final class KitSpec {
        private final int protectionLevel;
        private final int goldenApples;
        private final int spareTotems;
        private final boolean offhandTotem;

        private KitSpec(int protectionLevel, int goldenApples,
                        int spareTotems, boolean offhandTotem) {
            this.protectionLevel = protectionLevel;
            this.goldenApples = goldenApples;
            this.spareTotems = spareTotems;
            this.offhandTotem = offhandTotem;
        }

        int protectionLevel() { return protectionLevel; }
        int goldenApples() { return goldenApples; }
        int spareTotems() { return spareTotems; }
        boolean hasOffhandTotem() { return offhandTotem; }
    }
}
