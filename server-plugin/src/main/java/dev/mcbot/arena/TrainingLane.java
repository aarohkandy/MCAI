package dev.mcbot.arena;

/** One independently measured lane in the four-way parallel curriculum. */
final class TrainingLane {
    private final String id;
    private final ArenaMode mode;
    private final ArenaSizeCurriculum curriculum;

    TrainingLane(String id, ArenaMode mode, ArenaSizeCurriculum curriculum) {
        if (id == null || id.trim().isEmpty()) throw new IllegalArgumentException("lane id is required");
        if (mode == null || curriculum == null) throw new IllegalArgumentException("lane mode and curriculum are required");
        this.id = id;
        this.mode = mode;
        this.curriculum = curriculum;
    }

    String id() { return id; }
    ArenaMode mode() { return mode; }
    ArenaSizeCurriculum curriculum() { return curriculum; }
}
