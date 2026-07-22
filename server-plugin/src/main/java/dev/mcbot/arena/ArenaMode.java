package dev.mcbot.arena;

public enum ArenaMode {
    SCRIPTED,
    SWORD,
    CRYSTAL,
    COMBINED,
    TERRAIN;

    public boolean hasCrystalLayout() {
        return this == CRYSTAL || this == COMBINED || this == TERRAIN;
    }

    public boolean hasTerrainLayout() {
        return this == TERRAIN;
    }

    /**
     * A generic arm swing is unambiguous only in the sword-only curriculum.
     * Crystal/combined swings may be mining or crystal destruction and are
     * scored by their concrete block/entity events instead.
     */
    public boolean usesGenericArmSwingShaping() {
        return this == SWORD;
    }

    public static ArenaMode parse(String value) {
        if (value == null) return COMBINED;
        try {
            return ArenaMode.valueOf(value.trim().toUpperCase());
        } catch (IllegalArgumentException ignored) {
            return COMBINED;
        }
    }
}
