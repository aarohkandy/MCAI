package dev.mcbot.arena;

public enum ArenaMode {
    SCRIPTED,
    SWORD,
    CRYSTAL,
    COMBINED;

    public static ArenaMode parse(String value) {
        if (value == null) return COMBINED;
        try {
            return ArenaMode.valueOf(value.trim().toUpperCase());
        } catch (IllegalArgumentException ignored) {
            return COMBINED;
        }
    }
}
