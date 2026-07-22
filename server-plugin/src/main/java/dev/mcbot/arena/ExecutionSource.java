package dev.mcbot.arena;

import java.util.Locale;

/** Attribution supplied by the worker for the input that actually reached Minecraft. */
public enum ExecutionSource {
    POLICY,
    TEACHER_SWORD,
    TEACHER_CRYSTAL,
    TEACHER_BLOCK,
    SAFETY;

    String wireName() {
        return name().toLowerCase(Locale.ROOT);
    }

    boolean isAutonomous() {
        return this == POLICY;
    }

    static ExecutionSource parse(String value) {
        if (value == null) throw new IllegalArgumentException("execution source is required");
        try {
            return valueOf(value.trim().toUpperCase(Locale.ROOT));
        } catch (IllegalArgumentException error) {
            throw new IllegalArgumentException("unknown execution source: " + value);
        }
    }
}
