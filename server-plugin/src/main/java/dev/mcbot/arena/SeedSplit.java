package dev.mcbot.arena;

import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

final class SeedSplit {
    private SeedSplit() { }

    static boolean isHeldOut(long arenaSeed, int actionDelay, int observationDelay) {
        try {
            String value = arenaSeed + ":" + actionDelay + ":" + observationDelay;
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.US_ASCII));
            long prefix = Integer.toUnsignedLong(ByteBuffer.wrap(digest, 0, 4).getInt());
            return prefix % 100L < 20L;
        } catch (NoSuchAlgorithmException impossible) {
            throw new IllegalStateException("SHA-256 is unavailable", impossible);
        }
    }
}
