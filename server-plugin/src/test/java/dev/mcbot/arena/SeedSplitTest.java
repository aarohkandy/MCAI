package dev.mcbot.arena;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class SeedSplitTest {
    @Test
    void splitIsStableAndCloseToTwentyPercent() {
        assertTrue(SeedSplit.isHeldOut(0, 0, 0));
        assertEquals(false, SeedSplit.isHeldOut(42, 3, 1));
        int heldOut = 0;
        for (int seed = 0; seed < 1000; seed++) {
            boolean first = SeedSplit.isHeldOut(seed, seed % 6, (seed / 6) % 6);
            assertEquals(first, SeedSplit.isHeldOut(seed, seed % 6, (seed / 6) % 6));
            if (first) heldOut++;
        }
        assertTrue(heldOut >= 150 && heldOut <= 250, "unexpected held-out count: " + heldOut);
    }
}
