package dev.mcbot.arena;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class ExecutionSourceTest {
    @Test
    void parsesTheExactWorkerWireValues() {
        assertEquals(ExecutionSource.POLICY, ExecutionSource.parse("policy"));
        assertEquals(ExecutionSource.TEACHER_SWORD, ExecutionSource.parse("teacher_sword"));
        assertEquals(ExecutionSource.TEACHER_CRYSTAL, ExecutionSource.parse("teacher_crystal"));
        assertEquals(ExecutionSource.TEACHER_BLOCK, ExecutionSource.parse("teacher_block"));
        assertEquals(ExecutionSource.SAFETY, ExecutionSource.parse("safety"));
        assertThrows(IllegalArgumentException.class, () -> ExecutionSource.parse("camera"));
    }

    @Test
    void onlyPolicyIsAutonomous() {
        assertTrue(ExecutionSource.POLICY.isAutonomous());
        assertFalse(ExecutionSource.TEACHER_SWORD.isAutonomous());
        assertFalse(ExecutionSource.TEACHER_CRYSTAL.isAutonomous());
        assertFalse(ExecutionSource.TEACHER_BLOCK.isAutonomous());
        assertFalse(ExecutionSource.SAFETY.isAutonomous());
    }
}
