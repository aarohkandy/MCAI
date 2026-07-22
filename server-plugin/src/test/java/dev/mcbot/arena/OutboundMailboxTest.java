package dev.mcbot.arena;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class OutboundMailboxTest {
    @Test
    void replacesStaleSnapshotForTheSameArena() throws Exception {
        OutboundMailbox<String> mailbox = new OutboundMailbox<String>(4, 4);

        assertTrue(mailbox.offerSnapshot(1, "arena-1", "old"));
        assertTrue(mailbox.offerSnapshot(2, "arena-1", "latest"));

        assertEquals(1, mailbox.pendingSnapshots());
        assertEquals("latest", mailbox.take().getValue());
    }

    @Test
    void replacementKeepsLifecycleMessagesInSequenceOrder() throws Exception {
        OutboundMailbox<String> mailbox = new OutboundMailbox<String>(4, 4);

        mailbox.offerSnapshot(1, "arena-1", "stale snapshot");
        mailbox.offerOrdered(2, "match ended");
        mailbox.offerSnapshot(3, "arena-1", "new episode snapshot");

        assertEquals("match ended", mailbox.take().getValue());
        assertEquals("new episode snapshot", mailbox.take().getValue());
    }

    @Test
    void snapshotCapacityEvictsOldestArenaWithoutGrowing() throws Exception {
        OutboundMailbox<String> mailbox = new OutboundMailbox<String>(4, 2);

        mailbox.offerSnapshot(1, "arena-1", "one");
        mailbox.offerSnapshot(2, "arena-2", "two");
        mailbox.offerSnapshot(3, "arena-3", "three");

        assertEquals(2, mailbox.pendingSnapshots());
        assertEquals("two", mailbox.take().getValue());
        assertEquals("three", mailbox.take().getValue());
    }

    @Test
    void orderedQueueRejectsOverflowInsteadOfBacklogging() {
        OutboundMailbox<String> mailbox = new OutboundMailbox<String>(2, 2);

        assertTrue(mailbox.offerOrdered(1, "one"));
        assertTrue(mailbox.offerOrdered(2, "two"));
        assertFalse(mailbox.offerOrdered(3, "three"));
        assertEquals(2, mailbox.pendingOrdered());
    }

    @Test
    void closeDropsPendingWorkAndStopsConsumer() throws Exception {
        OutboundMailbox<String> mailbox = new OutboundMailbox<String>(2, 2);
        mailbox.offerOrdered(1, "one");

        mailbox.close();

        assertNull(mailbox.take());
        assertEquals(0, mailbox.pendingOrdered());
        assertEquals(0, mailbox.pendingSnapshots());
    }
}
