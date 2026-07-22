package dev.mcbot.arena;

import java.util.ArrayDeque;
import java.util.Deque;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A bounded per-client outbound mailbox.
 *
 * Ordered messages are retained in FIFO order. Snapshot messages are retained
 * only as the latest value for each key, so a client that cannot keep up never
 * accumulates an unbounded history of stale arena state.
 */
final class OutboundMailbox<T> {
    private final int orderedCapacity;
    private final int snapshotCapacity;
    private final Deque<Entry<T>> ordered = new ArrayDeque<Entry<T>>();
    private final Map<String, Entry<T>> snapshots = new LinkedHashMap<String, Entry<T>>();
    private boolean closed;

    OutboundMailbox(int orderedCapacity, int snapshotCapacity) {
        if (orderedCapacity < 1) throw new IllegalArgumentException("orderedCapacity must be positive");
        if (snapshotCapacity < 1) throw new IllegalArgumentException("snapshotCapacity must be positive");
        this.orderedCapacity = orderedCapacity;
        this.snapshotCapacity = snapshotCapacity;
    }

    synchronized boolean offerOrdered(long sequence, T value) {
        if (closed || ordered.size() >= orderedCapacity) return false;
        ordered.addLast(new Entry<T>(sequence, value));
        notifyAll();
        return true;
    }

    synchronized boolean offerSnapshot(long sequence, String key, T value) {
        if (closed) return false;
        if (!snapshots.containsKey(key) && snapshots.size() >= snapshotCapacity) {
            removeOldestSnapshot();
        }
        snapshots.put(key, new Entry<T>(sequence, value));
        notifyAll();
        return true;
    }

    /** Returns null after close; otherwise waits for the next sequence-aware message. */
    synchronized Entry<T> take() throws InterruptedException {
        while (!closed && ordered.isEmpty() && snapshots.isEmpty()) wait();
        if (closed) return null;

        Entry<T> orderedHead = ordered.peekFirst();
        String snapshotKey = oldestSnapshotKey();
        Entry<T> snapshotHead = snapshotKey == null ? null : snapshots.get(snapshotKey);
        if (orderedHead != null && (snapshotHead == null
                || orderedHead.getSequence() <= snapshotHead.getSequence())) {
            return ordered.removeFirst();
        }
        return snapshots.remove(snapshotKey);
    }

    synchronized void close() {
        closed = true;
        ordered.clear();
        snapshots.clear();
        notifyAll();
    }

    synchronized int pendingOrdered() {
        return ordered.size();
    }

    synchronized int pendingSnapshots() {
        return snapshots.size();
    }

    private void removeOldestSnapshot() {
        String key = oldestSnapshotKey();
        if (key != null) snapshots.remove(key);
    }

    private String oldestSnapshotKey() {
        String oldestKey = null;
        long oldestSequence = Long.MAX_VALUE;
        for (Map.Entry<String, Entry<T>> candidate : snapshots.entrySet()) {
            if (candidate.getValue().getSequence() < oldestSequence) {
                oldestSequence = candidate.getValue().getSequence();
                oldestKey = candidate.getKey();
            }
        }
        return oldestKey;
    }

    static final class Entry<T> {
        private final long sequence;
        private final T value;

        private Entry(long sequence, T value) {
            this.sequence = sequence;
            this.value = value;
        }

        long getSequence() {
            return sequence;
        }

        T getValue() {
            return value;
        }
    }
}
