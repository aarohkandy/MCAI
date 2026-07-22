package dev.mcbot.arena;

import com.google.gson.Gson;
import com.google.gson.JsonObject;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.net.InetAddress;
import java.net.ServerSocket;
import java.net.Socket;
import java.nio.charset.StandardCharsets;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.RejectedExecutionException;
import java.util.concurrent.ThreadFactory;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;

public final class ControlServer implements AutoCloseable {
    private static final int MAX_ORDERED_MESSAGES_PER_CLIENT = 512;
    private static final int MAX_PENDING_SNAPSHOTS_PER_CLIENT = 16;
    private static final String SNAPSHOT_EVENT = "arena_snapshot";
    private final MCAIPlugin plugin;
    private final ArenaManager manager;
    private final int port;
    private final Gson gson = new Gson();
    private final Set<Client> clients = ConcurrentHashMap.newKeySet();
    private final Object publicationLock = new Object();
    private final ExecutorService disconnectExecutor = Executors.newSingleThreadExecutor(new ThreadFactory() {
        @Override
        public Thread newThread(Runnable runnable) {
            Thread thread = new Thread(runnable, "mcai-control-disconnect");
            thread.setDaemon(true);
            return thread;
        }
    });
    private volatile boolean running;
    private long nextSequence;
    private ServerSocket server;
    private Thread acceptThread;

    public ControlServer(MCAIPlugin plugin, ArenaManager manager, int port) {
        this.plugin = plugin;
        this.manager = manager;
        this.port = port;
    }

    public void start() throws IOException {
        server = new ServerSocket(port, 32, InetAddress.getLoopbackAddress());
        running = true;
        acceptThread = new Thread(this::acceptLoop, "mcai-control-accept");
        acceptThread.setDaemon(true);
        acceptThread.start();
        plugin.getLogger().info("Rollout control listening only on " + server.getInetAddress() + ":" + port);
    }

    /**
     * Publishes without serializing or touching a socket on the caller thread.
     * Arena snapshots are coalesced per arena by each client's writer mailbox.
     */
    public void broadcast(JsonObject event) {
        String snapshotKey = snapshotKey(event);
        synchronized (publicationLock) {
            long sequence = nextSequence++;
            for (Client client : clients) {
                if (!client.enqueue(sequence, snapshotKey, event)) disconnectSlowClient(client);
            }
        }
    }

    @Override
    public void close() {
        running = false;
        try { if (server != null) server.close(); } catch (IOException ignored) { }
        for (Client client : clients) client.close();
        clients.clear();
        disconnectExecutor.shutdownNow();
    }

    private void acceptLoop() {
        while (running) {
            try {
                Client client = new Client(server.accept());
                clients.add(client);
                client.start();
            } catch (IOException error) {
                if (running) plugin.getLogger().warning("Control accept failed: " + error.getMessage());
            }
        }
    }

    private JsonObject dispatch(JsonObject request) throws Exception {
        if (!"command".equals(request.has("type") ? request.get("type").getAsString() : "")) {
            throw new IllegalArgumentException("expected command message");
        }
        String command = request.get("command").getAsString();
        JsonObject payload = request.has("payload") && request.get("payload").isJsonObject()
                ? request.getAsJsonObject("payload") : new JsonObject();
        Future<JsonObject> future = plugin.getServer().getScheduler().callSyncMethod(plugin,
                () -> manager.handleCommand(command, payload));
        return future.get(10, TimeUnit.SECONDS);
    }

    private final class Client implements AutoCloseable {
        private final Socket socket;
        private final BufferedReader reader;
        private final BufferedWriter writer;
        private final OutboundMailbox<JsonObject> outbound = new OutboundMailbox<JsonObject>(
                MAX_ORDERED_MESSAGES_PER_CLIENT, MAX_PENDING_SNAPSHOTS_PER_CLIENT);
        private final AtomicBoolean closed = new AtomicBoolean();
        private final AtomicBoolean closeScheduled = new AtomicBoolean();
        private Thread readerThread;
        private Thread writerThread;

        private Client(Socket socket) throws IOException {
            this.socket = socket;
            this.reader = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
            this.writer = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8));
        }

        private void start() {
            writerThread = new Thread(this::writeLoop, "mcai-control-writer");
            writerThread.setDaemon(true);
            writerThread.start();
            readerThread = new Thread(this::readLoop, "mcai-control-reader");
            readerThread.setDaemon(true);
            readerThread.start();
        }

        private void readLoop() {
            try {
                String line;
                while (running && (line = reader.readLine()) != null) {
                    if (line.trim().isEmpty()) continue;
                    JsonObject request = gson.fromJson(line, JsonObject.class);
                    JsonObject response = new JsonObject();
                    response.addProperty("type", "response");
                    if (request.has("id")) response.add("id", request.get("id"));
                    try {
                        response.addProperty("ok", true);
                        response.add("payload", dispatch(request));
                    } catch (Exception error) {
                        response.addProperty("ok", false);
                        response.addProperty("error", rootMessage(error));
                    }
                    send(response);
                }
            } catch (IOException error) {
                if (running) plugin.getLogger().fine("Control client closed: " + error.getMessage());
            } finally {
                close();
            }
        }

        private void writeLoop() {
            try {
                while (running && !closed.get()) {
                    OutboundMailbox.Entry<JsonObject> message = outbound.take();
                    if (message == null) break;
                    String line = gson.toJson(message.getValue());
                    try {
                        writer.write(line);
                        writer.newLine();
                        writer.flush();
                    } catch (IOException error) {
                        if (running) plugin.getLogger().fine("Control writer closed: " + error.getMessage());
                        break;
                    }
                }
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
            } finally {
                close();
            }
        }

        private boolean enqueue(long sequence, String snapshotKey, JsonObject message) {
            if (closed.get()) return false;
            return snapshotKey == null
                    ? outbound.offerOrdered(sequence, message)
                    : outbound.offerSnapshot(sequence, snapshotKey, message);
        }

        private void send(JsonObject message) {
            synchronized (publicationLock) {
                if (!enqueue(nextSequence++, null, message)) disconnectSlowClient(this);
            }
        }

        @Override
        public void close() {
            if (!closed.compareAndSet(false, true)) return;
            clients.remove(this);
            outbound.close();
            try { socket.close(); } catch (IOException ignored) { }
        }
    }

    private void disconnectSlowClient(final Client client) {
        if (!client.closeScheduled.compareAndSet(false, true)) return;
        try {
            disconnectExecutor.execute(new Runnable() {
                @Override
                public void run() {
                    if (running) plugin.getLogger().warning(
                            "Disconnecting slow control client after its bounded outbound queue filled");
                    client.close();
                }
            });
        } catch (RejectedExecutionException ignored) {
            // Server shutdown closes every client directly.
        }
    }

    private static String snapshotKey(JsonObject event) {
        if (event == null || !event.has("event")
                || !SNAPSHOT_EVENT.equals(event.get("event").getAsString())) return null;
        return event.has("arena_id") ? event.get("arena_id").getAsString() : SNAPSHOT_EVENT;
    }

    private static String rootMessage(Throwable error) {
        Throwable value = error;
        while (value.getCause() != null) value = value.getCause();
        return value.getMessage() == null ? value.getClass().getSimpleName() : value.getMessage();
    }
}
