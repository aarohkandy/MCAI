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
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

public final class ControlServer implements AutoCloseable {
    private final MCAIPlugin plugin;
    private final ArenaManager manager;
    private final int port;
    private final Gson gson = new Gson();
    private final Set<Client> clients = ConcurrentHashMap.newKeySet();
    private volatile boolean running;
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

    public void broadcast(JsonObject event) {
        String line = gson.toJson(event);
        for (Client client : clients) client.send(line);
    }

    @Override
    public void close() {
        running = false;
        try { if (server != null) server.close(); } catch (IOException ignored) { }
        for (Client client : clients) client.close();
        clients.clear();
    }

    private void acceptLoop() {
        while (running) {
            Socket accepted = null;
            try {
                accepted = server.accept();
                Client client = new Client(accepted);
                clients.add(client);
                client.start();
            } catch (IOException error) {
                if (accepted != null) {
                    try { accepted.close(); } catch (IOException ignored) { }
                }
                if (running) plugin.getLogger().warning("Control accept failed: " + error.getMessage());
            }
        }
    }

    private JsonObject dispatch(JsonObject request) throws Exception {
        if (!"command".equals(request.has("type") ? request.get("type").getAsString() : "")) {
            throw new IllegalArgumentException("expected command message");
        }
        if (!request.has("command") || request.get("command").isJsonNull()) {
            throw new IllegalArgumentException("command field is required");
        }
        String command = request.get("command").getAsString();
        JsonObject payload = request.has("payload") && request.get("payload").isJsonObject()
                ? request.getAsJsonObject("payload") : new JsonObject();
        Future<JsonObject> future = plugin.getServer().getScheduler().callSyncMethod(plugin,
                () -> manager.handleCommand(command, payload));
        try {
            return future.get(10, TimeUnit.SECONDS);
        } catch (TimeoutException timeout) {
            // Best effort: if the task has not started yet, drop it so it cannot mutate
            // state after we have already told the client the command failed.
            future.cancel(false);
            throw timeout;
        }
    }

    private final class Client implements AutoCloseable {
        private static final String POISON = "__mcai_control_close__";

        private final Socket socket;
        private final BufferedReader reader;
        private final BufferedWriter writer;
        private final BlockingQueue<String> outbox = new ArrayBlockingQueue<String>(512);
        private volatile boolean closed;
        private Thread readThread;
        private Thread writeThread;

        private Client(Socket socket) throws IOException {
            this.socket = socket;
            this.reader = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
            this.writer = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8));
        }

        private void start() {
            writeThread = new Thread(this::writeLoop, "mcai-control-writer");
            writeThread.setDaemon(true);
            writeThread.start();
            readThread = new Thread(this::readLoop, "mcai-control-client");
            readThread.setDaemon(true);
            readThread.start();
        }

        private void readLoop() {
            try {
                String line;
                while (running && !closed && (line = reader.readLine()) != null) {
                    if (line.trim().isEmpty()) continue;
                    JsonObject response = new JsonObject();
                    response.addProperty("type", "response");
                    try {
                        JsonObject request = gson.fromJson(line, JsonObject.class);
                        if (request == null) throw new IllegalArgumentException("request must be a JSON object");
                        if (request.has("id")) response.add("id", request.get("id"));
                        response.addProperty("ok", true);
                        response.add("payload", dispatch(request));
                    } catch (Exception error) {
                        response.addProperty("ok", false);
                        response.addProperty("error", rootMessage(error));
                    }
                    send(gson.toJson(response));
                }
            } catch (IOException error) {
                if (running) plugin.getLogger().fine("Control client closed: " + error.getMessage());
            } finally {
                close();
            }
        }

        // Runs on a dedicated thread so a stalled consumer can never block the main
        // server thread that produces broadcasts and command responses.
        private void writeLoop() {
            try {
                while (!closed) {
                    String line = outbox.take();
                    if (POISON.equals(line)) break;
                    writer.write(line);
                    writer.newLine();
                    writer.flush();
                }
            } catch (InterruptedException ignored) {
                Thread.currentThread().interrupt();
            } catch (IOException error) {
                if (running) plugin.getLogger().fine("Control client write failed: " + error.getMessage());
            } finally {
                close();
            }
        }

        private void send(String line) {
            if (closed) return;
            // Never block the caller (often the main thread). If the consumer has fallen
            // too far behind, drop the connection instead of stalling the server.
            if (!outbox.offer(line)) close();
        }

        @Override
        public void close() {
            synchronized (this) {
                if (closed) return;
                closed = true;
            }
            clients.remove(this);
            outbox.offer(POISON);
            try { socket.close(); } catch (IOException ignored) { }
        }
    }

    private static String rootMessage(Throwable error) {
        Throwable value = error;
        while (value.getCause() != null) value = value.getCause();
        return value.getMessage() == null ? value.getClass().getSimpleName() : value.getMessage();
    }
}
