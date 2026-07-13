package dev.mcbot.arena;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import org.bukkit.scheduler.BukkitTask;

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
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

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
        private final Object writeLock = new Object();
        private Thread thread;

        private Client(Socket socket) throws IOException {
            this.socket = socket;
            this.reader = new BufferedReader(new InputStreamReader(socket.getInputStream(), StandardCharsets.UTF_8));
            this.writer = new BufferedWriter(new OutputStreamWriter(socket.getOutputStream(), StandardCharsets.UTF_8));
        }

        private void start() {
            thread = new Thread(this::readLoop, "mcai-control-client");
            thread.setDaemon(true);
            thread.start();
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
                    send(gson.toJson(response));
                }
            } catch (IOException error) {
                if (running) plugin.getLogger().fine("Control client closed: " + error.getMessage());
            } finally {
                close();
            }
        }

        private void send(String line) {
            synchronized (writeLock) {
                try {
                    writer.write(line);
                    writer.newLine();
                    writer.flush();
                } catch (IOException error) {
                    close();
                }
            }
        }

        @Override
        public void close() {
            clients.remove(this);
            try { socket.close(); } catch (IOException ignored) { }
        }
    }

    private static String rootMessage(Throwable error) {
        Throwable value = error;
        while (value.getCause() != null) value = value.getCause();
        return value.getMessage() == null ? value.getClass().getSimpleName() : value.getMessage();
    }
}
