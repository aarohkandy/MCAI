package dev.mcbot.arena;

import org.bukkit.GameMode;
import org.bukkit.Location;
import org.bukkit.entity.Player;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.UUID;

public final class SpectatorController {
    private final ArenaManager manager;
    private final Map<UUID, View> views = new HashMap<UUID, View>();
    private long tick;

    public SpectatorController(ArenaManager manager) {
        this.manager = manager;
    }

    public boolean watch(Player spectator, String arenaId, String mode) {
        Arena arena = manager.byId(arenaId);
        if (arena == null || !arena.isActive()) return false;
        spectator.setGameMode(GameMode.SPECTATOR);
        View view = new View(arena, "orbit".equalsIgnoreCase(mode));
        views.put(spectator.getUniqueId(), view);
        if (!view.orbit) spectator.setSpectatorTarget(arena.players().iterator().next());
        return true;
    }

    public boolean next(Player spectator) {
        List<Arena> active = manager.activeArenas();
        if (active.isEmpty()) return false;
        View current = views.get(spectator.getUniqueId());
        int index = current == null ? -1 : active.indexOf(current.arena);
        Arena next = active.get((index + 1) % active.size());
        return watch(spectator, next.getId(), current != null && current.orbit ? "orbit" : "pov");
    }

    public void tick() {
        tick++;
        for (Player spectator : new ArrayList<Player>(org.bukkit.Bukkit.getOnlinePlayers())) {
            View view = views.get(spectator.getUniqueId());
            if (view == null) continue;
            if (!view.arena.isActive() || spectator.getGameMode() != GameMode.SPECTATOR) {
                views.remove(spectator.getUniqueId());
                continue;
            }
            if (view.orbit) {
                spectator.setSpectatorTarget(null);
                double angle = tick * 0.015;
                Location center = view.arena.getCenter();
                Location camera = center.clone().add(Math.cos(angle) * 12, 9, Math.sin(angle) * 12);
                camera.setYaw((float) Math.toDegrees(-angle + Math.PI / 2));
                camera.setPitch(25);
                spectator.teleport(camera);
            }
        }
    }

    private static final class View {
        private final Arena arena;
        private final boolean orbit;
        private View(Arena arena, boolean orbit) { this.arena = arena; this.orbit = orbit; }
    }
}
