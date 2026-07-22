package dev.mcbot.arena;

import net.md_5.bungee.api.ChatMessageType;
import net.md_5.bungee.api.chat.TextComponent;
import org.bukkit.GameMode;
import org.bukkit.Location;
import org.bukkit.entity.Player;
import org.bukkit.util.Vector;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.UUID;

public final class SpectatorController {
    // Auto-pairing deliberately leaves a short gap between episodes.  Keep the
    // camera on the same physical platform through that gap so Eaglercraft does
    // not bounce between far-apart arena chunks while matches recycle.
    static final long ARENA_RESTART_GRACE_TICKS = 60L;
    // Five camera teleports per second keeps Eaglercraft below its movement
    // packet limiter while the angle remains derived from real server ticks.
    static final long ORBIT_UPDATE_INTERVAL_TICKS = 4L;
    static final double ORBIT_RADIANS_PER_TICK = 0.015;

    private final ArenaManager manager;
    private final Map<UUID, View> views = new HashMap<UUID, View>();
    private long tick;

    public SpectatorController(ArenaManager manager) {
        this.manager = manager;
    }

    public boolean watch(Player spectator, String arenaId, String mode) {
        // Never turn a fighter into a camera or feed spectator movement back
        // into an assigned bot. Camera state exists only for unassigned users.
        if (manager.arenaFor(spectator) != null) return false;
        Arena arena = manager.byId(arenaId);
        if (arena == null || !arena.isActive()) return false;
        boolean orbit = "orbit".equalsIgnoreCase(mode);
        Player target = orbit ? null : firstAssignedFighter(arena);
        if (!orbit && target == null) return false;
        spectator.setGameMode(GameMode.SPECTATOR);
        View view = new View(arena, orbit);
        views.put(spectator.getUniqueId(), view);
        if (view.orbit) {
            spectator.setSpectatorTarget(null);
            updateOrbit(spectator, view);
        } else {
            spectator.setSpectatorTarget(target);
        }
        return true;
    }

    public SwitchResult next(Player spectator) {
        List<Arena> active = manager.activeArenas();
        if (active.isEmpty()) return SwitchResult.noActive();
        View current = views.get(spectator.getUniqueId());
        String currentId = current == null ? null : current.arena.getId();
        List<String> activeIds = new ArrayList<String>();
        for (Arena arena : active) activeIds.add(arena.getId());
        String nextId = selectNextArenaId(activeIds, currentId);

        // A single active match cannot be switched away from. Report that explicitly
        // instead of claiming success while leaving the camera in the same place.
        if (active.size() == 1 && nextId.equalsIgnoreCase(currentId == null ? "" : currentId)) {
            return SwitchResult.onlyActive(nextId, matchup(active.get(0)));
        }

        Arena next = manager.byId(nextId);
        boolean orbit = current == null || current.orbit;
        if (next == null || !watch(spectator, nextId, orbit ? "orbit" : "pov")) {
            return SwitchResult.noActive();
        }
        return SwitchResult.switched(nextId, matchup(next));
    }

    public void tick() {
        tick++;
        for (Player spectator : new ArrayList<Player>(org.bukkit.Bukkit.getOnlinePlayers())) {
            View view = views.get(spectator.getUniqueId());
            if (view == null) continue;
            if (spectator.getGameMode() != GameMode.SPECTATOR) {
                views.remove(spectator.getUniqueId());
                continue;
            }
            if (view.arena.isActive()) {
                view.inactiveSinceTick = -1L;
            } else {
                if (view.inactiveSinceTick < 0L) view.inactiveSinceTick = tick;
                List<Arena> active = manager.activeArenas();
                List<String> activeIds = new ArrayList<String>();
                for (Arena arena : active) activeIds.add(arena.getId());
                String selectedId = selectArenaForContinuity(activeIds, view.arena.getId(),
                        tick - view.inactiveSinceTick);
                if (selectedId != null && !selectedId.equalsIgnoreCase(view.arena.getId())) {
                    Arena selected = manager.byId(selectedId);
                    if (selected != null && selected.isActive()) {
                        view.arena = selected;
                        view.inactiveSinceTick = -1L;
                        announce(spectator, view.arena, view.orbit, "Match ended; moved to");
                    }
                }
            }
            if (view.orbit) {
                // Clearing an already-clear camera target sends a redundant camera
                // packet. Only detach when something has actually taken control,
                // then limit full teleports so Eaglercraft does not have to confirm
                // a new server teleport on every game tick.
                if (spectator.getSpectatorTarget() != null) spectator.setSpectatorTarget(null);
                if (shouldUpdateOrbit(tick)) updateOrbit(spectator, view);
            } else {
                Player target = spectator.getSpectatorTarget() instanceof Player
                        ? (Player) spectator.getSpectatorTarget() : null;
                if (target == null || manager.arenaFor(target) != view.arena) {
                    target = firstAssignedFighter(view.arena);
                    if (target != null) {
                        spectator.setSpectatorTarget(target);
                    } else {
                        // Do not remain attached to a recycled bot that may be
                        // assigned to another arena.  Hold a stable platform view
                        // until this arena's next pair appears.
                        if (spectator.getSpectatorTarget() != null) spectator.setSpectatorTarget(null);
                        if (shouldUpdateOrbit(tick)) updateOrbit(spectator, view);
                    }
                }
            }
            if (tick % 20 == 0) sendHud(spectator, view);
        }
    }

    public String matchup(String arenaId) {
        Arena arena = manager.byId(arenaId);
        return arena == null ? "fight" : matchup(arena);
    }

    public String currentArenaId(Player spectator) {
        View view = views.get(spectator.getUniqueId());
        return view == null ? null : view.arena.getId();
    }

    private void updateOrbit(Player spectator, View view) {
        double angle = orbitAngle(tick);
        Location focus = fightFocus(view.arena);
        double spread = fighterSpread(view.arena, focus);
        double radius = orbitRadius(view.arena.getOuterRadius(), spread);
        double height = orbitHeight(view.arena.getOuterRadius(), spread);
        Location camera = focus.clone().add(
                Math.cos(angle) * radius,
                height,
                Math.sin(angle) * radius);
        Vector lookDirection = focus.toVector().subtract(camera.toVector());
        if (lookDirection.lengthSquared() > 1.0e-8) camera.setDirection(lookDirection);
        spectator.teleport(camera);
    }

    static boolean shouldUpdateOrbit(long tick) {
        return tick % ORBIT_UPDATE_INTERVAL_TICKS == 0L;
    }

    static double orbitAngle(long tick) {
        return tick * ORBIT_RADIANS_PER_TICK;
    }

    static double orbitRadius(int arenaRadius, double fighterSpread) {
        double arenaScale = (arenaRadius * 2.0 + 1.0) / 5.0;
        return Math.min(24.0, Math.max(10.0 * arenaScale, 9.0 + fighterSpread * 0.8));
    }

    static double orbitHeight(int arenaRadius, double fighterSpread) {
        double arenaScale = (arenaRadius * 2.0 + 1.0) / 5.0;
        return Math.min(15.0, Math.max(6.0 * arenaScale, 6.0 + fighterSpread * 0.35));
    }

    private Location fightFocus(Arena arena) {
        Location fallback = arena.getCenter().add(0, 1.6, 0);
        double x = 0;
        double y = 0;
        double z = 0;
        int count = 0;
        for (Player fighter : assignedFighters(arena)) {
            Location eye = fighter.getEyeLocation();
            if (eye.getWorld() == null || !eye.getWorld().equals(fallback.getWorld())) continue;
            x += eye.getX();
            y += eye.getY();
            z += eye.getZ();
            count++;
        }
        return count == 0 ? fallback : new Location(fallback.getWorld(), x / count, y / count, z / count);
    }

    private double fighterSpread(Arena arena, Location focus) {
        double spread = 0;
        for (Player fighter : assignedFighters(arena)) {
            Location location = fighter.getLocation();
            if (location.getWorld() == null || !location.getWorld().equals(focus.getWorld())) continue;
            double dx = location.getX() - focus.getX();
            double dz = location.getZ() - focus.getZ();
            spread = Math.max(spread, Math.sqrt(dx * dx + dz * dz));
        }
        return spread;
    }

    private List<Player> assignedFighters(Arena arena) {
        List<Player> fighters = new ArrayList<Player>();
        for (Player fighter : arena.players()) {
            if (fighter != null && fighter.isOnline() && manager.arenaFor(fighter) == arena) fighters.add(fighter);
        }
        return fighters;
    }

    private Player firstAssignedFighter(Arena arena) {
        List<Player> fighters = assignedFighters(arena);
        return fighters.isEmpty() ? null : fighters.get(0);
    }

    private String matchup(Arena arena) {
        List<Player> fighters = assignedFighters(arena);
        if (fighters.size() >= 2) return fighters.get(0).getName() + " vs " + fighters.get(1).getName();
        if (fighters.size() == 1) return fighters.get(0).getName() + " waiting for opponent";
        return "fight preparing";
    }

    private void sendHud(Player spectator, View view) {
        String text = view.arena.getId() + " " + view.arena.getLane() + " S" + view.arena.getCurriculumStage()
                + "/" + view.arena.getCurriculumStageCount() + " R" + view.arena.getOuterRadius()
                + " | " + scoreLine(view.arena) + " | /ainext";
        spectator.spigot().sendMessage(ChatMessageType.ACTION_BAR, new TextComponent(text));
    }

    private String scoreLine(Arena arena) {
        List<Player> fighters = assignedFighters(arena);
        if (fighters.size() < 2) return matchup(arena);
        Player first = fighters.get(0);
        Player second = fighters.get(1);
        CombatStats firstStats = arena.stats(first);
        CombatStats secondStats = arena.stats(second);
        return first.getName() + " " + String.format(Locale.US, "%.1f", arena.points(first))
                + "-" + String.format(Locale.US, "%.1f", arena.points(second)) + " " + second.getName()
                + " | H " + counter(value(firstStats, Counter.HITS), value(secondStats, Counter.HITS))
                + " C(P/X) " + paired(value(firstStats, Counter.CRYSTALS_PLACED),
                        value(firstStats, Counter.CRYSTALS_EXPLODED),
                        value(secondStats, Counter.CRYSTALS_PLACED),
                        value(secondStats, Counter.CRYSTALS_EXPLODED))
                + " B(P/M) " + paired(value(firstStats, Counter.BLOCKS_PLACED),
                        value(firstStats, Counter.BLOCKS_MINED),
                        value(secondStats, Counter.BLOCKS_PLACED),
                        value(secondStats, Counter.BLOCKS_MINED));
    }

    private static String counter(long first, long second) {
        return first + "/" + second;
    }

    private static String paired(long firstA, long firstB, long secondA, long secondB) {
        return firstA + "/" + firstB + "-" + secondA + "/" + secondB;
    }

    private static long value(CombatStats stats, Counter counter) {
        if (stats == null) return 0;
        switch (counter) {
            case HITS: return stats.hitsLanded;
            case CRYSTALS_PLACED: return stats.crystalsPlaced;
            case CRYSTALS_EXPLODED: return stats.crystalsExploded;
            case BLOCKS_PLACED: return stats.blocksPlaced;
            case BLOCKS_MINED: return stats.blocksMined;
            default: return 0;
        }
    }

    private enum Counter {
        HITS,
        CRYSTALS_PLACED,
        CRYSTALS_EXPLODED,
        BLOCKS_PLACED,
        BLOCKS_MINED
    }

    private void announce(Player spectator, Arena arena, boolean orbit, String prefix) {
        String detail = matchup(arena);
        spectator.sendMessage(prefix + " " + arena.getId() + ": " + detail + ".");
        spectator.sendTitle(arena.getId(), detail + (orbit ? " (orbit)" : " (POV)"), 5, 35, 10);
    }

    static String selectNextArenaId(List<String> activeIds, String currentId) {
        if (activeIds.isEmpty()) return null;
        if (currentId == null) return activeIds.get(0);
        for (int index = 0; index < activeIds.size(); index++) {
            if (!activeIds.get(index).equalsIgnoreCase(currentId)) continue;
            return activeIds.get((index + 1) % activeIds.size());
        }
        return activeIds.get(0);
    }

    static String selectArenaForContinuity(List<String> activeIds, String currentId, long inactiveTicks) {
        for (String activeId : activeIds) {
            if (currentId != null && activeId.equalsIgnoreCase(currentId)) return activeId;
        }
        if (currentId != null && inactiveTicks < ARENA_RESTART_GRACE_TICKS) return currentId;
        return activeIds.isEmpty() ? currentId : activeIds.get(0);
    }

    public static final class SwitchResult {
        public enum Status { SWITCHED, ONLY_ACTIVE, NO_ACTIVE }

        private final Status status;
        private final String arenaId;
        private final String matchup;

        private SwitchResult(Status status, String arenaId, String matchup) {
            this.status = status;
            this.arenaId = arenaId;
            this.matchup = matchup;
        }

        static SwitchResult switched(String arenaId, String matchup) {
            return new SwitchResult(Status.SWITCHED, arenaId, matchup);
        }

        static SwitchResult onlyActive(String arenaId, String matchup) {
            return new SwitchResult(Status.ONLY_ACTIVE, arenaId, matchup);
        }

        static SwitchResult noActive() {
            return new SwitchResult(Status.NO_ACTIVE, null, null);
        }

        public Status getStatus() { return status; }
        public String getArenaId() { return arenaId; }
        public String getMatchup() { return matchup; }
    }

    private static final class View {
        private Arena arena;
        private final boolean orbit;
        private long inactiveSinceTick = -1L;
        private View(Arena arena, boolean orbit) { this.arena = arena; this.orbit = orbit; }
    }
}
