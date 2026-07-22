package dev.mcbot.arena;

import org.bukkit.ChatColor;
import org.bukkit.World;
import org.bukkit.WorldCreator;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.event.EventHandler;
import org.bukkit.event.Listener;
import org.bukkit.event.player.PlayerJoinEvent;
import org.bukkit.event.weather.ThunderChangeEvent;
import org.bukkit.event.weather.WeatherChangeEvent;
import org.bukkit.plugin.java.JavaPlugin;

import java.io.IOException;

public final class MCAIPlugin extends JavaPlugin implements CommandExecutor, Listener {
    private static final long ARENA_NOON_TIME = 6000L;

    private World arenaWorld;
    private ArenaManager manager;
    private SpectatorController spectators;
    private ControlServer control;

    @Override
    public void onEnable() {
        saveDefaultConfig();
        arenaWorld = createArenaWorld();
        manager = new ArenaManager(this, arenaWorld);
        spectators = new SpectatorController(manager);
        control = new ControlServer(this, manager, getConfig().getInt("control-port", 8765));
        manager.setControl(control);
        try {
            control.start();
        } catch (IOException error) {
            getLogger().severe("Cannot bind localhost rollout control: " + error.getMessage());
            getServer().getPluginManager().disablePlugin(this);
            return;
        }
        getServer().getPluginManager().registerEvents(new CombatListener(this, manager), this);
        getServer().getPluginManager().registerEvents(this, this);
        getCommand("aiwatch").setExecutor(this);
        getCommand("ainext").setExecutor(this);
        getCommand("aistop").setExecutor(this);
        getServer().getScheduler().runTaskTimer(this, () -> {
            manager.tick();
            spectators.tick();
        }, 1L, 1L);
        // Gamerules normally keep the arena at noon and clear. Reassert the
        // environment once per second as a guard against commands and plugins.
        getServer().getScheduler().runTaskTimer(this,
                () -> enforceArenaEnvironment(arenaWorld), 20L, 20L);
    }

    @EventHandler
    public void onPlayerJoin(PlayerJoinEvent event) {
        Player player = event.getPlayer();
        String prefix = getConfig().getString("bot-name-prefix", "MCAI_");
        // Eaglercraft finishes its WebSocket login handshake shortly after Bukkit's
        // join event.  Moving its camera immediately can make that handshake time
        // out, so let the client settle before switching it into spectator orbit.
        if (!player.getName().startsWith(prefix)) {
            getServer().getScheduler().runTaskLater(this, () -> autoWatch(player, 0), 100L);
        }
    }

    @EventHandler
    public void onWeatherChange(WeatherChangeEvent event) {
        if (arenaWorld != null && arenaWorld.equals(event.getWorld()) && event.toWeatherState()) {
            event.setCancelled(true);
            arenaWorld.setWeatherDuration(Integer.MAX_VALUE);
        }
    }

    @EventHandler
    public void onThunderChange(ThunderChangeEvent event) {
        if (arenaWorld != null && arenaWorld.equals(event.getWorld()) && event.toThunderState()) {
            event.setCancelled(true);
            arenaWorld.setThunderDuration(Integer.MAX_VALUE);
        }
    }

    private void autoWatch(Player player, int attempt) {
        if (!player.isOnline()) return;
        if (!manager.activeArenas().isEmpty()) {
            String arenaId = manager.activeArenas().get(0).getId();
            if (spectators.watch(player, arenaId, "orbit")) {
                player.sendMessage(ChatColor.GREEN + "Automatically watching " + arenaId + " in orbit mode. Use /ainext to switch fights.");
                return;
            }
        }
        if (attempt < 120) {
            getServer().getScheduler().runTaskLater(this, () -> autoWatch(player, attempt + 1), 5L);
        }
    }

    @Override
    public void onDisable() {
        if (manager != null) manager.stopAll("plugin_disable");
        if (control != null) control.close();
    }

    @Override
    public boolean onCommand(CommandSender sender, Command command, String label, String[] args) {
        if ("aistop".equalsIgnoreCase(command.getName())) {
            manager.stopAll("operator_emergency_stop");
            String prefix = getConfig().getString("bot-name-prefix", "MCAI_");
            for (Player player : getServer().getOnlinePlayers()) {
                if (player.getName().startsWith(prefix)) player.kickPlayer("MCAI emergency stop");
            }
            sender.sendMessage(ChatColor.RED + "All AI controls released and bot accounts disconnected.");
            return true;
        }
        if (!(sender instanceof Player)) {
            sender.sendMessage("This spectator command must be run by a player.");
            return true;
        }
        Player player = (Player) sender;
        if ("ainext".equalsIgnoreCase(command.getName())) {
            SpectatorController.SwitchResult result = spectators.next(player);
            if (result.getStatus() == SpectatorController.SwitchResult.Status.NO_ACTIVE) {
                player.sendMessage(ChatColor.RED + "No active arena.");
            } else if (result.getStatus() == SpectatorController.SwitchResult.Status.ONLY_ACTIVE) {
                player.sendMessage(ChatColor.YELLOW + "Only one fight is active; still watching "
                        + result.getArenaId() + ": " + result.getMatchup() + ".");
            } else {
                player.sendMessage(ChatColor.GREEN + "Now watching " + result.getArenaId()
                        + ": " + result.getMatchup() + ".");
                player.sendTitle(ChatColor.GOLD + result.getArenaId(), ChatColor.WHITE + result.getMatchup(), 5, 35, 10);
            }
            return true;
        }
        if (args.length < 1) return false;
        String mode = args.length >= 2 ? args[1] : "pov";
        if (!"pov".equalsIgnoreCase(mode) && !"orbit".equalsIgnoreCase(mode)) return false;
        player.sendMessage(spectators.watch(player, args[0], mode)
                ? ChatColor.GREEN + "Watching " + args[0] + " in " + mode + " mode."
                : ChatColor.RED + "That arena is not active.");
        return true;
    }

    private World createArenaWorld() {
        String name = getConfig().getString("world-name", "mcai_training");
        World world = getServer().getWorld(name);
        if (world == null) {
            WorldCreator creator = new WorldCreator(name);
            creator.generator(new VoidGenerator());
            creator.generateStructures(false);
            world = creator.createWorld();
        }
        world.setAutoSave(false);
        enforceArenaEnvironment(world);
        world.setGameRuleValue("doMobSpawning", "false");
        world.setGameRuleValue("keepInventory", "true");
        world.setGameRuleValue("mobGriefing", "false");
        // Damage must accumulate into terminal outcomes; healing should be a
        // learned golden-apple/totem decision rather than free passive regen.
        world.setGameRuleValue("naturalRegeneration", "false");
        return world;
    }

    static void enforceArenaEnvironment(World world) {
        if (world.getTime() != ARENA_NOON_TIME) world.setTime(ARENA_NOON_TIME);
        if (world.hasStorm()) world.setStorm(false);
        if (world.isThundering()) world.setThundering(false);
        world.setWeatherDuration(Integer.MAX_VALUE);
        world.setThunderDuration(Integer.MAX_VALUE);
        world.setGameRuleValue("doDaylightCycle", "false");
        world.setGameRuleValue("doWeatherCycle", "false");
    }
}
