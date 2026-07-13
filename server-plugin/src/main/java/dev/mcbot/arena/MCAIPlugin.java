package dev.mcbot.arena;

import org.bukkit.ChatColor;
import org.bukkit.World;
import org.bukkit.WorldCreator;
import org.bukkit.command.Command;
import org.bukkit.command.CommandExecutor;
import org.bukkit.command.CommandSender;
import org.bukkit.entity.Player;
import org.bukkit.plugin.java.JavaPlugin;

import java.io.IOException;

public final class MCAIPlugin extends JavaPlugin implements CommandExecutor {
    private ArenaManager manager;
    private SpectatorController spectators;
    private ControlServer control;

    @Override
    public void onEnable() {
        saveDefaultConfig();
        World world = createArenaWorld();
        manager = new ArenaManager(this, world);
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
        getCommand("aiwatch").setExecutor(this);
        getCommand("ainext").setExecutor(this);
        getCommand("aistop").setExecutor(this);
        getServer().getScheduler().runTaskTimer(this, () -> {
            manager.tick();
            spectators.tick();
        }, 1L, 1L);
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
            player.sendMessage(spectators.next(player) ? ChatColor.GREEN + "Switched arena." : ChatColor.RED + "No active arena.");
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
        world.setTime(6000);
        world.setStorm(false);
        world.setGameRuleValue("doMobSpawning", "false");
        world.setGameRuleValue("doDaylightCycle", "false");
        world.setGameRuleValue("keepInventory", "true");
        world.setGameRuleValue("mobGriefing", "false");
        world.setGameRuleValue("naturalRegeneration", "true");
        return world;
    }
}
