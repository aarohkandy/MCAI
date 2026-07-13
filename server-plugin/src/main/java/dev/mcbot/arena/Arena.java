package dev.mcbot.arena;

import com.google.gson.JsonObject;
import org.bukkit.GameMode;
import org.bukkit.Location;
import org.bukkit.Material;
import org.bukkit.World;
import org.bukkit.block.Block;
import org.bukkit.enchantments.Enchantment;
import org.bukkit.entity.Entity;
import org.bukkit.entity.Player;
import org.bukkit.inventory.ItemStack;
import org.bukkit.inventory.PlayerInventory;
import org.bukkit.potion.PotionEffect;
import org.bukkit.util.Vector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.UUID;

public final class Arena {
    private static final int FLOOR_Y = 63;
    private static final int OUTER_RADIUS = 17;
    private final ArenaManager manager;
    private final String id;
    private final Location center;
    private final Set<BlockKey> touchedBlocks = new HashSet<BlockKey>();
    private final Map<UUID, CombatStats> stats = new HashMap<UUID, CombatStats>();
    private Player first;
    private Player second;
    private long seed;
    private String episodeId;
    private long startedTick;
    private long deadlineTick;
    private ArenaMode mode = ArenaMode.COMBINED;
    private boolean active;
    private boolean basePrepared;

    public Arena(ArenaManager manager, String id, Location center) {
        this.manager = manager;
        this.id = id;
        this.center = center;
    }

    public void start(Player first, Player second, long seed, ArenaMode mode, long tick, int timeoutSeconds,
                      int actionDelay, int observationDelay) {
        this.first = first;
        this.second = second;
        this.seed = seed;
        this.mode = mode;
        this.episodeId = UUID.randomUUID().toString();
        this.startedTick = tick;
        this.deadlineTick = tick + timeoutSeconds * 20L;
        stats.clear();
        stats.put(first.getUniqueId(), new CombatStats(first.getUniqueId()));
        stats.put(second.getUniqueId(), new CombatStats(second.getUniqueId()));
        prepareBase();
        resetTouchedBlocks();
        clearEntities();
        Random random = new Random(seed);
        int size = mode == ArenaMode.SWORD ? 21 : 17 + 2 * random.nextInt(8);
        if (mode == ArenaMode.COMBINED || mode == ArenaMode.CRYSTAL) generateLayout(random, size);
        Location[] spawns = spawnLocations(random, size);
        clearSpawn(spawns[0]);
        clearSpawn(spawns[1]);
        preparePlayer(first, spawns[0]);
        preparePlayer(second, spawns[1]);
        active = true;
        manager.matchStarted(this, first, actionDelay, observationDelay);
        manager.matchStarted(this, second, actionDelay, observationDelay);
    }

    public void tick(long currentTick) {
        if (!active) return;
        flushRewards();
        if (!first.isOnline() || !second.isOnline()) {
            Player winner = first.isOnline() ? first : (second.isOnline() ? second : null);
            finish(winner, false, "disconnect");
        } else if (first.isDead() || second.isDead()) {
            finish(first.isDead() ? second : first, false, "death");
        } else if (currentTick >= deadlineTick) {
            finish(timeoutWinner(), true, "timeout");
        }
    }

    public void finish(Player winner, boolean truncated, String reason) {
        if (!active) return;
        flushRewards();
        active = false;
        manager.matchEnded(this, first, winner == null ? -0.05 : (winner.equals(first) ? 1.0 : -1.0), truncated, reason);
        manager.matchEnded(this, second, winner == null ? -0.05 : (winner.equals(second) ? 1.0 : -1.0), truncated, reason);
        for (Player player : players()) {
            if (player.isOnline()) {
                player.setVelocity(new Vector(0, 0, 0));
                player.getInventory().clear();
            }
        }
        manager.recycle(this, Arrays.asList(first, second));
    }

    public void addReward(Player player, double reward) {
        CombatStats value = stats.get(player.getUniqueId());
        if (value != null) value.pendingReward += reward * manager.getShapingScale();
    }

    public CombatStats stats(Player player) {
        return stats.get(player.getUniqueId());
    }

    public Player opponent(Player player) {
        if (first != null && first.equals(player)) return second;
        if (second != null && second.equals(player)) return first;
        return null;
    }

    public boolean contains(Location location) {
        return location != null && location.getWorld().equals(center.getWorld())
                && Math.abs(location.getBlockX() - center.getBlockX()) <= OUTER_RADIUS
                && Math.abs(location.getBlockZ() - center.getBlockZ()) <= OUTER_RADIUS
                && location.getY() >= FLOOR_Y - 1 && location.getY() <= FLOOR_Y + 8;
    }

    public void markTouched(Block block) {
        if (contains(block.getLocation())) touchedBlocks.add(new BlockKey(block.getX(), block.getY(), block.getZ()));
    }

    public Collection<Player> players() {
        List<Player> result = new ArrayList<Player>();
        if (first != null) result.add(first);
        if (second != null) result.add(second);
        return result;
    }

    public String getId() { return id; }
    public Location getCenter() { return center.clone(); }
    public long getSeed() { return seed; }
    public String getEpisodeId() { return episodeId; }
    public long getStartedTick() { return startedTick; }
    public ArenaMode getMode() { return mode; }
    public boolean isActive() { return active; }

    public JsonObject statsJson(Player player) {
        CombatStats value = stats(player);
        JsonObject json = new JsonObject();
        if (value == null) return json;
        json.addProperty("damage_dealt", value.damageDealt);
        json.addProperty("damage_taken", value.damageTaken);
        json.addProperty("healing", value.healing);
        json.addProperty("totem_pops", value.totemPops);
        json.addProperty("crystals_placed", value.crystalsPlaced);
        json.addProperty("crystals_destroyed", value.crystalsDestroyed);
        json.addProperty("blocks_placed", value.blocksPlaced);
        json.addProperty("blocks_mined", value.blocksMined);
        json.addProperty("invalid_interactions", value.invalidInteractions);
        return json;
    }

    private void flushRewards() {
        for (Player player : players()) {
            CombatStats value = stats(player);
            if (value == null || Math.abs(value.pendingReward) < 1e-12) continue;
            double clipped = Math.max(-0.05, Math.min(0.05, value.pendingReward));
            value.pendingReward = 0;
            manager.stepFeedback(this, player, clipped);
        }
    }

    private Player timeoutWinner() {
        int firstTotems = countTotems(first);
        int secondTotems = countTotems(second);
        if (firstTotems != secondTotems) return firstTotems > secondTotems ? first : second;
        double firstHealth = effectiveHealth(first);
        double secondHealth = effectiveHealth(second);
        if (Math.abs(firstHealth - secondHealth) > 1e-6) return firstHealth > secondHealth ? first : second;
        double firstDamage = stats(first).damageDealt;
        double secondDamage = stats(second).damageDealt;
        if (Math.abs(firstDamage - secondDamage) > 1e-6) return firstDamage > secondDamage ? first : second;
        return null;
    }

    private void prepareBase() {
        if (basePrepared) return;
        World world = center.getWorld();
        for (int x = -OUTER_RADIUS; x <= OUTER_RADIUS; x++) {
            for (int z = -OUTER_RADIUS; z <= OUTER_RADIUS; z++) {
                world.getBlockAt(center.getBlockX() + x, FLOOR_Y, center.getBlockZ() + z).setType(Material.STONE, false);
                for (int y = FLOOR_Y + 1; y <= FLOOR_Y + 7; y++) {
                    world.getBlockAt(center.getBlockX() + x, y, center.getBlockZ() + z).setType(Material.AIR, false);
                }
            }
        }
        for (int coordinate = -OUTER_RADIUS; coordinate <= OUTER_RADIUS; coordinate++) {
            for (int y = FLOOR_Y + 1; y <= FLOOR_Y + 5; y++) {
                world.getBlockAt(center.getBlockX() - OUTER_RADIUS, y, center.getBlockZ() + coordinate).setType(Material.BARRIER, false);
                world.getBlockAt(center.getBlockX() + OUTER_RADIUS, y, center.getBlockZ() + coordinate).setType(Material.BARRIER, false);
                world.getBlockAt(center.getBlockX() + coordinate, y, center.getBlockZ() - OUTER_RADIUS).setType(Material.BARRIER, false);
                world.getBlockAt(center.getBlockX() + coordinate, y, center.getBlockZ() + OUTER_RADIUS).setType(Material.BARRIER, false);
            }
        }
        basePrepared = true;
    }

    private void resetTouchedBlocks() {
        World world = center.getWorld();
        for (BlockKey key : touchedBlocks) {
            world.getBlockAt(key.x, key.y, key.z).setType(key.y == FLOOR_Y ? Material.STONE : Material.AIR, false);
        }
        touchedBlocks.clear();
    }

    private void clearEntities() {
        for (Entity entity : center.getWorld().getEntities()) {
            if (!(entity instanceof Player) && contains(entity.getLocation())) entity.remove();
        }
    }

    private void generateLayout(Random random, int size) {
        int half = size / 2;
        int holes = random.nextInt(5);
        for (int index = 0; index < holes; index++) {
            setLayoutBlock(randomOffset(random, half), FLOOR_Y, randomOffset(random, half), Material.AIR);
        }
        int structures = random.nextInt(13);
        for (int index = 0; index < structures; index++) {
            int x = randomOffset(random, half);
            int z = randomOffset(random, half);
            int height = 1 + random.nextInt(3);
            if (random.nextBoolean()) {
                for (int y = 1; y <= height; y++) setLayoutBlock(x, FLOOR_Y + y, z, Material.STONE);
            } else {
                int length = 2 + random.nextInt(3);
                boolean alongX = random.nextBoolean();
                for (int offset = 0; offset < length; offset++) {
                    for (int y = 1; y <= height; y++) {
                        setLayoutBlock(x + (alongX ? offset : 0), FLOOR_Y + y,
                                z + (alongX ? 0 : offset), Material.STONE);
                    }
                }
            }
        }
        int obsidian = random.nextInt(9);
        for (int index = 0; index < obsidian; index++) {
            setLayoutBlock(randomOffset(random, half), FLOOR_Y + 1, randomOffset(random, half), Material.OBSIDIAN);
        }
    }

    private int randomOffset(Random random, int half) {
        return random.nextInt(half * 2 - 3) - half + 2;
    }

    private void setLayoutBlock(int relativeX, int y, int relativeZ, Material material) {
        Block block = center.getWorld().getBlockAt(center.getBlockX() + relativeX, y, center.getBlockZ() + relativeZ);
        touchedBlocks.add(new BlockKey(block.getX(), block.getY(), block.getZ()));
        block.setType(material, false);
    }

    private Location[] spawnLocations(Random random, int size) {
        int separation = 6 + random.nextInt(7);
        double angle = random.nextDouble() * Math.PI * 2;
        double dx = Math.cos(angle) * separation / 2.0;
        double dz = Math.sin(angle) * separation / 2.0;
        Location a = center.clone().add(dx, FLOOR_Y + 1.01 - center.getY(), dz);
        Location b = center.clone().add(-dx, FLOOR_Y + 1.01 - center.getY(), -dz);
        a.setYaw(random.nextFloat() * 360.0F - 180.0F);
        b.setYaw(random.nextFloat() * 360.0F - 180.0F);
        return new Location[]{a, b};
    }

    private void clearSpawn(Location spawn) {
        int relativeX = spawn.getBlockX() - center.getBlockX();
        int relativeZ = spawn.getBlockZ() - center.getBlockZ();
        setLayoutBlock(relativeX, FLOOR_Y, relativeZ, Material.STONE);
        setLayoutBlock(relativeX, FLOOR_Y + 1, relativeZ, Material.AIR);
        setLayoutBlock(relativeX, FLOOR_Y + 2, relativeZ, Material.AIR);
    }

    private void preparePlayer(Player player, Location spawn) {
        player.setGameMode(GameMode.SURVIVAL);
        PlayerInventory inventory = player.getInventory();
        inventory.clear();
        inventory.setArmorContents(new ItemStack[4]);
        inventory.setItemInOffHand(null);
        for (PotionEffect effect : player.getActivePotionEffects()) player.removePotionEffect(effect.getType());
        player.setHealth(player.getMaxHealth());
        setAbsorption(player, 0.0);
        player.setFoodLevel(20);
        player.setSaturation(5);
        player.setFireTicks(0);
        player.setFallDistance(0);
        player.setVelocity(new Vector(0, 0, 0));
        equipKit(inventory, mode);
        player.teleport(spawn);
    }

    private void equipKit(PlayerInventory inventory, ArenaMode selectedMode) {
        ItemStack sword = new ItemStack(Material.DIAMOND_SWORD);
        sword.addUnsafeEnchantment(Enchantment.DAMAGE_ALL, 5);
        sword.addUnsafeEnchantment(Enchantment.KNOCKBACK, 1);
        inventory.setItem(0, sword);
        if (selectedMode != ArenaMode.SWORD) {
            ItemStack pickaxe = new ItemStack(Material.DIAMOND_PICKAXE);
            pickaxe.addUnsafeEnchantment(Enchantment.DIG_SPEED, 5);
            inventory.setItem(1, pickaxe);
            inventory.setItem(2, new ItemStack(Material.OBSIDIAN, 64));
            inventory.setItem(3, new ItemStack(Material.END_CRYSTAL, 64));
            inventory.setItem(4, new ItemStack(Material.GOLDEN_APPLE, 16, (short) 0));
            for (int slot = 5; slot <= 8; slot++) inventory.setItem(slot, new ItemStack(Material.TOTEM, 1));
            inventory.setItemInOffHand(new ItemStack(Material.TOTEM, 1));
        }
        inventory.setHelmet(armor(Material.DIAMOND_HELMET));
        inventory.setChestplate(armor(Material.DIAMOND_CHESTPLATE));
        inventory.setLeggings(armor(Material.DIAMOND_LEGGINGS));
        inventory.setBoots(armor(Material.DIAMOND_BOOTS));
        inventory.setHeldItemSlot(0);
    }

    private ItemStack armor(Material material) {
        ItemStack item = new ItemStack(material);
        item.addUnsafeEnchantment(Enchantment.PROTECTION_ENVIRONMENTAL, 4);
        item.addUnsafeEnchantment(Enchantment.DURABILITY, 3);
        return item;
    }

    private int countTotems(Player player) {
        int count = 0;
        for (int slot = 0; slot < 36; slot++) {
            ItemStack item = player.getInventory().getItem(slot);
            if (item != null && item.getType() == Material.TOTEM) count += item.getAmount();
        }
        ItemStack offhand = player.getInventory().getItemInOffHand();
        if (offhand != null && offhand.getType() == Material.TOTEM) count += offhand.getAmount();
        return count;
    }

    private double effectiveHealth(Player player) {
        return Math.max(0, player.getHealth()) + Math.max(0, absorption(player));
    }

    private double absorption(Player player) {
        try {
            Object result = player.getClass().getMethod("getAbsorptionAmount").invoke(player);
            return result instanceof Number ? ((Number) result).doubleValue() : 0.0;
        } catch (ReflectiveOperationException ignored) {
            return 0.0;
        }
    }

    private void setAbsorption(Player player, double amount) {
        try {
            player.getClass().getMethod("setAbsorptionAmount", double.class).invoke(player, amount);
        } catch (ReflectiveOperationException ignored) {
            // Bukkit 1.12 does not expose this method; removing potion effects is the portable reset path.
        }
    }

    private static final class BlockKey {
        private final int x;
        private final int y;
        private final int z;

        private BlockKey(int x, int y, int z) { this.x = x; this.y = y; this.z = z; }

        @Override
        public boolean equals(Object other) {
            if (!(other instanceof BlockKey)) return false;
            BlockKey key = (BlockKey) other;
            return x == key.x && y == key.y && z == key.z;
        }

        @Override
        public int hashCode() { return (x * 31 + y) * 31 + z; }
    }
}
