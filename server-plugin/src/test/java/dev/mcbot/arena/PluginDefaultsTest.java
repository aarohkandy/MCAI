package dev.mcbot.arena;

import org.bukkit.configuration.file.YamlConfiguration;
import org.junit.jupiter.api.Test;

import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.Arrays;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class PluginDefaultsTest {
    @Test
    void shipsFourParallelDeepRetentionAndProgressiveLanes() {
        InputStream stream = PluginDefaultsTest.class.getClassLoader().getResourceAsStream("config.yml");
        assertNotNull(stream);
        YamlConfiguration config = YamlConfiguration.loadConfiguration(
                new InputStreamReader(stream, StandardCharsets.UTF_8));

        assertEquals(4, config.getInt("max-concurrent-pairs"));
        assertEquals(35, config.getInt("match-timeout-seconds"));
        assertTrue(config.getInt("arena-depth") >= 12);
        assertEquals("combined", config.getString("default-mode"));
        assertEquals(Arrays.asList(5, 6, 7, 8),
                config.getIntegerList("curriculum.arena-radius-stages"));
        assertEquals(Arrays.asList(0, 64, 256, 1024),
                config.getIntegerList("curriculum.arena-radius-episode-thresholds"));
        assertEquals(32, config.getInt("curriculum.arena-radius-performance-window"));
        assertEquals(0.75, config.getDouble("curriculum.arena-radius-advance-engagement-rate"), 1.0e-9);
        assertEquals(0.10, config.getDouble("curriculum.arena-radius-advance-non-timeout-rate"), 1.0e-9);
        assertEquals(0.50, config.getDouble("curriculum.arena-radius-regress-engagement-rate"), 1.0e-9);
        assertEquals(0.05, config.getDouble("curriculum.arena-radius-regress-non-timeout-rate"), 1.0e-9);
        assertEquals(32, config.getInt("curriculum.arena-radius-stage-change-cooldown-episodes"));
        assertEquals(0.25, config.getDouble("reward.max-per-tick"), 1.0e-9);
        assertEquals(0.004, config.getDouble("reward.movement-per-block"), 1.0e-9);
        assertEquals(5.0, config.getDouble("reward.lock-on-range"), 1.0e-9);
        assertEquals(0.0, config.getDouble("reward.lock-on-per-tick"), 1.0e-9);
        assertEquals(0, config.getInt("reward.max-rewarded-lock-on-ticks"));
        assertEquals(0.004, config.getDouble("reward.valid-attack-swing"), 1.0e-9);
        assertEquals(40, config.getInt("reward.max-rewarded-attack-swings"));
        assertEquals(0.045, config.getDouble("reward.successful-hit"), 1.0e-9);
        assertEquals(10, config.getInt("reward.max-rewarded-hits"));
        assertEquals(0.180, config.getDouble("reward.damage-dealt-per-health"), 1.0e-9);
        assertEquals(-0.100, config.getDouble("reward.damage-taken-per-health"), 1.0e-9);
        assertEquals(0.12, config.getDouble("reward.forced-totem"), 1.0e-9);
        assertEquals(-0.12, config.getDouble("reward.own-totem"), 1.0e-9);
        assertEquals(20.0, config.getDouble("reward.policy-kill"), 1.0e-9);
        assertEquals(44.0, config.getDouble("reward.policy-kill-speed-bonus"), 1.0e-9);
        assertEquals(-20.0, config.getDouble("reward.death-loss"), 1.0e-9);
        assertEquals(-32.0, config.getDouble("reward.timeout-loss"), 1.0e-9);
        assertEquals(-32.0, config.getDouble("reward.disengaged-loss"), 1.0e-9);
        assertEquals(-15.0, config.getDouble("reward.double-ko-loss"), 1.0e-9);
        assertEquals(-0.900, config.getDouble("reward.own-crystal-self-hit"), 1.0e-9);
        assertEquals(-0.100,
                config.getDouble("reward.own-crystal-self-damage-per-health"), 1.0e-9);
        assertEquals(0.015, config.getDouble("reward.obsidian-placed"), 1.0e-9);
        assertEquals(4, config.getInt("reward.max-rewarded-obsidian"));
        assertEquals(0.200, config.getDouble("reward.obsidian-combo"), 1.0e-9);
        assertEquals(4, config.getInt("reward.max-rewarded-obsidian-combos"));
        assertEquals(120, config.getInt("reward.tactical-mine-place-max-ticks"));
        assertEquals(0.030, config.getDouble("reward.tactical-mine-place"), 1.0e-9);
        assertEquals(3, config.getInt("reward.max-rewarded-tactical-mine-place"));
        assertEquals(0.040, config.getDouble("reward.crystal-placed"), 1.0e-9);
        assertEquals(8, config.getInt("reward.max-rewarded-crystal-placements"));
        assertEquals(0.020, config.getDouble("reward.crystal-destroyed"), 1.0e-9);
        assertEquals(8, config.getInt("reward.max-rewarded-crystal-destructions"));
        assertEquals(0.050, config.getDouble("reward.crystal-exploded"), 1.0e-9);
        assertEquals(8, config.getInt("reward.max-rewarded-crystal-explosions"));
        assertEquals(40, config.getInt("reward.crystal-combo-max-ticks"));
        assertEquals(0.250, config.getDouble("reward.crystal-combo-damage"), 1.0e-9);
        assertEquals(0.500, config.getDouble("reward.crystal-combo-pop"), 1.0e-9);
        assertEquals(6, config.getInt("reward.max-rewarded-crystal-combos"));
        assertEquals(0.0, config.getDouble("reward.useful-block-mined"), 1.0e-9);
        assertEquals(0, config.getInt("reward.max-rewarded-mined-blocks"));
        assertEquals(40, config.getInt("reward.inaction-grace-ticks"));
        assertEquals(-0.0004, config.getDouble("reward.inaction-penalty-per-tick"), 1.0e-9);
        assertEquals(-0.002, config.getDouble("reward.max-inaction-penalty-per-tick"), 1.0e-9);
        assertEquals(200, config.getInt("reward.positive-reward-decay-start-ticks"));
        assertEquals(700, config.getInt("reward.positive-reward-decay-end-ticks"));
        assertEquals(0.35, config.getDouble("reward.minimum-positive-reward-multiplier"), 1.0e-9);
        assertEquals(20, config.getInt("reward.fight-time-pressure-start-ticks"));
        assertEquals(-0.001, config.getDouble("reward.fight-time-pressure-per-tick"), 1.0e-9);
        assertEquals(-0.010, config.getDouble("reward.max-fight-time-pressure-per-tick"), 1.0e-9);
    }
}
