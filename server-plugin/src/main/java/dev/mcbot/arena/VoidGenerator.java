package dev.mcbot.arena;

import org.bukkit.World;
import org.bukkit.generator.ChunkGenerator;

import java.util.Random;

public final class VoidGenerator extends ChunkGenerator {
    @Override
    public ChunkData generateChunkData(World world, Random random, int chunkX, int chunkZ, BiomeGrid biome) {
        return createChunkData(world);
    }
}
