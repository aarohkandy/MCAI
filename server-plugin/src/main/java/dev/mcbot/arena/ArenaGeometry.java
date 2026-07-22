package dev.mcbot.arena;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

/** Pure arena-bound calculations shared by configuration and tests. */
final class ArenaGeometry {
    static final int MINIMUM_RADIUS = 2;
    static final double TACTICAL_BASE_MIN_BUILDER_DISTANCE = 1.35;
    static final double TACTICAL_BASE_MAX_BUILDER_DISTANCE = 4.0;
    static final double TACTICAL_BASE_MIN_OPPONENT_DISTANCE = 1.5;
    static final double TACTICAL_BASE_MAX_OPPONENT_DISTANCE = 4.5;
    static final double TACTICAL_BASE_MAX_CORRIDOR_DISTANCE = 1.5;
    private static final long CRYSTAL_PAD_SEED_SALT = 0x4352595354414C50L;

    enum PreparationStep {
        GENERATE_LAYOUT,
        CHOOSE_SPAWNS,
        CLEAR_SPAWN_LANE,
        CLEAR_SPAWNS,
        GUARANTEE_CRYSTAL_PADS
    }

    private ArenaGeometry() {}

    static int clampRadius(int spacing, int requestedRadius) {
        int maximumRadius = Math.max(MINIMUM_RADIUS, spacing / 2 - 4);
        return Math.max(MINIMUM_RADIUS, Math.min(maximumRadius, requestedRadius));
    }

    static int maximumSpawnSeparation(int radius) {
        return Math.max(1, radius * 2 - 3);
    }

    static int clampSpawnMinimum(int radius, int requestedMinimum) {
        return Math.max(1, Math.min(maximumSpawnSeparation(radius), requestedMinimum));
    }

    static int clampSpawnMaximum(int radius, int minimum, int requestedMaximum) {
        return Math.max(minimum, Math.min(maximumSpawnSeparation(radius), requestedMaximum));
    }

    static int playableInteriorDiameter(int radius) {
        return Math.max(1, radius * 2 - 1);
    }

    /**
     * Four cardinal crystal bases one block inside the barrier ring. Radius 5
     * therefore yields the intended (+/-4, 0) and (0, +/-4) pads.
     */
    static int[][] guaranteedCrystalPadOffsets(int radius) {
        if (radius < MINIMUM_RADIUS) throw new IllegalArgumentException("arena radius is too small");
        int offset = radius - 1;
        return new int[][]{{offset, 0}, {-offset, 0}, {0, offset}, {0, -offset}};
    }

    static boolean strictlyInsideBoundary(int radius, int x, int z) {
        return Math.abs(x) < radius && Math.abs(z) < radius;
    }

    /**
     * Pure horizontal geometry for a useful player-built crystal base. Fighter
     * coordinates and block offsets are relative to the arena centre; the base
     * is evaluated at its block centre. Requiring it to sit near the assigned
     * pair's line segment prevents rewarding unrelated side-wall spam.
     */
    static boolean usefulTacticalObsidianGeometry(
            int radius, int blockOffsetX, int blockOffsetZ,
            double builderX, double builderZ, double opponentX, double opponentZ) {
        if (!strictlyInsideBoundary(radius, blockOffsetX, blockOffsetZ)
                || !finite(builderX) || !finite(builderZ)
                || !finite(opponentX) || !finite(opponentZ)) return false;
        double baseX = blockOffsetX + 0.5;
        double baseZ = blockOffsetZ + 0.5;
        double builderDistance = distance(baseX, baseZ, builderX, builderZ);
        double opponentDistance = distance(baseX, baseZ, opponentX, opponentZ);
        return builderDistance >= TACTICAL_BASE_MIN_BUILDER_DISTANCE
                && builderDistance <= TACTICAL_BASE_MAX_BUILDER_DISTANCE
                && opponentDistance >= TACTICAL_BASE_MIN_OPPONENT_DISTANCE
                && opponentDistance <= TACTICAL_BASE_MAX_OPPONENT_DISTANCE
                && pointToSegmentDistance(baseX, baseZ, builderX, builderZ,
                opponentX, opponentZ) <= TACTICAL_BASE_MAX_CORRIDOR_DISTANCE;
    }

    private static double pointToSegmentDistance(double px, double pz, double ax,
                                                  double az, double bx, double bz) {
        double dx = bx - ax;
        double dz = bz - az;
        double lengthSquared = dx * dx + dz * dz;
        if (lengthSquared <= 1.0e-12) return distance(px, pz, ax, az);
        double projection = ((px - ax) * dx + (pz - az) * dz) / lengthSquared;
        projection = Math.max(0.0, Math.min(1.0, projection));
        return distance(px, pz, ax + projection * dx, az + projection * dz);
    }

    private static double distance(double ax, double az, double bx, double bz) {
        double dx = ax - bx;
        double dz = az - bz;
        return Math.sqrt(dx * dx + dz * dz);
    }

    private static boolean finite(double value) {
        return !Double.isNaN(value) && !Double.isInfinite(value);
    }

    /** Pure reset plan used by Arena so tests lock the final pad pass after all spawn clearing. */
    static PreparationStep[] preparationSteps(boolean terrainLayout, boolean crystalLayout) {
        if (terrainLayout && crystalLayout) {
            return new PreparationStep[]{
                    PreparationStep.GENERATE_LAYOUT,
                    PreparationStep.CHOOSE_SPAWNS,
                    PreparationStep.CLEAR_SPAWN_LANE,
                    PreparationStep.CLEAR_SPAWNS,
                    PreparationStep.GUARANTEE_CRYSTAL_PADS
            };
        }
        if (terrainLayout) {
            return new PreparationStep[]{
                    PreparationStep.GENERATE_LAYOUT,
                    PreparationStep.CHOOSE_SPAWNS,
                    PreparationStep.CLEAR_SPAWN_LANE,
                    PreparationStep.CLEAR_SPAWNS
            };
        }
        if (crystalLayout) {
            return new PreparationStep[]{
                    PreparationStep.CHOOSE_SPAWNS,
                    PreparationStep.CLEAR_SPAWN_LANE,
                    PreparationStep.CLEAR_SPAWNS,
                    PreparationStep.GUARANTEE_CRYSTAL_PADS
            };
        }
        return new PreparationStep[]{
                PreparationStep.CHOOSE_SPAWNS,
                PreparationStep.CLEAR_SPAWN_LANE,
                PreparationStep.CLEAR_SPAWNS
        };
    }

    /** Maximum horizontal center distance after accounting for entity width. */
    static double maximumCenterDiagonal(int radius, double entityWidth) {
        double axisTravel = Math.max(0.0, playableInteriorDiameter(radius) - entityWidth);
        return Math.sqrt(2.0) * axisTravel;
    }

    /**
     * Produces three to seven episode-seeded crystal bases. Every layout keeps
     * one within two horizontal blocks of each spawn and one near the fight
     * midpoint; remaining pads are scattered through the playable interior.
     * Spawn blocks and the exact midpoint are deliberately excluded so neither
     * pad count nor one permanent look-down-and-use coordinate can be memorized.
     */
    static int[][] reachableCrystalPadOffsets(int radius, int firstX, int firstZ,
                                               int secondX, int secondZ, long episodeSeed) {
        int limit = radius - 1;
        firstX = clamp(firstX, -limit, limit);
        firstZ = clamp(firstZ, -limit, limit);
        secondX = clamp(secondX, -limit, limit);
        secondZ = clamp(secondZ, -limit, limit);
        int midpointX = clamp((int) Math.round((firstX + secondX) / 2.0), -limit, limit);
        int midpointZ = clamp((int) Math.round((firstZ + secondZ) / 2.0), -limit, limit);

        List<int[]> pads = new ArrayList<int[]>();
        Set<String> unavailable = new HashSet<String>();
        unavailable.add(coordinateKey(firstX, firstZ));
        unavailable.add(coordinateKey(secondX, secondZ));
        unavailable.add(coordinateKey(midpointX, midpointZ));

        Random random = new Random(crystalPadSeed(
                episodeSeed, radius, firstX, firstZ, secondX, secondZ));
        addRandomPadsNear(pads, unavailable, random, limit, firstX, firstZ, 1);
        addRandomPadsNear(pads, unavailable, random, limit, secondX, secondZ, 1);
        addRandomPadsNear(pads, unavailable, random, limit, midpointX, midpointZ, 1);
        int targetCount = 3 + random.nextInt(5);
        addRandomPadsAcrossInterior(pads, unavailable, random, limit,
                targetCount - pads.size());
        if (pads.size() != targetCount) {
            throw new IllegalStateException("could not place randomized reachable crystal pads");
        }
        return pads.toArray(new int[pads.size()][]);
    }

    private static void addRandomPadsAcrossInterior(
            List<int[]> pads, Set<String> unavailable, Random random, int limit, int count) {
        List<int[]> candidates = new ArrayList<int[]>();
        for (int x = -limit; x <= limit; x++) {
            for (int z = -limit; z <= limit; z++) {
                if (!unavailable.contains(coordinateKey(x, z))) {
                    candidates.add(new int[]{x, z});
                }
            }
        }
        Collections.shuffle(candidates, random);
        int remaining = count;
        for (int[] candidate : candidates) {
            if (remaining == 0) break;
            if (addUnique(pads, unavailable, candidate)) remaining--;
        }
    }

    /** Equal per-match supply, randomized widely enough to prevent kit memorization. */
    static int obsidianSupply(long episodeSeed) {
        Random random = new Random(episodeSeed ^ 0x4F4253494449414EL);
        return 16 + random.nextInt(49);
    }

    private static void addRandomPadsNear(List<int[]> pads, Set<String> unavailable,
                                          Random random, int limit, int anchorX,
                                          int anchorZ, int count) {
        List<int[]> candidates = new ArrayList<int[]>();
        for (int dx = -2; dx <= 2; dx++) {
            for (int dz = -2; dz <= 2; dz++) {
                int manhattanDistance = Math.abs(dx) + Math.abs(dz);
                if (manhattanDistance < 1 || manhattanDistance > 2) continue;
                int x = anchorX + dx;
                int z = anchorZ + dz;
                if (x < -limit || x > limit || z < -limit || z > limit) continue;
                if (!unavailable.contains(coordinateKey(x, z))) {
                    candidates.add(new int[]{x, z});
                }
            }
        }
        Collections.shuffle(candidates, random);
        int remaining = count;
        for (int[] candidate : candidates) {
            if (remaining == 0) break;
            if (addUnique(pads, unavailable, candidate)) remaining--;
        }
    }

    private static boolean addUnique(List<int[]> pads, Set<String> unavailable, int[] candidate) {
        if (!unavailable.add(coordinateKey(candidate[0], candidate[1]))) return false;
        pads.add(candidate);
        return true;
    }

    private static String coordinateKey(int x, int z) {
        return x + ":" + z;
    }

    private static long crystalPadSeed(long episodeSeed, int radius, int firstX,
                                       int firstZ, int secondX, int secondZ) {
        long value = episodeSeed ^ CRYSTAL_PAD_SEED_SALT;
        value = value * 31L + radius;
        value = value * 31L + firstX;
        value = value * 31L + firstZ;
        value = value * 31L + secondX;
        value = value * 31L + secondZ;
        value ^= value >>> 33;
        value *= 0xff51afd7ed558ccdL;
        value ^= value >>> 33;
        return value;
    }

    private static int clamp(int value, int minimum, int maximum) {
        return Math.max(minimum, Math.min(maximum, value));
    }
}
