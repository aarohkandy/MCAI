package dev.mcbot.arena;

import org.junit.jupiter.api.Test;

import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class ArenaGeometryTest {
    @Test
    void expandedSwordCurriculumCreatesSevenBySevenPlayableInterior() {
        assertEquals(4, ArenaGeometry.clampRadius(96, 4));
        assertEquals(7, ArenaGeometry.playableInteriorDiameter(4));
        assertEquals(5, ArenaGeometry.maximumSpawnSeparation(4));
        assertEquals(2, ArenaGeometry.clampSpawnMinimum(4, 2));
        assertEquals(5, ArenaGeometry.clampSpawnMaximum(4, 2, 5));
    }

    @Test
    void bootcampCornersRemainWithinConfiguredHorizontalSwordReach() {
        assertTrue(ArenaGeometry.maximumCenterDiagonal(2, 0.6) < 3.4);
    }

    @Test
    void unsafeOrOversizedRadiusIsClamped() {
        assertEquals(2, ArenaGeometry.clampRadius(96, -10));
        assertEquals(44, ArenaGeometry.clampRadius(96, 1000));
    }

    @Test
    void spawnRangeCannotEscapePlayableInterior() {
        assertEquals(1, ArenaGeometry.clampSpawnMinimum(2, 2));
        assertEquals(1, ArenaGeometry.clampSpawnMaximum(2, 1, 5));
        assertEquals(5, ArenaGeometry.clampSpawnMinimum(4, 99));
        assertEquals(5, ArenaGeometry.clampSpawnMaximum(4, 5, -1));
    }

    @Test
    void radiusFiveCrystalPadsUseFourInteriorCardinals() {
        int[][] pads = ArenaGeometry.guaranteedCrystalPadOffsets(5);
        Set<String> coordinates = new HashSet<String>();
        for (int[] pad : pads) {
            assertTrue(ArenaGeometry.strictlyInsideBoundary(5, pad[0], pad[1]));
            coordinates.add(pad[0] + "," + pad[1]);
        }

        assertEquals(4, coordinates.size());
        assertTrue(coordinates.contains("4,0"));
        assertTrue(coordinates.contains("-4,0"));
        assertTrue(coordinates.contains("0,4"));
        assertTrue(coordinates.contains("0,-4"));
    }

    @Test
    void guaranteedPadsStayStrictlyInsideEverySupportedRadius() {
        for (int radius = ArenaGeometry.MINIMUM_RADIUS; radius <= 44; radius++) {
            int[][] pads = ArenaGeometry.guaranteedCrystalPadOffsets(radius);
            assertEquals(4, pads.length);
            for (int[] pad : pads) {
                assertTrue(ArenaGeometry.strictlyInsideBoundary(radius, pad[0], pad[1]));
            }
        }
    }

    @Test
    void tacticalObsidianGeometryAcceptsReachableOffensiveCorridorBase() {
        assertTrue(ArenaGeometry.usefulTacticalObsidianGeometry(
                5, 0, 0, -2.0, 0.5, 3.0, 0.5));
    }

    @Test
    void tacticalObsidianGeometryRejectsFeetFarOpponentSideAndBoundarySpam() {
        assertFalse(ArenaGeometry.usefulTacticalObsidianGeometry(
                5, 0, 0, 0.5, 0.5, 3.0, 0.5), "under builder");
        assertFalse(ArenaGeometry.usefulTacticalObsidianGeometry(
                5, 0, 0, -4.0, 0.5, 3.0, 0.5), "outside builder reach");
        assertFalse(ArenaGeometry.usefulTacticalObsidianGeometry(
                5, 0, 0, -2.0, 0.5, 1.5, 0.5), "too close to opponent");
        assertFalse(ArenaGeometry.usefulTacticalObsidianGeometry(
                5, 0, 0, -2.0, 0.5, 5.5, 0.5), "too far from opponent");
        assertFalse(ArenaGeometry.usefulTacticalObsidianGeometry(
                5, 0, 2, -2.0, 0.5, 3.0, 0.5), "off assigned-pair corridor");
        assertFalse(ArenaGeometry.usefulTacticalObsidianGeometry(
                5, 5, 0, 2.0, 0.5, 4.0, 0.5), "not strictly interior");
        assertFalse(ArenaGeometry.usefulTacticalObsidianGeometry(
                5, 0, 0, Double.NaN, 0.5, 3.0, 0.5), "non-finite input");
    }

    @Test
    void progressiveLiveStagesGrowPlayableSpaceAndKeepSpawnsAndPadsSafe() {
        int[] radii = new int[]{5, 6, 7, 8};
        int[] diameters = new int[]{9, 11, 13, 15};
        for (int index = 0; index < radii.length; index++) {
            int radius = radii[index];
            assertEquals(diameters[index], ArenaGeometry.playableInteriorDiameter(radius));
            int minimum = ArenaGeometry.clampSpawnMinimum(radius, 2);
            int maximum = ArenaGeometry.clampSpawnMaximum(radius, minimum, 5);
            assertTrue(minimum >= 1);
            assertTrue(maximum >= minimum);
            assertTrue(maximum <= ArenaGeometry.maximumSpawnSeparation(radius));
            for (int[] pad : ArenaGeometry.guaranteedCrystalPadOffsets(radius)) {
                assertTrue(ArenaGeometry.strictlyInsideBoundary(radius, pad[0], pad[1]));
            }
        }
    }

    @Test
    void crystalPadsAreReappliedAfterLaneAndSpawnClearing() {
        assertArrayEquals(new ArenaGeometry.PreparationStep[]{
                        ArenaGeometry.PreparationStep.GENERATE_LAYOUT,
                        ArenaGeometry.PreparationStep.CHOOSE_SPAWNS,
                        ArenaGeometry.PreparationStep.CLEAR_SPAWN_LANE,
                        ArenaGeometry.PreparationStep.CLEAR_SPAWNS,
                        ArenaGeometry.PreparationStep.GUARANTEE_CRYSTAL_PADS
                },
                ArenaGeometry.preparationSteps(true, true));
        assertArrayEquals(new ArenaGeometry.PreparationStep[]{
                        ArenaGeometry.PreparationStep.CHOOSE_SPAWNS,
                        ArenaGeometry.PreparationStep.CLEAR_SPAWN_LANE,
                        ArenaGeometry.PreparationStep.CLEAR_SPAWNS,
                        ArenaGeometry.PreparationStep.GUARANTEE_CRYSTAL_PADS
                },
                ArenaGeometry.preparationSteps(false, true));
        assertArrayEquals(new ArenaGeometry.PreparationStep[]{
                        ArenaGeometry.PreparationStep.CHOOSE_SPAWNS,
                        ArenaGeometry.PreparationStep.CLEAR_SPAWN_LANE,
                        ArenaGeometry.PreparationStep.CLEAR_SPAWNS
                },
                ArenaGeometry.preparationSteps(false, false));
    }

    @Test
    void reachablePadsCoverBothSpawnsAndMidpointAtEveryLiveRadius() {
        for (int radius = 5; radius <= 8; radius++) {
            int[][] pads = ArenaGeometry.reachableCrystalPadOffsets(radius, -2, 0, 2, 0, 91L);
            Set<String> coordinates = new HashSet<String>();
            for (int[] pad : pads) {
                assertTrue(ArenaGeometry.strictlyInsideBoundary(radius, pad[0], pad[1]));
                coordinates.add(pad[0] + "," + pad[1]);
            }
            assertTrue(coordinates.size() >= 3 && coordinates.size() <= 7);
            assertTrue(hasPadNear(pads, -2, 0));
            assertTrue(hasPadNear(pads, 2, 0));
            assertTrue(countPadsNear(pads, 0, 0) >= 1);
            assertFalse(coordinates.contains("-2,0"));
            assertFalse(coordinates.contains("2,0"));
            assertFalse(coordinates.contains("0,0"));
        }
    }

    @Test
    void episodeSeedMakesPadLayoutReproducibleButNotFixed() {
        int[][] expected = ArenaGeometry.reachableCrystalPadOffsets(5, -2, 0, 2, 0, 1234L);
        assertArrayEquals(expected,
                ArenaGeometry.reachableCrystalPadOffsets(5, -2, 0, 2, 0, 1234L));

        Set<String> layouts = new HashSet<String>();
        Set<Integer> padCounts = new HashSet<Integer>();
        Set<String> commonCoordinates = null;
        for (long seed = 0; seed < 32; seed++) {
            int[][] pads = ArenaGeometry.reachableCrystalPadOffsets(5, -2, 0, 2, 0, seed);
            padCounts.add(pads.length);
            Set<String> coordinates = new HashSet<String>();
            StringBuilder signature = new StringBuilder();
            for (int[] pad : pads) {
                String coordinate = pad[0] + "," + pad[1];
                coordinates.add(coordinate);
                signature.append(coordinate).append(';');
            }
            layouts.add(signature.toString());
            if (commonCoordinates == null) commonCoordinates = new HashSet<String>(coordinates);
            else commonCoordinates.retainAll(coordinates);
        }

        assertTrue(layouts.size() >= 24, "episode seeds should create varied pad layouts");
        assertTrue(padCounts.size() >= 4, "episode seeds should vary obsidian pad count");
        assertTrue(commonCoordinates != null && commonCoordinates.isEmpty(),
                "no crystal coordinate should be present in every episode");
    }

    @Test
    void randomizedPadsStayReachableForEveryLiveInteriorSpawnPair() {
        for (int radius = 5; radius <= 8; radius++) {
            int limit = radius - 1;
            for (int firstX = -limit; firstX <= limit; firstX++) {
                for (int firstZ = -limit; firstZ <= limit; firstZ++) {
                    for (int secondX = -limit; secondX <= limit; secondX++) {
                        for (int secondZ = -limit; secondZ <= limit; secondZ++) {
                            int deltaX = secondX - firstX;
                            int deltaZ = secondZ - firstZ;
                            int distanceSquared = deltaX * deltaX + deltaZ * deltaZ;
                            if (distanceSquared < 9 || distanceSquared > 25) continue;

                            int midpointX = (int) Math.round((firstX + secondX) / 2.0);
                            int midpointZ = (int) Math.round((firstZ + secondZ) / 2.0);
                            for (long seed = 0; seed < 8; seed++) {
                                int[][] pads = ArenaGeometry.reachableCrystalPadOffsets(
                                        radius, firstX, firstZ, secondX, secondZ, seed);
                                assertTrue(pads.length >= 3 && pads.length <= 7);
                                Set<String> coordinates = new HashSet<String>();
                                for (int[] pad : pads) {
                                    assertTrue(ArenaGeometry.strictlyInsideBoundary(
                                            radius, pad[0], pad[1]));
                                    assertTrue(coordinates.add(pad[0] + "," + pad[1]),
                                            "crystal pads must be unique");
                                    assertFalse(pad[0] == firstX && pad[1] == firstZ);
                                    assertFalse(pad[0] == secondX && pad[1] == secondZ);
                                    assertFalse(pad[0] == midpointX && pad[1] == midpointZ);
                                }

                                assertTrue(isPadNear(pads[0], firstX, firstZ));
                                assertTrue(isPadNear(pads[1], secondX, secondZ));
                                assertTrue(isPadNear(pads[2], midpointX, midpointZ));
                            }
                        }
                    }
                }
            }
        }
    }

    @Test
    void obsidianSupplyIsFairReproducibleAndVariedAcrossEpisodes() {
        Set<Integer> supplies = new HashSet<Integer>();
        for (long seed = 0; seed < 128; seed++) {
            int amount = ArenaGeometry.obsidianSupply(seed);
            assertTrue(amount >= 16 && amount <= 64);
            assertEquals(amount, ArenaGeometry.obsidianSupply(seed));
            supplies.add(amount);
        }
        assertTrue(supplies.size() >= 40);
    }

    private static boolean hasPadNear(int[][] pads, int anchorX, int anchorZ) {
        return countPadsNear(pads, anchorX, anchorZ) >= 1;
    }

    private static int countPadsNear(int[][] pads, int anchorX, int anchorZ) {
        int count = 0;
        for (int[] pad : pads) {
            if (isPadNear(pad, anchorX, anchorZ)) count++;
        }
        return count;
    }

    private static boolean isPadNear(int[] pad, int anchorX, int anchorZ) {
        int manhattanDistance = Math.abs(pad[0] - anchorX) + Math.abs(pad[1] - anchorZ);
        return manhattanDistance >= 1 && manhattanDistance <= 2;
    }
}
