package dev.mcbot.arena;

import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.List;

/**
 * Bidirectional arena-size curriculum. Each lane owns one instance so sword,
 * crystal, combined, and terrain evidence can never unlock one another.
 */
final class ArenaSizeCurriculum {
    enum StageChange { NONE, ADVANCED, REGRESSED }

    private final List<Integer> radii;
    private final List<Integer> episodeThresholds;
    private final int performanceWindow;
    private final double advanceEngagementRate;
    private final double advanceNonTimeoutRate;
    private final double regressEngagementRate;
    private final double regressNonTimeoutRate;
    private final int stageChangeCooldownEpisodes;
    private final Deque<Boolean> recentEngagement = new ArrayDeque<Boolean>();
    private final Deque<Boolean> recentNonTimeout = new ArrayDeque<Boolean>();
    private int stageIndex;
    private long completedEpisodes;
    private long lastStageChangeEpisode;
    private long nextAdvanceEpisode;

    static ArenaSizeCurriculum create(int spacing, int fallbackRadius,
                                      List<Integer> configuredRadii,
                                      List<Integer> configuredThresholds,
                                      int performanceWindow,
                                      double advanceEngagementRate,
                                      double advanceNonTimeoutRate,
                                      double regressEngagementRate,
                                      double regressNonTimeoutRate,
                                      int stageChangeCooldownEpisodes) {
        List<Integer> radii = new ArrayList<Integer>();
        List<Integer> thresholds = new ArrayList<Integer>();
        if (configuredRadii != null) {
            for (int index = 0; index < configuredRadii.size(); index++) {
                int radius = ArenaGeometry.clampRadius(spacing, configuredRadii.get(index));
                if (!radii.isEmpty() && radius <= radii.get(radii.size() - 1)) continue;
                int requestedThreshold = configuredThresholds != null && index < configuredThresholds.size()
                        ? configuredThresholds.get(index)
                        : (thresholds.isEmpty() ? 0 : thresholds.get(thresholds.size() - 1) + 1);
                int threshold = thresholds.isEmpty() ? 0
                        : Math.max(thresholds.get(thresholds.size() - 1) + 1, requestedThreshold);
                radii.add(radius);
                thresholds.add(threshold);
            }
        }
        if (radii.isEmpty()) {
            radii.add(ArenaGeometry.clampRadius(spacing, fallbackRadius));
            thresholds.add(0);
        }
        return new ArenaSizeCurriculum(radii, thresholds, performanceWindow,
                advanceEngagementRate, advanceNonTimeoutRate,
                regressEngagementRate, regressNonTimeoutRate, stageChangeCooldownEpisodes);
    }

    ArenaSizeCurriculum(List<Integer> radii, List<Integer> episodeThresholds,
                        int performanceWindow, double advanceEngagementRate,
                        double advanceNonTimeoutRate, double regressEngagementRate,
                        double regressNonTimeoutRate, int stageChangeCooldownEpisodes) {
        if (radii == null || radii.isEmpty() || radii.size() != episodeThresholds.size()) {
            throw new IllegalArgumentException("arena radius stages and thresholds must be non-empty and aligned");
        }
        this.radii = Collections.unmodifiableList(new ArrayList<Integer>(radii));
        this.episodeThresholds = Collections.unmodifiableList(new ArrayList<Integer>(episodeThresholds));
        this.performanceWindow = Math.max(1, performanceWindow);
        this.advanceEngagementRate = unitRate(advanceEngagementRate);
        this.advanceNonTimeoutRate = unitRate(advanceNonTimeoutRate);
        this.regressEngagementRate = unitRate(regressEngagementRate);
        this.regressNonTimeoutRate = unitRate(regressNonTimeoutRate);
        this.stageChangeCooldownEpisodes = Math.max(0, stageChangeCooldownEpisodes);
    }

    StageChange recordCompletedEpisode(boolean autonomouslyEngaged, boolean nonTimeoutEnding) {
        completedEpisodes++;
        recentEngagement.addLast(autonomouslyEngaged);
        recentNonTimeout.addLast(nonTimeoutEnding);
        trim(recentEngagement);
        trim(recentNonTimeout);

        if (recentEngagement.size() < performanceWindow
                || completedEpisodes - lastStageChangeEpisode < stageChangeCooldownEpisodes) {
            return StageChange.NONE;
        }

        double engagementRate = recentEngagementRate();
        double nonTimeoutRate = recentNonTimeoutRate();
        if (stageIndex > 0 && (engagementRate < regressEngagementRate
                || nonTimeoutRate < regressNonTimeoutRate)) {
            stageIndex--;
            stageChanged(true);
            return StageChange.REGRESSED;
        }
        if (stageIndex + 1 < radii.size()
                && completedEpisodes >= nextAdvanceEpisode
                && completedEpisodes >= episodeThresholds.get(stageIndex + 1)
                && engagementRate >= advanceEngagementRate
                && nonTimeoutRate >= advanceNonTimeoutRate) {
            stageIndex++;
            stageChanged(false);
            return StageChange.ADVANCED;
        }
        return StageChange.NONE;
    }

    private void stageChanged(boolean regressed) {
        lastStageChangeEpisode = completedEpisodes;
        // A failed stage is retried only after two complete windows at the
        // easier stage: one ordinary cooldown/proof window plus one bounded
        // consolidation window.  This prevents 32-up/32-down oscillation
        // without permanently locking a lane out of the harder stage.
        nextAdvanceEpisode = regressed
                ? completedEpisodes + 2L * performanceWindow
                : completedEpisodes;
        // Every new radius must prove itself with a fresh complete window.
        recentEngagement.clear();
        recentNonTimeout.clear();
    }

    private void trim(Deque<Boolean> values) {
        while (values.size() > performanceWindow) values.removeFirst();
    }

    private static double rate(Deque<Boolean> values) {
        if (values.isEmpty()) return 0.0;
        int successes = 0;
        for (boolean value : values) if (value) successes++;
        return successes / (double) values.size();
    }

    private static double unitRate(double value) {
        return Math.max(0.0, Math.min(1.0, value));
    }

    int currentRadius() { return radii.get(stageIndex); }
    int maximumRadius() { return radii.get(radii.size() - 1); }
    int stageIndex() { return stageIndex; }
    int stageNumber() { return stageIndex + 1; }
    int stageCount() { return radii.size(); }
    long completedEpisodes() { return completedEpisodes; }
    long nextAdvanceEpisode() { return nextAdvanceEpisode; }
    List<Integer> radii() { return radii; }
    List<Integer> episodeThresholds() { return episodeThresholds; }
    double recentEngagementRate() { return rate(recentEngagement); }
    double recentNonTimeoutRate() { return rate(recentNonTimeout); }
}
