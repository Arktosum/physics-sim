import type { AlgorithmUIConfig, BaseMetrics } from '../MetricsPanel';

export interface QLearningMetrics extends BaseMetrics {
    statesVisited: number;
    coverageFraction: number; // 0..1, statesVisited / totalPossibleStates
    singleVisitFraction: number; // 0..1, fraction of visit-counts that are exactly 1 (noisy Q-estimates)
    totalVisits: number;
}

export const qLearningUIConfig: AlgorithmUIConfig<QLearningMetrics> = {
    extraStatGroups: [
        {
            heading: 'Agent Health (Q-Learning)',
            rows: [
                // No epsilon/"chaos" row: this agent runs UCB exploration by
                // default (docs/journey/01-q-learning.md), not epsilon-greedy —
                // epsilon is genuinely inert in that mode, so showing it would
                // just be a number that doesn't reflect what's actually
                // happening. Coverage is the real signal for this algorithm.
                {
                    label: 'States Visited', get: m => m.statesVisited.toLocaleString(),
                    info: 'This agent discretizes the continuous physics into a grid of buckets ("states") and keeps a table of learned values per state. This is how many distinct buckets it has ever landed in — a rough measure of how much of the problem it has actually experienced.',
                },
                {
                    label: 'Total Visits', get: m => m.totalVisits.toLocaleString(),
                    info: 'Cumulative count of every state-action pair the agent has updated, across the whole table — i.e. how many learning updates have happened in total.',
                },
                {
                    label: 'Coverage', get: m => (m.coverageFraction * 100).toFixed(1) + '%',
                    // Coverage growing much slower than table size as bins get
                    // finer is the actual ceiling of tabular Q-learning — worth
                    // seeing directly, not just in the write-up.
                    info: 'Percentage of the theoretically reachable grid of states the agent has ever visited. This is the number that exposes tabular Q-learning\'s real ceiling: it grows slower and slower as the grid gets finer, because a good policy avoids the states most in need of exploring.',
                    status: m => m.coverageFraction < 0.05 ? { text: 'SPARSE', kind: 'bad' }
                        : m.coverageFraction < 0.2 ? { text: 'GROWING', kind: 'warn' }
                            : { text: 'GOOD', kind: 'good' },
                },
                {
                    label: 'Single-Visit States', get: m => (m.singleVisitFraction * 100).toFixed(1) + '%',
                    info: 'Percentage of visited states that have been seen exactly once. A high number here means a large chunk of the table\'s values are one-sample estimates, not yet trustworthy.',
                    status: m => m.singleVisitFraction > 0.5 ? { text: 'NOISY ESTIMATES', kind: 'warn' } : null,
                },
            ],
        },
    ],
    extraCharts: [],
};
