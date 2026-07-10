import type { AlgorithmUIConfig, BaseMetrics } from '../MetricsPanel';

export interface DQNMetrics extends BaseMetrics {
    currentLoss: number;
    currentQ: number;
    lossHistory: number[];
    qValueHistory: number[];
    latestQValues: number[];
    currentActionIndex: number;
    epsilon: number;
}

// Matches config.ts THRUST_LEVELS: [-1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0]
const THRUST_BAR_LABELS = ['-100', '-66', '-33', '0', '33', '66', '100'];

export const dqnUIConfig: AlgorithmUIConfig<DQNMetrics> = {
    extraStatGroups: [
        {
            heading: 'Agent Health (DQN)',
            rows: [
                {
                    label: 'Loss', get: m => m.currentLoss.toFixed(4),
                    info: 'Mean-squared error between the network\'s predicted Q-value and the TD target it\'s being trained toward. Should generally trend down, though it can stay noisy — DQN\'s target itself keeps shifting as the target network periodically syncs.',
                },
                {
                    label: 'Avg Max Q', get: m => m.currentQ.toFixed(2),
                    info: 'Average of the network\'s highest predicted Q-value across recent states — a rough gauge of how optimistic the agent currently is about its own prospects.',
                },
                {
                    label: 'Chaos (ε)', get: m => (m.epsilon * 100).toFixed(1) + '%',
                    info: 'Epsilon-greedy exploration rate: the probability of taking a completely random action instead of the network\'s current best guess. Starts near 100% and decays over training as the agent shifts from exploring to exploiting what it has learned.',
                    status: m => m.epsilon < 0.05 ? { text: 'EXPLOITING', kind: 'good' }
                        : m.epsilon > 0.5 ? { text: 'EXPLORING', kind: 'warn' }
                            : null,
                },
            ],
        },
    ],
    extraCharts: [
        {
            title: 'Q-Values per Thrust Level',
            kind: 'bar',
            get: m => m.latestQValues,
            barLabels: THRUST_BAR_LABELS,
            highlightIndex: m => m.currentActionIndex,
            info: 'The network\'s predicted value for each of the fixed thrust levels it can choose from, for the current state. The highlighted bar is whichever action it just took.',
        },
        {
            title: 'Loss History', kind: 'line', color: '#ef4444', get: m => m.lossHistory,
            info: 'Training loss over recent learning updates. Trending down and stabilizing is healthy; wild, sustained spikes suggest the target network isn\'t syncing often enough to keep the target stable.',
        },
        {
            title: 'Avg Max Q History', kind: 'line', color: '#8b5cf6', get: m => m.qValueHistory,
            info: 'How the agent\'s average confidence in its own best action has changed over recent updates.',
        },
    ],
};
