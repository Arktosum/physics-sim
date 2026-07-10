import type { AlgorithmUIConfig, BaseMetrics } from '../MetricsPanel';

export interface PPOMetrics extends BaseMetrics {
    currentCriticLoss: number;
    currentAdvantage: number;
    currentClipFraction: number;
    currentKlDivergence: number;
    actionHistory: number[];
    currentActionMean: number;
    currentActionStd: number;
    currentThrustNewtons: number;
    maxThrustNewtons: number;
}

export const ppoUIConfig: AlgorithmUIConfig<PPOMetrics> = {
    extraStatGroups: [
        {
            heading: 'Actor Distribution',
            rows: [
                {
                    label: 'Mean (μ)', get: m => m.currentActionMean.toFixed(3),
                    info: 'The center of the action the policy currently intends to take, on a -1 to +1 scale. This is what the agent would do with zero exploration noise (same as the Live Demo view).',
                },
                {
                    label: 'Std Dev (σ)', get: m => m.currentActionStd.toFixed(3),
                    info: 'How much random noise the policy adds around its mean for exploration during training. Should generally shrink as the policy converges, without collapsing all the way to zero.',
                },
                {
                    label: 'Last Thrust', get: m => m.currentThrustNewtons.toFixed(0) + 'N',
                    info: 'The actual force, in Newtons, the last action translated into once scaled by the actuator\'s max thrust.',
                },
                {
                    label: 'Max Thrust (±)', get: m => m.maxThrustNewtons.toFixed(0) + 'N',
                    info: 'The actuator\'s hard force ceiling in either direction. If Last Thrust is frequently pinned at this value, the policy wants more control authority than the environment allows.',
                },
            ],
        },
        {
            heading: 'Agent Health (PPO)',
            rows: [
                {
                    label: 'Critic Loss', get: m => m.currentCriticLoss.toFixed(4),
                    info: 'Regression error of the Critic network, which predicts how much total reward is left from a given state. Lower means a more accurate baseline for the Actor to learn against.',
                },
                {
                    label: '|Advantage|', get: m => m.currentAdvantage.toFixed(2),
                    info: 'How surprised the Critic was by actual outcomes, on average — the gap between predicted and actual returns. Should shrink as the Critic improves.',
                    status: m => m.currentAdvantage < 2.0 ? { text: 'GOOD', kind: 'good' }
                        : m.currentAdvantage < 5.0 ? { text: 'LEARNING', kind: 'warn' }
                            : { text: 'HIGH', kind: 'bad' },
                },
                {
                    label: 'Clip Fraction', get: m => (m.currentClipFraction * 100).toFixed(1) + '%',
                    info: 'Percentage of recent updates where PPO\'s trust-region clip actually activated — i.e. where the policy wanted to change more than the allowed 20% and got capped. Near-zero means updates are too timid to matter; very high means the policy is changing so fast the clip is constantly fighting it.',
                    status: m => m.currentClipFraction < 0.02 ? { text: 'STALLED (Too Low)', kind: 'bad' }
                        : m.currentClipFraction > 0.30 ? { text: 'THRASHING (Too High)', kind: 'bad' }
                            : m.currentClipFraction > 0.20 ? { text: 'FAST', kind: 'warn' }
                                : { text: 'HEALTHY', kind: 'good' },
                },
                {
                    label: 'KL Divergence', get: m => m.currentKlDivergence.toFixed(4),
                    info: 'How much the policy actually changed during the last update, measured directly rather than inferred from the clip rate. Near-zero means training has stalled; a large spike means the policy moved further in one step than PPO is meant to allow.',
                    status: m => m.currentKlDivergence < 0.001 ? { text: 'STALLED', kind: 'bad' }
                        : m.currentKlDivergence > 0.08 ? { text: 'DANGER (Collapsing)', kind: 'bad' }
                            : m.currentKlDivergence > 0.03 ? { text: 'UNSTABLE', kind: 'warn' }
                                : { text: 'STABLE', kind: 'good' },
                },
            ],
        },
    ],
    extraCharts: [
        {
            title: 'Action Distribution (Last 30)',
            kind: 'histogramWithGaussian',
            get: m => m.actionHistory,
            gaussianMean: m => m.currentActionMean,
            gaussianStd: m => m.currentActionStd,
            info: 'A histogram of the actual clamped actions sent to the environment recently (bars), overlaid with the policy\'s current theoretical distribution (blue curve). The two should roughly match — the bars are a small, noisy sample, but a wildly different shape from the curve suggests something off in the training pipeline.',
        },
    ],
};
