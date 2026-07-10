import type { AlgorithmUIConfig, BaseMetrics } from '../MetricsPanel';

export interface ReinforceMetrics extends BaseMetrics {
    currentCriticLoss: number;
    currentAdvantage: number;
    currentEntropy: number;
    currentGradientClipRate: number;
    actionHistory: number[];
    currentMean: number;
    currentStd: number;
}

export const reinforceUIConfig: AlgorithmUIConfig<ReinforceMetrics> = {
    extraStatGroups: [
        {
            heading: 'Actor Distribution',
            rows: [
                {
                    label: 'Mean (μ)', get: m => m.currentMean.toFixed(3),
                    info: 'The center of the action the policy currently intends to take, on a -1 to +1 scale. This is what the agent would do with zero exploration noise.',
                },
                {
                    label: 'Std Dev (σ)', get: m => m.currentStd.toFixed(3),
                    info: 'How much random noise the policy adds around its mean for exploration. Higher means more random behavior; a healthy policy should narrow this over training without collapsing to zero.',
                },
            ],
        },
        {
            heading: 'Agent Health (REINFORCE)',
            rows: [
                {
                    label: 'Critic Loss', get: m => m.currentCriticLoss.toFixed(4),
                    info: 'Regression error of the Critic network, which tries to predict how much total reward is left in an episode from a given state. Lower means the Critic\'s baseline is a more accurate yardstick for the Actor to learn against.',
                },
                {
                    label: '|Advantage|', get: m => m.currentAdvantage.toFixed(2),
                    info: 'How surprised the Critic was by the actual outcome — the gap between what happened and what it predicted would happen. Should shrink over training as the Critic gets better at predicting.',
                },
                {
                    label: 'Policy Entropy', get: m => m.currentEntropy.toFixed(3),
                    // Entropy collapsing toward the LOG_STD_MIN floor (~-1.58 for this
                    // project's std clamp) is the exact failure mode documented in
                    // docs/journey/03-reinforce.md — total policy overconfidence.
                    info: 'A measure of how much randomness/uncertainty is left in the policy\'s action choice. This is the number that catches REINFORCE\'s signature failure mode: a lucky episode causing entropy to crash toward this project\'s clamp floor (~-1.58), meaning the policy has become permanently, rigidly overconfident rather than settling into a genuinely good, confident answer.',
                    status: m => m.currentEntropy < -1.4 ? { text: 'COLLAPSING', kind: 'bad' }
                        : m.currentEntropy < -0.5 ? { text: 'NARROWING', kind: 'warn' }
                            : { text: 'HEALTHY', kind: 'good' },
                },
                {
                    label: 'Gradient Clip Rate', get: m => (m.currentGradientClipRate * 100).toFixed(1) + '%',
                    info: 'Percentage of recent updates where the gradient had to be clamped to stop it from exploding the network weights. Occasional clipping is normal; a sustained high rate means training updates are becoming unstably large.',
                    status: m => m.currentGradientClipRate > 0.3 ? { text: 'EXPLODING', kind: 'bad' } : null,
                },
            ],
        },
    ],
    extraCharts: [
        {
            title: 'Action Distribution (Last 200)',
            kind: 'histogramWithGaussian',
            get: m => m.actionHistory,
            gaussianMean: m => m.currentMean,
            gaussianStd: m => m.currentStd,
            info: 'A histogram of the actual clamped actions sent to the environment recently (bars), overlaid with the policy\'s current theoretical distribution (blue curve). A healthy, still-exploring policy shows a spread-out bell shape; a collapsed one shows all the bars piled onto -1 or +1.',
        },
    ],
};
