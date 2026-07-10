import { defineConfig } from 'vite';
import { resolve } from 'path';

// Without this, `vite build` only bundles index.html by default — every
// other page here (Q-learning/DQN/REINFORCE/PPO x single/double pendulum)
// would silently be dropped from a production build even though `npm run
// dev` serves them all fine. Every HTML entry point needs to be listed here
// explicitly for `npm run build` to actually produce a deployable showcase.
export default defineConfig({
    build: {
        rollupOptions: {
            input: {
                index: resolve(__dirname, 'index.html'),
                qLearningSingle: resolve(__dirname, 'q-learning-single.html'),
                qLearningDouble: resolve(__dirname, 'q-learning-double.html'),
                dqnSingle: resolve(__dirname, 'dqn-single.html'),
                dqnDouble: resolve(__dirname, 'dqn-double.html'),
                reinforceSingle: resolve(__dirname, 'reinforce-single.html'),
                reinforceDouble: resolve(__dirname, 'reinforce-double.html'),
                ppoSingle: resolve(__dirname, 'ppo-single.html'),
                ppoDouble: resolve(__dirname, 'ppo-double.html'),
                journeyIndex: resolve(__dirname, 'docs/journey/index.html'),
                journeyQLearning: resolve(__dirname, 'docs/journey/01-q-learning.html'),
                journeyDqn: resolve(__dirname, 'docs/journey/02-dqn.html'),
                journeyReinforce: resolve(__dirname, 'docs/journey/03-reinforce.html'),
                journeyPpo: resolve(__dirname, 'docs/journey/04-ppo.html'),
            },
        },
    },
});
