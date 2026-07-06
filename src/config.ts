// ==========================================
// All the knobs for the single-pole DQN cart balancer live here.
// Nothing in here should require touching main.ts, CartPoleTask, or Trainer.
// ==========================================

export const CANVAS_WIDTH = 800;
export const CANVAS_HEIGHT = 600;
export const TRACK_HEIGHT = 400;

/** Discrete thrust levels the agent can pick between (fraction of motor.thrustPower). */
export const THRUST_LEVELS = [-1.0, -0.66, -0.33, 0.0, 0.33, 0.66, 1.0];

/** Subtracted from reward each step, scaled by |thrust|, to discourage thrashing. */
export const ENERGY_PENALTY_WEIGHT = 0.02;

/** Physics/agent step size in seconds. Decoupled from render framerate. */
export const FIXED_DT = 0.016;

/** How many ms of training work the loop does before yielding back to the event loop. */
export const TRAIN_TIME_BUDGET_MS = 8;

export const AGENT_CONFIG = {
    inputSize: 4,
    epsilonDecay: 0.995,
    learningRate: 0.001,
};