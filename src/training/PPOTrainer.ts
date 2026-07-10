// src/training/PPOTrainer.ts

import { PPOAgent } from '../engine/PPOAgent';
import { PPORolloutBuffer } from './PPORolloutBuffer';
import type { Task } from '../sim/Task';

const SCORE_WINDOW = 100;
const ACTION_HISTORY_LIMIT = 200;

// PPO SPECIFIC CONSTANTS
const HORIZON = 2048;             // How many steps to take before pausing to train
const MAX_EPISODE_STEPS = 2000;   // Truncate episodes that go on too long

function pushCapped(arr: number[], value: number, limit: number): void {
    arr.push(value);
    if (arr.length > limit) arr.shift();
}

export class PPOTrainer {
    public episode = 1;
    public score = 0;
    public currentState: number[];

    // UI Diagnostics
    public currentAction = 0;
    public readonly actionHistory: number[] = [];
    public readonly scoreHistory: number[] = [];
    public readonly survivalTimeHistory: number[] = []; // NEW
    public currentMovingAvg = 0;
    public maxSurvivalTime = 0; // NEW

    // PPO Health Metrics
    public currentActorLoss = 0;
    public currentCriticLoss = 0;
    public currentAdvantage = 0;
    public currentClipFraction = 0;
    public currentKlDivergence = 0;

    // Perf diagnostics — how long the actual training math takes per HORIZON,
    // and how many physics steps we're managing to push per second overall.
    // Exposed so we can watch these live instead of guessing where time goes.
    public totalSteps = 0;
    public lastTrainMs = 0;
    public avgTrainMs = 0;

    public timeBudgetMs: number;
    public stepsThisEpisode = 0;
    private stepsSinceLastTrain = 0; // Tracks the Horizon
    // Set by doOneStep() when the Horizon is hit; consumed by tick() BETWEEN
    // physics steps rather than mid-loop, so the (now async, chunked) train()
    // call is always awaited from a clean point instead of firing off inside
    // a synchronous while loop.
    private needsTraining = false;

    private readonly agent: PPOAgent;
    private readonly task: Task;
    private readonly buffer: PPORolloutBuffer;
    // The environment's actual next-state, captured BEFORE any episode-end reset
    // overwrites currentState. Needed so train() can bootstrap GAE correctly when
    // the HORIZON cuts a rollout off mid-episode rather than at a real termination.
    private lastNextState: number[];

    constructor(agent: PPOAgent, task: Task, timeBudgetMs: number) {
        this.agent = agent;
        this.task = task;
        this.timeBudgetMs = timeBudgetMs;
        this.buffer = new PPORolloutBuffer();
        this.currentState = task.reset();
        this.lastNextState = this.currentState;
    }

    private doOneStep(): void {
        // 1. Ask the agent for the action AND its state of mind
        const { rawAction, clampedAction, logProb, value } = this.agent.act(this.currentState);
        this.currentAction = clampedAction;

        pushCapped(this.actionHistory, clampedAction, ACTION_HISTORY_LIMIT);

        // 2. Step the physics
        const { nextState, reward, done } = this.task.step(clampedAction);
        this.lastNextState = nextState; // capture BEFORE the done-branch below potentially calls task.reset()

        // 3. Add to the Goldfish Memory (Using rawAction for accurate gradient math later!)
        this.buffer.add({
            state: this.currentState,
            action: rawAction,
            reward,
            value,
            logProb,
            done
        });

        this.stepsThisEpisode++;
        this.totalSteps++;
        this.score += reward;

        // 4. Handle Episode End (But DO NOT TRAIN YET)
        if (done || this.stepsThisEpisode >= MAX_EPISODE_STEPS) {
            pushCapped(this.scoreHistory, this.score, SCORE_WINDOW);
            this.currentMovingAvg = this.scoreHistory.reduce((a, b) => a + b, 0) / this.scoreHistory.length;

            // --- NEW SURVIVAL TRACKING ---
            const survivalSeconds = this.stepsThisEpisode * 0.016; // 0.016 is our fixed dt
            if (survivalSeconds > this.maxSurvivalTime) this.maxSurvivalTime = survivalSeconds;
            pushCapped(this.survivalTimeHistory, survivalSeconds, SCORE_WINDOW);
            // -----------------------------

            this.score = 0;
            this.stepsThisEpisode = 0;
            this.episode++;
            this.currentState = this.task.reset();
        } else {
            this.currentState = nextState;
        }

        // 5. THE HORIZON CHECK (Flag training; actually train() once tick()'s
        // physics loop below has stopped, not from inside it.)
        this.stepsSinceLastTrain++;
        if (this.stepsSinceLastTrain >= HORIZON) {
            this.stepsSinceLastTrain = 0;
            this.needsTraining = true;
        }
    }

    private async train(): Promise<void> {
        const trainStart = performance.now();

        // 1. Bootstrap: what does the Critic think the state AFTER our last stored
        // transition is worth? (Safe to compute even if that transition was terminal —
        // PPOAgent.learn's nextNonTerminal gate zeroes it out automatically in that case.)
        const bootstrapValue = this.agent.getValue(this.lastNextState);

        // 2. Run the PPO math on the 2048 steps. learn() yields internally every
        // few hundred samples so this doesn't block the thread's own timers for
        // its whole ~6000-sample duration.
        const metrics = await this.agent.learn(this.buffer, bootstrapValue);

        // 2. Update UI metrics
        if (!Number.isNaN(metrics.actorLoss)) {
            this.currentActorLoss = metrics.actorLoss;
            this.currentCriticLoss = metrics.criticLoss;
            this.currentAdvantage = metrics.avgAdvantage;
            this.currentClipFraction = metrics.clipFraction;
            this.currentKlDivergence = metrics.klDivergence;
        }

        // 3. BURN THE BUFFER. PPO only learns from fresh data.
        this.buffer.clear();

        // 4. Record how long that took. Exponential moving average so a single
        // slow call (e.g. first JIT-cold call) doesn't dominate the display.
        this.lastTrainMs = performance.now() - trainStart;
        const alpha = 0.2;
        this.avgTrainMs = this.avgTrainMs === 0
            ? this.lastTrainMs
            : alpha * this.lastTrainMs + (1 - alpha) * this.avgTrainMs;
    }

    public tick = async (): Promise<void> => {
        const start = performance.now();
        while (performance.now() - start < this.timeBudgetMs && !this.needsTraining) {
            this.doOneStep();
        }

        if (this.needsTraining) {
            this.needsTraining = false;
            await this.train();
        }

        setTimeout(this.tick, 0);
    };
}

/**
 * The subset of PPOTrainer's public fields PPODOMUI actually reads.
 * PPOTrainer satisfies this structurally already (no code change needed there).
 * This exists so a plain-data mirror object on the main thread — fed by
 * postMessage from a training Worker — can be handed to PPODOMUI too,
 * without PPODOMUI needing to know whether it's looking at a live trainer
 * or a snapshot relayed across a thread boundary.
 */
export interface PPOTrainerLike {
    episode: number;
    score: number;
    stepsThisEpisode: number;
    maxSurvivalTime: number;
    currentCriticLoss: number;
    currentAdvantage: number;
    currentClipFraction: number;
    currentKlDivergence: number;
    actionHistory: number[];
    scoreHistory: number[];
    survivalTimeHistory: number[];

    // Perf diagnostics. totalSteps/lastTrainMs/avgTrainMs come straight from
    // the trainer; stepsPerSecond and maxWorkerFrameGapMs are computed by the
    // Worker (it's the only place with a clock on both sides of the gap);
    // mainThreadFrameGapMs is written directly by the main thread's own
    // render loop, never touched by the 'metrics' postMessage handler.
    totalSteps: number;
    lastTrainMs: number;
    avgTrainMs: number;
    stepsPerSecond: number;
    maxWorkerFrameGapMs: number;
    mainThreadFrameGapMs: number;
}