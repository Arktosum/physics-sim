// src/training/PPOTrainer.ts

import { PPOAgent } from '../engine/PPOAgent';
import { PPORolloutBuffer } from './PPORolloutBuffer';
import type { Task } from '../sim/Task';

const SCORE_WINDOW = 100;
const CHART_HISTORY_LIMIT = 200;
const ACTION_HISTORY_LIMIT = 200; 

// PPO SPECIFIC CONSTANTS
const HORIZON = 4096;             // How many steps to take before pausing to train
const MAX_EPISODE_STEPS = 4000;   // Truncate episodes that go on too long

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

    public timeBudgetMs: number;
    public stepsThisEpisode = 0;
    private stepsSinceLastTrain = 0; // Tracks the Horizon

    private readonly agent: PPOAgent;
    private readonly task: Task;
    private readonly buffer: PPORolloutBuffer;

    constructor(agent: PPOAgent, task: Task, timeBudgetMs: number) {
        this.agent = agent;
        this.task = task;
        this.timeBudgetMs = timeBudgetMs;
        this.buffer = new PPORolloutBuffer();
        this.currentState = task.reset();
    }

    private doOneStep(): void {
        // 1. Ask the agent for the action AND its state of mind
        const { rawAction, clampedAction, logProb, value } = this.agent.act(this.currentState);
        this.currentAction = clampedAction;

        pushCapped(this.actionHistory, clampedAction, ACTION_HISTORY_LIMIT);

        // 2. Step the physics
        const { nextState, reward, done } = this.task.step(clampedAction);

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

        // 5. THE HORIZON CHECK (Train when the buffer is full)
        this.stepsSinceLastTrain++;
        if (this.stepsSinceLastTrain >= HORIZON) {
            this.train();
        }
    }

    private train(): void {
        // 1. Run the PPO math on the 2048 steps
        const metrics = this.agent.learn(this.buffer);

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
        this.stepsSinceLastTrain = 0;
    }

    public tick = (): void => {
        const start = performance.now();
        while (performance.now() - start < this.timeBudgetMs) {
            this.doOneStep();
        }
        setTimeout(this.tick, 0);
    };
}