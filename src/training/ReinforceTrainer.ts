import { ReinforceAgent } from '../engine/ReinforceAgent';
import { EpisodeBuffer } from './EpisodeBuffer';
import type { Task } from '../sim/Task';

const SCORE_WINDOW = 100;
const CHART_HISTORY_LIMIT = 200;
const EMA_WEIGHT = 0.01;
const ACTION_HISTORY_LIMIT = 200; // For the rolling histogram
const MAX_EPISODE_STEPS = 2000;   // The truncation safety net

function pushCapped(arr: number[], value: number, limit: number): void {
    arr.push(value);
    if (arr.length > limit) arr.shift();
}

/**
 * Same job as Trainer.ts (drive the act/step/learn cycle, expose diagnostics
 * fields, stay decoupled from rendering), but reshaped around REINFORCE's
 * fundamental difference: nothing gets trained mid-episode. Every step just
 * records a Transition; the actual learning call happens once, in
 * onEpisodeEnd(), against the whole trajectory at once.
 */
export class ReinforceTrainer {
    public episode = 1;
    public score = 0;
    public maxScore = 0;
    public currentState: number[];
    public currentAction = 0;      // last CLAMPED action actually sent to the environment
    public currentMean = 0;
    public currentStd = 0;

    public readonly scoreHistory: number[] = [];
    public readonly actorLossHistory: number[] = [];
    public readonly criticLossHistory: number[] = [];
    public readonly movingAverageHistory: number[] = [];

    public currentActorLoss = 0;
    public currentCriticLoss = 0;
    public currentAdvantage = 0;
    public currentMovingAvg = 0;
    public maxMovingAvg = 0;

    public stepsPerSecond = 0;
    private stepsThisSecond = 0;
    private lastThroughputCheck = performance.now();

    private readonly agent: ReinforceAgent;
    private readonly task: Task;
    private readonly buffer: EpisodeBuffer;
    public timeBudgetMs: number;

    public stepsThisEpisode = 0;
    public maxSurvivalTime = 0;
    public currentAvgSurvivalTime = 0;
    public readonly survivalTimeHistory: number[] = [];
    private readonly dt = 0.016;

    public readonly actionHistory: number[] = [];
    public currentGradientClipRate = 0;
    public currentEntropy = 0;

    constructor(agent: ReinforceAgent, task: Task, timeBudgetMs: number) {
        this.agent = agent;
        this.task = task;
        this.timeBudgetMs = timeBudgetMs;
        this.buffer = new EpisodeBuffer();
        this.currentState = task.reset();
    }

    private doOneStep(): void {
        const { rawAction, clampedAction, mean, std } = this.agent.act(this.currentState);
        this.currentAction = clampedAction;
        this.currentMean = mean;
        this.currentStd = std;

        // Keep the rolling histogram of actions updated every single step
        pushCapped(this.actionHistory, clampedAction, ACTION_HISTORY_LIMIT);

        const { nextState, reward, done } = this.task.step(clampedAction);
        this.buffer.add({ state: this.currentState, rawAction, mean, std, reward });
        this.stepsThisEpisode++;

        // THE SAFETY NET: If the agent survives perfectly, cut it off at MAX_EPISODE_STEPS
        if (done || this.stepsThisEpisode >= MAX_EPISODE_STEPS) {
            this.score += reward;
            this.onEpisodeEnd();
        } else {
            this.score += reward;
            this.currentState = nextState;
        }

        this.stepsThisSecond++;
    }

    private onEpisodeEnd(): void {
        const episodeData = this.buffer.getEpisode();
        if (episodeData.length > 0) {
            const metrics = this.agent.learn(episodeData);
            if (!Number.isNaN(metrics.actorLoss) && !Number.isNaN(metrics.criticLoss)) {
                this.currentActorLoss = this.currentActorLoss * (1 - EMA_WEIGHT) + metrics.actorLoss * EMA_WEIGHT;
                this.currentCriticLoss = this.currentCriticLoss * (1 - EMA_WEIGHT) + metrics.criticLoss * EMA_WEIGHT;

                this.currentAdvantage = metrics.avgAbsoluteAdvantage;
                this.currentGradientClipRate = metrics.gradientClipRate;
                this.currentEntropy = metrics.avgEntropy;

                pushCapped(this.actorLossHistory, this.currentActorLoss, CHART_HISTORY_LIMIT);
                pushCapped(this.criticLossHistory, this.currentCriticLoss, CHART_HISTORY_LIMIT);
            }
        }
        this.buffer.clear();

        // --- Survival + score bookkeeping, same shape as Trainer.ts ---
        const survivalSeconds = this.stepsThisEpisode * this.dt;
        if (survivalSeconds > this.maxSurvivalTime) this.maxSurvivalTime = survivalSeconds;
        pushCapped(this.survivalTimeHistory, survivalSeconds, 100);
        this.currentAvgSurvivalTime = this.survivalTimeHistory.reduce((a, b) => a + b, 0) / this.survivalTimeHistory.length;
        this.stepsThisEpisode = 0;

        if (this.score > this.maxScore) this.maxScore = this.score;
        pushCapped(this.scoreHistory, this.score, SCORE_WINDOW);
        this.currentMovingAvg = this.scoreHistory.reduce((a, b) => a + b, 0) / this.scoreHistory.length;
        if (this.currentMovingAvg > this.maxMovingAvg) this.maxMovingAvg = this.currentMovingAvg;
        pushCapped(this.movingAverageHistory, this.currentMovingAvg, CHART_HISTORY_LIMIT);

        this.currentState = this.task.reset();
        this.episode++;
        this.score = 0;
    }

    public tick = (): void => {
        const start = performance.now();
        while (performance.now() - start < this.timeBudgetMs) {
            this.doOneStep();
        }

        const now = performance.now();
        if (now - this.lastThroughputCheck >= 1000) {
            this.stepsPerSecond = this.stepsThisSecond;
            this.stepsThisSecond = 0;
            this.lastThroughputCheck = now;
        }

        setTimeout(this.tick, 0);
    };
}