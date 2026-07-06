import { DQNAgent } from '../engine/DQNAgent';
import type { Task } from '../sim/Task';

const SCORE_WINDOW = 100; // episodes averaged for the moving-average score
const CHART_HISTORY_LIMIT = 200; // points kept for the loss/Q/moving-avg charts
const METRIC_SAMPLE_RATE = 0.05; // only sample ~5% of steps into chart history, keeps it light
const EMA_WEIGHT = 0.01; // smoothing applied to per-step loss/Q before charting

function pushCapped(arr: number[], value: number, limit: number): void {
    arr.push(value);
    if (arr.length > limit) arr.shift();
}

/**
 * Drives the train/act/remember/replay cycle. Runs as many env steps as fit
 * in a time budget per tick, then yields via setTimeout(0) so the render
 * loop and the rest of the page stay responsive.
 *
 * Knows nothing about canvas/rendering — DiagnosticsPanel reads its public
 * fields to draw the UI, but Trainer never draws anything itself.
 */
export class Trainer {
    public episode = 1;
    public score = 0;
    public maxScore = 0;
    public currentActionIndex = 0;
    public currentState: number[];

    public readonly scoreHistory: number[] = [];
    public readonly lossHistory: number[] = [];
    public readonly qValueHistory: number[] = [];
    public readonly movingAverageHistory: number[] = [];

    public currentLoss = 0;
    public currentQ = 0;
    public currentMovingAvg = 0;
    public maxMovingAvg = 0;

    public stepsPerSecond = 0;
    private stepsThisSecond = 0;
    private lastThroughputCheck = performance.now();

    private readonly agent: DQNAgent;
    private readonly task: Task;
    private readonly thrustLevels: number[];
    public timeBudgetMs: number;

    private totalSteps = 0;

    public stepsThisEpisode = 0;
    public maxSurvivalTime = 0;
    public currentAvgSurvivalTime = 0;
    public readonly survivalTimeHistory: number[] = [];
    private readonly dt = 0.016; // Your FIXED_DT


    constructor(agent: DQNAgent, task: Task, thrustLevels: number[], timeBudgetMs: number) {
        this.agent = agent;
        this.task = task;
        this.thrustLevels = thrustLevels;
        this.timeBudgetMs = timeBudgetMs;
        this.currentState = task.reset();
    }

    private doOneStep(): void {
        this.currentActionIndex = this.agent.getAction(this.currentState);
        const thrustFraction = this.thrustLevels[this.currentActionIndex];

        const { nextState, reward, done } = this.task.step(thrustFraction);
        this.agent.remember(this.currentState, this.currentActionIndex, reward, nextState, done);

        this.totalSteps++; // <--- Increment step counter
        this.stepsThisEpisode++;

        // ONLY TRAIN EVERY 4 STEPS
        let metrics = null;
        if (this.totalSteps % 4 === 0) {
            metrics = this.agent.replay();
        }
        if (metrics && !Number.isNaN(metrics.loss)) {
            this.currentLoss = this.currentLoss * (1 - EMA_WEIGHT) + metrics.loss * EMA_WEIGHT;
            this.currentQ = this.currentQ * (1 - EMA_WEIGHT) + metrics.qValue * EMA_WEIGHT;
            if (Math.random() < METRIC_SAMPLE_RATE) {
                pushCapped(this.lossHistory, this.currentLoss, CHART_HISTORY_LIMIT);
                pushCapped(this.qValueHistory, this.currentQ, CHART_HISTORY_LIMIT);
            }
        }

        if (done) {
            this.onEpisodeEnd();
        } else {
            this.score += reward;
            this.currentState = nextState;
        }

        this.stepsThisSecond++;
    }

    private onEpisodeEnd(): void {

        // --- NEW: Calculate survival time ---
        const survivalSeconds = this.stepsThisEpisode * this.dt;
        if (survivalSeconds > this.maxSurvivalTime) this.maxSurvivalTime = survivalSeconds;

        pushCapped(this.survivalTimeHistory, survivalSeconds, 100); // SCORE_WINDOW
        this.currentAvgSurvivalTime = this.survivalTimeHistory.reduce((a, b) => a + b, 0) / this.survivalTimeHistory.length;

        this.stepsThisEpisode = 0; // Reset for next episode
        // ------------------------------------

        if (this.score > this.maxScore) this.maxScore = this.score;
        pushCapped(this.scoreHistory, this.score, SCORE_WINDOW);

        this.currentMovingAvg = this.scoreHistory.reduce((a, b) => a + b, 0) / this.scoreHistory.length;
        if (this.currentMovingAvg > this.maxMovingAvg) this.maxMovingAvg = this.currentMovingAvg;
        pushCapped(this.movingAverageHistory, this.currentMovingAvg, CHART_HISTORY_LIMIT);

        this.agent.decayEpsilon();
        this.currentState = this.task.reset();
        this.episode++;
        this.score = 0;
    }

    /** Runs as much training as fits in the time budget, then reschedules itself. */
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