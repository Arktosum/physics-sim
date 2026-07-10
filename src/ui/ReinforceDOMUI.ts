import { ReinforceTrainer } from '../training/ReinforceTrainer';

export class ReinforceDOMUI {
    private trainer: ReinforceTrainer;

    // DOM Element references
    private elEp = document.getElementById('ui-ep')!;
    private elScore = document.getElementById('ui-score')!;
    private elScoreAvg = document.getElementById('ui-score-avg')!;
    private elSurvival = document.getElementById('ui-survival')!;

    private elCriticLoss = document.getElementById('ui-critic-loss')!;
    private elAdvantage = document.getElementById('ui-advantage')!;
    private elEntropy = document.getElementById('ui-entropy')!;
    private elClip = document.getElementById('ui-clip')!;

    private actionCtx: CanvasRenderingContext2D;
    private scoreCtx: CanvasRenderingContext2D;

    constructor(trainer: ReinforceTrainer) {
        this.trainer = trainer;
        const actionCanvas = document.getElementById('ui-action-chart') as HTMLCanvasElement;
        const scoreCanvas = document.getElementById('ui-score-chart') as HTMLCanvasElement;
        this.actionCtx = actionCanvas.getContext('2d')!;
        this.scoreCtx = scoreCanvas.getContext('2d')!;
    }

    public update(): void {
        // 1. Update Text Readouts
        this.elEp.textContent = this.trainer.episode.toString();
        this.elScore.textContent = this.trainer.score.toFixed(1);
        this.elScoreAvg.textContent = this.trainer.currentMovingAvg.toFixed(1);
        this.elSurvival.textContent = (this.trainer.stepsThisEpisode * 0.016).toFixed(2) + 's';

        this.elCriticLoss.textContent = this.trainer.currentCriticLoss.toFixed(4);
        this.elAdvantage.textContent = this.trainer.currentAdvantage.toFixed(2);
        this.elEntropy.textContent = this.trainer.currentEntropy.toFixed(3);
        this.elClip.textContent = (this.trainer.currentGradientClipRate * 100).toFixed(1) + '%';

        // 2. Draw Mini Charts
        this.drawActionHistogram();
        this.drawScoreHistory();
    }

    private drawActionHistogram(): void {
        const ctx = this.actionCtx;
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;

        ctx.clearRect(0, 0, width, height);

        const history = this.trainer.actionHistory;
        if (history.length === 0) return;

        // Create 21 bins for range [-1, 1]
        const numBins = 21;
        const bins = new Array(numBins).fill(0);

        for (let i = 0; i < history.length; i++) {
            // Map [-1, 1] to bin index [0, 20]
            let binIdx = Math.floor(((history[i] + 1) / 2) * numBins);
            if (binIdx >= numBins) binIdx = numBins - 1;
            if (binIdx < 0) binIdx = 0;
            bins[binIdx]++;
        }

        const maxBin = Math.max(...bins, 1); // Avoid div by 0
        const barWidth = width / numBins;

        ctx.fillStyle = '#4fc1ff';
        for (let i = 0; i < numBins; i++) {
            const barHeight = (bins[i] / maxBin) * height;
            ctx.fillRect(i * barWidth, height - barHeight, barWidth - 1, barHeight);
        }
    }

    private drawScoreHistory(): void {
        const ctx = this.scoreCtx;
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;

        ctx.clearRect(0, 0, width, height);

        const history = this.trainer.scoreHistory;
        if (history.length < 2) return;

        const maxScore = Math.max(...history, 10);
        const minScore = Math.min(...history, 0);
        const range = maxScore - minScore || 1;

        ctx.beginPath();
        ctx.strokeStyle = '#ce9178';
        ctx.lineWidth = 2;

        for (let i = 0; i < history.length; i++) {
            const x = (i / (history.length - 1)) * width;
            // Normalize score to height
            const y = height - ((history[i] - minScore) / range) * height;

            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();
    }
}