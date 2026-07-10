import { PPOTrainer } from '../training/PPOTrainer';

export class PPODOMUI {
    private trainer: PPOTrainer;

    // Text Elements
    private elEp = document.getElementById('ui-ep')!;
    private elScore = document.getElementById('ui-score')!;
    private elSurvival = document.getElementById('ui-survival')!;
    private elMaxSurvival = document.getElementById('ui-max-survival')!;

    // PPO Specific Metrics
    private elCriticLoss = document.getElementById('ui-critic-loss')!;

    private elAdvantage = document.getElementById('ui-advantage')!;
    private elAdvStatus = document.getElementById('ui-adv-status')!;

    private elClip = document.getElementById('ui-clip')!;
    private elClipStatus = document.getElementById('ui-clip-status')!;

    private elKl = document.getElementById('ui-kl')!;
    private elKlStatus = document.getElementById('ui-kl-status')!;

    // Canvases
    private actionCtx: CanvasRenderingContext2D;
    private scoreCtx: CanvasRenderingContext2D;
    private survivalCtx: CanvasRenderingContext2D;

    constructor(trainer: PPOTrainer) {
        this.trainer = trainer;
        this.actionCtx = (document.getElementById('ui-action-chart') as HTMLCanvasElement).getContext('2d')!;
        this.scoreCtx = (document.getElementById('ui-score-chart') as HTMLCanvasElement).getContext('2d')!;
        this.survivalCtx = (document.getElementById('ui-survival-chart') as HTMLCanvasElement).getContext('2d')!;
    }

    public update(): void {
        this.elEp.textContent = this.trainer.episode.toString();
        this.elScore.textContent = this.trainer.score.toFixed(1);

        const currentSurvival = this.trainer.stepsThisEpisode * 0.016;
        this.elSurvival.textContent = currentSurvival.toFixed(2) + 's';
        this.elMaxSurvival.textContent = this.trainer.maxSurvivalTime.toFixed(2) + 's';

        // 1. Critic Loss (No status needed, just needs to generally trend down)
        this.elCriticLoss.textContent = this.trainer.currentCriticLoss.toFixed(4);

        // 2. Absolute Advantage (The Coach's Error)
        const adv = this.trainer.currentAdvantage;
        this.elAdvantage.textContent = adv.toFixed(2);
        if (adv < 2.0) this.setStatus(this.elAdvStatus, 'GOOD', 'status-good');
        else if (adv < 5.0) this.setStatus(this.elAdvStatus, 'LEARNING', 'status-warn');
        else this.setStatus(this.elAdvStatus, 'HIGH', 'status-bad');

        // 3. Clip Fraction (The Speed Limit)
        const clip = this.trainer.currentClipFraction;
        this.elClip.textContent = (clip * 100).toFixed(1) + '%';
        if (clip < 0.02) this.setStatus(this.elClipStatus, 'STALLED (Too Low)', 'status-bad');
        else if (clip > 0.30) this.setStatus(this.elClipStatus, 'THRASHING (Too High)', 'status-bad');
        else if (clip > 0.20) this.setStatus(this.elClipStatus, 'FAST', 'status-warn');
        else this.setStatus(this.elClipStatus, 'HEALTHY', 'status-good');

        // 4. KL Divergence (The Brain's Stability)
        const kl = this.trainer.currentKlDivergence;
        this.elKl.textContent = kl.toFixed(4);
        if (kl < 0.001) this.setStatus(this.elKlStatus, 'STALLED', 'status-bad');
        else if (kl > 0.08) this.setStatus(this.elKlStatus, 'DANGER (Collapsing)', 'status-bad');
        else if (kl > 0.03) this.setStatus(this.elKlStatus, 'UNSTABLE', 'status-warn');
        else this.setStatus(this.elKlStatus, 'STABLE', 'status-good');

        // Draw Charts
        this.drawActionHistogram();
        this.drawLineChart(this.scoreCtx, this.trainer.scoreHistory, '#ce9178');
        this.drawLineChart(this.survivalCtx, this.trainer.survivalTimeHistory, '#4fc1ff');
    }

    /**
     * Helper to easily swap classes and text for our status badges
     */
    private setStatus(element: HTMLElement, text: string, className: string): void {
        element.textContent = text;
        element.className = `status-badge ${className}`;
    }

    private drawActionHistogram(): void {
        const ctx = this.actionCtx;
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        ctx.clearRect(0, 0, width, height);

        const history = this.trainer.actionHistory;
        if (history.length === 0) return;

        const numBins = 21;
        const bins = new Array(numBins).fill(0);

        for (let i = 0; i < history.length; i++) {
            let binIdx = Math.floor(((history[i] + 1) / 2) * numBins);
            if (binIdx >= numBins) binIdx = numBins - 1;
            if (binIdx < 0) binIdx = 0;
            bins[binIdx]++;
        }

        const maxBin = Math.max(...bins, 1);
        const barWidth = width / numBins;

        ctx.fillStyle = '#c586c0';
        for (let i = 0; i < numBins; i++) {
            const barHeight = (bins[i] / maxBin) * height;
            ctx.fillRect(i * barWidth, height - barHeight, barWidth - 1, barHeight);
        }

        // Draw Labels for Action Chart
        ctx.fillStyle = '#888';
        ctx.font = '10px Courier New';
        ctx.fillText('-1.0', 2, 10);
        ctx.fillText('0.0', width / 2 - 10, 10);
        ctx.fillText('+1.0', width - 25, 10);
    }

    // Generic function to draw labeled line charts for Score and Survival
    private drawLineChart(ctx: CanvasRenderingContext2D, data: number[], color: string): void {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        ctx.clearRect(0, 0, width, height);

        if (data.length < 2) return;

        const maxVal = Math.max(...data, 1);
        const minVal = Math.min(...data, 0);
        const range = (maxVal - minVal) || 1;

        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;

        for (let i = 0; i < data.length; i++) {
            const x = (i / (data.length - 1)) * width;
            const y = height - ((data[i] - minVal) / range) * height;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
        }
        ctx.stroke();

        // Draw Auto-scaling Text Labels
        ctx.fillStyle = '#888';
        ctx.font = '10px Courier New';
        ctx.fillText(maxVal.toFixed(1), 5, 12);
        ctx.fillText(minVal.toFixed(1), 5, height - 5);
    }
}