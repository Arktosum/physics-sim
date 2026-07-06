export interface DiagnosticsData {
    episode: number;
    score: number;
    currentMovingAvg: number;
    currentLoss: number;
    currentQ: number;
    stepsPerSecond: number;
    lossHistory: number[];
    qValueHistory: number[];
    latestQValues: number[];
    currentActionIndex: number;
    thrustFraction: number;
    cartX: number;
}

/**
 * Draws the on-canvas HUD: the stats box, the loss/Q history charts, the
 * per-thrust-level Q-value bars, and the thrust gauge under the cart.
 * Purely a renderer — every number it draws is handed to it via draw().
 */
export class DiagnosticsPanel {
    private readonly ctx: CanvasRenderingContext2D;
    private readonly canvasWidth: number;
    private readonly trackHeight: number;
    private readonly thrustLevels: number[];

    constructor(ctx: CanvasRenderingContext2D, canvasWidth: number, trackHeight: number, thrustLevels: number[]) {
        this.ctx = ctx;
        this.canvasWidth = canvasWidth;
        this.trackHeight = trackHeight;
        this.thrustLevels = thrustLevels;
    }

    public draw(data: DiagnosticsData): void {
        this.drawThrustGauge(data.cartX, data.thrustFraction);
        this.drawStatsPanel(data);
        this.drawActionQValues(data.latestQValues, data.currentActionIndex);
    }

    private drawThrustGauge(cartX: number, thrustFraction: number): void {
        const ctx = this.ctx;
        const y = this.trackHeight + 30;
        const maxBarWidth = 80;

        ctx.strokeStyle = '#4b5563';
        ctx.strokeRect(cartX - maxBarWidth, y - 6, maxBarWidth * 2, 12);

        const barWidth = Math.abs(thrustFraction) * maxBarWidth;
        ctx.fillStyle = thrustFraction < 0 ? '#ef4444' : thrustFraction > 0 ? '#22c55e' : '#6b7280';
        if (thrustFraction < 0) {
            ctx.fillRect(cartX - barWidth, y - 6, barWidth, 12);
        } else if (thrustFraction > 0) {
            ctx.fillRect(cartX, y - 6, barWidth, 12);
        }

        ctx.fillStyle = 'white';
        ctx.font = '11px monospace';
        const label = thrustFraction === 0 ? 'COAST' : `${(thrustFraction * 100).toFixed(0)}%`;
        ctx.fillText(label, cartX - 18, y + 25);
    }

    private drawStatsPanel(data: DiagnosticsData): void {
        const ctx = this.ctx;
        ctx.fillStyle = 'rgba(0,0,0,0.7)';
        ctx.fillRect(10, 10, 250, 230);
        ctx.fillStyle = 'white';
        ctx.font = '14px monospace';
        ctx.fillText(`Episode:   ${data.episode}`, 20, 35);
        ctx.fillText(`Score:     ${data.score.toFixed(1)}`, 20, 55);
        ctx.fillText(`MovAvg:    ${data.currentMovingAvg.toFixed(1)}`, 20, 75);
        ctx.fillText(`Loss:      ${data.currentLoss.toFixed(4)}`, 20, 95);
        ctx.fillText(`Q-Val:     ${data.currentQ.toFixed(2)}`, 20, 115);
        ctx.fillStyle = '#facc15';
        ctx.fillText(`Steps/sec: ${data.stepsPerSecond}`, 20, 135);
        ctx.fillStyle = 'white';

        this.drawChart(data.lossHistory, 20, 150, 220, 40, 'MSE Loss', '#ef4444');
        this.drawChart(data.qValueHistory, 20, 195, 220, 40, 'Avg Max Q', '#8b5cf6');
    }

    private drawChart(
        data: number[],
        x: number,
        y: number,
        w: number,
        h: number,
        label: string,
        color: string,
    ): void {
        const ctx = this.ctx;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.6)';
        ctx.fillRect(x, y, w, h);
        ctx.strokeStyle = '#4b5563';
        ctx.strokeRect(x, y, w, h);

        ctx.fillStyle = 'white';
        ctx.font = '10px monospace';
        ctx.fillText(label, x + 5, y + 12);

        if (data.length < 2) return;

        const max = Math.max(...data, 1);
        const min = Math.min(...data, 0);
        const range = max - min || 1;

        ctx.beginPath();
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;

        const stepX = w / (data.length - 1);
        for (let i = 0; i < data.length; i++) {
            const px = x + i * stepX;
            const py = y + h - ((data[i] - min) / range) * h;
            if (i === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
        }
        ctx.stroke();
    }

    private drawActionQValues(latestQValues: number[], currentActionIndex: number): void {
        const ctx = this.ctx;
        const panelX = this.canvasWidth - 260;
        const panelY = 10;
        const panelW = 250;
        const panelH = 140;

        ctx.fillStyle = 'rgba(0,0,0,0.7)';
        ctx.fillRect(panelX, panelY, panelW, panelH);
        ctx.fillStyle = 'white';
        ctx.font = '11px monospace';
        ctx.fillText('Q-values per thrust level', panelX + 8, panelY + 16);

        const maxQ = Math.max(...latestQValues, 1);
        const minQ = Math.min(...latestQValues, 0);
        const range = maxQ - minQ || 1;

        const barAreaW = panelW - 20;
        const barW = barAreaW / this.thrustLevels.length - 4;
        const baseY = panelY + panelH - 20;
        const maxBarH = panelH - 45;

        for (let i = 0; i < this.thrustLevels.length; i++) {
            const x = panelX + 10 + i * (barW + 4);
            const h = ((latestQValues[i] - minQ) / range) * maxBarH;
            const isChosen = i === currentActionIndex;

            ctx.fillStyle = isChosen ? '#facc15' : '#3b82f6';
            ctx.fillRect(x, baseY - h, barW, h);

            ctx.fillStyle = '#9ca3af';
            ctx.font = '8px monospace';
            const levelLabel = this.thrustLevels[i] === 0 ? '0' : (this.thrustLevels[i] * 100).toFixed(0);
            ctx.fillText(levelLabel, x, baseY + 12);
        }
    }
}