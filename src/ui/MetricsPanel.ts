// src/ui/MetricsPanel.ts
//
// Replaces DiagnosticsPanel/ReinforceDOMUI/PPODOMUI (three near-identical
// classes, one per algorithm) with one config-driven panel used by all 8
// scenes. The bigger change from those three: this one BUILDS its own
// sidebar DOM from the config at construction time, instead of reading
// `document.getElementById('ui-xyz')` against hand-authored HTML. That's
// what prevents the exact bug hit earlier in this project's history — a new
// stat field added to a worker but forgotten in one of two near-duplicate
// HTML files. Now there's exactly one place a new stat gets declared: the
// algorithm's config object.
//
// Episode Status, Perf Diagnostics, and score/survival history charts are
// universal across every algorithm (all 4 now go through the same worker
// harness and share that shape) and are built unconditionally. Each
// algorithm's config only needs to describe what's DIFFERENT about it —
// its own health-metric rows and its own special chart(s).

export interface StatusBadge {
    text: string;
    kind: 'good' | 'warn' | 'bad';
}

export interface BaseMetrics {
    episode: number;
    score: number;
    stepsThisEpisode: number;
    maxSurvivalTime: number;
    evalSurvivalSeconds: number;
    evalMaxSurvivalSeconds: number;
    scoreHistory: number[];
    survivalTimeHistory: number[];
    stepsPerSecond: number;
    lastTrainMs: number;
    avgTrainMs: number;
    maxWorkerFrameGapMs: number;
    mainThreadFrameGapMs: number;
}

export interface StatRow<M> {
    label: string;
    get(m: M): string;
    status?(m: M): StatusBadge | null;
    info?: string; // shown in a hover/focus tooltip next to the label
}

export interface StatGroup<M> {
    heading: string;
    rows: StatRow<M>[];
}

export interface ChartSpec<M> {
    title: string;
    kind: 'line' | 'histogram' | 'histogramWithGaussian' | 'bar';
    get(m: M): number[];
    color?: string;
    info?: string; // shown in a hover/focus tooltip next to the chart title
    // bar chart only
    barLabels?: string[];
    highlightIndex?(m: M): number;
    // histogramWithGaussian only
    gaussianMean?(m: M): number;
    gaussianStd?(m: M): number;
}

export interface AlgorithmUIConfig<M extends BaseMetrics> {
    extraStatGroups: StatGroup<M>[];
    extraCharts: ChartSpec<M>[];
}

export interface MetricsPanelHandlers {
    onRenderToggle(enabled: boolean): void;
    onPreviewToggle(live: boolean): void;
    onSave(): void;
    onLoadFile(json: string): void;
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, className?: string): HTMLElementTagNameMap[K] {
    const e = document.createElement(tag);
    if (className) e.className = className;
    return e;
}

// pure-CSS hover/focus tooltip — no positioning JS, so it's cheap to attach
// to every stat row and chart title without a perf concern
function infoIcon(text: string): HTMLElement {
    const icon = el('span', 'info-icon');
    icon.textContent = 'i';
    icon.tabIndex = 0;
    const tip = el('span', 'info-tooltip');
    tip.textContent = text;
    icon.appendChild(tip);
    return icon;
}

export class MetricsPanel<M extends BaseMetrics> {
    private metrics: M;
    private config: AlgorithmUIConfig<M>;

    private rowBindings: { value: HTMLElement; badge: HTMLElement | null; row: StatRow<M> }[] = [];
    private chartBindings: { ctx: CanvasRenderingContext2D; spec: ChartSpec<M> }[] = [];

    // Fixed (non-configurable) elements every algorithm shares.
    private elEp!: HTMLElement;
    private elScore!: HTMLElement;
    private elSurvival!: HTMLElement;
    private elMaxSurvival!: HTMLElement;
    private elEvalSurvival!: HTMLElement;
    private elEvalMaxSurvival!: HTMLElement;
    private elStepsPerSec!: HTMLElement;
    private elLastTrainMs!: HTMLElement;
    private elAvgTrainMs!: HTMLElement;
    private elWorkerGap!: HTMLElement;
    private elWorkerGapStatus!: HTMLElement;
    private elRenderGap!: HTMLElement;
    private elRenderGapStatus!: HTMLElement;

    constructor(
        container: HTMLElement,
        metrics: M,
        config: AlgorithmUIConfig<M>,
        handlers: MetricsPanelHandlers,
    ) {
        this.metrics = metrics;
        this.config = config;
        this.buildDOM(container, handlers);
    }

    private buildDOM(container: HTMLElement, handlers: MetricsPanelHandlers): void {
        container.appendChild(this.buildControls(handlers));
        container.appendChild(this.buildEpisodeStatus());

        for (const group of this.config.extraStatGroups) {
            container.appendChild(this.buildStatGroup(group));
        }

        container.appendChild(this.buildPerfGroup());

        for (const spec of this.config.extraCharts) {
            container.appendChild(this.buildChart(spec));
        }
        container.appendChild(this.buildChart({
            title: 'Survival History (Seconds)', kind: 'line', color: '#4fc1ff',
            get: m => m.survivalTimeHistory,
            info: 'How long each of the last several training episodes lasted, oldest to newest (left to right). An upward trend means the policy is getting better at not falling.',
        }));
        container.appendChild(this.buildChart({
            title: 'Score History', kind: 'line', color: '#ce9178',
            get: m => m.scoreHistory,
            info: 'Total reward earned in each of the last several training episodes, oldest to newest. Similar to Survival History, but reflects the exact reward signal the agent is optimizing, not just time-alive.',
        }));
    }

    private buildControls(handlers: MetricsPanelHandlers): HTMLElement {
        const group = el('div', 'metric-group');
        group.appendChild(el('h3')).textContent = 'Controls';

        const renderLabel = el('label', 'control-row');
        const renderCheckbox = el('input');
        renderCheckbox.type = 'checkbox';
        renderCheckbox.checked = true;
        renderCheckbox.addEventListener('change', () => handlers.onRenderToggle(renderCheckbox.checked));
        renderLabel.appendChild(renderCheckbox);
        renderLabel.appendChild(document.createTextNode('Render Physics (Uncheck for Max Speed)'));
        group.appendChild(renderLabel);

        const previewLabel = el('label', 'control-row');
        const previewCheckbox = el('input');
        previewCheckbox.type = 'checkbox';
        previewCheckbox.checked = true; // matches workerHarness's default previewMode: 'live'
        previewCheckbox.addEventListener('change', () => handlers.onPreviewToggle(previewCheckbox.checked));
        previewLabel.appendChild(previewCheckbox);
        previewLabel.appendChild(document.createTextNode('Live Demo (greedy, no exploration noise)'));
        group.appendChild(previewLabel);

        const buttonRow = el('div', 'control-buttons');
        const saveButton = el('button');
        saveButton.textContent = 'Save Model';
        saveButton.addEventListener('click', () => handlers.onSave());
        buttonRow.appendChild(saveButton);

        const loadLabel = el('label', 'file-label');
        loadLabel.textContent = 'Load Model';
        const loadInput = el('input');
        loadInput.type = 'file';
        loadInput.accept = 'application/json';
        loadInput.style.display = 'none';
        loadInput.addEventListener('change', () => {
            const file = loadInput.files?.[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = () => handlers.onLoadFile(reader.result as string);
            reader.readAsText(file);
            loadInput.value = '';
        });
        loadLabel.appendChild(loadInput);
        buttonRow.appendChild(loadLabel);
        group.appendChild(buttonRow);

        return group;
    }

    private buildEpisodeStatus(): HTMLElement {
        const group = el('div', 'metric-group');
        group.appendChild(el('h3')).textContent = 'Episode Status';

        this.elEp = this.addRow(group, 'Episode:', 'How many training episodes have run so far. One episode is one attempt, from reset to falling (or timing out).');
        this.elScore = this.addRow(group, 'Score:', 'Cumulative reward earned in the training episode currently in progress. Includes exploration noise, so it undersells how good the policy actually is.');
        this.elSurvival = this.addRow(group, 'Survival:', 'How long, in simulated seconds, the current training episode has lasted.');
        this.elMaxSurvival = this.addRow(group, 'Max Survival:', 'The longest any training episode has lasted so far.');
        this.elEvalSurvival = this.addRow(group, 'Live Demo Survival:', 'How long the noise-free Live Demo run (a separate environment, driven by the policy\'s best guess with no randomness) has survived.');
        this.elEvalMaxSurvival = this.addRow(group, 'Live Demo Best:', 'The longest the noise-free Live Demo has ever survived — the most honest single number for "how good is this policy right now."');

        return group;
    }

    private buildPerfGroup(): HTMLElement {
        const group = el('div', 'metric-group');
        group.appendChild(el('h3')).textContent = 'Performance';

        this.elStepsPerSec = this.addRow(group, 'Steps/sec:', 'Physics steps the training loop is getting through per second — raw throughput, not learning quality.');
        this.elLastTrainMs = this.addRow(group, 'Last Train Time:', 'Wall-clock time the most recent learning update took.');
        this.elAvgTrainMs = this.addRow(group, 'Avg Train Time:', 'Rolling average of how long each learning update takes.');
        const [workerVal, workerBadge] = this.addRowWithBadge(group, 'Worker Max Stall (1s):', 'Longest gap between the training thread\'s own internal heartbeat over the last second. High values mean the training thread was busy long enough to freeze its own timers — a sign training is CPU-heavy, not that anything is broken.');
        this.elWorkerGap = workerVal;
        this.elWorkerGapStatus = workerBadge;
        const [renderVal, renderBadge] = this.addRowWithBadge(group, 'Render Max Stall (1s):', 'Longest gap between rendered frames on this page over the last second. Since training runs on a separate thread, this should stay smooth (~16ms) regardless of how hard the agent is training.');
        this.elRenderGap = renderVal;
        this.elRenderGapStatus = renderBadge;

        return group;
    }

    private buildStatGroup(group: StatGroup<M>): HTMLElement {
        const groupEl = el('div', 'metric-group');
        groupEl.appendChild(el('h3')).textContent = group.heading;

        for (const row of group.rows) {
            if (row.status) {
                const [value, badge] = this.addRowWithBadge(groupEl, row.label + ':', row.info);
                this.rowBindings.push({ value, badge, row });
            } else {
                const value = this.addRow(groupEl, row.label + ':', row.info);
                this.rowBindings.push({ value, badge: null, row });
            }
        }

        return groupEl;
    }

    private addRow(group: HTMLElement, label: string, info?: string): HTMLElement {
        const row = el('div', 'metric-row');
        const labelWrap = el('span', 'metric-label');
        labelWrap.appendChild(document.createTextNode(label));
        if (info) labelWrap.appendChild(infoIcon(info));
        row.appendChild(labelWrap);
        const value = el('span', 'metric-value');
        value.textContent = '0';
        row.appendChild(value);
        group.appendChild(row);
        return value;
    }

    private addRowWithBadge(group: HTMLElement, label: string, info?: string): [HTMLElement, HTMLElement] {
        const row = el('div', 'metric-row');
        const labelWrap = el('span', 'metric-label');
        labelWrap.appendChild(document.createTextNode(label));
        if (info) labelWrap.appendChild(infoIcon(info));
        row.appendChild(labelWrap);
        const wrap = el('div');
        const value = el('span', 'metric-value');
        value.textContent = '0';
        const badge = el('span', 'status-badge');
        wrap.appendChild(value);
        wrap.appendChild(badge);
        row.appendChild(wrap);
        group.appendChild(row);
        return [value, badge];
    }

    private buildChart(spec: ChartSpec<M>): HTMLElement {
        const container = el('div', 'chart-container');
        const heading = el('h3');
        heading.appendChild(document.createTextNode(spec.title));
        if (spec.info) heading.appendChild(infoIcon(spec.info));
        container.appendChild(heading);
        const canvas = el('canvas', 'sidebar-canvas');
        canvas.width = 298;
        canvas.height = 100;
        container.appendChild(canvas);
        const ctx = canvas.getContext('2d')!;
        this.chartBindings.push({ ctx, spec });
        return container;
    }

    private setStatus(badge: HTMLElement, status: StatusBadge | null): void {
        if (!status) {
            badge.textContent = '';
            badge.className = 'status-badge';
            return;
        }
        badge.textContent = status.text;
        badge.className = `status-badge status-${status.kind}`;
    }

    public update(): void {
        const m = this.metrics;

        this.elEp.textContent = m.episode.toString();
        this.elScore.textContent = m.score.toFixed(1);
        this.elSurvival.textContent = (m.stepsThisEpisode * 0.016).toFixed(2) + 's';
        this.elMaxSurvival.textContent = m.maxSurvivalTime.toFixed(2) + 's';
        this.elEvalSurvival.textContent = m.evalSurvivalSeconds.toFixed(2) + 's';
        this.elEvalMaxSurvival.textContent = m.evalMaxSurvivalSeconds.toFixed(2) + 's';

        this.elStepsPerSec.textContent = Math.round(m.stepsPerSecond).toLocaleString();
        this.elLastTrainMs.textContent = m.lastTrainMs.toFixed(0) + 'ms';
        this.elAvgTrainMs.textContent = m.avgTrainMs.toFixed(0) + 'ms';

        const workerGap = m.maxWorkerFrameGapMs;
        this.elWorkerGap.textContent = workerGap.toFixed(0) + 'ms';
        this.setStatus(this.elWorkerGapStatus,
            workerGap < 100 ? { text: 'SMOOTH', kind: 'good' } :
                workerGap < 300 ? { text: 'HITCHING', kind: 'warn' } :
                    { text: 'STALLED', kind: 'bad' });

        const renderGap = m.mainThreadFrameGapMs;
        this.elRenderGap.textContent = renderGap.toFixed(0) + 'ms';
        this.setStatus(this.elRenderGapStatus,
            renderGap < 50 ? { text: 'SMOOTH', kind: 'good' } :
                renderGap < 150 ? { text: 'HITCHING', kind: 'warn' } :
                    { text: 'STALLED', kind: 'bad' });

        for (const { value, badge, row } of this.rowBindings) {
            value.textContent = row.get(m);
            if (row.status && badge) this.setStatus(badge, row.status(m));
        }

        for (const { ctx, spec } of this.chartBindings) {
            this.drawChart(ctx, spec);
        }
    }

    private drawChart(ctx: CanvasRenderingContext2D, spec: ChartSpec<M>): void {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        ctx.clearRect(0, 0, width, height);

        switch (spec.kind) {
            case 'line':
                this.drawLineChart(ctx, spec.get(this.metrics), spec.color ?? '#ce9178');
                break;
            case 'histogram':
                this.drawHistogram(ctx, spec.get(this.metrics), spec.color ?? '#c586c0');
                break;
            case 'histogramWithGaussian':
                this.drawHistogram(ctx, spec.get(this.metrics), spec.color ?? '#c586c0');
                this.drawGaussianOverlay(ctx, spec.gaussianMean!(this.metrics), spec.gaussianStd!(this.metrics));
                break;
            case 'bar':
                this.drawBarChart(ctx, spec.get(this.metrics), spec.barLabels, spec.highlightIndex?.(this.metrics));
                break;
        }
    }

    private drawLineChart(ctx: CanvasRenderingContext2D, data: number[], color: string): void {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
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
            if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }
        ctx.stroke();

        ctx.fillStyle = '#888';
        ctx.font = '10px Courier New';
        ctx.fillText(maxVal.toFixed(1), 5, 12);
        ctx.fillText(minVal.toFixed(1), 5, height - 5);
    }

    private drawHistogram(ctx: CanvasRenderingContext2D, history: number[], color: string): void {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
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

        ctx.fillStyle = color;
        for (let i = 0; i < numBins; i++) {
            const barHeight = (bins[i] / maxBin) * height;
            ctx.fillRect(i * barWidth, height - barHeight, barWidth - 1, barHeight);
        }

        ctx.fillStyle = '#888';
        ctx.font = '10px Courier New';
        ctx.fillText('-1.0', 2, 10);
        ctx.fillText('0.0', width / 2 - 10, 10);
        ctx.fillText('+1.0', width - 25, 10);
    }

    private drawGaussianOverlay(ctx: CanvasRenderingContext2D, mean: number, stdRaw: number): void {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        const std = Math.max(stdRaw, 1e-4);

        ctx.beginPath();
        ctx.strokeStyle = '#4fc1ff';
        ctx.lineWidth = 1.5;
        for (let px = 0; px <= width; px++) {
            const x = (px / width) * 2 - 1;
            const normalized = Math.exp(-((x - mean) ** 2) / (2 * std * std));
            const y = height - normalized * height;
            if (px === 0) ctx.moveTo(px, y); else ctx.lineTo(px, y);
        }
        ctx.stroke();
    }

    private drawBarChart(ctx: CanvasRenderingContext2D, values: number[], labels?: string[], highlightIndex?: number): void {
        const width = ctx.canvas.width;
        const height = ctx.canvas.height;
        if (values.length === 0) return;

        const maxV = Math.max(...values, 1);
        const minV = Math.min(...values, 0);
        const range = maxV - minV || 1;

        const barAreaW = width - 10;
        const barW = barAreaW / values.length - 4;
        const baseY = height - 16;
        const maxBarH = height - 30;

        for (let i = 0; i < values.length; i++) {
            const x = 5 + i * (barW + 4);
            const h = ((values[i] - minV) / range) * maxBarH;
            ctx.fillStyle = i === highlightIndex ? '#facc15' : '#3b82f6';
            ctx.fillRect(x, baseY - h, barW, h);

            if (labels && labels[i] !== undefined) {
                ctx.fillStyle = '#9ca3af';
                ctx.font = '8px Courier New';
                ctx.fillText(labels[i], x, baseY + 12);
            }
        }
    }
}
