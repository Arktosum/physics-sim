import { Environment } from '../engine/Environment';

/**
 * The subset of Environment's shape CanvasRenderer actually reads.
 * Environment satisfies this structurally already. This exists so a plain
 * data snapshot posted from a training Worker (points + constraints, no
 * class instances, no methods) can be rendered too — postMessage's
 * structured-clone can't send class instances with their prototypes intact
 * anyway, so the render path was always going to need to work off plain data
 * once training moved off the main thread.
 */
export interface RenderablePoint {
    position: { x: number; y: number };
    mass: number;
    isPinned: boolean;
}
export interface RenderableConstraint {
    p1?: { position: { x: number; y: number } };
    p2?: { position: { x: number; y: number } };
    lockedY?: number;
}
export interface RenderableEnvironment {
    points: RenderablePoint[];
    constraints: RenderableConstraint[];
}

export class CanvasRenderer {
    private ctx: CanvasRenderingContext2D;
    private canvas: HTMLCanvasElement;

    constructor(canvas: HTMLCanvasElement) {
        this.canvas = canvas;
        this.ctx = canvas.getContext('2d')!;
    }

    render(env: RenderableEnvironment | Environment) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // 1. Draw Constraints (The Bones/Strings) FIRST
        this.ctx.strokeStyle = '#555577'; // A sleek, visible slate-blue
        this.ctx.lineWidth = 2;

        for (const constraint of env.constraints) {
            // Check if it's a structural bone (string/stick)
            if ('p1' in constraint && 'p2' in constraint) {
                const c = constraint as any; // Quick cast
                this.ctx.strokeStyle = '#555577';
                this.ctx.beginPath();
                this.ctx.moveTo(c.p1.position.x, c.p1.position.y);
                this.ctx.lineTo(c.p2.position.x, c.p2.position.y);
                this.ctx.stroke();
            } 
            // Check if it's our new Axis Track
            else if ('lockedY' in constraint) {
                const c = constraint as any;
                this.ctx.strokeStyle = '#334433'; // Faint green laser line for the track
                this.ctx.beginPath();
                this.ctx.moveTo(0, c.lockedY);
                this.ctx.lineTo(this.canvas.width, c.lockedY);
                this.ctx.stroke();
            }
        }

        // 2. Draw Point Masses (The Joints/Weights) SECOND
        for (const point of env.points) {
            this.ctx.beginPath();

            // Pinned points are neon blue, free points are neon pink
            this.ctx.fillStyle = point.isPinned ? '#38bdf8' : '#ff0055';

            const visualRadius = Math.max(5, Math.min(30, point.mass * 5));

            this.ctx.arc(point.position.x, point.position.y, visualRadius, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }
}