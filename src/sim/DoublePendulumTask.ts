import { Environment } from '../engine/Environment';
import { PointMass } from '../state/PointMass';
import { AxisConstraint, BoundaryConstraint, DistanceConstraint } from '../engine/Constraint';
import { Actuator } from '../engine/Actuator';
import { clamp } from '../lib/mathUtils';
import type { StepResult, Task } from './Task';



export class DoublePendulumTask implements Task {
    public readonly env: Environment;
    public readonly cart: PointMass;
    public readonly pole1: PointMass;
    public readonly pole2: PointMass;
    public readonly motor: Actuator;

    private readonly trackHeight: number;
    private readonly fixedDt: number;
    private readonly centerX: number;
    private readonly rightEdge: number;
    private readonly leftEdge = 50;

    constructor(
        canvasWidth: number,
        canvasHeight: number,
        trackHeight: number,
        fixedDt: number,
    ) {
        this.trackHeight = trackHeight;
        this.fixedDt = fixedDt;
        this.centerX = canvasWidth / 2;
        this.rightEdge = canvasWidth - 50;

        this.env = new Environment(9.81);
        this.cart = new PointMass(this.centerX, trackHeight, 10, false);
        this.pole1 = new PointMass(this.centerX, trackHeight - 100, 2, false);
        this.pole2 = new PointMass(this.centerX, trackHeight - 200, 2, false);

        const link1 = new DistanceConstraint(this.cart, this.pole1, 100);
        const link2 = new DistanceConstraint(this.pole1, this.pole2, 100);
        const track = new AxisConstraint(this.cart, trackHeight);
        const screenBounds = new BoundaryConstraint(this.cart, canvasWidth, canvasHeight, 0);

        this.env.addPoint(this.cart);
        this.env.addPoint(this.pole1);
        this.env.addPoint(this.pole2);
        this.env.addConstraint(link1);
        this.env.addConstraint(link2);
        this.env.addConstraint(track);
        this.env.addConstraint(screenBounds);

        this.motor = new Actuator(this.cart, 1500);
    }

    public reset(): number[] {
        this.cart.position.x = this.centerX;
        this.cart.position.y = this.trackHeight;
        this.cart.oldPosition.x = this.centerX;
        this.cart.oldPosition.y = this.trackHeight;

        // Wobble both poles
        const wobble1 = (Math.random() - 0.5) * 0.3;
        const wobble2 = (Math.random() - 0.5) * 0.3;

        this.pole1.position.x = this.centerX + Math.sin(wobble1) * 100;
        this.pole1.position.y = this.trackHeight - Math.cos(wobble1) * 100;
        this.pole1.oldPosition.x = this.pole1.position.x;
        this.pole1.oldPosition.y = this.pole1.position.y;

        this.pole2.position.x = this.pole1.position.x + Math.sin(wobble1 + wobble2) * 100;
        this.pole2.position.y = this.pole1.position.y - Math.cos(wobble1 + wobble2) * 100;
        this.pole2.oldPosition.x = this.pole2.position.x;
        this.pole2.oldPosition.y = this.pole2.position.y;

        this.motor.activeDirection = 0;
        return this.senseAndNormalize();
    }

    public step(thrustFraction: number): StepResult {
        this.motor.activeDirection = thrustFraction;
        this.motor.apply();
        this.env.update(this.fixedDt);

        // 1. Calculate the TRUE raw physics values first
        const dx1 = this.pole1.position.x - this.cart.position.x;
        const dy1 = this.cart.position.y - this.pole1.position.y;
        const rawA1 = Math.atan2(dx1, dy1);

        const dx2 = this.pole2.position.x - this.pole1.position.x;
        const dy2 = this.pole1.position.y - this.pole2.position.y;
        const rawA2 = Math.atan2(dx2, dy2);

        const rawX = this.cart.position.x;

        // 2. Generate the normalized state for the Neural Network
        const nextState = this.senseAndNormalize();

        // 3. Judge the true physics state
        let done = false;
        if (isNaN(rawX) || isNaN(rawA1) || isNaN(rawA2)) {
            done = true; // Physics glitch failsafe
        } else {
            done = Math.abs(rawA1) > 0.8 || Math.abs(rawA2) > 0.8 || rawX < this.leftEdge || rawX > this.rightEdge;
        }

        // 4. THE NEW REWARD FUNCTION
        const centerPenalty = Math.abs(rawX - this.centerX) / this.centerX;
        const energyPenalty = 0.02 * Math.abs(thrustFraction);

        const reward = done
            ? -20
            : 1.0 - (0.5 * Math.abs(rawA1)) - (0.5 * Math.abs(rawA2)) - (0.5 * centerPenalty) - energyPenalty;

        return { nextState, reward, done };
    }

    private senseAndNormalize(): number[] {
        const dx1 = this.pole1.position.x - this.cart.position.x;
        const dy1 = this.cart.position.y - this.pole1.position.y;
        const dx2 = this.pole2.position.x - this.pole1.position.x;
        const dy2 = this.pole1.position.y - this.pole2.position.y;

        const a1 = Math.atan2(dx1, dy1);
        const a2 = Math.atan2(dx2, dy2);

        const oldDx1 = this.pole1.oldPosition.x - this.cart.oldPosition.x;
        const oldDy1 = this.cart.oldPosition.y - this.pole1.oldPosition.y;
        const oldDx2 = this.pole2.oldPosition.x - this.pole1.oldPosition.x;
        const oldDy2 = this.pole1.oldPosition.y - this.pole2.oldPosition.y;

        const oldA1 = Math.atan2(oldDx1, oldDy1);
        const oldA2 = Math.atan2(oldDx2, oldDy2);

        let da1 = a1 - oldA1;
        if (da1 > Math.PI) da1 -= 2 * Math.PI;
        if (da1 < -Math.PI) da1 += 2 * Math.PI;
        const v1 = da1 / this.fixedDt;

        let da2 = a2 - oldA2;
        if (da2 > Math.PI) da2 -= 2 * Math.PI;
        if (da2 < -Math.PI) da2 += 2 * Math.PI;
        const v2 = da2 / this.fixedDt;

        const cartV = (this.cart.position.x - this.cart.oldPosition.x) / this.fixedDt;

        return [
            clamp((this.cart.position.x - this.centerX) / this.centerX, -1, 1),
            clamp(cartV / 500.0, -1, 1),
            clamp(a1 / 1.0, -1, 1),
            clamp(a2 / 1.0, -1, 1),
            clamp(v1 / 10.0, -1, 1),
            clamp(v2 / 10.0, -1, 1)
        ];
    }
}