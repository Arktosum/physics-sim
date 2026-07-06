import { Environment } from '../engine/Environment';
import { PointMass } from '../state/PointMass';
import { AxisConstraint, BoundaryConstraint, DistanceConstraint } from '../engine/Constraint';
import { Actuator } from '../engine/Actuator';
import { clamp } from '../lib/mathUtils'; // Assuming you have a mathUtils file for clamp
import type { StepResult, Task } from './Task';


/**
 * Wraps the from-scratch Verlet/PBD physics engine into a small Gym-style
 * task: reset() -> initial state, step(thrustFraction) -> { nextState, reward, done }.
 *
 * Everything specific to *this* task (state normalization, fail condition,
 * reward shaping) lives here. main.ts and Trainer never touch PointMass,
 * Constraint, or Actuator directly.
 */
export class CartPoleTask implements Task {
    public readonly env: Environment;
    public readonly cart: PointMass;
    public readonly pole: PointMass;
    public readonly motor: Actuator;

    private readonly trackHeight: number;
    private readonly fixedDt: number;
    private readonly energyPenaltyWeight: number;

    private readonly centerX: number;
    private readonly rightEdge: number;
    private readonly leftEdge = 50;

    constructor(
        canvasWidth: number,
        canvasHeight: number,
        trackHeight: number,
        fixedDt: number,
        energyPenaltyWeight: number,
    ) {
        this.trackHeight = trackHeight;
        this.fixedDt = fixedDt;
        this.energyPenaltyWeight = energyPenaltyWeight;

        this.centerX = canvasWidth / 2;
        this.rightEdge = canvasWidth - 50;

        this.env = new Environment(9.81);
        this.cart = new PointMass(this.centerX, trackHeight, 10, false);
        this.pole = new PointMass(this.centerX, trackHeight - 100, 2, false);

        const link = new DistanceConstraint(this.cart, this.pole, 100);
        const track = new AxisConstraint(this.cart, trackHeight);
        const screenBounds = new BoundaryConstraint(this.cart, canvasWidth, canvasHeight, 0);

        this.env.addPoint(this.cart);
        this.env.addPoint(this.pole);
        this.env.addConstraint(link);
        this.env.addConstraint(track);
        this.env.addConstraint(screenBounds);

        this.motor = new Actuator(this.cart, 1500);
    }

    /** Resets cart/pole to center with a small random wobble. Returns the normalized state. */
    public reset(): number[] {
        this.cart.position.x = this.centerX;
        this.cart.position.y = this.trackHeight;
        this.cart.oldPosition.x = this.centerX;
        this.cart.oldPosition.y = this.trackHeight;

        const wobble = (Math.random() - 0.5) * 0.2;
        this.pole.position.x = this.centerX + Math.sin(wobble) * 100;
        this.pole.position.y = this.trackHeight - Math.cos(wobble) * 100;
        this.pole.oldPosition.x = this.pole.position.x;
        this.pole.oldPosition.y = this.pole.position.y;

        this.motor.activeDirection = 0;

        return this.senseAndNormalize();
    }

    /** Applies the chosen thrust, steps physics one tick, and reports the consequences. */
    public step(thrustFraction: number): StepResult {
        this.motor.activeDirection = thrustFraction;
        this.motor.apply();
        this.env.update(this.fixedDt);

        const nextState = this.senseAndNormalize();
        const rawAngle = nextState[2];

        const done =
            Math.abs(rawAngle) > 0.8 ||
            this.cart.position.x < this.leftEdge ||
            this.cart.position.x > this.rightEdge;

        // THE FIX: Shaped Reward. 
        // Math.cos(0) is 1.0 (perfectly upright). 
        // Math.cos(0.8) is ~0.69 (about to fall over). 
        // This gives the network a smooth gradient to climb toward perfectly vertical.
        const uprightBonus = Math.cos(rawAngle);
        const reward = done ? -10 : uprightBonus - (this.energyPenaltyWeight * Math.abs(thrustFraction));

        return { nextState, reward, done };
    }

    private senseAndNormalize(): number[] {
        const dx = this.pole.position.x - this.cart.position.x;
        const dy = this.cart.position.y - this.pole.position.y;
        const angle = Math.atan2(dx, dy);

        const oldDx = this.pole.oldPosition.x - this.cart.oldPosition.x;
        const oldDy = this.cart.oldPosition.y - this.pole.oldPosition.y;
        const oldAngle = Math.atan2(oldDx, oldDy);

        let deltaAngle = angle - oldAngle;
        if (deltaAngle > Math.PI) deltaAngle -= 2 * Math.PI;
        if (deltaAngle < -Math.PI) deltaAngle += 2 * Math.PI;

        const angularVelocity = deltaAngle / this.fixedDt;
        const cartVelocity = (this.cart.position.x - this.cart.oldPosition.x) / this.fixedDt;

        return [
            clamp((this.cart.position.x - this.centerX) / this.centerX, -1, 1),
            clamp(cartVelocity / 500.0, -1, 1),
            clamp(angle / 1.0, -1, 1),
            clamp(angularVelocity / 10.0, -1, 1),
        ];
    }
}