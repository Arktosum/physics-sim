import { Vector } from './Vector';
import { PointMass } from '../state/PointMass';
import { type Constraint } from './Constraint'; // Import our new constraint class


export class Environment {
    public points: PointMass[] = [];
    public constraints: Constraint[] = []; // New array to store rules
    public gravity: Vector;

    constructor(gravityY: number = 9.8) {
        // Gravity acts downwards, so it's a positive Y force in screen coordinates
        this.gravity = new Vector(0, gravityY * 100); // Scale by 100 to make pixels look like meters
    }

    // Add a point mass to our simulation world
    addPoint(point: PointMass) {
        this.points.push(point);
    }
    // New method to add structural rules to our world
    addConstraint(constraint: Constraint) {
        this.constraints.push(constraint);
    }
    // This updates the entire world state by one time-step
    update(dt: number) {
        // Prevent massive time jumps (like if you switch browser tabs) from breaking the math
        if (dt > 0.1) dt = 0.1;

        // Step A: Apply global forces to all objects
        for (const point of this.points) {
            // Force = mass * acceleration. We multiply gravity by mass so that
            // heavy and light objects fall at the exact same rate under gravity.
            const gravityForce = this.gravity.mult(point.mass);
            point.applyForce(gravityForce);
        }

        // Step B: Run the Verlet integration step to move positions based on forces
        for (const point of this.points) {
            point.update(dt);
        }

        // Step C: Constrain the positions (The Relaxation Loop)
        // We run this 5 times per frame to ensure structural rigidity
        const iterations = 50;
        for (let i = 0; i < iterations; i++) {
            for (const constraint of this.constraints) {
                constraint.resolve();
            }
        }

    }
}