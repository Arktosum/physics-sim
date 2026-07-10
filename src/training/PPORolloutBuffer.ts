// src/training/PPORolloutBuffer.ts

export interface PPOTransition {
    state: number[];
    action: number;       // The raw, unclamped action taken
    reward: number;
    value: number;        // The Critic's baseline prediction at this exact moment
    logProb: number;      // The Actor's mathematical probability of taking this action
    done: boolean;        // Did the pendulum fall on this step?
}

export class PPORolloutBuffer {
    public transitions: PPOTransition[] = [];

    public add(transition: PPOTransition) {
        this.transitions.push(transition);
    }

    public get(): PPOTransition[] {
        return this.transitions;
    }

    public clear() {
        this.transitions = [];
    }

    public get length(): number {
        return this.transitions.length;
    }
}