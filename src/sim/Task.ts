export interface StepResult {
    nextState: number[];
    reward: number;
    done: boolean;
}

/**
 * The universal contract for any RL environment. 
 * The Trainer only needs these two methods to train the agent.
 */
export interface Task {
    reset(): number[];
    step(actionValue: number): StepResult;
}