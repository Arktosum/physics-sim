import type { Transition } from "../engine/ReinforceAgent";

/**
 * REINFORCE's memory, structurally opposite to ReplayBuffer:
 *   - ReplayBuffer: huge, capped, sampled randomly, kept across episodes.
 *   - EpisodeBuffer: one episode's worth, used in ORDER, thrown away every time.
 *
 * This is what makes REINFORCE "on-policy" in code, not just in theory —
 * there is no mechanism here to hold onto old data at all.
 */
export class EpisodeBuffer {
    private transitions: Transition[] = [];

    public add(transition: Transition): void {
        this.transitions.push(transition);
    }

    public getEpisode(): Transition[] {
        return this.transitions;
    }

    public get length(): number {
        return this.transitions.length;
    }

    /** Called immediately after learn() consumes the episode. */
    public clear(): void {
        this.transitions = [];
    }
}