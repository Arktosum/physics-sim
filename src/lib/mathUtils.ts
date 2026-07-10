export function clamp(val: number, min: number, max: number): number {
    return isNaN(val) ? 0 : Math.max(min, Math.min(max, val));
}

/**
 * Draws a single sample from a Standard Normal distribution (mean 0, std 1)
 * using the Box-Muller transform. Used by the REINFORCE/PPO Actor to turn
 * a (mean, std) pair into an actual continuous action.
 */
export function sampleStandardNormal(): number {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}