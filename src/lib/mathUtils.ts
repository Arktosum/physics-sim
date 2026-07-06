export function clamp(val: number, min: number, max: number): number {
    return isNaN(val) ? 0 : Math.max(min, Math.min(max, val));
}