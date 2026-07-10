export class CartPoleEnv {
    // Physical Constants of the Universe
    private gravity: number = 9.8;
    private masscart: number = 1.0;
    private masspole: number = 0.1;
    private totalMass: number = 1.1; // masscart + masspole
    private length: number = 0.5; // half the pole's length
    private polemassLength: number = 0.05; // masspole * length
    private forceMag: number = 10.0;
    private tau: number = 0.02; // seconds between state updates

    // The Wolves (Fail conditions)
    private thetaThresholdRadians: number = (24 * 2 * Math.PI) / 360; // 12 degrees
    private xThreshold: number = 2.4;

    public state: [number, number, number, number];

    constructor() {
        this.state = [0, 0, 0, 0];
    }

    /**
     * Resets the universe to a random, slightly wobbly starting position.
     * Returns: [CartX, CartVelocityX, PoleAngle, PoleVelocityAngle]
     */
    public reset(): [number, number, number, number] {
        this.state = [0, 0, (Math.random() - 0.5) * 0.1, 0];
        return this.state;
    }

    /**
     * Applies a force (Action 0 = Left, Action 1 = Right) and steps time forward.
     * Returns the exact consequences of that action.
     */
    public step(action: number): { nextState: [number, number, number, number], reward: number, done: boolean } {
        let [x, x_dot, theta, theta_dot] = this.state;

        // 0 = Push Left (-10), 1 = Push Right (+10)
        let force = action === 1 ? this.forceMag : -this.forceMag;

        // Kinematics math (how gravity and force affect the pendulum)
        let costheta = Math.cos(theta);
        let sintheta = Math.sin(theta);
        let temp = (force + this.polemassLength * theta_dot * theta_dot * sintheta) / this.totalMass;
        let thetaacc = (this.gravity * sintheta - costheta * temp) / (this.length * (4.0 / 3.0 - (this.masspole * costheta * costheta) / this.totalMass));
        let xacc = temp - (this.polemassLength * thetaacc * costheta) / this.totalMass;

        // Update state based on acceleration and time step (tau)
        x = x + this.tau * x_dot;
        x_dot = x_dot + this.tau * xacc;
        theta = theta + this.tau * theta_dot;
        theta_dot = theta_dot + this.tau * thetaacc;

        this.state = [x, x_dot, theta, theta_dot];

        // JUDGE: Did the cart fall off the rails or did the pole fall over?
        let done = x < -this.xThreshold ||
            x > this.xThreshold ||
            theta < -this.thetaThresholdRadians ||
            theta > this.thetaThresholdRadians;

        // Hand out the Apples (+1) and the Wolves (-100)
        let reward = done ? -100 : 1;

        return { nextState: this.state, reward, done };
    }
}