import numpy as np
import matplotlib.pyplot as plt

# Set seed for reproducible noise in the experiment
np.random.seed(42)

# ================================================================
# 1. Base Neuron Definitions
# ================================================================

class LIFNeuron:
    """
    Classic Leaky Integrate-and-Fire (LIF) neuron.
    Membrane potential decays exponentially (leak) and integrates
    incoming input. A spike fires when potential exceeds the threshold,
    then the membrane is reset.
    """
    def __init__(self, lambda_leak=0.5, threshold=2.0):
        """
        Parameters
        ----------
        lambda_leak : float
            Leak factor (0 < lambda_leak < 1). Controls membrane time constant.
        threshold : float
            Spike threshold (higher for LIF to reduce noise sensitivity).
        """
        self.lambda_leak = lambda_leak
        self.threshold = threshold
        self.h = 0.0      # membrane potential

    def forward(self, current_input):
        """
        One time step of LIF dynamics.

        Parameters
        ----------
        current_input : float
            Instantaneous sensory (light) intensity.

        Returns
        -------
        spike : int
            1 if spike emitted, 0 otherwise.
        """
        # Leaky integration
        self.h = self.lambda_leak * self.h + current_input
        # Spike generation and reset
        if self.h > self.threshold:
            spike = 1
            self.h = 0.0  # reset after firing
        else:
            spike = 0
        return spike


# ================================================================
# 2. Robust TAN Neuron with Noise Tolerance
# ================================================================

class TANNeuronRobust:
    """
    Temporal Attention Neuron with noise‑tolerant surprise gating.
    In addition to rectified surprise, a dead zone (noise_tolerance)
    is introduced: only prediction errors larger than the expected
    noise level are considered as true sensory gradients.
    """
    def __init__(self, window_size=5, lambda_leak=0.5, threshold=0.5,
                 noise_tolerance=0.5):
        """
        Parameters
        ----------
        window_size : int
            Length of the local temporal receptive window.
        lambda_leak : float
            Membrane leak factor.
        threshold : float
            Spike threshold.
        noise_tolerance : float
            Sensory dead zone. Surprise values below this magnitude are
            suppressed to zero, filtering out environmental noise.
        """
        self.window_size = window_size
        self.lambda_leak = lambda_leak
        self.threshold = threshold
        self.noise_tolerance = noise_tolerance
        self.h = 0.0

        # Synaptic projection weights (scalars for simplicity)
        self.W_q = np.array([2.0])   # amplifies surprise
        self.W_k = np.array([1.0])   # historical features
        self.W_v = np.array([1.0])   # historical energy

        # Initialize history buffer (no prior stimulation)
        self.history = np.zeros(window_size)
        self.beta = 1.0              # inverse temperature for attention

    def softmax(self, x):
        """
        Numerically stable softmax (Boltzmann distribution).

        Parameters
        ----------
        x : np.ndarray
            Unnormalised attention scores.

        Returns
        -------
        np.ndarray
            Normalised attention weights (probability distribution).
        """
        e_x = np.exp(self.beta * (x - np.max(x)))
        return e_x / (e_x.sum(axis=0) + 1e-9)

    def forward(self, current_input):
        """
        One time step of the robust TAN neuron.

        Steps:
        1. Update history window and compute prediction error.
        2. Rectify surprise (only positive changes matter) and subtract
           the noise tolerance, creating a sensory dead zone.
        3. Project Q, K, V and compute attention weights.
        4. Integrate context and apply gating.
        5. Update membrane potential and emit spike if threshold crossed.

        Parameters
        ----------
        current_input : float
            Instantaneous sensory (light) intensity.

        Returns
        -------
        spike : int
            1 if spike emitted, 0 otherwise.
        """
        # 1. Update sliding window
        self.history = np.roll(self.history, -1)
        self.history[-1] = current_input

        # 2. Compute prediction error (surprise)
        S_t = current_input - np.mean(self.history)

        # ---- Core innovation: Sensory dead zone ----
        # Only surprise values above the noise tolerance are considered as
        # genuine intensity gradients. This prevents the neuron from
        # responding to random fluctuations.
        S_t = np.maximum(0, S_t - self.noise_tolerance)

        # 3. Temporal attention projections
        Q = S_t * self.W_q             # Query driven by gated surprise
        K = self.history * self.W_k    # Keys: historical patterns
        V = self.history * self.W_v    # Values: historical energy

        # Attention scores and normalisation
        attention_scores = Q * K
        attention_weights = self.softmax(attention_scores)

        # 4. Context integration and gating
        C_t = np.sum(attention_weights * V)    # base context
        gating = np.tanh(S_t)                  # smooth gate (S_t already non‑negative)
        A_t = gating * C_t                     # effective synaptic current

        # 5. Membrane update and spike generation
        self.h = self.lambda_leak * self.h + A_t
        if self.h > self.threshold:
            spike = 1
            self.h = 0.0                       # reset membrane
        else:
            spike = 0
        return spike


# ================================================================
# 3. Environment with Noise
# ================================================================

def get_clean_light_intensity(x):
    """
    Clean (noise-free) light intensity distribution.
    A Gaussian peak centred at x = 80 models a light source.

    Parameters
    ----------
    x : float or np.ndarray
        Position(s) on the 1D track.

    Returns
    -------
    intensity : float or np.ndarray
        Light intensity at the given position(s).
    """
    light_pos = 80.0                      # centre of the light source
    sigma = 15.0                          # width of the beam
    max_intensity = 10.0                  # peak intensity
    return max_intensity * np.exp(-((x - light_pos)**2) / (2 * sigma**2))


def get_noisy_light_intensity(x, noise_level=0.5):
    """
    Light intensity with additive Gaussian noise.
    The noise simulates an unpredictable natural environment.

    Parameters
    ----------
    x : float
        Position on the track.
    noise_level : float
        Standard deviation of the Gaussian noise added to the clean signal.

    Returns
    -------
    intensity : float
        Noisy light intensity (clipped to non‑negative values).
    """
    clean = get_clean_light_intensity(x)
    noisy = clean + np.random.normal(0, noise_level)
    return max(0.0, noisy)  # ensure non‑negative light intensity


# ================================================================
# 4. Robust Agent (Bioton)
# ================================================================

class BiotonRobust:
    """
    Simple agent with a robust neuron brain (TAN or LIF) moving in a 1D world.
    """
    def __init__(self, brain_type='TAN', start_pos=50.0):
        """
        Parameters
        ----------
        brain_type : str
            'TAN' or 'LIF' – neuron type to use as brain.
        start_pos : float
            Initial position on the track.
        """
        if brain_type == 'TAN':
            # Use a noise tolerance slightly above the environment's noise level
            self.brain = TANNeuronRobust(noise_tolerance=0.6)
        else:
            # LIF threshold is also raised to improve noise immunity
            self.brain = LIFNeuron(threshold=2.5)

        self.pos = start_pos
        self.velocity = 0.0

        # Logs for analysis and plotting
        self.pos_log = []
        self.spike_log = []

    def step(self, light_intensity):
        """
        Advance the agent by one time step.

        Parameters
        ----------
        light_intensity : float
            Perceived (noisy) light intensity at the current position.
        """
        spike = self.brain.forward(light_intensity)

        # Simple locomotion model
        friction = 0.6       # strong fluid drag
        boost = 0.8          # thrust per spike
        self.velocity = friction * self.velocity + boost * spike
        self.pos += self.velocity

        self.pos_log.append(self.pos)
        self.spike_log.append(spike)


# ================================================================
# 5. Run the Noisy Competition
# ================================================================
NOISE_LEVEL = 0.5
time_steps = 250
agent_lif = BiotonRobust(brain_type='LIF', start_pos=50.0)
agent_tan = BiotonRobust(brain_type='TAN', start_pos=50.0)

for t in range(time_steps):
    # Both agents perceive the same noisy light field
    intensity_lif = get_noisy_light_intensity(agent_lif.pos, noise_level=NOISE_LEVEL)
    intensity_tan = get_noisy_light_intensity(agent_tan.pos, noise_level=NOISE_LEVEL)
    agent_lif.step(intensity_lif)
    agent_tan.step(intensity_tan)


# ================================================================
# 6. Publication‑Quality Visualization
# ================================================================
fig, axs = plt.subplots(3, 1, figsize=(10, 10),
                        gridspec_kw={'height_ratios': [1, 2, 1]},
                        sharex=False)

# ---- Panel 1: Environment and final positions ----
x_space = np.linspace(0, 130, 500)
clean_light_space = get_clean_light_intensity(x_space)
noisy_light_sample = [get_noisy_light_intensity(x, NOISE_LEVEL) for x in x_space]

axs[0].plot(x_space, noisy_light_sample, color='gold', alpha=0.4,
            label=f"Noisy Environment (std={NOISE_LEVEL})")
axs[0].plot(x_space, clean_light_space, color='orange', linewidth=2,
            linestyle='--', label="Clean Underlying Gradient")
axs[0].axvline(x=80, color='red', linestyle=':',
               label="Optimal Zone (x=80)")
axs[0].set_title("Experiment D: System Robustness in Noisy Environment")
axs[0].set_ylabel("Sensory Input")
axs[0].set_xlabel("Physical Space (1D Track)")
axs[0].legend(loc='upper left')
axs[0].grid(True, linestyle='--', alpha=0.6)

# Mark final positions of the two agents
axs[0].scatter([agent_lif.pos], [get_clean_light_intensity(agent_lif.pos)],
               color='red', s=100, zorder=5, label="LIF Final Position")
axs[0].scatter([agent_tan.pos], [get_clean_light_intensity(agent_tan.pos)],
               color='black', s=100, zorder=5, label="TAN Final Position")
axs[0].legend(loc='upper right')

# ---- Panel 2: Trajectories over time ----
time_axis = np.arange(time_steps)
axs[1].plot(time_axis, agent_lif.pos_log, color='red', linewidth=2,
            label="LIF Agent Trajectory")
axs[1].plot(time_axis, agent_tan.pos_log, color='black', linewidth=2,
            label="TAN Agent Trajectory")
axs[1].axhline(y=80, color='orange', linestyle='--', linewidth=2,
               label="Optimal Zone (x=80)")
axs[1].set_ylabel("Position on Track")
axs[1].set_xlabel("Time Steps")
axs[1].legend(loc='upper left')
axs[1].grid(True, linestyle='--', alpha=0.6)

# ---- Panel 3: Spike rasters ----
axs[2].eventplot(np.where(np.array(agent_lif.spike_log) == 1)[0],
                 lineoffsets=1.5, linelengths=0.5, color='red',
                 label="LIF Motor Spikes")
axs[2].eventplot(np.where(np.array(agent_tan.spike_log) == 1)[0],
                 lineoffsets=0.5, linelengths=0.5, color='black',
                 label="TAN Motor Spikes")
axs[2].set_yticks([0.5, 1.5])
axs[2].set_yticklabels(['TAN', 'LIF'])
axs[2].set_xlabel("Time Steps")
axs[2].set_title("Motor Neuron Spiking Activity")
axs[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("exp_d_noisy_robust.png")
plt.show()