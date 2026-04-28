import numpy as np
import matplotlib.pyplot as plt

# ================================================================
# 1. Neuron Definitions (Parameters Rebalanced)
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
            Leak factor (0 < lambda_leak < 1). Controls membrane time constant.
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


class TANNeuron:
    """
    Temporal Attention Neuron (TAN).
    Features a non‑Markovian memory window, attention‑based context
    integration, and a surprise‑gated modulation for habituation.
    """
    def __init__(self, window_size=5, lambda_leak=0.5, threshold=0.5):
        """
        Parameters
        ----------
        window_size : int
            Length of the local temporal receptive field.
        lambda_leak : float
            Membrane leak factor.
        threshold : float
            Spike threshold (tuned to capture attentive responses).
        """
        self.window_size = window_size
        self.lambda_leak = lambda_leak
        self.threshold = threshold
        self.h = 0.0                     # membrane potential

        # Synaptic projection weights (scalars for simplicity)
        self.W_q = np.array([2.0])       # amplifies surprise
        self.W_k = np.array([1.0])       # historical features
        self.W_v = np.array([1.0])       # historical energy

        # Initialize history buffer (no prior stimulation)
        self.history = np.zeros(window_size)
        self.beta = 1.0                  # inverse temperature for attention

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
        One time step of the TAN neuron.

        Steps:
        1. Update history window and compute prediction error (surprise).
        2. Apply asymmetric biological rectification: only light increases 
           (positive surprise) are excitatory.
        3. Project into Q, K, V and compute attention weights.
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

        # --- Key evolutionary refinement: Asymmetric biological rectification ---
        # Only light increments (positive surprise) excite the neuron.
        # Light decrements are ignored → neuron stops pushing when light fades.
        S_t = np.maximum(0, S_t)       # ReLU activation on surprise

        # 3. Temporal attention projections
        Q = S_t * self.W_q             # Query driven by rectified surprise
        K = self.history * self.W_k    # Keys: historical feature patterns
        V = self.history * self.W_v    # Values: historical energy magnitudes

        # Attention scores and normalisation
        attention_scores = Q * K
        attention_weights = self.softmax(attention_scores)

        # 4. Context integration and gating
        C_t = np.sum(attention_weights * V)    # base context
        gating = np.tanh(S_t)                  # smooth gate (no need for abs, S_t≥0)
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
# 2. Three‑Cell Organism (Bioton)
# ================================================================
class Bioton:
    """
    Simple agent with a single neuron brain and a 1D position.
    The neuron receives light intensity as input and controls
    a motor output (velocity).
    """
    def __init__(self, brain_type='TAN', start_pos=50.0):
        """
        Parameters
        ----------
        brain_type : str
            'TAN' or 'LIF' – neuron type to use as brain.
        start_pos : float
            Initial position on the 1D track.
        """
        if brain_type == 'TAN':
            self.brain = TANNeuron()
        else:
            self.brain = LIFNeuron()

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
            Perceived light intensity at the current position.
        """
        # Feed light intensity to the neuron and get a spike decision
        spike = self.brain.forward(light_intensity)

        # Simple kinematics:
        #   - friction: high fluid drag, agent decelerates without spikes
        #   - boost:    gentle thrust from each spike
        friction = 0.6       # large drag — deceleration when no spike
        boost = 0.8          # moderate propulsion per spike
        self.velocity = friction * self.velocity + boost * spike
        self.pos += self.velocity

        # Record for later visualization
        self.pos_log.append(self.pos)
        self.spike_log.append(spike)


# ================================================================
# 3. Physical Environment (Make the Sun Brighter!)
# ================================================================
def get_light_intensity(x):
    """
    Light intensity distribution along the 1D track.
    A Gaussian peak centred at x = 80 models a light source.

    Parameters
    ----------
    x : float or np.ndarray
        Position(s) on the track.

    Returns
    -------
    intensity : float or np.ndarray
        Light intensity at the given position(s).
    """
    light_pos = 80.0                          # centre of the light source
    sigma = 15.0                              # standard deviation (width of the beam)
    max_intensity = 10.0                      # peak intensity — strong enough
                                              # to create distinct surprise transients
    return max_intensity * np.exp(-((x - light_pos)**2) / (2 * sigma**2))


# ================================================================
# 4. Run the Simulation Competition
# ================================================================
time_steps = 300
agent_lif = Bioton(brain_type='LIF', start_pos=50.0)
agent_tan = Bioton(brain_type='TAN', start_pos=50.0)

for t in range(time_steps):
    # Both agents perceive the light based on their current positions
    intensity_lif = get_light_intensity(agent_lif.pos)
    intensity_tan = get_light_intensity(agent_tan.pos)

    # Update each agent
    agent_lif.step(intensity_lif)
    agent_tan.step(intensity_tan)


# ================================================================
# 5. Publication‑Quality Plot (Three Panels)
# ================================================================
fig, axs = plt.subplots(3, 1, figsize=(10, 10),
                        gridspec_kw={'height_ratios': [1, 2, 1]},
                        sharex=False)

# --- Spatial map of light and final positions ---
x_space = np.linspace(0, 130, 500)
light_space = get_light_intensity(x_space)
axs[0].plot(x_space, light_space, color='gold', linewidth=3,
            label="Light Source Distribution")
axs[0].axvline(x=80, color='orange', linestyle='--',
               label="Optimal Zone (Light Peak)")
axs[0].set_title("Experiment C: Environment and Agent Trajectories")
axs[0].set_ylabel("Light Intensity")
axs[0].set_xlabel("Physical Space (1D Track)")
axs[0].grid(True, linestyle='--', alpha=0.6)

# Mark final positions
axs[0].scatter([agent_lif.pos], [get_light_intensity(agent_lif.pos)],
               color='red', s=100, zorder=5, label="LIF Final Position")
axs[0].scatter([agent_tan.pos], [get_light_intensity(agent_tan.pos)],
               color='black', s=100, zorder=5, label="TAN Final Position")
axs[0].legend(loc='upper right')

# --- Trajectories over time ---
time_axis = np.arange(time_steps)
axs[1].plot(time_axis, agent_lif.pos_log, color='red', linewidth=2,
            label="LIF Agent Trajectory")
axs[1].plot(time_axis, agent_tan.pos_log, color='black', linewidth=2,
            label="TAN Agent Trajectory (Proposed)")
axs[1].axhline(y=80, color='orange', linestyle='--', linewidth=2,
               label="Optimal Zone (x=80)")
axs[1].set_ylabel("Position on Track")
axs[1].set_xlabel("Time Steps")
axs[1].legend(loc='upper left')
axs[1].grid(True, linestyle='--', alpha=0.6)

# --- Spike rasters ---
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
plt.show()