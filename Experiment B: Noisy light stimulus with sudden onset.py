import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Baseline 1: Classic Leaky Integrate-and-Fire (LIF) neuron
# ============================================================
class LIFNeuron:
    """
    Standard leaky integrate-and-fire neuron with Markovian dynamics.
    The membrane potential decays exponentially (leak) and integrates
    incoming current. A spike is fired when the potential exceeds the
    threshold, after which the potential is reset.
    """
    def __init__(self, lambda_leak=0.8, threshold=1.0):
        """
        Parameters
        ----------
        lambda_leak : float
            Leak factor (0 < lambda_leak < 1). Models the membrane time constant.
        threshold : float
            Firing threshold for spike generation.
        """
        self.lambda_leak = lambda_leak
        self.threshold = threshold
        self.h = 0.0  # membrane potential

    def forward(self, current_input):
        """
        Advance the neuron by one time step.

        Parameters
        ----------
        current_input : float
            Input current (sensory signal).

        Returns
        -------
        spike : int
            1 if a spike was emitted, 0 otherwise.
        h : float
            Membrane potential after updating (and possible reset).
        """
        # Markovian state update: leaky integration + instantaneous input
        self.h = self.lambda_leak * self.h + current_input

        # Spike generation and reset
        if self.h > self.threshold:
            spike = 1
            self.h = 0.0  # reset membrane potential after firing
        else:
            spike = 0

        return spike, self.h


# ============================================================
# Proposed Model: Temporal Attention Neuron (TAN)
# ============================================================
class TANNeuron:
    """
    Temporal Attention Neuron (TAN) with surprise-gated modulation
    and a non-Markovian memory window. This neuron uses an attention
    kernel derived from statistical mechanics to integrate historical
    context, and a gating mechanism based on prediction error (surprise)
    to implement sensory habituation.
    """
    def __init__(self, window_size=10, lambda_leak=0.8, threshold=1.0):
        """
        Parameters
        ----------
        window_size : int
            Length of the local temporal receptive window (memory).
        lambda_leak : float
            Membrane leak factor (0 < lambda_leak < 1).
        threshold : float
            Firing threshold.
        """
        self.window_size = window_size
        self.lambda_leak = lambda_leak
        self.threshold = threshold
        self.h = 0.0  # membrane potential

        # Projection weights for Query, Key, Value (scalars in minimal model)
        self.W_q = np.array([2.0])   # Query: amplifies surprise
        self.W_k = np.array([1.0])   # Key: historical features
        self.W_v = np.array([1.0])   # Value: historical energy

        # Initialize history buffer (no prior stimulation)
        self.history = np.zeros(window_size)

        # Thermodynamic sensitivity (inverse temperature)
        self.beta = 1.0

    def softmax(self, x):
        """
        Numerically stable softmax (Boltzmann distribution) with temperature scaling.

        Parameters
        ----------
        x : np.ndarray
            Raw attention scores.

        Returns
        -------
        np.ndarray
            Attention weights (probabilities) summing to 1.
        """
        e_x = np.exp(self.beta * (x - np.max(x)))
        return e_x / (e_x.sum(axis=0) + 1e-9)

    def forward(self, current_input):
        """
        Perform one time step of the TAN neuron.

        Steps:
        1. Update history window and compute surprise.
        2. Project Query (from surprise), Key and Value (from history).
        3. Compute attention weights via Boltzmann kernel.
        4. Integrate base context and apply surprise gating.
        5. Update membrane potential (leak + gated current) and generate spike.

        Parameters
        ----------
        current_input : float
            Sensory input at the current time step.

        Returns
        -------
        spike : int
            1 if spike emitted, 0 otherwise.
        h : float
            Membrane potential after the update (and possible reset).
        """
        # 1. Update temporal receptive window (sliding window)
        self.history = np.roll(self.history, -1)
        self.history[-1] = current_input

        # 2. Compute sensory surprise (prediction error)
        # Expected value is the mean of the window; surprise is deviation.
        S_t = current_input - np.mean(self.history)

        # 3. Project into temporal attention space (Q, K, V)
        Q = S_t * self.W_q          # Query: driven by surprise, not raw input
        K = self.history * self.W_k # Keys: features of past events
        V = self.history * self.W_v # Values: energies of past events

        # 4. Compute attention scores and Boltzmann weights
        attention_scores = Q * K
        attention_weights = self.softmax(attention_scores)

        # 5. Base context integration (attention-weighted sum of V)
        C_t = np.sum(attention_weights * V)

        # 6. Neuromodulatory gating (surprise-gated modulation)
        # Use a smooth nonlinearity (tanh) to filter out minor noise
        gating = np.tanh(abs(S_t))
        A_t = gating * C_t          # Effective synaptic current

        # 7. Non-Markovian dynamical evolution: leak + attention-gated current
        self.h = self.lambda_leak * self.h + A_t

        # 8. Spike generation and reset
        if self.h > self.threshold:
            spike = 1
            self.h = 0.0           # reset after spike
        else:
            spike = 0

        return spike, self.h


# ============================================================
# Experiment B: Noisy light stimulus with sudden onset
# ============================================================
time_steps = 60

# Base signal: 15 steps of darkness (0), then 45 steps of bright light (1.5)
base_signal = np.concatenate([np.zeros(15), np.ones(45) * 1.5])

# Add Gaussian white noise (mean=0, std=0.4) to simulate noisy environment
noise = np.random.normal(0, 0.4, time_steps)
noisy_signal = np.clip(base_signal + noise, 0, None)  # ensure intensity >= 0

# Instantiate neurons with appropriate parameters
lif_neuron = LIFNeuron(lambda_leak=0.8, threshold=2.0)
tan_neuron = TANNeuron(window_size=10, lambda_leak=0.8, threshold=1.0)

# Data logging
lif_spikes, lif_h_log = [], []
tan_spikes, tan_h_log = [], []

# Run simulation loop
for x in noisy_signal:
    l_spike, l_h = lif_neuron.forward(x)
    t_spike, t_h = tan_neuron.forward(x)

    lif_spikes.append(l_spike)
    lif_h_log.append(l_h)
    tan_spikes.append(t_spike)
    tan_h_log.append(t_h)

# ============================================================
# Publication‑quality visualization (3 panels)
# ============================================================
fig, axs = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

# ----- Panel 1: Noisy stimulus and clean ground truth -----
axs[0].plot(noisy_signal, color='orange', drawstyle='steps-post',
            label="Noisy Light Stimulus")
axs[0].plot(base_signal, color='gray', linestyle='--', alpha=0.5,
            label="Clean Ground Truth")
axs[0].set_title("Experiment B: Neural Response to Noisy Stimulus")
axs[0].set_ylabel("Intensity")
axs[0].legend(loc="upper left")
axs[0].grid(True, linestyle='--', alpha=0.6)

# ----- Panel 2: LIF baseline dynamics -----
# Trace of membrane potential
axs[1].plot(lif_h_log, color='gray', alpha=0.5,
            label="LIF Membrane Potential (h)")
# Mark spike times with vertical red lines
lif_spike_times = np.where(np.array(lif_spikes) == 1)[0]
for t in lif_spike_times:
    axs[1].axvline(x=t, color='red', linestyle='-', alpha=0.8)
axs[1].set_ylabel("LIF Dynamics")
# Add an invisible line for legend entry of LIF Spikes
axs[1].plot([], [], color='red', label='LIF Spikes')
axs[1].legend(loc="upper left")
axs[1].grid(True, linestyle='--', alpha=0.6)

# ----- Panel 3: TAN dynamics -----
# Trace of membrane potential
axs[2].plot(tan_h_log, color='dodgerblue', alpha=0.5,
            label="TAN Membrane Potential (h)")
# Mark spike times with vertical black lines
tan_spike_times = np.where(np.array(tan_spikes) == 1)[0]
for t in tan_spike_times:
    axs[2].axvline(x=t, color='black', linestyle='-', ymin=0, ymax=1,
                   linewidth=2)
axs[2].set_ylabel("TAN Dynamics")
axs[2].set_xlabel("Time Steps")
# Add invisible line for legend entry of TAN Spikes
axs[2].plot([], [], color='black', linewidth=2,
            label='TAN Spikes (Filtered & Habituated)')
axs[2].legend(loc="upper left")
axs[2].grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()