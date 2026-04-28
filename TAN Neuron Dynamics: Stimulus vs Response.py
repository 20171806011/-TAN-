import numpy as np
import matplotlib.pyplot as plt

class TANNeuron:
    """
    Temporal Attention Neuron (TAN) with surprise-gated modulation.

    This implementation incorporates the crucial mathematical fix that 
    eliminates the "Softmax ghost" effect: when the input becomes constant,
    the surprise term vanishes, pulling the total activation to zero and
    producing a natural habituation curve.
    """
    def __init__(self, window_size=10):
        """
        Parameters
        ----------
        window_size : int
            Length of the local temporal receptive window.
            Larger windows create smoother habituation curves.
        """
        self.window_size = window_size
        # Projection weights for Query, Key, Value (scalars in this minimal model)
        self.W_q = np.array([2.0])   # Query weight, amplifies surprise
        self.W_k = np.array([1.0])   # Key weight for historical features
        self.W_v = np.array([1.0])   # Value weight for historical energy
        # Firing threshold for spike generation
        self.threshold = 0.3
        # Initialize history buffer with zeros (no previous stimulus)
        self.history = np.zeros(window_size)
        
    def softmax(self, x):
        """
        Numerically stable softmax with a small epsilon to avoid division by zero.

        Parameters
        ----------
        x : np.ndarray
            Raw attention scores (logits).

        Returns
        -------
        np.ndarray
            Normalised attention weights that sum to 1.
        """
        e_x = np.exp(x - np.max(x))
        return e_x / (e_x.sum(axis=0) + 1e-9)

    def forward(self, current_input):
        """
        Execute one time step of the neuron.

        Steps:
        1. Update temporal window and compute surprise (prediction error).
        2. Project Q from surprise, K and V from history.
        3. Compute attention scores and weights.
        4. Integrate context and apply surprise gating.
        5. Generate spike via threshold comparison.

        Parameters
        ----------
        current_input : float
            The scalar sensory input at the present time step.

        Returns
        -------
        spike : int (0 or 1)
            Binary spike output.
        activation : float
            The gated internal activation energy (before thresholding).
        attention_weights : np.ndarray
            Attention distribution over the memory window.
        """
        # 1. Update sliding window: shift left and insert current input at the end
        self.history = np.roll(self.history, -1)
        self.history[-1] = current_input
        
        # --- Surprise computation (prediction error) ---
        # The neuron predicts the current input as the mean of its history,
        # and the deviation from this prediction is the "surprise".
        # Biologically, this models sensory novelty detection.
        expected = np.mean(self.history)
        surprise = current_input - expected
        
        # 2. Temporal attention projections
        # Query is driven by surprise, not by the raw input!
        # This is the key divergence from standard Transformer attention.
        Q = surprise * self.W_q          # Query: "What changed?"
        K = self.history * self.W_k      # Keys: historical features
        V = self.history * self.W_v      # Values: historical energy
        
        # 3. Attention mechanism
        # Compute dot-product scores between Q and each key
        attention_scores = Q * K 
        # Softmax normalisation to obtain attention weights
        attention_weights = self.softmax(attention_scores)
        
        # 4. Base activation: attention-weighted integration of the value vector
        base_activation = np.sum(attention_weights * V)
        
        # ===== Core mathematical fix: Surprise gating =====
        # Without this gate, even when Q=0 (constant stimulus), the softmax
        # would spread weights uniformly and produce non-zero activation,
        # failing to capture habituation.
        #
        # By multiplying by |surprise|, we implement sensory gating:
        #   - When surprise is high (sudden change), the gate opens wide.
        #   - When surprise → 0 (constant stimulus), the gate shuts down
        #     entirely, and activation plummets exponentially due to the
        #     absence of new input energy.
        activation = base_activation * abs(surprise)
        
        # 5. Spike generation using a Heaviside step function
        spike = 1 if activation > self.threshold else 0
        
        return spike, activation, attention_weights


# =============================================================================
# Simulation: constant light stimulus after a dark period
# =============================================================================

# Instantiate the TAN neuron with a 10-step memory window
neuron = TANNeuron(window_size=10)

# Construct the stimulus: 5 steps of darkness (0) followed by 25 steps of bright light (1)
signal_input = np.concatenate([np.zeros(5), np.ones(25)])

# Storage for analysis and plotting
spikes = []            # Binary spike train
activations = []       # Internal activation energy over time
attention_maps = []    # Attention weight vectors for each time step

# ================= Main temporal loop =================
for t, x in enumerate(signal_input):
    spike, act, att_w = neuron.forward(x)
    spikes.append(spike)
    activations.append(act)
    attention_maps.append(att_w)

# ================= Publication-quality figure =================
fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

# --- Subplot 1: Environmental stimulus (Input) ---
axs[0].plot(signal_input, drawstyle='steps-post', color='orange', linewidth=2,
            label="Light Stimulus")
axs[0].set_ylabel("Intensity")
axs[0].set_title("TAN Neuron Dynamics: Stimulus vs Response")
axs[0].grid(True, linestyle='--', alpha=0.6)
axs[0].legend(loc="upper left")

# --- Subplot 2: Internal activation energy ---
axs[1].plot(activations, color='dodgerblue', linewidth=2,
            label="Internal Activation Energy")
# Draw the firing threshold as a horizontal dashed line
axs[1].axhline(y=neuron.threshold, color='red', linestyle='--', linewidth=1.5,
               label=f"Spike Threshold ({neuron.threshold})")
axs[1].set_ylabel("Activation")
axs[1].grid(True, linestyle='--', alpha=0.6)
axs[1].legend(loc="upper left")

# --- Subplot 3: Output spikes (action potentials) ---
axs[2].stem(spikes, linefmt='k-', markerfmt='ko', basefmt=" ",
            label="Action Potentials (Spikes)")
axs[2].set_xlabel("Time Steps")
axs[2].set_ylabel("Spike Output")
axs[2].set_yticks([0, 1])
axs[2].grid(True, linestyle='--', alpha=0.6)
axs[2].legend(loc="upper left")

plt.tight_layout()
plt.show()

