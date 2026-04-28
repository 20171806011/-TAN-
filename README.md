# Temporal Attention Neuron (TAN) Model Experiments

This repository implements a minimalist **Temporal Attention Neuron (TAN)** model. The model integrates three core ideas:  
- Predictive coding (surprise-based signal processing)  
- Attention mechanism (selection over temporal context)  
- Spiking neuron dynamics (threshold firing mechanism)

This repository contains a series of computational neuroscience experiments demonstrating the dynamics and advantages of the **Temporal Attention Neuron (TAN)** compared to classic Leaky Integrate-and-Fire (LIF) models. The TAN incorporates a non-Markovian memory window, attention-based context integration (Q/K/V projections), and surprise-gated modulation to achieve natural biological habituation and high noise tolerance.

## 📂 Code Structure

The repository comprises four progressive experiments:

*   **`TAN Neuron Dynamics: Stimulus vs Response.py`**: Introduces the core TAN architecture. Demonstrates its habituation property using a simple step-light stimulus, showing how surprise-gated modulation organically filters out constant stimuli without a "softmax ghost" effect.
*   **`Experiment B: Noisy light stimulus with sudden onset.py`**: Compares TAN and LIF neurons under a noisy light stimulus. Highlights the TAN's ability to filter out temporal noise while maintaining sharp responsiveness to sudden environmental onset changes.
*   **`Experiment C: Environment and Agent Trajectories.py`**: Simulates "Bioton" agents navigating a 1D spatial light gradient. Compares the movement trajectories and motor spiking activity of TAN-driven vs. LIF-driven agents to demonstrate task-oriented efficiency.
*   **`Experiment D: System Robustness in Noisy Environment.py`**: Advances the 1D environment by introducing severe Gaussian noise and a `noise_tolerance` dead-zone mechanism in the TAN. Demonstrates TAN's superior navigation robustness compared to the erratic behavior of LIF.

## 🚀 Run Instructions

1.  **Prerequisites**: Ensure you have Python installed (Python 3.7+ recommended).
2.  **Install Dependencies**: The experiments rely on standard scientific libraries. Install them via pip:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Execution**: Run any of the experiment scripts directly from your terminal. Each script will run the simulation and automatically open a Matplotlib window displaying the results.
    ```bash
    python "TAN Neuron Dynamics: Stimulus vs Response.py"
    python "Experiment B: Noisy light stimulus with sudden onset.py"
    python "Experiment C: Environment and Agent Trajectories.py"
    python "Experiment D: System Robustness in Noisy Environment.py"
    ```

## 📊 Figures and Visualizations

Running the scripts generates comprehensive, publication-quality figures comprising multiple panels for deep analysis:

*   **Basic Dynamics Figure**: Displays the environmental light stimulus, the internal activation energy (with threshold markers), and the resulting discrete action potentials (spikes). It visually captures the concept of habituation.
*   **Experiment B Figure (Noisy Stimulus)**: Plots the clean vs. noisy stimulus alongside the membrane potentials ($h$) and discrete spike trains for both LIF and TAN neurons.
*   **Experiment C Figure (Agent Trajectories)**: Visualizes the 1D spatial light distribution, the time-series position trajectories of both agents attempting to reach the optimal zone, and a raster plot of their motor spikes.
*   **Experiment D Figure (Noise Robustness)**: Shows the extremely noisy spatial light gradient, mapping out how the TAN agent smoothly and reliably reaches the optimal zone while the LIF agent's trajectory gets disrupted by the noise, supported by their respective motor spike rasters.
