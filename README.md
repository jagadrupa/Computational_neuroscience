# Computational Neuroscience

My study notes for computational neuroscience, written in LaTeX.

## The course in brief

Four questions, each building on the last:

1. **What is a neuron, and how does it compute?** — the biological substrate, and the three levels at which it can be modelled.
2. **Encoding** — how does a stimulus become spikes? Build a forward model and fit it from data.
3. **Decoding** — how do you recover the stimulus from spikes? Two alternatives, a continuous parameter, then a whole time-varying signal.
4. **Information** — how good is the code, independent of any particular decoder, and what shapes it?

The recurring theme is that these are the same problem seen from different sides: the spike-triggered average is both an encoding tool and the optimal single-spike decoder, and modern stimulus reconstruction works by running an encoding model in a loop.

## Skills

### Neuroscience

- Recording methods and their trade-offs — fMRI, EEG, multi-electrode arrays, calcium imaging, patch clamp — and what analysis each makes possible
- Receptive fields: centre–surround, orientation-selective, spatiotemporal; how they are measured and what they compute
- Tuning curves (Gaussian, cosine), functional maps, and where the tuning-curve picture breaks down
- Spike statistics: Poisson firing, Fano factor, interspike intervals, refractoriness
- Synaptic transmission and plasticity: Hebbian learning, LTP/LTD, spike-timing dependent plasticity
- Retinal circuitry and single-photon detection; why the rod → rod bipolar synapse is where thresholding must happen
- Reading real results from the literature: MT and perceptual decisions, LIP and evidence accumulation, LGN coding precision, M1 population vectors, hippocampal place fields

### Mathematics

- **Linear systems** — convolution, impulse responses, linearity and time invariance, why a filter is a projection
- **Probability** — conditional distributions, Bayes' rule, and keeping likelihood, prior, marginal and posterior straight
- **Signal detection theory** — thresholds, hits and false alarms, the likelihood ratio test, the Neyman–Pearson lemma
- **Estimation** — maximum likelihood, maximum a posteriori, the conditional mean as the least-squares estimator, and the bias–variance trade-off
- **Distributions** — Gaussian, Bernoulli, binomial and the Poisson limit; white noise and why it is the right stimulus
- **Calculus in anger** — differentiating log-likelihoods, discarding terms that vanish, solving for an extremum
- **Random walks** — drift to a bound, and the speed–accuracy trade-off it produces
- **Dimensionality reduction** — PCA via the spike-triggered covariance, projection onto feature subspaces, maximally informative dimensions
- **Information theory** — entropy, KL divergence, mutual information, and why the logarithm makes evidence additive
- **Model fitting** — linear–nonlinear models, generalized linear models, the time-rescaling theorem

### Computing

- Python and NumPy for spike-train analysis: the spike-triggered average, boolean masking, slicing, and vectorising a loop with broadcasting
- Working with real recorded data, and the practical traps — off-by-one windows, undersampling bias, division by zero
- Numerical methods: grid search for a threshold, and checking an analytic result against a brute-force one
- Generating figures programmatically with Matplotlib
- LaTeX for technical writing

## Contents

### 01 — Basic Neurobiology

📄 **[compneuro_notes.pdf](01_basic_neurobiology/compneuro_notes.pdf)**

*Lecture 1 — what computational neuroscience is.* The goal of the field · recording from a neuron in an intact brain · receptive fields as the worked example · the descriptive model, or *what* receptive fields look like.

*Lecture 2 — mechanistic and interpretive models.* *How* receptive fields are built, from LGN input to V1 in the Hubel and Wiesel model · *why* they are shaped that way · the efficient coding hypothesis · learning receptive fields from natural images · the three levels side by side.

*Lecture 3 — the neuron.* The neuronal zoo and the idealized neuron · the membrane potential · ionic channels as gatekeepers · the action potential, its propagation along the axon, and myelination.

*Lecture 4 — synapses and plasticity.* The synapse · the synaptic doctrine and how brains learn · Hebbian learning · spike-timing dependent plasticity · what remains unknown.

*Lecture 5 — the nervous system.* Peripheral and central nervous system · a tour of the brain · the cerebral cortex as a layered sheet · neural versus digital computing.

### 02 — Neural Encoding

📄 **[neural_encoding_notes.pdf](02_neural_encoding/neural_encoding_notes.pdf)**

*Part I — What is the neural code?* Recording techniques from fMRI to patch clamp · the retina on a multi-electrode array · raster plots · encoding vs. decoding · tuning curves · functional maps · concept cells · hierarchy and top-down effects.

*Part II — Constructing response models.* Linear response · temporal filtering and convolution · running and leaky averages · spatial filtering and centre–surround receptive fields · difference of Gaussians · spatiotemporal filters · the linear–nonlinear model.

*Part III — Finding the model from data.* Stimuli as vectors · why white noise · the spike-triggered average · recovering the nonlinearity · PCA for multiple features · maximally informative dimensions.

*Part IV — From firing rates to spikes.* Bernoulli to binomial to Poisson · Fano factor and interval distributions · where Poisson breaks · generalized linear models · the time-rescaling theorem.

*Part V — The graded quiz: theory against data.* Computing a spike-triggered average from a real fly H1 recording, and reading what it says about the neuron.

### 03 — Neural Decoding

📄 **[neural_decoding_notes.pdf](03_neural_decoding/neural_decoding_notes.pdf)**

*Lecture 1 — deciding between two alternatives.* The random-dot motion task and area MT · signal detection theory and where to put the threshold · the likelihood ratio and the Neyman–Pearson lemma · accumulating evidence to a bound and LIP ramping · the role of priors and the single-photon response · building in the cost of a mistake.

*Lecture 2 — population codes.* Cricket cercal cells and the population vector · cosine tuning in M1 · why the population vector is neither general nor optimal · Bayesian inference · maximum likelihood with Gaussian tuning and Poisson firing · inverse-variance weighting · maximum a posteriori · correlations.

*Lecture 3 — reconstruction.* The conditional mean as the least-squares estimator · why it reduces to the spike-triggered average for a single spike · reconstruction by pasting · the sign problem and the fly's two H1 neurons · reading movies from LGN spike trains and from human fMRI.

*Guest lecture (Fred Rieke) — seeing by starlight.* Sparse signals in noisy detectors · why averaging fails · where the threshold must live · and why the retina discards most genuine single-photon responses.

### 04 — Information and Coding Principles

📄 **[information_theory_notes.pdf](04_information_theory_and_neural_coding/information_theory_notes.pdf)**

*Lecture 1 — entropy and information.* Information as surprise and why it is a logarithm · entropy as average information and as a count of yes/no questions · entropy as a code's raw capacity, maximal for a binary code at p = 1/2 · noise entropy and what it costs · mutual information as total entropy minus average noise entropy · its two limits · the continuous case · information as KL divergence between the joint and the product of marginals, and its symmetry.

*Lecture 2 — computing information from data.* Chopping spike trains into binary words · total and noise entropy from repeats of a frozen stimulus · ergodicity, or averaging over time instead of over stimuli · the LGN worked example and its 2 ms resolution limit · undersampling bias and extrapolation in 1/L · information per spike from the rate ratio, with no coding model required · what timing precision and mean rate each contribute · place fields.

*Lecture 3 — coding efficiency.* The dynamic range and power-law scaling of natural stimuli · histogram equalisation, or why the input/output curve should be the cumulative integral of the stimulus distribution · Laughlin's test in the fly · real-time adaptation, and the collapse of response curves when the input is normalised by its standard deviation · feature adaptation and the retinal filter from differentiating to integrating · redundancy reduction and why correlations are not simply waste · sparse coding and the emergence of V1-like receptive fields.

*The graded quiz — theory against data.* Entropy and mutual information worked by hand · four simulated cricket cercal neurons · identifying the half-wave rectified cosine tuning curve from its flat zeros · the Fano factor test for Poisson firing, and why converting rates to counts is the whole question · decoding a mystery stimulus with the population vector.

### 05 — Computing in Carbon

📄 **[computing_in_carbon_notes.pdf](05_compuating_in_carbon/computing_in_carbon_notes.pdf)**

*Lecture 1 — neuroelectronics.* Turning a neuron into an equivalent circuit · Kirchhoff's and Ohm's laws, and why neuroscience prefers conductance to resistance · the lipid bilayer as a capacitor with a leak resistor in parallel · deriving the passive membrane equation · ionic gradients as batteries and the Nernst potential · the membrane time constant τ = RC, and the leaky integrator recovered from physics rather than assumed · ion channels as stochastic binary pores · each ion's own battery, with sodium depolarising and potassium hyperpolarising · why a linear membrane computes nothing, and the first sign of excitability.

*Lecture 2 — excitability and Hodgkin–Huxley.* Why fixed conductances leave the circuit linear · the potassium channel's four subunits and P_K ~ n⁴ · gate kinetics as a two-state rate equation, rewritten in τ and steady-state form · sodium's separate inactivation gate, P_Na ~ m³h, and why voltage raises m but lowers h · the four coupled Hodgkin–Huxley equations · anatomy of a spike as a separation of timescales, with fast sodium feedback caught by slower inactivation and potassium · what the model assumed without knowing ion channels existed.

*Lecture 3 — simplified models.* Writing every model as dV/dt = f(V) + I(t) and asking what shape f should take · reading stability off the slope · integrate-and-fire, where the spike is pasted on by rule · bending f(V) upward to get an unstable fixed point that acts as an intrinsic threshold · quadratic and exponential forms, and the theta neuron whose phase gives a free reset · why a third fixed point makes the model bistable, and why escaping it needs a second recovery variable · the phase plane and nullclines · Izhikevich's simple model, and subthreshold resonance as the behaviour one variable cannot produce.

*Lecture 4 — dendritic computation.* Why a point neuron is hard to defend · how location on the tree changes a synapse's influence · Kelvin's cable equation as diffusion plus a leak · the space constant λ = √(r_m/r_i) and propagation at 2λ/τ · passive signals falling to ~5% within two space constants · compartmental models, and Rall's 3/2 rule for collapsing passive trees · what dendrites add — synaptic scaling, sublinear and superlinear summation, dendritic spikes, back-propagating action potentials · delay lines for sound localisation and sequence sensitivity for direction selectivity.

### 06 — Computing with Networks

📄 **[computing_with_networks_notes.pdf](06_computing_with_networks/computing_with_networks_notes.pdf)**

*Lecture 1 — modelling synapses.* What excitatory and inhibitory synapses actually do · recapping the RC membrane and its steady state V_ss = E_L + I_e R_m · adding a synapse as one more conductance branch, g_s(V − E_s), exactly as Hodgkin–Huxley added ionic ones · why excitatory versus inhibitory is a single parameter, whether E_s sits above or below E_L · the factorisation g_s = g_s,max · P_rel · P_s · the kinetic model for channel opening, the same two-state equation as HH gating but driven by transmitter rather than voltage · exponential time courses for AMPA and GABA_A against the delayed alpha function for NMDA · handling spike trains by linear filtering · and a two-neuron network where excitation gives alternating firing while inhibition gives synchrony.

*Lecture 2 — from spikes to firing rates.* The trade-off between spiking and firing-rate models · replacing spike trains with instantaneous rates, and when that fails · turning the synaptic convolution into a differential equation with an exponential kernel · the firing-rate model and its two timescale regimes · why v_ss = F(w·u) is the artificial neuron of machine learning, revealed as the steady state of a much richer system · weight matrices W and M, with feedforward as the special case M = 0 · mutual inhibition from negative off-diagonal entries · and a linear feedforward network that computes spatial derivatives, which is what V1 simple cells appear to do.

*Lecture 3 — recurrent networks.* Decoupling the network by writing the output in the eigenvector basis of a symmetric M · eigenvalues setting stability, with any λ > 1 causing the linear network to explode · amplification by 1/(1−λ₁), selectively along one eigenvector · cosine connectivity, exciting nearby and inhibiting distant neurons as cortex appears to · λ = 1 turning a mode into an integrator that holds a memory of past input, as real eye-position neurons do · rectification keeping a network with λ₁ = 1.9 stable, and enabling winner-take-all selection, gain modulation and working-memory persistence · non-symmetric excitatory–inhibitory circuits analysed through the Jacobian · and one time constant carrying the network from a point attractor through a Hopf bifurcation to a limit cycle.

*The graded quiz — networks in practice.* Reading a weight matrix as an image filter, and using row sums to tell dimming, blurring, edge detection and blocking apart · what the firing-rate abstraction discards, which turns out to be everything about time and nothing about magnitude · the alpha synapse, where t_peak raises the firing rate sublinearly by slowing the synaptic conductance rather than the membrane · why inhibition is E_syn below threshold rather than below zero, and shunting inhibition as the case in between · and solving a recurrent network directly as (I − M)⁻¹Wu, with a warning about eigenvector degeneracy.

### 07 — Networks that Learn: Plasticity in the Brain and Learning

📄 **[plasticity_and_learning_notes.pdf](07_networks_that_learn--plasticity_in_the_brain_and_learning/plasticity_and_learning_notes.pdf)**

*Lecture 1 — synaptic plasticity and Hebbian learning.* LTP and LTD as the same experiment with the postsynaptic response switched on or off · Hebb's 1949 prediction, made before either was observed · formalising it as the product uv, a rule local enough for a synapse to follow · the discrete update and its descent to machine learning · the averaged rule as the input correlation matrix Q, a statistic nothing in the system computes · the covariance rule, thresholding at the neuron's own mean output to get both LTP and LTD · why being able to weaken synapses does not confer stability, and both rules explode · Oja's rule and its −αv²w decay term, converging to ‖w‖ = 1/√α · solving the averaged rule in the eigenvector basis, and why direction converges long before length does · Hebbian learning as principal component analysis, finding the direction of maximum variance · why the plain Hebb rule fails on data with a non-zero mean, which non-negative firing rates guarantee · and clustered data, which one output neuron cannot describe at all.

*Lecture 2 — competitive learning and unsupervised learning.* One output neuron per cluster, with the weight vector as a prototype rather than a direction · why the most active neuron is the closest one, and why that equivalence collapses without normalisation · the running-average update Δw = ε(u − w), a difference rather than a product, and therefore stable · competitive learning as a neural k-means · the dead-unit problem, where a neuron that never wins can never improve · self-organising maps, updating the winner's neighbours to produce topology-preserving representations, and orientation preference maps in V1 · generative models, which ask what process produced the data rather than what feature to extract · recognition as the inverse problem solved by Bayes' rule · mixtures of Gaussians · and the EM algorithm, alternating soft posterior responsibilities with weighted re-fitting, learning the spreads and priors that competitive learning cannot express.

*Lecture 3 — sparse coding and predictive coding.* Eigenfaces, and why PCA compresses a million pixels into ten numbers but cannot extract local parts · dropping orthogonality for an overcomplete dictionary with more features than pixels, and why that ambiguity is what the prior exists to resolve · Gaussian white noise making squared reconstruction error identical to log likelihood · sparse, super-Gaussian priors, which differ from a Gaussian almost entirely in the tails · the objective F trading reconstruction against sparseness, with neither extreme usable · gradient ascent, alternating an inference step and a learning step as EM does · the inference step as a recurrent network that predicts its own input, measures the error and corrects · the learning step as a Hebbian rule almost identical to Oja's, escaping PCA only because of the sparse prior · Olshausen and Field's result, where the learnt features resemble V1 simple-cell receptive fields — an interpretive model of why receptive fields take the form they do · and predictive coding, which assigns roles to the reciprocal feedforward and feedback connections of visual cortex: errors up, predictions down.

*The graded quiz — Hebbian learning in practice.* Reading the principal eigenvector off a correlation matrix, and why only the direction is determined · what a converged weight vector does and does not tell you about the data · the learning rate as a single knob controlling both adaptation speed and noise sensitivity · why EM must alternate, and why monotonic improvement is not the same as global optimality · discretising Oja's rule and the constants that survive · implementing it on the Dayan and Abbott c10p1 data, where it converges to one of two unit vectors along the principal axis · what goes wrong without mean-centring, and the two traps in testing that · and the pure Hebb rule, which finds the same direction while its length runs to 10^100.

### 08 — Learning from Supervision and Rewards

📄 **[supervision_and_rewards_notes.pdf](08_learning_from_supervision_and_rewards/supervision_and_rewards_notes.pdf)**

*Lecture 1 — neurons as classifiers and supervised learning.* Classification as geometry, with images as points in a space of as many dimensions as pixels · labels as the new ingredient, and what supervision assumes · the perceptron, which is week 1's idealised neuron with a name and a learning rule · why setting a weighted sum equal to a threshold defines a hyperplane, so the boundary is always flat · the learning rule driven by output error rather than by the neuron's own activity, and why the threshold moves opposite to the weights · XOR, which no perceptron can learn — a limit of representation, not of training · multilayer perceptrons, and the two-layer network that composes flat boundaries into a curved one · sigmoid units for regression, adopted for differentiability rather than realism · the delta rule for the output layer and backpropagation for the hidden layers via the chain rule · why backpropagation is hard to defend biologically, needing information no real synapse has · the difference between a network that cannot represent a function and one that merely fails to find it · and backing a truck into a loading dock, where imitation of a teacher is also a ceiling.

## Code

| File | What it does |
|---|---|
| [compute_sta.py](02_neural_encoding/compute_sta.py) | Spike-triggered average, annotated line by line, with a worked ten-step example and a runnable demo |
| [sta_analysis.ipynb](02_neural_encoding/sta_analysis.ipynb) | Computing an STA from the fly H1 recording and reading what it says about the neuron |
| [make_figs.py](03_neural_decoding/make_figs.py) | Generates the decoding figures that are not on the slides — the evidence walk, the prior shift, ML vs MAP |
| [decoding_quiz.ipynb](03_neural_decoding/decoding_quiz.ipynb) | Likelihood ratio test with asymmetric costs, solved analytically and by brute force as a check |
| [tuning_curves.ipynb](04_information_theory_and_neural_coding/tuning_curves.ipynb) | Cricket cercal analysis: tuning curve shape, the Fano factor test for Poisson firing, and the population vector |
| [make_figs.py](04_information_theory_and_neural_coding/make_figs.py) | Generates the quiz figures in the information theory notes from the supplied data |
| [rc_membrane.ipynb](05_compuating_in_carbon/rc_membrane.ipynb) | The passive membrane equation integrated numerically, and how τ = RC sets the response |
| [integrate_and_fire.ipynb](05_compuating_in_carbon/integrate_and_fire.ipynb) | Leaky integrate-and-fire with a current sweep, from silence through threshold to saturation |
| [alpha_synapse.ipynb](06_computing_with_networks/alpha_synapse.ipynb) | An alpha-function synapse driving an integrate-and-fire cell: sweeping t_peak and the reversal potential |
| [recurrent_steady_state.ipynb](06_computing_with_networks/recurrent_steady_state.ipynb) | Steady state of a recurrent network by direct solve and by eigendecomposition, and why eigh is required |
| [make_figs.py](06_computing_with_networks/make_figs.py) | Generates the network quiz figures — image filters, row sums, the t_peak sweep and the reversal potential |
| [make_figs.py](07_networks_that_learn--plasticity_in_the_brain_and_learning/make_figs.py) | Simulates every computed figure in the Week 7 notes — the three Hebbian rules, competitive learning, EM, and sparse coding on image patches |
| [oja_rule.ipynb](07_networks_that_learn--plasticity_in_the_brain_and_learning/oja_rule.ipynb) | Oja's rule implemented and run on real data: the two solutions, the effect of not mean-centring, and pure Hebb compared against it |
| [make_figs.py](08_learning_from_supervision_and_rewards/make_figs.py) | Perceptron learning, the XOR limit, sigmoid slopes, and backpropagation implemented from scratch |

## Building

Each folder holds the LaTeX source alongside its PDF:

```bash
cd 02_neural_encoding
latexmk -pdf neural_encoding_notes.tex
```

LaTeX sources and figures are kept local, so the `.tex` files are not in the repo — the built PDFs above are the readable output. Lecture slides, videos, transcripts and course datasets are third-party material and are not redistributed here.
