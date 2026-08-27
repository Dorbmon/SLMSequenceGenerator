// Shared numerical constants for the CPU/Wasm and WebGPU WGS implementations.
// Keep this module dependency-free so WebGPU-only entry points do not load or
// initialize the Wasm backend merely to reuse the same coefficients.

// A coherent sum below this fraction has no numerically meaningful phase.
export const WGS_INITIALIZATION_CANCELLATION_RATIO = 1e-3;

// Phase-locked trap coefficients are strongly coupled after the phase-only
// projection. A large multiplicative WGS step therefore creates a two-cycle.
// Reference/free-phase WGS follows the measured phase and can safely use the
// configured feedback gain.
export const WGS_MAX_STABLE_TRAP_AMPLITUDE_GAIN = 0.1;
export const WGS_REFERENCE_TRAP_AMPLITUDE_GAIN = 0.85;
export const WGS_LOCKED_PHASE_PRECOMPENSATION_GAIN = 0.7;
export const WGS_SOFT_PHASE_PRECOMPENSATION_GAIN = 0.2;
