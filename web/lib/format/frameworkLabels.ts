/**
 * Canonical framework name → display label mappings.
 *
 * This module provides consistent formatting for all frameworks in the catalog.
 * Import from here instead of duplicating mappings across components.
 */

const FRAMEWORK_LABELS: Record<string, string> = {
  // Core ML frameworks
  pytorch: 'PyTorch',
  tensorflow: 'TensorFlow',
  jax: 'JAX',
  keras: 'Keras',

  // PyTorch ecosystem
  torchvision: 'TorchVision',
  torchaudio: 'Torchaudio',
  'transformer-engine': 'Transformer Engine',
  apex: 'APEX',

  // LLM serving
  vllm: 'vLLM',
  ollama: 'Ollama',
  'llama.cpp': 'llama.cpp',
  'text-generation-inference': 'Text Generation Inference',
  'triton-inference-server': 'Triton Inference Server',
  tensorrt: 'TensorRT',
  'tensorrt-llm': 'TensorRT-LLM',

  // HuggingFace
  transformers: 'Transformers',

  // Notebooks & interactive
  jupyter: 'Jupyter',

  // NVIDIA NeMo
  nemo: 'NeMo',

  // RAPIDS (GPU data science)
  rapids: 'RAPIDS',
  cudf: 'cuDF',
  cuml: 'cuML',
  cugraph: 'cuGraph',

  // Scientific Python
  numpy: 'NumPy',
  scipy: 'SciPy',
  pandas: 'pandas',
  'scikit-learn': 'scikit-learn',

  // Big data & distributed
  spark: 'Spark',
  pyspark: 'PySpark',
  xgboost: 'XGBoost',

  // Languages
  r: 'R'
};

/**
 * Format a raw framework name for user-facing display.
 *
 * Uses the canonical mapping if available, otherwise applies intelligent
 * title-casing to hyphen/underscore-separated parts.
 *
 * @param rawName - The framework name (case-insensitive)
 * @returns Human-friendly display label
 */
export function formatFrameworkLabel(rawName: string): string {
  const normalized = rawName.toLowerCase();

  if (normalized in FRAMEWORK_LABELS) {
    return FRAMEWORK_LABELS[normalized];
  }

  // Fallback: title-case each hyphen/underscore-separated part
  return normalized
    .split(/[-_]+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

/**
 * Get all known framework names (lowercase).
 * Useful for type checking or validation.
 */
export function getKnownFrameworks(): string[] {
  return Object.keys(FRAMEWORK_LABELS);
}
