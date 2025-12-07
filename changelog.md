# GPU Doctor Changelog

## 2025-12-07 ~13:30 UTC

### Created JSON Schema and Initial Data

**Files created:**
- `data/schema.json` - JSON Schema 2020-12 defining the image catalog structure
- `data/images.json` - Initial catalog with 15 ML Docker images

**Schema highlights:**
- Nested structure with logical groupings: `metadata`, `cuda`, `runtime`, `capabilities`, `cloud`, `security`, `size`, `urls`
- Controlled vocabularies (enums) for providers, registries, roles, workloads
- Optional fields handled via `oneOf` with `null` (e.g., `cuda` is null for CPU-only images)
- Validation tested with `jsonschema` library

**Images added (15 total):**
- PyTorch: 2.5.1 runtime, 2.5.1 devel, 2.4.0 CUDA 11.8
- TensorFlow: 2.17.0 GPU, 2.17.0 GPU Jupyter, 2.17.0 CPU
- NVIDIA NGC: PyTorch 24.12, TensorFlow 24.12
- Inference/Serving: vLLM, TGI 3.3.5, Triton 25.11, Ollama
- Base CUDA: 12.4 runtime, 12.4 devel, 12.4 cudnn-devel

**Coverage by role:**
- Training: 7
- Serving: 4
- Base: 3
- Notebook: 1

**Next steps:**
- Add JAX images
- Add AWS/GCP Deep Learning Containers
- Add Jupyter notebook stacks
- Populate security scan data via CI automation
