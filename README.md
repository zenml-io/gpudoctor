# GPU Doctor

[![maintained-by-zenml](assets/maintained-by-zenml.png)](https://github.com/zenml-io/zenml)

A curated catalog of ML Docker base images to help engineers escape "CUDA hell" and choose the right container for their GPU workloads.

## The Problem

Selecting a Docker base image for ML is painful:
- CUDA/cuDNN version matrices are confusing
- Image sizes range from 1GB to 20GB+ with unclear trade-offs
- Documentation is scattered across NVIDIA, cloud providers, and framework maintainers
- Security scan data is hard to find

## The Solution

GPU Doctor provides:
- **Structured catalog** of vetted ML images (PyTorch, TensorFlow, vLLM, TGI, Triton, etc.)
- **Consistent metadata** including CUDA versions, driver requirements, sizes, and security ratings
- **Guided picker** to find the right image for your use case
- **Searchable table** with filtering by framework, role, and cloud affinity

## Quick Start

```bash
# Browse the catalog
cat data/images.json | jq '.images[] | {id, name, role: .capabilities.role}'

# Validate after edits
.venv/bin/python -c "
import json
from jsonschema import validate
with open('data/schema.json') as f: schema = json.load(f)
with open('data/images.json') as f: data = json.load(f)
validate(instance=data, schema=schema)
print('Valid')
"
```

## Project Status

- [x] JSON Schema design
- [x] Initial catalog (15 images)
- [x] Website with guided picker (5-step wizard) and table view
- [ ] CI automation for metadata/security scans
- [ ] Claude skill for recommendations

## Website

The website is a Next.js app in `web/`:

```bash
cd web
yarn install
yarn dev
```

**Pages:**
- `/guide` - 5-step wizard to find the right image based on workload, environment, frameworks, and priorities
- `/table` - Searchable/filterable table of all images
- `/images/[id]` - Detailed view of a single image with specs, security info, and quick-start commands

## License

MIT
