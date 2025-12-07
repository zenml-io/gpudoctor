# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU Doctor is a curated catalog of ML Docker base images to help engineers select the right container for their GPU workloads. The project consists of:

1. **Data layer**: JSON Schema + catalog of ML Docker images (`data/`)
2. **Website** (planned): SPA with guided picker and searchable table
3. **CI automation** (planned): Auto-populate image metadata and security scans
4. **Claude skill** (planned): Recommendations based on the catalog

## Key Files

- `data/schema.json` - JSON Schema 2020-12 defining image catalog structure
- `data/images.json` - The actual image catalog (validates against schema)
- `design/plan.md` - High-level roadmap and field specifications
- `changelog.md` - Session notes tracking work done (update this when you get done with a task block)

## Commands

### Validate data against schema
```bash
.venv/bin/python -c "
import json
from jsonschema import validate, Draft202012Validator
with open('data/schema.json') as f: schema = json.load(f)
with open('data/images.json') as f: data = json.load(f)
validate(instance=data, schema=schema)
print(f'Valid: {len(data[\"images\"])} images')
"
```

### Install dependencies
```bash
uv pip install jsonschema
```

## Data Schema Architecture

Images use a nested structure with these top-level groups:
- `metadata`: provider, registry, status, maintenance, license
- `cuda`: version, cudnn, min_driver, compute_capabilities (null for CPU-only)
- `runtime`: python version, os, architectures
- `frameworks`: array of {name, version} objects
- `capabilities`: gpu_vendors, image_type (base/runtime/devel), role, workloads
- `cloud`: affinity, exclusive_to, cloud-specific IDs (optional)
- `security`: CVE counts by severity, rating, scan date (populated by automation)
- `size`: compressed_mb, uncompressed_mb
- `urls`: registry, documentation, source

Required fields: `id`, `name`, `metadata`, `runtime`, `capabilities`, `urls`

## Adding New Images

1. Add entry to `data/images.json` following the schema
2. Use kebab-case for `id` (e.g., `pytorch-2-5-1-cuda12-4-runtime`)
3. Set `cuda: null` for CPU-only images
4. Set `security: null` until CI automation populates it
5. Run validation to ensure schema compliance

## Design Docs

The `design/` folder contains research and planning docs (gitignored from commits):
- `plan.md` - Roadmap and schema field specifications
- `web_design_ideas.md` - Website UX specifications
- `problem_validation/` - User research on Docker image selection pain points
