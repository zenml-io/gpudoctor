# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU Doctor is a curated catalog of ML Docker base images to help engineers select the right container for their GPU workloads. The project consists of:

1. **Data layer**: JSON Schema + catalog of ML Docker images (`data/`)
2. **Website**: Next.js 14 static site with 5-step guided picker, searchable table, and image detail pages (`web/`)
3. **CI automation** (planned): Auto-populate image metadata and security scans
4. **Claude skill** (planned): Recommendations based on the catalog

## Commands

### Website development
```bash
cd web
yarn install        # Install dependencies
yarn dev            # Start dev server with hot reload
yarn build          # Production build (static export)
yarn lint           # ESLint check
npx tsc --noEmit    # Type-check without emitting
```

### Validate data against schema
```bash
.venv/bin/python -c "
import json
from jsonschema import validate
with open('data/schema.json') as f: schema = json.load(f)
with open('data/images.json') as f: data = json.load(f)
validate(instance=data, schema=schema)
print(f'Valid: {len(data[\"images\"])} images')
"
```

### Install Python validation dependencies
```bash
uv pip install jsonschema
```

## Architecture

### Static Export
The site uses `output: 'export'` in `next.config.mjs`, generating static HTML at build time. Data is loaded from `../data/images.json` relative to `web/`.

### URL-Driven State (Guide Wizard)
The 5-step guide wizard persists all user selections to URL query params using short keys:
- `wk` (workload), `fw` (frameworks), `cl` (clouds), `role`, `gpu`, `lic` (license), `cs` (cloud specificity), `sec` (security rating), `py` (python), `prio` (priorities)

This enables shareable recommendation URLs and browser back/forward navigation. See:
- `web/lib/url/guideSearchParams.ts` - State type, parsing, and serialization
- `web/lib/filters/guideFilters.ts` - Filtering and scoring logic

### Import Alias
Use `@/` for imports from the `web/` root (configured in `tsconfig.json`):
```typescript
import { ImageEntry } from '@/lib/types/images';
```

### Key Files

**Data:**
- `data/schema.json` - JSON Schema 2020-12 defining image catalog structure
- `data/images.json` - The actual image catalog (validates against schema)

**Website:**
- `web/app/guide/GuideClient.tsx` - Multi-step wizard orchestrating the 5 step components
- `web/app/table/TableClient.tsx` - Searchable/filterable table view
- `web/app/images/[id]/page.tsx` - Image detail page (dynamic route)
- `web/lib/types/images.ts` - TypeScript types mirroring the JSON schema
- `web/components/ui/` - Reusable UI primitives (Button, Card, Badge, etc.)

**Project:**
- `design/plan.md` - High-level roadmap and field specifications (gitignored)
- `changelog.md` - Session notes tracking work done

## Data Schema

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
5. Run schema validation to ensure compliance
