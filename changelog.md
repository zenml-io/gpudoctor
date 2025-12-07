# GPU Doctor Changelog

## 2025-12-07 ~18:15 UTC

### Table View - Expanded Metadata Columns

**Files modified:**
- `web/lib/url/tableSearchParams.ts` - Extended `TableSortBy` with 5 new sortable keys: `role`, `imageType`, `os`, `arch`, `size`
- `web/lib/filters/tableFilters.ts` - Added sorting logic for new columns with rank-based comparators for enum fields
- `web/components/table/DataTable.tsx` - Expanded header row from 6 to 13 columns; added horizontal scroll wrapper
- `web/components/table/DataRow.tsx` - Added 7 new cells with formatting helpers for each metadata field

**New columns (13 total):**
1. Image (sortable) - existing
2. Framework (display-only) - existing
3. Role (sortable) - Training, Inference, Serving, Notebook, Base
4. Type (sortable) - Runtime, Devel, Base
5. CUDA (sortable) - existing
6. cuDNN (display-only) - NEW
7. Python (sortable) - existing
8. OS (sortable) - e.g., "Ubuntu 22.04"
9. Arch (sortable) - e.g., "x86_64" or "x86_64 / ARM64"
10. Size (sortable, right-aligned) - e.g., "2.9 GB"
11. Status (sortable) - existing
12. Provider (sortable) - existing
13. License (display-only) - e.g., "BSD-3-Clause"

**Implementation details:**
- Rank-based sorting for enum fields (role, imageType, status) ensures semantic ordering
- Array fields (architectures) sorted and joined for deterministic comparison
- Null handling: display "—", sort as empty string or 0 for numeric fields
- Table wrapped in `overflow-x-auto` for horizontal scrolling on narrower desktop screens

---

## 2025-12-07 ~16:30 UTC

### Guide Wizard - Multi-Step Navigation UX

**Files modified:**
- `web/components/guide/GuideStepper.tsx` - Step indicators now clickable for backward navigation
- `web/components/guide/PrioritiesStep.tsx` - "Find Images" button scrolls to results
- `web/components/guide/GuideResults.tsx` - Added `id="guide-results"` scroll target
- `web/app/guide/GuideClient.tsx` - Wired up `onStepClick` and `scrollToResults` handlers

**UX improvements:**
- Completed step circles (steps 1-5) are now clickable buttons that allow users to jump back
- Forward navigation is intentionally disabled—users must progress step by step
- The "Find Images" button on step 5 smoothly scrolls to the recommendations section
- Added proper `aria-label` for accessibility on step navigation buttons

---

## 2025-12-07 ~15:00 UTC

### Guide Wizard - 5-Step Flow

**Files created:**
- `web/components/guide/GuideStepper.tsx` - Horizontal progress indicator with step numbers and labels
- `web/components/guide/UseCaseStep.tsx` - Step 1: Select ML workload (CV, LLM training, inference, etc.)
- `web/components/guide/EnvironmentStep.tsx` - Step 2: GPU preference and cloud provider selection
- `web/components/guide/FrameworkRoleStep.tsx` - Step 3: Framework and image role selection
- `web/components/guide/RuntimeComplianceStep.tsx` - Step 4: Python version, license, and security preferences
- `web/components/guide/PrioritiesStep.tsx` - Step 5: Drag-to-reorder priority ranking

**Files modified:**
- `web/app/guide/GuideClient.tsx` - Refactored from two-panel layout to multi-step wizard with `currentStep` state
- `web/components/guide/GuideResults.tsx` - Enhanced summary with GPU, license, Python, security, and priority labels
- `web/lib/url/guideSearchParams.ts` - Extended `GuideState` with new fields (gpuPreference, cloudSpecificity, licensePreference, minSecurityRating, pythonVersion, priorities)

**Architecture:**
- URL-driven state persistence via `parseGuideState`/`serializeGuideState`
- Live filtering: `matchingImages` updates as users progress through steps
- Each step receives only its required props (state down, callbacks up pattern)

---

## 2025-12-07 ~14:00 UTC

### Image Detail Page - Full Metadata Surfacing

**Files modified:**
- `web/components/images/ImageDetailClient.tsx` - Enhanced header badges and new sections
- `web/components/images/ImageSpecs.tsx` - Additional specification rows

**Enhancements:**
- License badge in header
- Last-updated metadata
- Improved size display (compressed with hover showing uncompressed)
- Full recommended_for list (was showing only first item)
- Security CVE breakdown with severity pills (Critical/High/Medium/Low)
- New "Metadata & links" card with external URLs (Registry, Documentation, Source)
- New "System packages & notes" card

---

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
