# Repository Guidelines

## Project Structure & Module Organization
- `data/`: Source of truth for the catalog. Edit `images.json` and keep it consistent with `schema.json`.
- `web/`: Next.js 14 + TypeScript app (app router) that renders the picker, table, and image detail pages. Shared UI lives in `web/components`, data helpers in `web/lib`.
- `design/`: Product notes and UX ideas; helpful context but not shipped code.
- `scripts/`: Catalog automation scripts for updating images from container registries (Docker Hub, GHCR, NGC). Key files:
  - `update_catalog.py`: Main CLI (`--dry-run`, `--source {dockerhub,ghcr,ngc,all}`)
  - `config.py`: Loads `data/tracked_images.yaml`
  - `tag_parsers.py` + `builders.py`: Parse tags and build catalog entries
  - `merge.py`: Safe merge preserving curated fields
  - `fetchers/`: Registry API clients
  - `enrichers/`: Placeholder hooks for future security/CUDA enrichment
  - `audit_catalog.py`: Image audit CLI (uses skopeo/Trivy, writes audit data under `data/.audit`)
  - `audit/`: Helpers (models, cache, tools, parsers) for image audit overlays

## Build, Test, and Development Commands
- Install web deps once: `cd web && npm install`.
- Local dev server: `cd web && npm run dev` (Next dev server with hot reload).
- Production build: `cd web && npm run build`; run `npm start` to smoke-test the built app.
- Lint for style correctness: `cd web && npm run lint`.
- Validate catalog data:
  ```bash
  python -m venv .venv && . .venv/bin/activate
  pip install jsonschema
  python - <<'PY'
  import json
  from jsonschema import validate
  validate(json.load(open("data/images.json")), json.load(open("data/schema.json")))
  print("Valid")
  PY
  ```
- Update catalog from registries:
  ```bash
  pip install -r scripts/requirements.txt
  # Dry-run to preview changes
  python scripts/update_catalog.py --dry-run --source all
  # Apply updates
  python scripts/update_catalog.py --source all
  ```

## Coding Style & Naming Conventions
- TypeScript + React function components; favor small, pure components and props typed with explicit interfaces.
- Use PascalCase for component files (`ImageCard.tsx`), camelCase for variables/functions, SCREAMING_SNAKE_CASE for constants.
- Prefer named exports; keep files cohesive by domain (e.g., guide/table/images features under `web/components` and `web/app`).
- Styling uses Tailwind utility classes plus `clsx`; keep variants and sizes centralized (see `components/ui`).
- Single quotes and semicolons match existing code; rely on `npm run lint` to catch deviations.

## Testing Guidelines
- No automated UI test suite yet; treat `npm run lint` and the JSON schema validation as required pre-flight checks.
- For data changes, include a quick sanity run of the validation snippet above; for UI changes, at least load `/guide`, `/table`, and `/images/[id]` locally to confirm filters and cards render.

## Commit & Pull Request Guidelines
- Commit messages follow short imperative phrases, e.g., `Add initial data`, `Update README.md`.
- For PRs, include: what changed, why, and how to verify (commands or URLs). Attach screenshots/GIFs for UI changes and note any data updates or migrations.
- Reference related issues or TODOs when relevant; keep PRs focused and small where possible.

## Security & Configuration Tips
- Never commit secrets. If environment variables are needed, use `NEXT_PUBLIC_*` only for safe client-side values; keep sensitive keys in local env files not checked in.
- Validate third-party image sources before adding them to the catalog; prefer official vendor registries.
