import Fuse from 'fuse.js';

import type {
  ImageEntry,
  ImageProvider,
  MaintenanceStatus,
  Workload,
  ImageRole,
  ImageType,
  CpuArchitecture
} from '@/lib/types/images';
import type {
  TableState,
  TableSortBy
} from '@/lib/url/tableSearchParams';

/**
 * Applies text search (via Fuse.js), structured filters, and sorting to the
 * full image list based on the current TableState.
 */
export function filterTableImages(
  images: ImageEntry[],
  state: TableState
): ImageEntry[] {
  if (images.length === 0) {
    return [];
  }

  const baseImages: ImageEntry[] =
    state.ids.length > 0
      ? (() => {
          const allowed = new Set(state.ids);
          return images.filter((image) => allowed.has(image.id));
        })()
      : images;

  if (baseImages.length === 0) {
    return [];
  }

  let working: ImageEntry[] = baseImages;

  // Text search using Fuse.js when a query is present.
  if (state.query.trim().length > 0) {
    const fuse = new Fuse(baseImages, {
      keys: [
        'name',
        'metadata.provider',
        'frameworks.name',
        'frameworks.version',
        'capabilities.workloads',
        'urls.registry',
        'recommended_for'
      ],
      threshold: 0.35,
      ignoreLocation: true,
      minMatchCharLength: 2
    });

    const results = fuse.search(state.query.trim());
    working = results.map((result) => result.item);
  }

  // Structured filters.
  if (state.frameworks.length > 0) {
    const desired = new Set(state.frameworks.map((fw) => fw.toLowerCase()));
    working = working.filter((image) => {
      const imageFrameworks = image.frameworks.map((f) =>
        f.name.toLowerCase()
      );
      return imageFrameworks.some((fw) => desired.has(fw));
    });
  }

  if (state.providers.length > 0) {
    const desired = new Set<ImageProvider>(state.providers);
    working = working.filter((image) =>
      desired.has(image.metadata.provider)
    );
  }

  if (state.workloads.length > 0) {
    const desired = new Set<Workload>(state.workloads);
    working = working.filter((image) =>
      image.capabilities.workloads.some((wk) => desired.has(wk))
    );
  }

  if (state.status.length > 0) {
    const desired = new Set<MaintenanceStatus>(state.status);
    working = working.filter((image) =>
      desired.has(image.metadata.maintenance)
    );
  }

  if (state.cudaVersions.length > 0) {
    const desired = new Set(state.cudaVersions);
    working = working.filter((image) => {
      const version = image.cuda?.version;
      if (!version) {
        return false;
      }
      return desired.has(version);
    });
  }

  // Sorting.
  const sorted = [...working].sort((a, b) => {
    const compare = compareByColumn(a, b, state.sortBy);
    return state.sortDir === 'asc' ? compare : -compare;
  });

  return sorted;
}

function compareByColumn(
  a: ImageEntry,
  b: ImageEntry,
  sortBy: TableSortBy
): number {
  switch (sortBy) {
    case 'provider':
      return a.metadata.provider.localeCompare(b.metadata.provider);
    case 'cuda': {
      const va = a.cuda?.version ?? '';
      const vb = b.cuda?.version ?? '';
      if (va === vb) {
        return fallbackNameCompare(a, b);
      }
      return va.localeCompare(vb, undefined, { numeric: true });
    }
    case 'python': {
      const va = a.runtime.python ?? '';
      const vb = b.runtime.python ?? '';
      if (va === vb) {
        return fallbackNameCompare(a, b);
      }
      return va.localeCompare(vb, undefined, { numeric: true });
    }
    case 'status': {
      const sa = maintenanceRank(a.metadata.maintenance);
      const sb = maintenanceRank(b.metadata.maintenance);
      if (sa === sb) {
        return fallbackNameCompare(a, b);
      }
      return sa - sb;
    }
    case 'role': {
      const ra = roleRank(a.capabilities.role);
      const rb = roleRank(b.capabilities.role);
      if (ra === rb) {
        return fallbackNameCompare(a, b);
      }
      return ra - rb;
    }
    case 'imageType': {
      const ta = imageTypeRank(a.capabilities.image_type);
      const tb = imageTypeRank(b.capabilities.image_type);
      if (ta === tb) {
        return fallbackNameCompare(a, b);
      }
      return ta - tb;
    }
    case 'os': {
      const na = a.runtime.os.name;
      const nb = b.runtime.os.name;
      const nameCompare = na.localeCompare(nb);
      if (nameCompare !== 0) {
        return nameCompare;
      }

      const va = a.runtime.os.version ?? '';
      const vb = b.runtime.os.version ?? '';
      if (va === vb) {
        return fallbackNameCompare(a, b);
      }
      return va.localeCompare(vb, undefined, { numeric: true });
    }
    case 'arch': {
      const ka = architectureKey(a.runtime.architectures);
      const kb = architectureKey(b.runtime.architectures);
      if (ka === kb) {
        return fallbackNameCompare(a, b);
      }
      return ka.localeCompare(kb);
    }
    case 'size': {
      const sa = a.size?.compressed_mb ?? 0;
      const sb = b.size?.compressed_mb ?? 0;
      if (sa === sb) {
        return fallbackNameCompare(a, b);
      }
      return sa - sb;
    }
    case 'name':
    default:
      return fallbackNameCompare(a, b);
  }
}

function fallbackNameCompare(a: ImageEntry, b: ImageEntry): number {
  const providerCompare = a.metadata.provider.localeCompare(
    b.metadata.provider
  );
  if (providerCompare !== 0) {
    return providerCompare;
  }
  return a.name.localeCompare(b.name);
}

function maintenanceRank(status: MaintenanceStatus): number {
  // Lower is "better" so that Active appears before Deprecated/EOL.
  switch (status) {
    case 'active':
      return 0;
    case 'deprecated':
      return 1;
    case 'end-of-life':
      return 2;
    default:
      return 3;
  }
}

function roleRank(role: ImageRole): number {
  switch (role) {
    case 'base':
      return 0;
    case 'training':
      return 1;
    case 'inference':
      return 2;
    case 'notebook':
      return 3;
    case 'serving':
      return 4;
    default:
      return 5;
  }
}

function imageTypeRank(t: ImageType): number {
  switch (t) {
    case 'base':
      return 0;
    case 'runtime':
      return 1;
    case 'devel':
      return 2;
    default:
      return 3;
  }
}

function architectureKey(archs: CpuArchitecture[]): string {
  if (!archs || archs.length === 0) {
    return '';
  }
  return [...archs].sort().join(',');
}