import type {
  ImageProvider,
  MaintenanceStatus,
  Workload
} from '@/lib/types/images';

export type TableSortBy =
  | 'name'
  | 'provider'
  | 'cuda'
  | 'python'
  | 'status'
  | 'role'
  | 'imageType'
  | 'os'
  | 'arch'
  | 'size';
export type TableSortDir = 'asc' | 'desc';

export interface TableState {
  /**
   * Free-text search query applied via Fuse.js against image name and metadata.
   */
  query: string;
  /**
   * Selected ML frameworks (normalized to lowercase framework names).
   */
  frameworks: string[];
  /**
   * Selected image providers (e.g. pytorch, nvidia-ngc).
   */
  providers: ImageProvider[];
  /**
   * Selected workloads (llm, computer-vision, etc.) from the schema.
   */
  workloads: Workload[];
  /**
   * Selected maintenance statuses (active, deprecated, end-of-life).
   */
  status: MaintenanceStatus[];
  /**
   * Selected CUDA versions (exact matches, e.g. "12.4", "11.8").
   */
  cudaVersions: string[];
  /**
   * Active sort column for the table.
   */
  sortBy: TableSortBy;
  /**
   * Sort direction for the active sort column.
   */
  sortDir: TableSortDir;
  /**
   * Optional hard filter limiting results to a specific set of image IDs.
   * When non-empty, only these images are considered by the table.
   */
  ids: string[];
}

/**
 * Minimal read-only subset of URLSearchParams so this helper can work with
 * both the browser implementation and Next.js ReadonlyURLSearchParams.
 */
export type SearchParamsLike = Pick<URLSearchParams, 'get'>;

export const DEFAULT_TABLE_STATE: TableState = Object.freeze({
  query: '',
  frameworks: [],
  providers: [],
  workloads: [],
  status: [],
  cudaVersions: [],
  sortBy: 'name' as TableSortBy,
  sortDir: 'asc' as TableSortDir,
  ids: []
});

const PROVIDER_VALUES: ImageProvider[] = [
  'nvidia-ngc',
  'aws-dlc',
  'gcp-dlc',
  'azure-ml',
  'pytorch',
  'tensorflow',
  'jax',
  'huggingface',
  'vllm',
  'ollama',
  'jupyter',
  'community'
];

const WORKLOAD_VALUES: Workload[] = [
  'classical-ml',
  'llm',
  'multimodal',
  'computer-vision',
  'nlp',
  'audio',
  'reinforcement-learning',
  'scientific-computing',
  'generic'
];

const STATUS_VALUES: MaintenanceStatus[] = ['active', 'deprecated', 'end-of-life'];

const SORT_BY_VALUES: TableSortBy[] = [
  'name',
  'provider',
  'cuda',
  'python',
  'status',
  'role',
  'imageType',
  'os',
  'arch',
  'size'
];
const SORT_DIR_VALUES: TableSortDir[] = ['asc', 'desc'];

/**
 * Parses URL search params into a normalized TableState. Unknown values are
 * ignored rather than causing errors so stale links remain usable.
 */
export function parseTableState(searchParams: SearchParamsLike): TableState {
  const query = (searchParams.get('q') ?? '').trim();

  const frameworks = parseListParam(searchParams, 'fw').map((fw) =>
    fw.toLowerCase()
  );

  const providerTokens = parseListParam(searchParams, 'prov');
  const providers = providerTokens.filter(
    (token): token is ImageProvider =>
      PROVIDER_VALUES.includes(token as ImageProvider)
  );

  const workloadTokens = parseListParam(searchParams, 'wk');
  const workloads = workloadTokens.filter(
    (token): token is Workload =>
      WORKLOAD_VALUES.includes(token as Workload)
  );

  const statusTokens = parseListParam(searchParams, 'st');
  const status = statusTokens.filter(
    (token): token is MaintenanceStatus =>
      STATUS_VALUES.includes(token as MaintenanceStatus)
  );

  const cudaVersions = parseListParam(searchParams, 'cu');

  const sortByParam = searchParams.get('sort') as TableSortBy | null;
  const sortBy = SORT_BY_VALUES.includes(sortByParam as TableSortBy)
    ? (sortByParam as TableSortBy)
    : DEFAULT_TABLE_STATE.sortBy;

  const sortDirParam = searchParams.get('dir') as TableSortDir | null;
  const sortDir = SORT_DIR_VALUES.includes(sortDirParam as TableSortDir)
    ? (sortDirParam as TableSortDir)
    : DEFAULT_TABLE_STATE.sortDir;

  const ids = parseListParam(searchParams, 'ids');

  return {
    query,
    frameworks,
    providers,
    workloads,
    status,
    cudaVersions,
    sortBy,
    sortDir,
    ids
  };
}

/**
 * Serializes a TableState into a compact query string suitable for use in
 * the /table URL. Only non-default values are included so links stay short.
 */
export function serializeTableState(state: TableState): string {
  const params = new URLSearchParams();

  if (state.query.trim().length > 0) {
    params.set('q', state.query.trim());
  }
  if (state.frameworks.length > 0) {
    params.set('fw', state.frameworks.join(','));
  }
  if (state.providers.length > 0) {
    params.set('prov', state.providers.join(','));
  }
  if (state.workloads.length > 0) {
    params.set('wk', state.workloads.join(','));
  }
  if (state.status.length > 0) {
    params.set('st', state.status.join(','));
  }
  if (state.cudaVersions.length > 0) {
    params.set('cu', state.cudaVersions.join(','));
  }
  if (state.ids.length > 0) {
    params.set('ids', state.ids.join(','));
  }

  if (state.sortBy !== DEFAULT_TABLE_STATE.sortBy) {
    params.set('sort', state.sortBy);
  }
  if (state.sortDir !== DEFAULT_TABLE_STATE.sortDir) {
    params.set('dir', state.sortDir);
  }

  return params.toString();
}

/**
 * Returns true when no filters or query are active (aside from default sort).
 */
export function isTableStateEmpty(state: TableState): boolean {
  return (
    state.query.trim().length === 0 &&
    state.frameworks.length === 0 &&
    state.providers.length === 0 &&
    state.workloads.length === 0 &&
    state.status.length === 0 &&
    state.cudaVersions.length === 0 &&
    state.ids.length === 0
  );
}

function parseListParam(
  searchParams: SearchParamsLike,
  key: string
): string[] {
  const raw = searchParams.get(key);
  if (!raw) {
    return [];
  }
  return Array.from(
    new Set(
      raw
        .split(',')
        .map((value) => value.trim())
        .filter(Boolean)
    )
  );
}