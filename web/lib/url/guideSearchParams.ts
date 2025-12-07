import type {
  CloudProviderAffinity,
  ImageRole
} from '@/lib/types/images';

/**
 * Workload options exposed in the guide UI.
 * These are mapped onto the underlying schema workloads in the filters module.
 */
export type GuideWorkload =
  | 'computer-vision'
  | 'llm-train'
  | 'llm-inference'
  | 'multimodal'
  | 'classical-ml'
  | 'general';

export type GuideCloud = CloudProviderAffinity;
export type GuideRole = ImageRole;

export type GuideGpuPreference = 'any' | 'gpu-required' | 'cpu-only';
export type GuideLicensePreference = 'any' | 'oss-only';
export type GuideCloudSpecificity = 'either' | 'portable' | 'cloud-optimized';
export type SecurityRating = 'A' | 'B' | 'C' | 'D' | 'F';
export type GuidePriorityKey =
  | 'security'
  | 'size'
  | 'license'
  | 'gpu'
  | 'cloud'
  | 'freshness';
export type GuidePriorities = GuidePriorityKey[];

export interface GuideState {
  /**
   * Primary workload the user is targeting, or null if not yet selected.
   */
  workload: GuideWorkload | null;
  /**
   * Preferred ML frameworks (normalized to lowercase framework names).
   */
  frameworks: string[];
  /**
   * Preferred cloud providers where the image should run well.
   */
  clouds: GuideCloud[];
  /**
   * Desired primary image role (training, serving, notebook, base), or null for any.
   */
  role: GuideRole | null;
  /**
   * High-level GPU requirement; defaults to 'any' when the user is flexible.
   */
  gpuPreference: GuideGpuPreference;
  /**
   * Whether the user prefers portable images vs cloud-optimized ones.
   */
  cloudSpecificity: GuideCloudSpecificity;
  /**
   * Open-source vs proprietary license preference; 'any' for no preference.
   */
  licensePreference: GuideLicensePreference;
  /**
   * Minimum acceptable security rating, or null if security is not a constraint.
   */
  minSecurityRating: SecurityRating | null;
  /**
   * Target Python version (e.g. "3.11"), or null if any version is acceptable.
   */
  pythonVersion: string | null;
  /**
   * Ordered list of ranking priorities (first entry has highest weight).
   */
  priorities: GuidePriorities;
}

/**
 * Small subset of URLSearchParams used by the parser.
 * Accepts both browser URLSearchParams and Next.js ReadonlyURLSearchParams.
 */
export type SearchParamsLike = Pick<URLSearchParams, 'get'>;

const WORKLOAD_VALUES: GuideWorkload[] = [
  'computer-vision',
  'llm-train',
  'llm-inference',
  'multimodal',
  'classical-ml',
  'general'
];

const ROLE_VALUES: GuideRole[] = [
  'base',
  'training',
  'inference',
  'notebook',
  'serving'
];

const CLOUD_VALUES: GuideCloud[] = ['aws', 'gcp', 'azure', 'any'];

const GPU_VALUES: GuideGpuPreference[] = ['any', 'gpu-required', 'cpu-only'];
const LICENSE_VALUES: GuideLicensePreference[] = ['any', 'oss-only'];
const CLOUD_SPEC_VALUES: GuideCloudSpecificity[] = [
  'either',
  'portable',
  'cloud-optimized'
];
const PRIORITY_VALUES: GuidePriorityKey[] = [
  'security',
  'gpu',
  'cloud',
  'size',
  'license',
  'freshness'
];
const DEFAULT_PRIORITIES: GuidePriorities = [
  'security',
  'gpu',
  'cloud',
  'size',
  'license',
  'freshness'
];
const SECURITY_VALUES: SecurityRating[] = ['A', 'B', 'C', 'D', 'F'];

export const EMPTY_GUIDE_STATE: GuideState = Object.freeze({
  workload: null,
  frameworks: [],
  clouds: [],
  role: null,
  gpuPreference: 'any',
  cloudSpecificity: 'either',
  licensePreference: 'any',
  minSecurityRating: null,
  pythonVersion: null,
  priorities: DEFAULT_PRIORITIES
});

/**
 * Parses the current URL search params into a normalized GuideState.
 * Unknown or malformed values are ignored rather than causing errors.
 */
export function parseGuideState(searchParams: SearchParamsLike): GuideState {
  const wkParam = searchParams.get('wk');
  const workload = WORKLOAD_VALUES.includes(wkParam as GuideWorkload)
    ? (wkParam as GuideWorkload)
    : null;

  const frameworksParam = searchParams.get('fw');
  const frameworks =
    frameworksParam && frameworksParam.trim().length > 0
      ? Array.from(
          new Set(
            frameworksParam
              .split(',')
              .map((fw) => fw.trim().toLowerCase())
              .filter(Boolean)
          )
        )
      : [];

  const cloudsParam = searchParams.get('cl');
  const cloudTokens =
    cloudsParam && cloudsParam.trim().length > 0
      ? cloudsParam
          .split(',')
          .map((cl) => cl.trim().toLowerCase())
          .filter(Boolean)
      : [];
  const clouds = Array.from(
    new Set(
      cloudTokens.filter((cl): cl is GuideCloud =>
        CLOUD_VALUES.includes(cl as GuideCloud)
      )
    )
  );

  const roleParam = searchParams.get('role');
  const role = ROLE_VALUES.includes(roleParam as GuideRole)
    ? (roleParam as GuideRole)
    : null;

  const gpuParam = searchParams.get('gpu');
  const gpuToken =
    gpuParam && gpuParam.trim().length > 0
      ? gpuParam.trim().toLowerCase()
      : '';
  const gpuPreference = GPU_VALUES.includes(gpuToken as GuideGpuPreference)
    ? (gpuToken as GuideGpuPreference)
    : 'any';

  const licenseParam = searchParams.get('lic');
  const licenseToken =
    licenseParam && licenseParam.trim().length > 0
      ? licenseParam.trim().toLowerCase()
      : '';
  const licensePreference = LICENSE_VALUES.includes(
    licenseToken as GuideLicensePreference
  )
    ? (licenseToken as GuideLicensePreference)
    : 'any';

  const cloudSpecParam = searchParams.get('cs');
  const cloudSpecToken =
    cloudSpecParam && cloudSpecParam.trim().length > 0
      ? cloudSpecParam.trim().toLowerCase()
      : '';
  const cloudSpecificity = CLOUD_SPEC_VALUES.includes(
    cloudSpecToken as GuideCloudSpecificity
  )
    ? (cloudSpecToken as GuideCloudSpecificity)
    : 'either';

  const secParam = searchParams.get('sec');
  const secToken =
    secParam && secParam.trim().length > 0
      ? secParam.trim().toUpperCase()
      : '';
  const minSecurityRating = SECURITY_VALUES.includes(
    secToken as SecurityRating
  )
    ? (secToken as SecurityRating)
    : null;

  const pyParam = searchParams.get('py');
  const pythonVersion =
    pyParam && pyParam.trim().length > 0 ? pyParam.trim() : null;

  const prioParam = searchParams.get('prio');
  let priorities: GuidePriorities;
  if (prioParam && prioParam.trim().length > 0) {
    const tokens = prioParam
      .split(',')
      .map((token) => token.trim().toLowerCase())
      .filter(Boolean);

    const base = tokens.filter(
      (token): token is GuidePriorityKey =>
        PRIORITY_VALUES.includes(token as GuidePriorityKey)
    );

    const uniqueBase = Array.from(new Set(base));

    if (uniqueBase.length === 0) {
      priorities = [...DEFAULT_PRIORITIES];
    } else {
      priorities = [...uniqueBase];
      for (const key of DEFAULT_PRIORITIES) {
        if (!priorities.includes(key)) {
          priorities.push(key);
        }
      }
    }
  } else {
    priorities = [...DEFAULT_PRIORITIES];
  }

  return {
    workload,
    frameworks,
    clouds,
    role,
    gpuPreference,
    cloudSpecificity,
    licensePreference,
    minSecurityRating,
    pythonVersion,
    priorities
  };
}

/**
 * Serializes a GuideState into a compact query string using short keys:
 *  • wk   – workload
 *  • fw   – frameworks (comma-separated)
 *  • cl   – clouds (comma-separated)
 *  • role – image role
 *  • gpu  – GPU preference
 *  • lic  – license preference
 *  • cs   – cloud specificity
 *  • sec  – minimum security rating
 *  • py   – Python version
 *  • prio – ranking priorities (comma-separated)
 */
export function serializeGuideState(state: GuideState): string {
  const params = new URLSearchParams();

  if (state.workload) {
    params.set('wk', state.workload);
  }
  if (state.frameworks.length > 0) {
    params.set('fw', state.frameworks.join(','));
  }
  if (state.clouds.length > 0) {
    params.set('cl', state.clouds.join(','));
  }
  if (state.role) {
    params.set('role', state.role);
  }
  if (state.gpuPreference !== 'any') {
    params.set('gpu', state.gpuPreference);
  }
  if (state.licensePreference !== 'any') {
    params.set('lic', state.licensePreference);
  }
  if (state.cloudSpecificity !== 'either') {
    params.set('cs', state.cloudSpecificity);
  }
  if (state.minSecurityRating) {
    params.set('sec', state.minSecurityRating);
  }
  if (state.pythonVersion && state.pythonVersion.trim().length > 0) {
    params.set('py', state.pythonVersion.trim());
  }
  if (!prioritiesEqual(state.priorities, DEFAULT_PRIORITIES)) {
    params.set('prio', state.priorities.join(','));
  }

  return params.toString();
}

/**
 * Returns true when the user has not provided any guide inputs yet.
 */
export function isGuideStateEmpty(state: GuideState): boolean {
  const empty = EMPTY_GUIDE_STATE;

  return (
    state.workload === empty.workload &&
    state.role === empty.role &&
    state.frameworks.length === 0 &&
    state.clouds.length === 0 &&
    state.gpuPreference === empty.gpuPreference &&
    state.cloudSpecificity === empty.cloudSpecificity &&
    state.licensePreference === empty.licensePreference &&
    state.minSecurityRating === empty.minSecurityRating &&
    state.pythonVersion === empty.pythonVersion &&
    prioritiesEqual(state.priorities, empty.priorities)
  );
}

/**
 * Compares two priority lists for strict order and membership equality.
 */
export function prioritiesEqual(
  a: GuidePriorities,
  b: GuidePriorities
): boolean {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i += 1) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
}

/**
 * Shallow clone helper so callers can avoid mutating GuideState objects in place.
 */
export function cloneGuideState(state: GuideState): GuideState {
  return {
    workload: state.workload,
    frameworks: [...state.frameworks],
    clouds: [...state.clouds],
    role: state.role,
    gpuPreference: state.gpuPreference,
    cloudSpecificity: state.cloudSpecificity,
    licensePreference: state.licensePreference,
    minSecurityRating: state.minSecurityRating,
    pythonVersion: state.pythonVersion,
    priorities: [...state.priorities]
  };
}