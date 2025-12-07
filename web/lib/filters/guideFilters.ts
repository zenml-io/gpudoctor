import type {
  ImageEntry,
  Workload,
  CloudProviderAffinity,
  SecurityRating
} from '@/lib/types/images';
import type {
  GuideState,
  GuideWorkload,
  GuidePriorityKey
} from '@/lib/url/guideSearchParams';
import { isGuideStateEmpty } from '@/lib/url/guideSearchParams';

interface ScoredImage {
  image: ImageEntry;
  score: number;
}

/**
 * Filters images based on the current GuideState and returns them sorted
 * by match score (best matches first).
 *
 * Filtering rules:
 *  • workload   → image.capabilities.workloads must contain the mapped workload
 *  • frameworks → at least one selected framework must be present (if any selected)
 *  • clouds     → image should work on at least one selected cloud (if any selected)
 *  • role       → image.capabilities.role must match (if selected)
 */
export function filterImages(images: ImageEntry[], state: GuideState): ImageEntry[] {
  if (images.length === 0) {
    return [];
  }

  // When no filters are set, show a stable default ordering.
  if (isGuideStateEmpty(state)) {
    return [...images].sort(defaultSort);
  }

  const filtered = images.filter((image) => {
    if (state.workload && !matchesWorkload(image, state.workload)) {
      return false;
    }
    if (state.frameworks.length > 0 && !matchesFrameworks(image, state.frameworks)) {
      return false;
    }
    if (state.clouds.length > 0 && !matchesClouds(image, state.clouds)) {
      return false;
    }
    if (state.role && image.capabilities.role !== state.role) {
      return false;
    }
    if (!matchesGpuPreference(image, state.gpuPreference)) {
      return false;
    }
    if (!matchesLicense(image, state.licensePreference)) {
      return false;
    }
    if (!matchesSecurityFloor(image, state.minSecurityRating)) {
      return false;
    }
    return true;
  });

  if (filtered.length === 0) {
    // If strict filtering yields nothing, return an empty list rather than
    // silently relaxing constraints. The UI will show an empty state.
    return [];
  }

  const scored: ScoredImage[] = filtered.map((image) => ({
    image,
    score: scoreImage(image, state)
  }));

  return scored
    .sort((a, b) => {
      if (b.score !== a.score) {
        return b.score - a.score;
      }
      return defaultSort(a.image, b.image);
    })
    .map((entry) => entry.image);
}

/**
 * Computes a rough relevance score for an image given the current guide state.
 * Workload and role are weighted more heavily than secondary preferences.
 */
export function scoreImage(image: ImageEntry, state: GuideState): number {
  if (isGuideStateEmpty(state)) {
    return 0;
  }

  let score = 0;

  if (state.workload && matchesWorkload(image, state.workload)) {
    // Inference-focused LLM workloads get slightly higher weight for serving images.
    if (state.workload === 'llm-inference') {
      score += 6;
      if (image.capabilities.role === 'serving' || image.capabilities.role === 'inference') {
        score += 2;
      }
    } else {
      score += 5;
    }
  }

  if (state.role && image.capabilities.role === state.role) {
    score += 4;
  }

  if (state.frameworks.length > 0) {
    const imageFrameworks = image.frameworks.map((f) => f.name.toLowerCase());
    let frameworkMatches = 0;
    for (const fw of state.frameworks) {
      if (imageFrameworks.includes(fw)) {
        frameworkMatches += 1;
      }
    }
    // Reward multiple framework matches, but cap the contribution to avoid dominating other factors.
    if (frameworkMatches > 0) {
      score += Math.min(frameworkMatches * 2, 6);
    }
  }

  if (state.clouds.length > 0 && matchesClouds(image, state.clouds)) {
    // Cloud alignment is useful but less critical than workload and framework.
    score += 2;
  }

  // Security aspect
  const secWeight = getPriorityWeight(state, 'security');
  if (secWeight > 0 && image.security) {
    score += securityRatingScore(image.security.rating) * secWeight;
  }

  // Size aspect (prefer smaller)
  const sizeWeight = getPriorityWeight(state, 'size');
  if (sizeWeight > 0 && image.size?.compressed_mb) {
    const sizeScore = 1 / Math.log10(image.size.compressed_mb + 10);
    score += sizeScore * 5 * sizeWeight;
  }

  // License aspect
  const licWeight = getPriorityWeight(state, 'license');
  if (licWeight > 0 && isOssLicense(image)) {
    score += 2 * licWeight;
  }

  // GPU aspect
  const gpuWeight = getPriorityWeight(state, 'gpu');
  if (gpuWeight > 0 && isGpuImage(image)) {
    score += 2 * gpuWeight;
  }

  // GPU preference nudge
  if (state.gpuPreference === 'cpu-only' && !isGpuImage(image)) {
    score += 1;
  }
  if (state.gpuPreference === 'gpu-required' && isGpuImage(image)) {
    score += 1;
  }

  // Cloud specificity aspect
  const cloudSpecWeight = getPriorityWeight(state, 'cloud');
  if (cloudSpecWeight > 0) {
    const cls = classifyCloudSpecificity(image);
    if (state.cloudSpecificity === 'cloud-optimized' && cls === 'cloud-optimized') {
      score += 2 * cloudSpecWeight;
    } else if (state.cloudSpecificity === 'portable' && cls === 'portable') {
      score += 2 * cloudSpecWeight;
    }
  }

  // Freshness aspect
  const freshWeight = getPriorityWeight(state, 'freshness');
  if (freshWeight > 0) {
    score += freshnessScore(image) * freshWeight;
  }

  // Python version soft preference
  if (state.pythonVersion && image.runtime.python) {
    if (image.runtime.python === state.pythonVersion) {
      score += 3;
    } else if (image.runtime.python.split('.')[0] === state.pythonVersion.split('.')[0]) {
      score += 1;
    }
  }

  return score;
}

function matchesWorkload(image: ImageEntry, workload: GuideWorkload): boolean {
  const mappedWorkloads = mapGuideWorkloadToSchemaWorkloads(workload);
  const hasWorkload = mappedWorkloads.some((wk) =>
    image.capabilities.workloads.includes(wk)
  );

  if (!hasWorkload) {
    return false;
  }

  // For LLM inference, prefer images whose primary role is serving or inference.
  if (workload === 'llm-inference') {
    const role = image.capabilities.role;
    return role === 'serving' || role === 'inference';
  }

  return true;
}

function mapGuideWorkloadToSchemaWorkloads(workload: GuideWorkload): Workload[] {
  switch (workload) {
    case 'computer-vision':
      return ['computer-vision'];
    case 'llm-train':
    case 'llm-inference':
      return ['llm'];
    case 'multimodal':
      return ['multimodal'];
    case 'classical-ml':
      return ['classical-ml'];
    case 'general':
      return ['generic'];
    default:
      return [];
  }
}

function matchesFrameworks(image: ImageEntry, frameworks: string[]): boolean {
  if (frameworks.length === 0) {
    return true;
  }
  const imageFrameworks = image.frameworks.map((f) => f.name.toLowerCase());
  return frameworks.some((fw) => imageFrameworks.includes(fw));
}

function matchesClouds(
  image: ImageEntry,
  preferredClouds: CloudProviderAffinity[]
): boolean {
  if (preferredClouds.length === 0) {
    return true;
  }

  const cloud = image.cloud;

  // If cloud metadata is missing, treat the image as generally portable.
  const supported: CloudProviderAffinity[] = [];
  if (!cloud) {
    supported.push('any');
  } else {
    for (const affinity of cloud.affinity) {
      supported.push(affinity);
    }
    if (cloud.exclusive_to) {
      supported.push(cloud.exclusive_to);
    }
  }

  return preferredClouds.some((desired) => {
    if (desired === 'any') {
      return supported.includes('any');
    }
    // Images marked as "any" are considered compatible with specific clouds as well.
    return supported.includes(desired) || supported.includes('any');
  });
}

function isGpuImage(image: ImageEntry): boolean {
  return image.capabilities.gpu_vendors.some((v) => v !== 'none') || image.cuda !== null;
}

function matchesGpuPreference(
  image: ImageEntry,
  pref: GuideState['gpuPreference']
): boolean {
  switch (pref) {
    case 'gpu-required':
      return isGpuImage(image);
    case 'cpu-only':
      return !isGpuImage(image);
    case 'any':
    default:
      return true;
  }
}

function isOssLicense(image: ImageEntry): boolean {
  const lic = image.metadata.license;
  if (!lic) return false;
  return !lic.toLowerCase().includes('proprietary');
}

function matchesLicense(
  image: ImageEntry,
  pref: GuideState['licensePreference']
): boolean {
  if (pref === 'any') return true;
  return isOssLicense(image);
}

function securityRatingScore(rating: SecurityRating): number {
  switch (rating) {
    case 'A':
      return 5;
    case 'B':
      return 4;
    case 'C':
      return 3;
    case 'D':
      return 2;
    case 'F':
      return 1;
  }
}

function matchesSecurityFloor(
  image: ImageEntry,
  minRating: SecurityRating | null
): boolean {
  if (!minRating) return true;
  const minScore = securityRatingScore(minRating);
  if (!image.security) return false;
  return securityRatingScore(image.security.rating) >= minScore;
}

type CloudSpecificityClass = 'cloud-optimized' | 'portable';

function classifyCloudSpecificity(image: ImageEntry): CloudSpecificityClass {
  const provider = image.metadata.provider;
  const isProviderOptimized =
    provider === 'aws-dlc' || provider === 'gcp-dlc' || provider === 'azure-ml';
  const hasExclusiveCloud = !!image.cloud?.exclusive_to;

  if (isProviderOptimized || hasExclusiveCloud) {
    return 'cloud-optimized';
  }
  return 'portable';
}

function freshnessScore(image: ImageEntry): number {
  const dateStr = image.metadata.last_updated;
  if (!dateStr) return 0;
  const date = new Date(dateStr);
  const now = new Date();
  const days = (now.getTime() - date.getTime()) / (1000 * 60 * 60 * 24);
  if (days <= 90) return 3;
  if (days <= 180) return 2;
  if (days <= 365) return 1;
  return 0.5;
}

function getPriorityWeight(state: GuideState, key: GuidePriorityKey): number {
  const idx = state.priorities.indexOf(key);
  if (idx === -1) return 0;
  if (idx === 0) return 3;
  if (idx === 1) return 2;
  if (idx === 2) return 1;
  return 0.5;
}

function defaultSort(a: ImageEntry, b: ImageEntry): number {
  const providerCompare = a.metadata.provider.localeCompare(b.metadata.provider);
  if (providerCompare !== 0) {
    return providerCompare;
  }
  return a.name.localeCompare(b.name);
}