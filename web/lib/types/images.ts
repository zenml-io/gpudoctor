export type ImageStatus = 'official' | 'community' | 'verified';

export type ImageProvider =
  | 'nvidia-ngc'
  | 'aws-dlc'
  | 'gcp-dlc'
  | 'azure-ml'
  | 'pytorch'
  | 'tensorflow'
  | 'jax'
  | 'huggingface'
  | 'vllm'
  | 'ollama'
  | 'jupyter'
  | 'community';

export type ImageRegistry = 'dockerhub' | 'ecr' | 'gcr' | 'ngc' | 'ghcr' | 'mcr';

export type MaintenanceStatus = 'active' | 'deprecated' | 'end-of-life';

export type OsName = 'ubuntu' | 'debian' | 'centos' | 'rhel' | 'alpine' | 'rockylinux';

export type CpuArchitecture = 'amd64' | 'arm64';

export type GpuVendor = 'nvidia' | 'amd' | 'intel' | 'none';

export type ImageType = 'base' | 'runtime' | 'devel';

export type ImageRole = 'base' | 'training' | 'inference' | 'notebook' | 'serving';

export type Workload =
  | 'classical-ml'
  | 'llm'
  | 'multimodal'
  | 'computer-vision'
  | 'nlp'
  | 'audio'
  | 'reinforcement-learning'
  | 'scientific-computing'
  | 'generic';

export type CloudProviderAffinity = 'aws' | 'gcp' | 'azure' | 'any';

export type ExclusiveCloudProvider = 'aws' | 'gcp' | 'azure' | null;

export type SecurityRating = 'A' | 'B' | 'C' | 'D' | 'F';

export type SecurityScanner = 'trivy' | 'grype' | 'snyk' | 'clair';

export interface ImageCatalog {
  /**
   * Reference to the JSON schema used to validate this catalog.
   */
  $schema?: string;
  /**
   * Schema version for the data file (e.g., "0.1.0").
   */
  version: string;
  /**
   * Optional date the catalog was last updated (YYYY-MM-DD).
   */
  last_updated?: string;
  /**
   * Array of Docker image entries.
   */
  images: ImageEntry[];
}

export interface ImageEntry {
  /**
   * Unique identifier for this image (kebab-case).
   */
  id: string;
  /**
   * Full image name with tag, e.g. "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime".
   */
  name: string;
  /**
   * Metadata and provenance information.
   */
  metadata: ImageMetadata;
  /**
   * CUDA configuration, or null for CPU-only images.
   */
  cuda: CudaConfig | null;
  /**
   * Runtime environment configuration (OS, architectures, Python).
   */
  runtime: RuntimeConfig;
  /**
   * Frameworks included in this image.
   */
  frameworks: Framework[];
  /**
   * Capabilities and intended use.
   */
  capabilities: Capabilities;
  /**
   * Cloud provider specific configuration, or null if not cloud-specific.
   */
  cloud: CloudConfig | null;
  /**
   * Security scan results, or null if not yet scanned.
   */
  security: SecurityInfo | null;
  /**
   * Image size information, or null if not measured.
   */
  size: ImageSize | null;
  /**
   * Related URLs (registry, docs, source).
   */
  urls: Urls;
  /**
   * Short descriptions of recommended use cases.
   */
  recommended_for: string[];
  /**
   * Notable system packages included.
   */
  system_packages: string[];
  /**
   * Additional notes or observations.
   */
  notes: string | null;
}

export interface ImageMetadata {
  status: ImageStatus;
  provider: ImageProvider;
  registry: ImageRegistry;
  maintenance: MaintenanceStatus;
  /**
   * Date when this image was last updated (YYYY-MM-DD), if known.
   */
  last_updated?: string | null;
  /**
   * License identifier (e.g., BSD-3-Clause, Apache-2.0, MIT), if known.
   */
  license?: string | null;
}

export interface CudaConfig {
  /**
   * CUDA toolkit version (e.g., 12.4, 11.8).
   */
  version: string;
  /**
   * cuDNN version if included.
   */
  cudnn: string | null;
  /**
   * Minimum required NVIDIA driver version.
   */
  min_driver: string | null;
  /**
   * Supported CUDA compute capabilities (e.g., 7.0, 8.0, 9.0).
   */
  compute_capabilities: string[];
}

export interface RuntimeOs {
  name: OsName;
  /**
   * OS version (e.g., "22.04", "11").
   */
  version: string;
}

export interface RuntimeConfig {
  /**
   * Python version (null for non-Python images).
   */
  python: string | null;
  os: RuntimeOs;
  /**
   * Supported CPU architectures (at least one).
   */
  architectures: CpuArchitecture[];
}

export interface Framework {
  /**
   * Framework name (e.g., pytorch, tensorflow, jax).
   */
  name: string;
  /**
   * Framework version.
   */
  version: string;
}

export interface Capabilities {
  /**
   * Supported GPU vendors (use "none" for CPU-only).
   */
  gpu_vendors: GpuVendor[];
  /**
   * Image type: base (minimal), runtime (libraries only), devel (includes compilers).
   */
  image_type: ImageType;
  /**
   * Primary intended role for this image.
   */
  role: ImageRole;
  /**
   * Types of ML workloads this image is suited for.
   */
  workloads: Workload[];
}

export interface CloudConfig {
  /**
   * Cloud providers this image works well with.
   */
  affinity: CloudProviderAffinity[];
  /**
   * If set, image only works on this cloud provider.
   */
  exclusive_to: ExclusiveCloudProvider;
  aws_ami: string | null;
  gcp_image: string | null;
  azure_image: string | null;
}

export interface SecurityInfo {
  /**
   * Total number of CVEs found.
   */
  total_cves: number;
  /**
   * Number of critical severity CVEs (optional in schema).
   */
  critical?: number;
  /**
   * Number of high severity CVEs (optional in schema).
   */
  high?: number;
  /**
   * Number of medium severity CVEs (optional in schema).
   */
  medium?: number;
  /**
   * Number of low severity CVEs (optional in schema).
   */
  low?: number;
  rating: SecurityRating;
  /**
   * Date of last security scan (YYYY-MM-DD).
   */
  last_scan: string;
  scanner: SecurityScanner;
}

export interface ImageSize {
  /**
   * Compressed image size in megabytes (as stored in registry).
   */
  compressed_mb: number;
  /**
   * Uncompressed image size in megabytes (on disk), if known.
   */
  uncompressed_mb: number | null;
}

export interface Urls {
  /**
   * URL to image on container registry.
   */
  registry: string;
  /**
   * URL to official documentation, if available.
   */
  documentation: string | null;
  /**
   * URL to source repository (Dockerfile, etc.), if available.
   */
  source: string | null;
}