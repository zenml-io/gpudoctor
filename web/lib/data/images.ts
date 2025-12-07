import fs from 'node:fs';
import path from 'node:path';

import type { ImageCatalog, ImageEntry } from '@/lib/types/images';

/**
 * In-memory cache for the parsed image catalog.
 * Safe to use during static generation since the dataset is small and immutable at runtime.
 */
let cachedCatalog: ImageCatalog | null = null;

/**
 * Returns the full image catalog loaded from ../data/images.json.
 * The result is cached in memory for the lifetime of the Node.js process.
 */
export function getImageCatalog(): ImageCatalog {
  if (cachedCatalog) {
    return cachedCatalog;
  }

  const filePath = path.join(process.cwd(), '..', 'data', 'images.json');
  const raw = fs.readFileSync(filePath, 'utf-8');
  const parsed = JSON.parse(raw) as ImageCatalog;

  cachedCatalog = parsed;
  return parsed;
}

/**
 * Convenience helper returning all image entries in the catalog.
 */
export function getAllImages(): ImageEntry[] {
  return getImageCatalog().images;
}

/**
 * Finds a single image by its id, or undefined if no match is found.
 */
export function getImageById(id: string): ImageEntry | undefined {
  return getAllImages().find((image) => image.id === id);
}