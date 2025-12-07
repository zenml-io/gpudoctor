import { redirect } from 'next/navigation';

/**
 * Root route simply redirects to the Guide view so users land directly
 * in the primary experience.
 */
export default function RootPage() {
  redirect('/guide');
}