import { redirect } from 'next/navigation';

/**
 * Root route redirects to the Table view so users land directly
 * in the searchable catalog experience.
 */
export default function RootPage() {
  redirect('/table');
}