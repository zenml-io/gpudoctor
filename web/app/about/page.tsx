import type { Metadata } from 'next';

import { PageShell } from '@/components/layout/PageShell';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Button } from '@/components/ui/Button';

export const metadata: Metadata = {
  title: 'About',
  description:
    'Why ML teams struggle with CUDA compatibility, 10–20GB Docker images, fragmented image catalogs and security debt — and how GPU Doctor helps them choose the right container.'
};

export default function AboutPage() {
  return (
    <PageShell activeTab={null}>
      <div className="space-y-16 md:space-y-20 lg:space-y-24">
        <HeroSection />
        <ProblemSection />
        <CompatibilityInfographic />
        <SolutionSection />
        <SolutionFlowInfographic />
        <WhyZenMLCaresSection />
      </div>
    </PageShell>
  );
}

function HeroSection() {
  return (
    <section className="relative overflow-hidden rounded-2xl border border-neutral-200 bg-[var(--gradient-subtle)] px-6 py-12 sm:px-10 sm:py-16">
      <div className="relative max-w-3xl space-y-6">
        <p className="text-xs font-semibold uppercase tracking-[0.2em] text-primary-600">
          ML containers, under the microscope
        </p>
        <h1 className="text-3xl font-semibold tracking-tight text-neutral-900 sm:text-4xl lg:text-5xl">
          The Hidden Crisis in ML Infrastructure
        </h1>
        <p className="text-base text-neutral-700 sm:text-lg">
          CUDA compatibility hell. 10–20GB bloated images. Fragmented discovery
          scattered across NGC, Docker Hub, AWS DLC and GCP containers. Vendor
          bases shipping with dozens of unpatched CVEs. ML teams are burning
          weeks on container decisions before they can even train a model.
          GPU Doctor exists because this invisible tax on GPU work has gone on
          for far too long.
        </p>
      </div>

      {/* Subtle purple gradient accent blob to give the hero a magazine-style backdrop without overwhelming the content */}
      <div className="pointer-events-none absolute -right-24 -top-24 hidden h-80 w-80 rounded-full bg-gradient-to-br from-primary-200/70 via-primary-400/40 to-magenta-300/40 opacity-80 blur-3xl sm:block" />
      <div className="pointer-events-none absolute -bottom-24 right-0 hidden h-64 w-64 rounded-full bg-gradient-to-tr from-primary-100/60 via-primary-300/30 to-magenta-200/40 opacity-70 blur-3xl md:block" />
    </section>
  );
}

function ProblemSection() {
  return (
    <section className="space-y-6">
      <div className="max-w-2xl space-y-3">
        <h2 className="text-2xl font-semibold tracking-tight text-neutral-900 sm:text-3xl">
          The problem everyone feels, but nobody owns
        </h2>
        <p className="text-sm text-neutral-600 sm:text-base">
          Containers were supposed to make ML environments boring. Instead,
          teams are trapped in a maze of CUDA matrices, oversized images,
          scattered catalogs and quiet security landmines. The result is a
          constant background hum of build failures, mysterious GPU errors and
          slow, expensive deployments.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card padding="lg" className="flex flex-col gap-3">
          <Badge variant="purple" size="sm">
            CUDA compatibility hell
          </Badge>
          <h3 className="text-lg font-semibold text-neutral-900">
            A six-dimensional guessing game
          </h3>
          <p className="text-sm text-neutral-600">
            Real GPU containers live inside a 6‑dimensional dependency matrix:
            host drivers, CUDA toolkit, cuDNN, framework version, Python
            version and GPU architecture. Get any edge of that cube wrong and
            you are rewarded with errors like{' '}
            <code>CUDA_ERROR_UNSUPPORTED_PTX_VERSION</code>, mismatched
            <code>torch.cuda.is_available()</code> signals or, worse, silent
            numerical failures that only surface in production.
          </p>
        </Card>

        <Card padding="lg" className="flex flex-col gap-3">
          <Badge variant="purple" size="sm">
            Image bloat
          </Badge>
          <h3 className="text-lg font-semibold text-neutral-900">
            10–20GB containers as the new normal
          </h3>
          <p className="text-sm text-neutral-600">
            ML images have quietly exploded in size. It is now routine to see
            10–20GB containers for workloads where the actual models fit in a
            few hundred megabytes. A single PyTorch GPU install can weigh in at
            roughly 2.8GB. Every pull slows CI, drags out blue/green deploys
            and quietly drives up cloud storage and egress costs.
          </p>
        </Card>

        <Card padding="lg" className="flex flex-col gap-3">
          <Badge variant="purple" size="sm">
            Fragmented discovery
          </Badge>
          <h3 className="text-lg font-semibold text-neutral-900">
            Five catalogs, zero unified view
          </h3>
          <p className="text-sm text-neutral-600">
            NVIDIA NGC, Docker Hub, AWS Deep Learning Containers, GCP Deep
            Learning images, official framework images and abandoned community
            images all compete for attention. Each has its own opaque tag
            schema, release cadence and fine print. There is no single place to
            compare trade‑offs or answer a basic question: "Which image should I
            actually use?"
          </p>
        </Card>

        <Card padding="lg" className="flex flex-col gap-3">
          <Badge variant="purple" size="sm">
            Security debt
          </Badge>
          <h3 className="text-lg font-semibold text-neutral-900">
            Bloated bases, endless CVEs
          </h3>
          <p className="text-sm text-neutral-600">
            Vendor and framework images ship as full operating systems with SSH,
            systemd, unused daemons and years of accumulated packages. Security
            scanners routinely find dozens or hundreds of CVEs in "official"
            images. The larger the base, the bigger the attack surface — and the
            harder it is for ML teams to argue that shipping models should also
            mean owning Linux patch management.
          </p>
        </Card>
      </div>
    </section>
  );
}

function CompatibilityInfographic() {
  return (
    <section
      aria-labelledby="compatibility-matrix-title"
      className="space-y-4"
    >
      <div className="flex flex-col gap-2 sm:flex-row sm:items-baseline sm:justify-between">
        <div>
          <h2
            id="compatibility-matrix-title"
            className="text-sm font-semibold uppercase tracking-[0.2em] text-neutral-500"
          >
            Infographic
          </h2>
          <p className="mt-1 text-base font-medium text-neutral-900">
            The Compatibility Matrix Nightmare
          </p>
        </div>

      </div>

      <div className="overflow-hidden rounded-xl border border-dashed border-primary-200 bg-white">
        <img
          src="/about/compatibility-matrix.svg"
          alt="Placeholder diagram illustrating the dependency chain between GPU drivers, CUDA, cuDNN, frameworks, Python versions and GPU architectures."
          className="w-full"
        />
      </div>
    </section>
  );
}

function SolutionSection() {
  return (
    <section className="space-y-8">
      <div className="max-w-2xl space-y-3">
        <h2 className="text-2xl font-semibold tracking-tight text-neutral-900 sm:text-3xl">
          What GPU Doctor does about it
        </h2>
        <p className="text-sm text-neutral-600 sm:text-base">
          GPU Doctor is a curated, opinionated layer on top of the messy
          container ecosystem. It does not invent a new base image format — it
          reads the fine print across vendors, frameworks and clouds and gives
          ML engineers a single, trustworthy interface for choosing a base
          image.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        <Card padding="lg" className="flex flex-col gap-3">
          <h3 className="text-sm font-semibold text-neutral-900 sm:text-base">
            Curated catalog of ML Docker images
          </h3>
          <p className="text-sm text-neutral-600">
            We ingest and normalize GPU‑aware images from NVIDIA NGC, official
            framework repos, cloud providers and carefully selected community
            projects. Each image is enriched with structured metadata — CUDA,
            cuDNN, framework, Python, architecture, size and lifecycle status —
            so you can compare options side by side instead of flipping between
            half a dozen docs pages.
          </p>
        </Card>

        <Card padding="lg" className="flex flex-col gap-3">
          <h3 className="text-sm font-semibold text-neutral-900 sm:text-base">
            5‑step guided picker for the &ldquo;just tell me what to use&rdquo;
            moment
          </h3>
          <p className="text-sm text-neutral-600">
            The guided picker walks you through five sharp questions about your
            workload: training vs. inference, framework, CUDA expectations,
            environment (cloud / on‑prem) and priorities like size vs.
            convenience. The result is a short, defensible list of images that
            actually fit your constraints — not another generic recommendation
            to &ldquo;start from the latest NGC PyTorch image.&rdquo;
          </p>
        </Card>

        <Card padding="lg" className="flex flex-col gap-3">
          <h3 className="text-sm font-semibold text-neutral-900 sm:text-base">
            Searchable, filterable table for power users
          </h3>
          <p className="text-sm text-neutral-600">
            When you need full control, the table view exposes the entire
            catalog with rich filters for provider, framework, CUDA, Python,
            image family, size, status and more. It is designed like a data
            tool, not a marketing site — keyboard‑friendly, dense but readable,
            and built for engineers who want to slice the catalog in their own
            way.
          </p>
        </Card>

        <Card padding="lg" className="flex flex-col gap-3">
          <h3 className="text-sm font-semibold text-neutral-900 sm:text-base">
            Clear compatibility and security signals
          </h3>
          <p className="text-sm text-neutral-600">
            Each image comes with human‑readable compatibility context: minimum
            driver versions, CUDA runtime vs. devel capabilities, JIT kernel
            support expectations and basic security posture. Instead of
            discovering incompatibilities at runtime, GPU Doctor pushes those
            constraints to the surface at selection time — before you burn hours
            on a broken base.
          </p>
        </Card>
      </div>
    </section>
  );
}

function SolutionFlowInfographic() {
  return (
    <section aria-labelledby="solution-flow-title" className="space-y-4">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-baseline sm:justify-between">
        <div>
          <h2
            id="solution-flow-title"
            className="text-sm font-semibold uppercase tracking-[0.2em] text-neutral-500"
          >
            Infographic
          </h2>
          <p className="mt-1 text-base font-medium text-neutral-900">
            How GPU Doctor turns confusion into clarity
          </p>
        </div>

      </div>

      <div className="overflow-hidden rounded-xl border border-dashed border-primary-200 bg-white">
        <img
          src="/about/solution-flow.svg"
          alt="Placeholder flow diagram showing how GPU Doctor turns scattered documentation and guesswork into a confident base image decision."
          className="w-full"
        />
      </div>
    </section>
  );
}

function WhyZenMLCaresSection() {
  return (
    <section className="space-y-8 border-t border-neutral-200 pt-10">
      <div className="max-w-3xl space-y-4">
        <h2 className="text-2xl font-semibold tracking-tight text-neutral-900 sm:text-3xl">
          Why ZenML cares so much about your base image
        </h2>
        <p className="text-sm text-neutral-600 sm:text-base">
          ZenML builds MLOps infrastructure: production‑grade pipelines, stacks
          and reproducible ML workflows that need to run the same way on a
          laptop, in a CI job and on a GPU cluster. Every one of those stacks
          sits on top of a container image. When that base is wrong, the entire
          workflow shakes — from broken builds to failed orchestrator jobs to
          mysterious GPU errors that only show up under load.
        </p>
        <p className="text-sm text-neutral-600 sm:text-base">
          We kept seeing the same pattern: teams could design beautiful
          pipelines but still got blocked on &ldquo;which image do we even
          start from?&rdquo; GPU Doctor is our response — a free, vendor‑neutral
          tool that encodes everything we have learned from helping teams ship
          ML stacks in the real world.
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)] md:items-start">
        <Card padding="lg" className="space-y-3">
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-neutral-500">
            Made by ZenML
          </p>
          <p className="text-sm text-neutral-700 sm:text-base">
            GPU Doctor is free to use because we want the base image decision to
            stop being a hidden tax on every ML project. If you just need to
            pick a sane, compatible container for your next GPU workload, use
            GPU Doctor and move on with your life.
          </p>
          <p className="text-sm text-neutral-700 sm:text-base">
            If you want to wire that decision into a full MLOps stack — with
            pipelines, experiment tracking, artifact stores and production
            deployments — that&apos;s where ZenML comes in. We take the same
            obsession with clarity and reproducibility and apply it to the rest
            of your ML workflow.
          </p>
        </Card>

        <Card padding="lg" className="flex flex-col gap-4">
          <div className="space-y-1">
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-neutral-500">
              See ZenML in action
            </p>
            <p className="text-sm text-neutral-700">
              Talk to the ZenML team about how to turn GPU Doctor from a handy
              lookup tool into a fully integrated part of your production ML
              platform.
            </p>
          </div>
          <div>
            <a
              href="https://www.zenml.io/book-a-demo"
              target="_blank"
              rel="noreferrer"
              className="inline-flex"
            >
              <Button variant="primary" size="lg">
                Book a ZenML demo
              </Button>
            </a>
          </div>
          <p className="text-xs text-neutral-500">
            GPU Doctor is free to use. If you want help wiring it into a real
            MLOps stack, that&apos;s where ZenML comes in.
          </p>
        </Card>
      </div>
    </section>
  );
}