---
name: release-new-version
description: Release a new version of the ai Rust crate from this repository by pulling latest, confirming CI is green, bumping Cargo.toml, building to refresh Cargo.lock, committing, pushing, waiting for commit CI, then pushing a tag that triggers GitHub Actions publishing. Use for explicit invocations like "/release-new-version" or requests like "release a new version", "bump Cargo.toml", or "publish ai".
---

# Release New Version

## Overview

Release the `ai` crate from the `ai.rs` workspace with a pull-first flow, explicit version bump, build-refreshed lockfile, semantic release commit, and a release tag push that triggers automated publishing.

## Preconditions

- Work from the repository root unless the user explicitly directs otherwise.
- Run this release flow from the `main` branch only. If the current branch is not `main`, stop and ask before switching branches.
- Treat the release as live unless the user asks for a dry run.
- Inspect `git status --short --branch` before changing files.
- If the worktree has user changes, do not overwrite them. Ask before mixing release edits into a dirty worktree.
- Pull before editing when the branch tracks a remote: `git pull --ff-only`.
- Keep unrelated untracked files, such as local skill work, out of the release commit.

## Upstream CI Gate

After pulling latest, check that CI for the current `main` commit is green before making release edits.

Use GitHub CLI when available:

```bash
gh run list --branch main --limit 5 --json databaseId,headSha,status,conclusion,workflowName,displayTitle,createdAt,url
```

Confirm the latest CI run for `HEAD` completed with `success`. If the latest CI status cannot be determined or is not green, stop and ask the user whether to run the full local CI-equivalent checks before continuing.

## Version Bump

1. Read the current version from `crates/ai/Cargo.toml`.
2. Inspect tags with `git tag --sort=-version:refname | head` and commits since the latest release tag.
3. Choose the semver bump from commit impact unless the user specified a version.
4. Update `crates/ai/Cargo.toml` package `version`.
5. Build to refresh `Cargo.lock`:

```bash
cargo build --all --locked --verbose
```

If `--locked` fails because the version bump requires a lockfile update, run `cargo build --all --verbose`, verify the lockfile changed only for the package version, then rerun `cargo build --all --locked --verbose`.

6. Verify only the intended release files changed before continuing.

## Validation

When the upstream `HEAD` CI was green and the release edit is only the version bump plus lockfile refresh, the required local build step above is enough before committing.

If upstream CI was not green, could not be confirmed, or the release edit includes anything beyond the version bump and lockfile refresh, ask the user before running the full CI-equivalent checks from `.github/workflows/ci.yaml`:

```bash
cargo fmt --all -- --check
cargo build --all --locked --verbose
cargo test --all --verbose
```

## Publishing

Do not run `cargo publish` manually. This repository publishes from GitHub Actions when a matching release tag is pushed.

The workflow tag patterns are:

```bash
v0.[0-9]+.[0-9]+
v0.[0-9]+.[0-9]+-beta.[0-9]+
v0.[0-9]+.[0-9]+-alpha.[0-9]+
```

## Git Commit And Tag

After the build and any requested validation pass:

1. Commit the version bump with `chore: release vX.Y.Z`.
2. Push the commit: `git push origin HEAD`.
3. Wait for CI on that pushed commit to complete successfully.
4. Create an annotated tag: `git tag -a vX.Y.Z -m "vX.Y.Z"`.
5. Push the tag: `git push origin vX.Y.Z`.
6. Check the GitHub Actions run for the tag, because that run publishes to crates.io.

Do not create or push the release tag until the release commit CI is green. Do not tell the user to publish manually unless the GitHub Actions release path is intentionally unavailable.

## Reporting

Tell the user:

- Previous version and new version.
- Whether upstream CI for the previous commit was green.
- Build and validation commands run and whether they passed.
- That publishing is performed by GitHub Actions after the tag push.
- Commit hash, tag name, and whether the tag workflow was triggered when created.
- Any skipped steps and why.
