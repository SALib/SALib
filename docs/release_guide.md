# Release Guide

This document outlines the process for publishing a new release of SALib to PyPI.

## Prerequisites

- Maintainer-level access to the SALib GitHub repository
- PyPI and TestPyPI accounts with permissions to publish to the `SALib` project
- A configured API token for both PyPI and TestPyPI (see [PyPI docs](https://pypi.org/help/#apitoken))
- A complete local development environment (see the [Developers Guide](./developers_guide.md))

## Overview

Releases follow this sequence:

1. Verify the main branch is release-ready
2. Bump the version
3. Build and verify the distribution
4. Publish to TestPyPI
5. Publish to PyPI
6. Create a GitHub Release

---

## Step 1: Verify the main branch is release-ready

Ensure all intended changes are merged into `main` and that CI passes. Run the
full test suite locally to confirm:

```bash
$ uv run pytest    # equivalent: pytest (but runs within the uv-managed environment)
```

Also run pre-commit checks to confirm there are no outstanding formatting issues:

```bash
$ pre-commit run --all-files
```

Do not proceed until all tests and checks pass.

---

## Step 2: Bump the version

SALib uses `hatch` to manage versioning. Version numbers follow
[Semantic Versioning](https://semver.org/): `MAJOR.MINOR.PATCH`.

| Change type                       | Command               | Example: 1.4.2 → |
| --------------------------------- | --------------------- | ---------------- |
| Bug fixes / patches               | `hatch version patch` | `1.4.3`          |
| New backwards-compatible features | `hatch version minor` | `1.5.0`          |
| Breaking changes                  | `hatch version major` | `2.0.0`          |

To inspect the current version without changing it:

```bash
$ hatch version
```

To bump the version:

```bash
$ hatch version patch    # or: minor, major
```

This updates the version string in the project source directly. Verify the
change looks correct before continuing, then commit:

```bash
$ git add .
$ git commit -m "Release vX.Y.Z"
```

---

## Step 3: Build the distribution

Build the source distribution (sdist) and binary wheel:

```bash
$ hatch build
```

Hatch places the build artefacts in the `dist/` directory. Verify that both a
`.tar.gz` and a `.whl` file are present:

```bash
$ ls dist/
SALib-X.Y.Z.tar.gz
SALib-X.Y.Z-py3-none-any.whl
```

---

## Step 4: Publish to TestPyPI

Publishing to [TestPyPI](https://test.pypi.org) first allows verification that
the package installs correctly before pushing to the main index.

```bash
$ hatch publish -r test
```

Hatch will prompt for TestPyPI credentials. When complete, verify the release
looks correct at `https://test.pypi.org/project/SALib/`.

Optionally, test the install from TestPyPI in a temporary environment:

```bash
$ uv venv .venv-test
$ .venv-test/Scripts/activate        # or: source .venv-test/bin/activate on *nix
$ uv pip install --index-url https://test.pypi.org/simple/ SALib==X.Y.Z
```

---

## Step 5: Publish to PyPI

Once satisfied with the TestPyPI release, publish to the main index:

```bash
$ hatch publish
```

Hatch will prompt for PyPI credentials. Verify the release at
`https://pypi.org/project/SALib/`.

---

## Step 6: Create a GitHub Release

GitHub Releases serve as the canonical record of each release and are the
recommended way to notify users of new versions.

First, push the release commit to `main`:

```bash
$ git push origin main
```

Then navigate to the SALib repository on GitHub and open
**Releases → Draft a new release**.

1. In the **Choose a tag** field, type `vX.Y.Z` and select
   **Create new tag: vX.Y.Z on publish**. This creates the Git tag automatically
   when the release is published, so there is no need to create it manually.
2. Set the **Target** branch to `main`.
3. Set the **Release title** to `vX.Y.Z`.
4. Click **Generate release notes**. GitHub will automatically populate the
   release body with a categorised list of merged pull requests and
   contributors since the previous tag. Review and edit the generated notes
   as needed — for example, to highlight breaking changes or call out
   particularly significant additions.
5. If this is a pre-release or release candidate, check
   **Set as a pre-release** to avoid it being presented to users as the latest
   stable version.
6. Click **Publish release**.

---

## Troubleshooting

**Build artefacts from a previous release are present in `dist/`**
Remove the `dist/` directory before building to avoid uploading stale files:
```bash
$ rm -rf dist/
$ hatch build
```

**`hatch publish` fails with a 403 error**
This typically indicates an authentication issue. Confirm that the API token is
correctly configured, and that it has upload permissions for the `SALib` project
on the target index.

**Version already exists on PyPI**
PyPI does not allow re-uploading a release under an existing version number.
If a release needs to be corrected after publishing, a new patch version must
be issued.

---

## conda-forge

No manual action is required to release on conda-forge. After the PyPI
release is published, the `regro-cf-autotick-bot` will automatically open a
pull request on the [salib-feedstock](https://github.com/conda-forge/salib-feedstock)
repository with the updated version. A maintainer should review and merge
that PR, after which conda-forge will build and publish the package
automatically.

If the new release changes any dependencies, the feedstock recipe may need
manual adjustment before the bot PR can be merged.
