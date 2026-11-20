# The hive in CI

The hive gates the publication of a dataset with one command, run in the **dataset's own repo**:

```bash
hive new.zarr --spec checks/<dataset>.yaml --strict --out hive/
```

It runs every bee (meta, stats, spot): each builds its dump (meta-bee: metadata and structure;
stats-bee: statistics of the sample) and compares it with the spec and the committed reference.
`--out hive/` writes `report.md` (for the merge request or the job summary), `report.json` (every
finding) and `ref.yaml` (the dump, the next reference), also when checks fail. The exit code tells
the job what happened:

| code | meaning | job |
|---|---|---|
| 0 | no finding reaches `--fail-on` (default `error`), and with `--strict` no check was skipped | passes |
| 1 | the dataset failed the checks (a crashing check counts as an ERROR finding) | fails: look at the findings |
| 2 | configuration error: missing dataset, invalid spec, missing `--reference` or `calibrated_tolerances`, unknown `--select` name | fails: fix the job or the spec |
| 3 | internal error of the hive | fails: report it with the traceback |

- `--strict` fails when a check is skipped, e.g. because no variable in the spec has a range, or the
  spec's reference does not exist yet. Otherwise a misconfigured job stays green.
- `--out` refuses (exit 2) to write over the reference it compares with: write to another directory
  and copy `ref.yaml` after review.
- stdout gets the text report (`--format text`, meant for the log); `--format markdown` or `json`
  print the same as `report.md` / `report.json`.
- `--fail-on warning` also fails on warnings. Known, accepted warnings can be demoted to INFO in the
  spec, so it stays usable: `ignore_warnings: {statistics_consistent: [insolation]}`.
- `--bees meta stats` runs only some bees, e.g. on a runner that cannot read the source data. spot-bee
  needs the source (e.g. the GRIB archive and its grib-index DBs with the `grib` extra, or an FDB with
  the `fdb` extra, run inside the FDB uenv); without a `source` in the spec it is skipped, which
  fails under `--strict`.

## Publishing a dataset

The spec, the ref file and the CI job all live in the dataset repo, next to the recipe.

All three steps are the same command; only what you do with `hive/` differs.

1. **Set up the spec** once, from the recipe: see [Setting up a dataset](../README.md#setting-up-a-dataset).
   Set `reference: <dataset>.ref.yaml` before the file exists.
2. **First reference.** Build the first version and run `hive data.zarr --spec checks/<dataset>.yaml
   --strict --out hive/`. The reference checks report "skipped: reference … not found" (exit 1 under
   `--strict`, not a configuration error). Review every other finding; fix the dataset or the spec.
   Accept it by committing the dump with the spec:

   ```bash
   cp hive/ref.yaml checks/<dataset>.ref.yaml
   ```

   The ref file holds meta-bee's dump (`meta`) and stats-bee's dump (`stats`).
3. **Gate every rebuild** in CI with the command above. Each rebuild is compared with the accepted
   version: structure and metadata exactly ([meta-bee](meta-bee.md#the-reference-the-meta-section-and-compare)),
   statistics on the same sample, defined by the spec ([stats-bee](stats-bee.md#stats-files)).
4. **Intended changes** (a new variable, a new period, a new anemoi version) show up as differences
   in `hive/report.md`. After review, accept them by copying the job's `hive/ref.yaml` (an artifact,
   so accepting a change is a download) and commit it together with the spec change:

   ```bash
   cp hive/ref.yaml checks/<dataset>.ref.yaml
   ```

   The sample is defined by the spec, so later runs keep comparing the same dates; a change of the
   sampling or the period in the spec is such an intended change. `hive compare hive/ref.yaml checks/<dataset>.ref.yaml` shows the differences of two dumps
   again, without the data. Accepted metadata churn can also be ignored for good with
   `reference_policy` in the spec, see [meta-bee](meta-bee.md#the-reference-the-meta-section-and-compare).

## Installing the hive in a job

Either way, pin a version so a hive update cannot change a release gate unnoticed:

- **As a tool**, without touching the dataset repo's environment:
  `uvx --from "git+https://github.com/MeteoSwiss/varda-dataset-checker-hive@<tag>" hive ...`
- **As a dependency** of the dataset repo (`uv add "varda-dataset-checker-hive @ git+https://github.com/MeteoSwiss/varda-dataset-checker-hive@<tag>"`),
  then `uv run hive ...` after the environment is synced.

## Example CSCS-CI job

A job for the dataset repo's `.cscs-ci/*.yml`, using the shared runner and script fragments of
[varda-cscs-ci-workflows](https://github.com/MeteoSwiss/varda-cscs-ci-workflows). It installs the
hive as a tool; with the hive as a dependency, reference `.mch-uv-sync` and call `uv run hive` instead.

```yaml
include:
  - remote: 'https://raw.githubusercontent.com/MeteoSwiss/varda-cscs-ci-workflows/main/v1/.ci-ext-mch.yml'

check-dataset:
  extends: [.baremetal-runner-mch]
  stage: test
  variables:
    SLURM_TIMELIMIT: '00:30:00'
    MCH_WORK_NAME: 'my-dataset-ci'
    DATASET: '/scratch/mch/.../my-dataset.zarr'   # the freshly built dataset
    HIVE_REF: 'main'                              # better: a release tag
  script:
    - !reference [.mch-sh-preamble, script]
    - !reference [.mch-log-ci-context, script]
    - !reference [.mch-env-paths, script]
    - !reference [.mch-cleanup-on-exit, script]
    - !reference [.mch-install-uv, script]
    - |
      uvx --from "git+https://github.com/MeteoSwiss/varda-dataset-checker-hive@${HIVE_REF}" \
        hive "$DATASET" --spec checks/my-dataset.yaml --strict --out hive/
  artifacts:
    when: always
    paths: [hive/]
```

The job fails with the hive's exit code (see the table above), and keeps `hive/` even when it fails:
`report.md`, ready to paste into the merge request, `report.json`, with every finding, and
`ref.yaml`, to commit as the new reference after an intended change.
