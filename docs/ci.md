# The hive in CI

The hive gates the publication of a dataset with one command, run in the **dataset's own repo**:

```bash
hive new.zarr --spec checks/<dataset>.yaml --strict --report hive.json --format markdown > hive.md
```

It runs every bee (meta, stats, spot) against the spec and the committed references, and its exit
code tells the job what happened:

| code | meaning | job |
|---|---|---|
| 0 | no finding reaches `--fail-on` (default `error`), and with `--strict` no check was skipped | passes |
| 1 | the dataset failed the checks (a crashing check counts as an ERROR finding) | fails: look at the findings |
| 2 | configuration error: missing dataset, invalid spec, missing reference, unknown `--select` name | fails: fix the job or the spec |
| 3 | internal error of the hive | fails: report it with the traceback |

- `--strict` fails when a check is skipped, e.g. because no variable in the spec has a range, or the
  spec has no reference. Otherwise a misconfigured job stays green.
- `--report hive.json` writes every finding, while `--format markdown` prints a summary for the
  merge request or the job summary (`--format text`, the default, is meant for the log).
- `--fail-on warning` also fails on warnings. Known, accepted warnings can be demoted to INFO in the
  spec, so it stays usable: `ignore_warnings: {statistics_consistent: [insolation]}`.
- `--bees meta stats` runs only some bees, e.g. on a runner that cannot read the source data. spot-bee
  needs the source (e.g. the GRIB archive and its grib-index DBs with the `grib` extra, or an FDB with
  the `fdb` extra, run inside the FDB uenv); without a `source` in the spec it is skipped, which
  fails under `--strict`.

## Publishing a dataset

The spec, the ref file and the CI job all live in the dataset repo, next to the recipe.

1. **Set up the spec** once, from the recipe: see [Setting up a dataset](../README.md#setting-up-a-dataset).
2. **Build and check** the first version: `hive data.zarr --spec checks/<dataset>.yaml`. Review every
   finding; fix the dataset or the spec.
3. **Accept it** by writing its ref file, and commit it with the spec:

   ```bash
   hive dump data.zarr --spec checks/<dataset>.yaml -o checks/<dataset>.ref.yaml
   ```

   and set `reference: <dataset>.ref.yaml` in the spec. The ref file holds meta-bee's meta file and
   stats-bee's stats file.
4. **Gate every rebuild** in CI with the command above. Each rebuild is compared with the accepted
   version: structure and metadata exactly ([meta-bee](meta-bee.md#the-reference-meta-files-and-compare)),
   statistics on the same sample ([stats-bee](stats-bee.md#stats-files)).
5. **Intended changes** (a new variable, a new period, a new anemoi version) show up as differences.
   Review them with the compare commands, then write a new ref file and commit it together with the
   spec change:

   ```bash
   hive dump new.zarr --spec checks/<dataset>.yaml --dates-from checks/<dataset>.ref.yaml -o new.ref.yaml
   meta-bee compare new.ref.yaml checks/<dataset>.ref.yaml
   stats-bee compare new.ref.yaml checks/<dataset>.ref.yaml
   ```

   The CI job can write `new.ref.yaml` as an artifact, so accepting a change is a download.

   Accepted metadata churn can also be ignored for good with `reference_policy` in the spec, see
   [meta-bee](meta-bee.md#the-reference-meta-files-and-compare).

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
      status=0
      uvx --from "git+https://github.com/MeteoSwiss/varda-dataset-checker-hive@${HIVE_REF}" \
        hive "$DATASET" --spec checks/my-dataset.yaml --strict \
        --report hive.json --format markdown > hive.md || status=$?
      cat hive.md
      exit "$status"
  artifacts:
    when: always
    paths: [hive.md, hive.json]
```

The job fails with the hive's exit code (see the table above), and keeps `hive.md`, ready to paste
into the merge request, and `hive.json`, with every finding, even when it fails.
