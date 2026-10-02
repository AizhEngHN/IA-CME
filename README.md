# IA-CME: Interaction-Aware Coupled MAP-Elites

IA-CME constructs high-coverage software product-line test suites for multiple suite-size budgets. The implementation connects two stages through interaction support:

1. **Support-adaptive product-pool construction.** Randomized SAT exploration produces valid, unique products. Targeted SAT generation then replenishes interactions with insufficient support until the requested support threshold is reached, subject to the pool-size limit.
2. **Interaction-aware quality-diversity search.** A size-indexed MAP-Elites archive maintains one elite suite per size. Pool support and archive misses guide loss-aware removal and gain-aware addition. Tensorized operations evaluate exact coverage on CPU or CUDA.

SAT solving is confined to Stage I. After qualification, the pool is frozen and Stage II selects products from that pool without further SAT calls.

This is a source-code release. It includes the IA-CME runner, its supporting Python and Java modules, and unit tests. It does **not** include benchmark datasets, published result files, third-party JARs, a Python environment, or a complete experiment campaign runner.

## Contents

- [1. Repository structure](#1-repository-structure)
- [2. Requirements and installation](#2-requirements-and-installation)
- [3. Input files](#3-input-files)
  - [Data sources](#33-data-sources)
- [4. Running IA-CME](#4-running-ia-cme)
- [5. Small input example](#5-small-input-example)
- [6. Command-line parameters](#6-command-line-parameters)
- [7. Results and interpretation](#7-results-and-interpretation)
- [8. Repeated runs and reproducibility](#8-repeated-runs-and-reproducibility)
- [9. Tests](#9-tests)
- [10. Troubleshooting](#10-troubleshooting)
- [11. Validation status](#11-validation-status)

## 1. Repository structure

```text
IA-CME/
|-- README.md
|-- requirements.txt
|-- run_iacme.py
|-- src/
|   |-- iacme/
|   |   |-- __init__.py
|   |   |-- pipeline.py
|   |   |-- pool_construction.py
|   |   |-- sampler.py
|   |   +-- sat.py
|   +-- spl/
|       |-- __init__.py
|       |-- data_model.py
|       |-- dimacs_validation.py
|       |-- evaluator.py
|       |-- evox_paper_aligned.py
|       |-- interaction_guidance.py
|       |-- output_parser.py
|       |-- tensor_archive.py
|       |-- tensor_coverage.py
|       +-- tensor_variation.py
|-- tools/
|   +-- java_runner/
|       |-- DeterministicSAT4JProductSampler.java
|       +-- TargetedSAT4JProductGenerator.java
+-- tests/
    |-- test_evox_paper_aligned.py
    |-- test_interaction_guidance.py
    |-- test_pool_construction.py
    |-- test_release_interface.py
    |-- test_spl_tensor_archive.py
    |-- test_spl_tensor_coverage.py
    +-- test_spl_tensor_variation.py
```

### Entry point and orchestration

| File | Purpose |
| --- | --- |
| `run_iacme.py` | Parses command-line arguments, invokes the two-stage pipeline, and writes one JSON result. |
| `requirements.txt` | Specifies Python dependency constraints. Java and SAT4J must be installed separately. |
| `src/iacme/pipeline.py` | Validates inputs and settings, compiles the Java helpers, constructs the pool, builds the coverage representation, runs search, and independently checks the returned elites. |
| `src/iacme/pool_construction.py` | Computes support deficits, selects replenishment targets, and records support-qualified pool checkpoints. |
| `src/iacme/sampler.py` | Resolves the external SAT4J JAR and builds commands for initial SAT sampling. |
| `src/iacme/sat.py` | Runs targeted SAT generation and checks generated products for model validity, target satisfaction, and uniqueness. |

### Search and coverage components

| File | Purpose |
| --- | --- |
| `src/spl/data_model.py` | Defines products, interactions, and test suites using signed feature literals. |
| `src/spl/dimacs_validation.py` | Parses DIMACS CNF and independently checks complete product assignments against model constraints. |
| `src/spl/evaluator.py` | Provides a set-based coverage evaluator used to check final results independently. |
| `src/spl/evox_paper_aligned.py` | Implements the EvoX problem, size-indexed MAP-Elites search, evaluation-budget accounting, and search history. |
| `src/spl/interaction_guidance.py` | Builds the product-interaction incidence matrix and implements interaction hardness and guided variation. |
| `src/spl/output_parser.py` | Parses semicolon-separated products and target interactions; also contains legacy result-file readers. |
| `src/spl/tensor_archive.py` | Stores elite candidates and fitness values in a tensor archive and computes archive summaries. |
| `src/spl/tensor_coverage.py` | Creates tensor representations and computes batched coverage. |
| `src/spl/tensor_variation.py` | Implements suite initialization, cardinality handling, and unguided variation. |

`DeterministicSAT4JProductSampler.java` performs seeded initial sampling with uniqueness enforcement. `TargetedSAT4JProductGenerator.java` generates products covering requested interactions. Their source files are bundled; compilation is performed automatically by the Python runner.

Some supporting modules expose lower-level controls and retain historical implementation-oriented names. The supported two-stage entry point for this release is `run_iacme.py`; running a lower-level search function alone does not perform Stage I.

## 2. Requirements and installation

### 2.1 Software requirements

The Python requirements file specifies:

| Dependency | Constraint | Role |
| --- | --- | --- |
| Python | 3.10 or newer, with compatible dependency wheels | Runtime; Python 3.11 or 3.12 is a practical starting point. |
| PyTorch | `>=2.6.0,<3.0` | Tensor operations on CPU or CUDA. |
| NumPy | `>=2.0.0,<3.0` | Numerical support. |
| EvoX | `==1.3.0` | Evolutionary workflow integration. |
| pytest | `>=7.0,<10.0` | Unit tests; not needed for the main algorithm itself, but included in the requirements file. |
| Java Development Kit | Must provide `java` and `javac` | Compiles and runs the SAT helpers. |
| SAT4J core JAR | External dependency | Provides the Java SAT solver API. |

A JRE alone is insufficient because the runner compiles Java source files. The exact JDK and SAT4J versions are not pinned by this release. The bundled helpers use SAT4J core classes, including solver-order classes, so retain and record the JAR version used for your runs.

No Maven or Gradle build is required. Installing Python packages does not install Java or SAT4J.

### 2.2 Create a Python environment

Extract the archive and enter the directory containing `run_iacme.py` and `requirements.txt`:

```text
cd IA-CME
```

Create an isolated environment:

```text
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```bash
source .venv/bin/activate
```

Then install dependencies:

```text
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

If PowerShell blocks activation, you can invoke the environment's Python directly rather than changing your system execution policy:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use the same Python executable for installation, testing, and running IA-CME.

### 2.3 CPU and CUDA setup

CPU execution does not require an NVIDIA GPU. CUDA execution requires a compatible NVIDIA driver and a CUDA-enabled PyTorch build.

Use the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/) to select a CPU or CUDA wheel appropriate for your platform. Keep its version within the `torch>=2.6.0,<3.0` constraint, then install the remaining requirements. Installing `torchvision` or `torchaudio` is not required by IA-CME.

Check Python dependencies and CUDA availability:

```text
python -c "import torch, numpy, evox; from importlib.metadata import version; print('torch:', torch.__version__); print('numpy:', numpy.__version__); print('evox:', version('evox')); print('CUDA available:', torch.cuda.is_available())"
```

`CUDA available: False` is acceptable for CPU runs. The runner's `--device auto` setting selects CUDA when PyTorch reports it as available and CPU otherwise. Explicit `--device cuda` does not silently fall back to CPU.

The documented device choices are `auto`, `cpu`, `cuda`, and `cuda:N`. Other accelerator backends are not documented or validated by this release.

### 2.4 Java and SAT4J

Install a JDK and ensure both executables are available:

```text
java -version
javac -version
```

Obtain a SAT4J core JAR from the [SAT4J project](https://www.sat4j.org/) or its linked distribution sources. Use the binary core JAR, not a sources or Javadoc JAR. The JAR is not bundled with IA-CME.

Place it wherever convenient, for example:

```text
dependencies/sat4j-core.jar
```

This is an example location and filename, not an existing bundled file. Supply the actual path through `--sat4j-jar` on every CLI invocation. A versioned JAR filename can be used directly; renaming is unnecessary.

If Java is not on `PATH`, supply its executables explicitly:

```text
--java "C:/path/to/jdk/bin/java.exe" --javac "C:/path/to/jdk/bin/javac.exe"
```

The runner compiles both helpers into a temporary directory on each invocation. It also uses that temporary directory for intermediate SAT files, which are removed when the pipeline leaves the temporary workspace.

## 3. Input files

Every run requires two externally supplied input files. Coverage is computed over the interaction set in the supplied file.

### 3.1 Feature model: DIMACS CNF

Supply a satisfiable feature model encoded in DIMACS CNF. The header has the form:

```text
p cnf <number_of_variables> <number_of_clauses>
```

Each clause ends with `0`. Positive literal `i` means feature `i` is selected; negative literal `-i` means it is deselected. Variables must use identifiers from `1` through the declared number of variables.

For example:

```text
c Features 1 and 2 cannot both be deselected.
p cnf 3 1
1 2 0
```

This model has three features and requires feature 1 or feature 2 to be selected. Feature 3 is unconstrained.

The Python parser checks the header, literal ranges, terminated clauses, and declared clause count. Empty clauses are rejected by this parser. Blank lines and lines beginning with `c` or `%` are ignored. Save input files as UTF-8 without a byte-order mark.

### 3.2 Target interactions: signed-literal tuples

Use a plain-text file with one interaction per line. Separate its signed literals with semicolons:

```text
1;2
1;-2
-1;2
```

For 3-wise interactions, each line contains three distinct feature literals, for example:

```text
1;-2;3
```

Requirements:

- The set must be non-empty.
- All interactions in one file must have the same positive strength `t`.
- Each interaction must refer to distinct features using non-zero signed integers within the model's variable range.
- An interaction must not contain both `i` and `-i`.
- Every supplied interaction must be satisfiable together with the feature-model constraints.
- Do not add a header, comment lines, or DIMACS-style trailing zeros to this file. Blank lines are ignored.

Repeated interaction lines are deduplicated by the parser. Do not rely on duplicate literals within a line: literals are also stored with set semantics.

The runner infers `t` from the input; there is no separate `--t` argument. It does not enumerate all valid interactions from the feature model. To evaluate complete `t`-wise coverage, supply the complete valid set. If you supply a subset, the reported percentage is coverage of that subset.

The pipeline checks interaction consistency and range, but does not independently pre-solve every interaction before pool construction. Supplying invalid interactions can prevent support qualification or cause replenishment to fail.

### 3.3 Data sources

The feature models in the accompanying study are drawn from established SPL-testing benchmarks. A public source of feature models and supplied interaction sets is the repository released with the following work:

> Yi Xiang, Han Huang, Sizhe Li, Miqing Li, Chuan Luo, and Xiaowei Yang. **Automated Test Suite Generation for Software Product Lines Based on Quality-Diversity Optimization.** ACM Transactions on Software Engineering and Methodology, 33(2), Article 46, 2024. [DOI: 10.1145/3628158](https://doi.org/10.1145/3628158).

- **Original repository:** [SPLTestingMAP](https://github.com/gzhuxiangyi/SPLTestingMAP).
- **Model and interaction directory:** [all_FM/Selected](https://github.com/gzhuxiangyi/SPLTestingMAP/tree/c9befe9fd9e09b637cd98f2513b1dfbbef20855e/all_FM/Selected).
- **Original-study supplementary materials and raw results:** [Zenodo record 7805017, version v1](https://zenodo.org/records/7805017). This record contains materials for Xiang et al.'s study, not IA-CME experiment results.

The directory link above uses upstream commit `c9befe9fd9e09b637cd98f2513b1dfbbef20855e`, inspected on 2026-10-02. To obtain that source snapshot with Git:

```text
git clone https://github.com/gzhuxiangyi/SPLTestingMAP.git SPLTestingMAP-data
git -C SPLTestingMAP-data checkout c9befe9fd9e09b637cd98f2513b1dfbbef20855e
```

Alternatively, use GitHub's **Code > Download ZIP** option to obtain the current branch. Record the downloaded revision and retain checksums of the input files used in your runs.

Under `all_FM/Selected`, matching inputs use these naming conventions:

| Upstream filename | IA-CME argument | Content |
| --- | --- | --- |
| `<model>.dimacs` | `--model` | Feature-model constraints in DIMACS CNF. |
| `<model>.dimacs.valid2-Set` | `--interactions` | Supplied valid 2-wise interactions. |
| `<model>.dimacs.valid3-Set` | `--interactions` | Supplied valid 3-wise interactions, where available. |

The filename extension does not need to be changed: the IA-CME runner accepts a DIMACS model named `.dimacs` and a semicolon-separated interaction file named `.valid2-Set` or `.valid3-Set` directly. Keep each interaction file paired with its corresponding model so that feature identifiers agree.

For example, after obtaining the source snapshot, run the `Printers` model using its supplied pairwise set:

```text
python run_iacme.py --model SPLTestingMAP-data/all_FM/Selected/Printers.dimacs --interactions SPLTestingMAP-data/all_FM/Selected/Printers.dimacs.valid2-Set --sat4j-jar dependencies/sat4j-core.jar --output results/Printers_seed0.json --device cpu --seed 0
```

This is a usage example, not a reported IA-CME benchmark result. The external JAR and Python dependencies are still required.

The accompanying evaluation uses 56 feature models with supplied 2-wise sets and ten models with supplied 3-wise sets. The upstream repository contains a broader collection; select the subjects listed in the paper for paper-specific comparisons. Coverage is measured over the supplied sets, some of which are capped. Preserve those sets when comparing coverage values rather than replacing them with newly enumerated interaction universes.

This IA-CME release distributes code and documentation rather than copies of the upstream dataset. Retain the original data attribution and consult the upstream distribution terms before redistributing its files.

## 4. Running IA-CME

Run commands from the release root. Paths below are examples and must refer to files you have prepared.

### 4.1 Inspect the command-line interface

```text
python run_iacme.py --help
```

This displays the available arguments without requiring model data or invoking the algorithm. Successful `--help` output alone does not verify Python runtime dependencies, Java, or SAT4J.

### 4.2 CPU run

```text
python run_iacme.py --model inputs/model.cnf --interactions inputs/valid_2wise.txt --sat4j-jar dependencies/sat4j-core.jar --output results/cpu_seed0.json --device cpu --seed 0
```

### 4.3 GPU run

```text
python run_iacme.py --model inputs/model.cnf --interactions inputs/valid_2wise.txt --sat4j-jar dependencies/sat4j-core.jar --output results/gpu_seed0.json --device cuda --pool-device cpu --seed 0
```

To select a particular visible GPU, use `--device cuda:0`, `--device cuda:1`, and so on. GPU indices follow PyTorch's visible-device numbering.

By default, Stage I support calculations run on CPU, while `--device` determines Stage II tensor operations. SAT solving runs through Java regardless of either device setting. `--pool-device auto` makes Stage I tensor calculations use the resolved Stage II device.

### 4.4 3-wise run

```text
python run_iacme.py --model inputs/model.cnf --interactions inputs/valid_3wise.txt --sat4j-jar dependencies/sat4j-core.jar --output results/3wise_seed0.json --device cuda --seed 0
```

The interaction file, rather than a CLI strength parameter, determines whether this is a 2-wise or 3-wise run.

### 4.5 What one invocation does

1. Validates the configuration and parses the model and interactions.
2. Compiles the bundled Java helpers against the supplied SAT4J JAR.
3. Samples the requested number of valid, unique initial products.
4. Measures pool support and performs targeted replenishment when necessary.
5. Freezes a pool meeting the support threshold and constructs its product-interaction incidence matrix.
6. Initializes one archive cell per suite size and runs the requested evolutionary evaluations.
7. Checks returned suite cardinalities and uniqueness, and reevaluates their coverage with the independent set-based evaluator.
8. Writes the result JSON and prints best fitness and QD-score.

The output directory is created automatically. An existing file at `--output` is overwritten on successful completion, so use distinct filenames for runs you want to preserve. Results are written at completion, not as periodic checkpoints; this CLI does not provide automatic resume.

## 5. Small input example

The following example is provided as documentation, not as bundled benchmark data. It illustrates how to prepare a small model and a matching interaction set before attempting a larger run.

Create `inputs/toy.cnf` with:

```text
p cnf 3 1
1 2 0
```

Create `inputs/toy_2wise.txt` with:

```text
1;2
1;-2
-1;2
1;3
1;-3
-1;3
-1;-3
2;3
2;-3
-2;3
-2;-3
```

This is the complete valid pairwise set for the toy model: 11 interactions. The omitted tuple `-1;-2` violates the model. There are six valid complete products.

After installing dependencies, run:

```text
python run_iacme.py --model inputs/toy.cnf --interactions inputs/toy_2wise.txt --sat4j-jar dependencies/sat4j-core.jar --output results/toy.json --device cpu --initial-pool-size 6 --maximum-pool-size 6 --lower-bound 2 --upper-bound 4 --initial-assumption-count 3 --random-assumption-count 2 --search-evaluations 64 --batch-size 16 --seed 0
```

These settings deliberately replace the larger defaults: the model has only six products, so an initial pool of 100 cannot be constructed. Because all six products are requested, the initial pool already supports every valid target interaction. This example exercises initial SAT sampling and Stage II, but not a nontrivial support-deficit replenishment round.

On successful execution, `results/toy.json` records interaction strength 2, 11 interactions, a pool size of 6, and 64 search evaluations. The archive contains suite-size cells 2 through 4. Coverage values and timings should be inspected from the generated result rather than assumed in advance.

The input set and these configuration relationships have been checked; a fresh end-to-end execution of this example has not been completed in the local inspection environment described in Section 11.

## 6. Command-line parameters

### 6.1 Inputs, output, and devices

| Argument | Default | Meaning |
| --- | --- | --- |
| `--model` | Required | DIMACS CNF feature-model path. |
| `--interactions` | Required | Valid target-interaction file. |
| `--sat4j-jar` | Required | External SAT4J core JAR path. |
| `--output` | Required | Result JSON path. |
| `--java` | `java` | Java runtime executable. |
| `--javac` | `javac` | Java compiler executable. |
| `--device` | `auto` | Stage II device: `auto`, `cpu`, `cuda`, or `cuda:N`. |
| `--pool-device` | `cpu` | Device for Stage I tensor support calculations; `auto` uses the resolved Stage II device. |

### 6.2 Pool construction

| Argument | Default | Meaning |
| --- | --- | --- |
| `--initial-pool-size` | `100` | Number of valid, unique products requested by initial SAT sampling. |
| `--maximum-pool-size` | `3000` | Upper limit on activated pool size during construction; not a fixed final pool size. |
| `--support-threshold` | `1` | Minimum number of pool products required to cover each supplied interaction. |
| `--targets-per-batch` | `32` | Requested number of replenishment targets per SAT batch, also bounded by available targets and remaining pool capacity. |
| `--initial-assumption-count` | `16` | Number of random feature assumptions used by the initial sampler, capped by the model's variable count. |
| `--random-assumption-count` | `8` | Number of additional random assumptions used during targeted generation. |
| `--max-attempts-per-product` | `10000` | Attempt limit for finding each new initial product. |
| `--max-attempts-per-target` | `1000` | Attempt limit used by targeted generation for each requested interaction. |

At threshold 1, every supplied interaction is reachable from the qualified pool. This does not mean every individual suite covers every interaction. Higher thresholds require more distinct supporting products and can be infeasible for some interactions.

Construction can stop below the maximum pool size as soon as the threshold is reached. If the initial pool already qualifies, it is used directly. Search is not started if support qualification fails. If the qualified pool is smaller than the requested upper suite-size bound, the runner reports an error rather than padding it with duplicate products.

### 6.3 Search

| Argument | Default | Meaning |
| --- | --- | --- |
| `--lower-bound` | `2` | Smallest suite-size archive cell. |
| `--upper-bound` | `10` | Largest suite-size archive cell, inclusive. |
| `--search-evaluations` | `4096` | Number of evolutionary candidate evaluations after archive initialization. |
| `--batch-size` | `64` | Number of offspring evaluated per regular search batch; the last batch is truncated to meet the exact budget. |
| `--guidance-eta` | `0.5` | Probability of using interaction-aware choices; `0` gives unguided variation and `1` requests guidance for every offspring. |
| `--candidate-sample-size` | `64` | Maximum number of pool products considered in sampled gain-aware addition choices. |
| `--interaction-chunk-size` | `64` | Product-row chunk size passed when building the incidence matrix in this pipeline; the CLI name is historical. |

The number of archive cells is `upper_bound - lower_bound + 1`. Initialization evaluates one suite for each size. The default configuration therefore performs nine initialization evaluations plus exactly 4,096 search evaluations. SAT calls, pool construction, and representation setup are not included in that evolutionary evaluation counter.

`--search-evaluations 0` returns the initialized archive without evolutionary iterations. Configuration requires positive pool sizes, bounds, support, batch and sampling settings, non-negative evaluation and assumption counts, and `0 <= guidance_eta <= 1`.

### 6.4 Seeds

| Argument | Default | Meaning |
| --- | --- | --- |
| `--seed` | `0` | Shared fallback seed for components without an explicit override. |
| `--initial-seed` | Falls back to `--seed` | Initial SAT sampling. |
| `--target-seed` | Falls back to `--seed` | Support-deficit target selection. |
| `--sat-seed` | Falls back to `--seed` | Targeted SAT generation. |
| `--search-seed` | Falls back to `--seed` | Archive initialization and evolutionary variation. |

## 7. Results and interpretation

The output is a JSON object. Important fields are:

| Field | Meaning |
| --- | --- |
| `method` | Algorithm name. |
| `config` | Resolved configuration, including the four component seeds. |
| `device` | Resolved Stage II device. |
| `interaction_strength` | Input strength `t`. |
| `num_interactions` | Number of distinct supplied interactions. |
| `pool_size` | Number of products in the frozen pool. |
| `minimum_support` | Minimum interaction support in the incidence matrix. |
| `pool_ceiling_percent` | Percentage of supplied interactions covered by the whole constructed pool. |
| `best_fitness` | Highest suite coverage percentage among occupied archive cells. |
| `qd_score` | Sum of elite coverage percentages over occupied suite-size cells; empty cells contribute zero. |
| `initial_evaluations` | Archive-initialization evaluations. |
| `search_evaluations` | Completed evolutionary evaluations. |
| `guided_mutations` | Count of offspring selected for guided variation. |
| `timings_seconds` | Stage timing measurements described below. |
| `archive` | Elite suites indexed by their size as string keys, such as `"2"` or `"10"`. |
| `history` | Per-search-batch archive summaries and evaluation counters. |

Each `archive` entry contains:

- `fitness`: coverage percentage of the elite suite.
- `product_indices`: zero-based indices into the frozen pool.
- `products`: complete signed-literal assignments for the suite's selected products.

The complete pool is not exported as a separate file. Elite product assignments are included in JSON, so each returned suite can be inspected without a separate pool file. Product indices are pool-local and should not be compared across independently generated pools.

For a suite `S` and supplied target set `T`, fitness is:

```text
fitness(S) = 100 * number_of_targets_covered_by_S / number_of_supplied_targets
```

QD-score is not an average percentage and can exceed 100. For the default nine cells, its maximum is 900. Compare QD-scores only when the suite-size domain and evaluation target set are aligned.

In `history`, `coverage` means the fraction of occupied archive cells, not interaction-coverage fitness. Other history fields include `generation`, `initial_evaluations`, `search_evaluations`, `total_evaluations`, `qd_score`, `best_fitness`, and `occupied_cells`. Search generations are numbered from zero; initialization is counted separately and is not a standalone entry in the exported `history` list.

### 7.1 Timing fields

`timings_seconds` contains:

| Field | Timed scope |
| --- | --- |
| `pool_construction` | Initial SAT sampling, support-adaptive replenishment, and pool processing and checks inside Stage I. |
| `representation` | Stage II tensorization, product-interaction incidence-matrix construction, and its support check. |
| `archive_initialization` | Initial archive seeding and evaluation. |
| `search` | Evolutionary iterations after initialization. |

CUDA is synchronized at the main timing boundaries. Java compilation is outside these timers. Input loading, some setup, final elite checking, and JSON writing are also outside them, so their sum is not complete process wall time. Measure the whole process separately if end-to-end timing is needed. Timing results depend on hardware and concurrent load.

### 7.2 Inspect a result

Format the JSON for viewing:

```text
python -m json.tool results/cpu_seed0.json
```

Or read a short summary:

```python
import json

with open("results/cpu_seed0.json", encoding="utf-8") as handle:
    result = json.load(handle)

print("Best coverage:", result["best_fitness"])
print("QD-score:", result["qd_score"])
print("Pool size:", result["pool_size"])
for size, elite in sorted(result["archive"].items(), key=lambda item: int(item[0])):
    print("Suite size:", size, "Coverage:", elite["fitness"])
```

## 8. Repeated runs and reproducibility

One invocation produces one run, not a multi-model or multi-replicate campaign. For repetitions, change seeds and output paths explicitly.

For example, Windows PowerShell can run three seeds sequentially:

```powershell
0..2 | ForEach-Object {
    $runSeed = $_
    python run_iacme.py --model inputs/model.cnf --interactions inputs/valid_2wise.txt --sat4j-jar dependencies/sat4j-core.jar --output "results/seed_$runSeed.json" --device cuda --seed $runSeed
    if ($LASTEXITCODE -ne 0) { throw "IA-CME failed for seed $runSeed" }
}
```

For Linux/macOS, use:

```bash
for run_seed in 0 1 2; do
    python run_iacme.py --model inputs/model.cnf --interactions inputs/valid_2wise.txt --sat4j-jar dependencies/sat4j-core.jar --output "results/seed_${run_seed}.json" --device cpu --seed "$run_seed" || break
done
```

Changing `--seed` changes all four component seeds unless overridden. To keep pool-construction seeds fixed while varying search seeds, provide fixed `--initial-seed`, `--target-seed`, and `--sat-seed` values and vary `--search-seed`. Each invocation still reconstructs the pool; the CLI does not load a saved prebuilt pool.

For reproducibility, retain:

- The exact CNF and interaction files, preferably with checksums.
- The command, JSON configuration, and component seeds.
- Python, PyTorch, NumPy, EvoX, JDK, and SAT4J versions.
- Device model, driver information, and whether the run used CPU or CUDA.
- All successful outputs and failure logs under distinct names.

Seeds control the implemented random choices, but identical results across different solver versions, devices, or software stacks are not guaranteed. Dependencies and benchmark inputs must be aligned when reconstructing an experiment. This source-only release does not by itself reproduce the paper's benchmark campaign or comparison tables.

## 9. Tests

After installing the Python requirements, run from the release root:

```text
python -m pytest -q tests
```

The tests cover archive behavior, tensorized coverage, variation, interaction guidance, support-deficit pool construction, EvoX search integration, and the release interface. The interface tests check command construction and missing-dependency handling; they do not execute a complete Java/SAT4J pipeline.

The EvoX integration test module skips when EvoX is absent. The release-interface module imports the pipeline and therefore requires EvoX. Inspect pytest's summary rather than treating skipped integration tests as executed tests.

For a component-only check without EvoX or SAT4J, with PyTorch, NumPy, and pytest installed:

```text
python -m pytest -q tests/test_spl_tensor_archive.py tests/test_spl_tensor_coverage.py tests/test_spl_tensor_variation.py tests/test_interaction_guidance.py tests/test_pool_construction.py
```

A successful full run additionally requires the external inputs, JDK, SAT4J JAR, and compatible Python runtime described above.

## 10. Troubleshooting

| Symptom | Checks and next steps |
| --- | --- |
| `ModuleNotFoundError` for `torch`, `numpy`, or `evox` | Install `requirements.txt` with the same Python executable used to run the script; check the active environment. |
| Missing `java` or `javac` executable | Install a JDK, check `PATH`, or pass `--java` and `--javac` explicitly. |
| `Supply an existing SAT4J core JAR using --sat4j-jar.` | Check the supplied path. A JAR is not bundled or installed by pip. |
| Java compilation cannot resolve `org.sat4j` classes | Use a compatible binary SAT4J core JAR, not a sources or Javadoc archive. Check its contents and version. |
| Invalid DIMACS header, out-of-range literal, or clause count mismatch | Fix the CNF syntax and header counts; ensure every clause ends with `0`. |
| Interaction parsing fails | Remove headers, comments, trailing zeros, and non-integer tokens; use semicolon-separated signed literals. |
| Interaction strength or consistency error | Use one positive strength per file, distinct features per tuple, and no contradictory literals. |
| Initial sampling cannot produce enough unique products | The model may have fewer products than requested, or the assumption/attempt settings may be too restrictive. Reduce `--initial-pool-size` for small models and review sampler settings. |
| `support-deficit replenishment produced no accepted products` | Check target satisfiability, uniqueness constraints, remaining pool capacity, and the SAT generator's assumptions and attempt limits. |
| Pool limit reached before support qualification | Inspect whether the target set is valid and whether the threshold is feasible; review the maximum pool size or threshold for a new configuration. |
| Qualified pool smaller than the largest suite size | Reduce the suite-size bound or increase the initial pool size where the model has enough valid products. Raising the maximum alone does not force construction past an already qualified pool. |
| CUDA unavailable or initialization error | Confirm a compatible PyTorch CUDA wheel and driver, check `torch.cuda.is_available()`, or use `--device cpu`. |
| CUDA out of memory | Reduce batch size or candidate sampling for a new run, consider CPU execution, and inspect pool and target-set dimensions. Incidence-matrix storage remains proportional to pool size times interaction count; chunking does not remove that storage requirement. |
| No output JSON after failure or interruption | JSON is written only after successful completion. Retain the terminal traceback; the CLI does not save intermediate resumable checkpoints. |

Parameter changes constitute a new run configuration. Keep changes explicit when comparing results. The CLI has no whole-run timeout argument; SAT attempt limits are not process wall-clock deadlines.

## 11. Validation status

During the local release inspection on 2026-10-02:

- The original archive passed its ZIP integrity check and matched the corresponding release source files.
- All 23 bundled Python files passed a syntax check.
- `python run_iacme.py --help` executed successfully.
- The five component test modules listed in Section 9 completed with **37 tests passed**.

Those component tests used Python 3.12.7, PyTorch 2.5.1+cu121, and NumPy 1.26.4 in an existing local environment. That environment is not the dependency configuration specified by `requirements.txt`; it verifies the tested components, not installation of the advertised runtime stack.

A fresh complete two-stage run, the small CLI example, and the full test suite were not executed during that inspection because EvoX and the external SAT4J JAR were unavailable in the inspection environment. This README documents the source interface and verified input/configuration relationships; it does not claim that an end-to-end CPU or CUDA execution has passed in that environment.
