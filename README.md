# Methylation Analysis

Refactored methylation analysis script with configurable group comparisons using Hydra.

## Installation

Install required dependencies:

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install hydra-core pandas numpy scipy matplotlib joblib
```

## Usage

Run the analysis with default configuration:

```bash
python methylation/methylation_analysis.py
```

Or from the methylation directory:

```bash
cd methylation
python methylation_analysis.py
```

Override configuration parameters via command line:

```bash
python methylation/methylation_analysis.py group_comparison.group_a=[1,2] group_comparison.group_b=[3,4] output.folder=results
```

Or create a custom configuration file in `conf/` and specify it:

```bash
python methylation/methylation_analysis.py --config-name=my_config
```

### Running Sweeps

Hydra supports parameter sweeps (multirun) for exploring multiple configurations. You can define sweeps in two ways:

#### Option 1: Command Line (Quick sweeps)

```bash
# Sweep over different group comparisons
python methylation/methylation_analysis.py --multirun \
  group_comparison.group_a=[1],[1,3] \
  group_comparison.group_b=[3],[2,4]

# Sweep over targets
python methylation/methylation_analysis.py --multirun \
  targets=[all],[male],[female]

# Combine sweeps
python methylation/methylation_analysis.py --multirun \
  group_comparison.group_a=[1],[1,3] \
  group_comparison.group_b=[3],[2,4] \
  targets=[all],[male]

# Sweep over number of permutations (for speed vs accuracy trade-off)
python methylation/methylation_analysis.py --multirun \
  num_permutations=100,1000
```

#### Option 2: Config File (Recommended for complex sweeps)

Edit `conf/sweep.yaml` to define your sweep parameters:

```yaml
defaults:
  - config  # Inherit from base config

hydra:
  sweeper:
    params:
      # For list parameters, use YAML list syntax with quoted strings
      group_comparison.group_a:
        - "[1]"
        - "[1,3]"
      group_comparison.group_b:
        - "[3]"
        - "[2,4]"
      # For single values, use comma-separated: value1,value2
      targets: all,male,female
      num_permutations: 100,1000
```

Then run with the `--multirun` flag:

```bash
python methylation/methylation_analysis.py --config-name=sweep --multirun
```

**Important Notes:**
- You **must** still use the `--multirun` flag on the command line
- For list parameters (like `group_a`), use quoted strings with brackets: `"[1]","[1,3]"`
- For scalar parameters (like `targets`), use comma-separated values: `all,male,female`
- The sweep config inherits all other settings from `config.yaml`
- See `conf/sweep_example.yaml` for more detailed examples

**Note:** The sweep config file (`sweep.yaml`) inherits all settings from `config.yaml` and only overrides the sweep parameters. This makes it easy to maintain and version control your sweep configurations.

Sweep results are saved in separate directories under `outputs/` with descriptive names based on the parameter combinations.

## Configuration

Edit `conf/config.yaml` to customize:

- **group_comparison**: Specify which groups to combine for comparison
  - `group_a`: List of groups to combine into group A (e.g., [1, 3])
  - `group_b`: List of groups to combine into group B (e.g., [2, 4])

- **input**: File paths
  - `feature_vector_path`: Path to feature vector joblib file
  - `groups_path`: Path to groups CSV file
  - `global_set_path`: Path to global_set.npy file

- **output**: Output configuration
  - `folder`: Output folder for results and visualizations

- **targets**: List of targets to process (e.g., ["all", "male", "female"])

- **delta_beta_threshold**: Minimum delta_beta value to keep

- **num_permutations**: Number of permutations for FDR calculation
  - **1000 (default)**: Recommended for stable FDR estimates and publication-quality results. The method examines the upper tail (80th-99.9th percentiles) of the delta_beta distribution, so more permutations provide more stable FDR estimates in this critical region.
  - **100**: ~10x faster for exploratory analysis, but FDR estimates have higher variance. Acceptable for quick exploration during development.
  - **500**: Good middle ground between speed and stability

- **visualization**: Visualization parameters
  - `exclude_control`: Whether to exclude control groups from visualization

## Output

The script generates:
- CSV files with methylation differences: `mouse_differences_{target}.csv`
- Visualization plots: `methylation_visualization_{target}.png`

All outputs are saved to the folder specified in `output.folder`.
