"""Main script for methylation analysis with configurable group comparisons."""
import logging
import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from hydra import compose, initialize, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from tqdm import tqdm
import hydra

from methylation_stats import permutation_fdr_delta_beta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def format_group_label(group_list):
    """
    Format group list for display, handling control group syntax.
    
    Args:
        group_list: List of groups, can include strings like "c:3" for control of group 3
    
    Returns:
        Formatted string for display (e.g., "3", "Control of 3", "3+4", "Control of 3+2")
    """
    labels = []
    for g in group_list:
        g_str = str(g)
        if g_str.startswith("c:"):
            # Control group syntax
            try:
                control_group = g_str.split(":")[1]
                labels.append(f"Control of {control_group}")
            except IndexError:
                labels.append(g_str)
        else:
            labels.append(g_str)
    
    return '+'.join(labels)


def create_modified_groups(
    groups_df: pd.DataFrame,
    sample_names: list,
    group_comparison: dict,
    target: str = "all"
):
    """
    Create modified groups based on group comparison configuration.
    
    Supports both simple format (backward compatible) and control group comparisons:
    - Simple: group_a: [3] compares treatment groups
    - Control: group_a: ["c:3"] compares control of group 3
    - Mixed: group_a: [3], group_b: ["c:2"] compares group 3 vs control of group 2
    
    Args:
        groups_df: DataFrame with sample, group, and Sex columns
        sample_names: List of sample names (without .joblib extension)
        group_comparison: Dict with keys 'group_a' and 'group_b' containing lists of groups to combine
                         Can include strings like "c:3" to specify control of group 3
        target: Target subset ('all', 'male', 'female')
    
    Returns:
        List of modified group labels
    """
    groups_dict = dict(zip(groups_df["sample"].astype(str).values, groups_df.group.values))
    sex_dict = dict(zip(groups_df["sample"].astype(str).values, groups_df.Sex.values))
    
    groups = [groups_dict[name] for name in sample_names]
    
    # Parse group_a and group_b to extract treatment groups and control specifications
    def parse_group_spec(group_list):
        """Parse group specification, handling both simple and control syntax."""
        treatment_groups = []
        control_of_groups = []
        
        for g in group_list:
            g_str = str(g)
            if g_str.startswith("c:"):
                # Control group syntax: "c:3" means control of group 3
                try:
                    control_group = int(g_str.split(":")[1])
                    control_of_groups.append(control_group)
                except (ValueError, IndexError):
                    logger.warning(f"Invalid control group syntax: {g_str}. Expected format: 'c:X' where X is a group number.")
            else:
                # Regular treatment group
                try:
                    treatment_groups.append(int(g_str))
                except ValueError:
                    # Keep as string if not a number
                    treatment_groups.append(g_str)
        
        return treatment_groups, control_of_groups
    
    group_a_treatment, group_a_control_of = parse_group_spec(group_comparison['group_a'])
    group_b_treatment, group_b_control_of = parse_group_spec(group_comparison['group_b'])
    
    if target == 'all':
        modified_groups = []
        group_a_treatment_str = [str(g) for g in group_a_treatment]
        group_b_treatment_str = [str(g) for g in group_b_treatment]
        
        for g in groups:
            g_str = str(g)
            # Check if it's a control group that should be in group_a
            if g_str == 'c' and group_a_control_of:
                # Include all control samples when comparing controls (no filtering by associated group for now)
                modified_groups.append(f"group_a_{target}")
            # Check if it's a control group that should be in group_b
            elif g_str == 'c' and group_b_control_of:
                modified_groups.append(f"group_b_{target}")
            # Check if it's a treatment group in group_a
            elif g_str in group_a_treatment_str:
                modified_groups.append(f"group_a_{target}")
            # Check if it's a treatment group in group_b
            elif g_str in group_b_treatment_str:
                modified_groups.append(f"group_b_{target}")
            else:
                modified_groups.append(g_str)
    else:
        sex_groups = [f"{groups_dict[name]}_{sex_dict[name]}" for name in sample_names]
        modified_groups = []
        group_a_treatment_str = [f"{str(g)}_{target}" for g in group_a_treatment]
        group_b_treatment_str = [f"{str(g)}_{target}" for g in group_b_treatment]
        
        for g in sex_groups:
            # Check if it's a control group of the target sex
            if g == f'c_{target}':
                # Check if control should be in group_a or group_b
                if group_a_control_of:
                    modified_groups.append(f'group_a_{target}')
                elif group_b_control_of:
                    modified_groups.append(f'group_b_{target}')
                else:
                    # Keep control separate if not being compared
                    modified_groups.append(f'c_{target}')
            # Check if it's a treatment group in group_a
            elif g in group_a_treatment_str:
                modified_groups.append(f'group_a_{target}')
            # Check if it's a treatment group in group_b
            elif g in group_b_treatment_str:
                modified_groups.append(f'group_b_{target}')
            else:
                modified_groups.append('other')
    
    return modified_groups


def get_differences_between_groups(
    data: dict,
    groups_df: pd.DataFrame,
    names: np.ndarray,
    group_comparison: dict,
    target: str = "all",
    delta_beta_threshold: float = 20.0,
    num_permutations: int = 1000
):
    """
    Calculate differences between groups.
    
    Args:
        data: Dictionary of feature vectors (from joblib)
        groups_df: DataFrame with group information
        names: Pre-loaded global_set.npy array
        group_comparison: Dict with group_a and group_b lists
        target: Target subset ('all', 'male', 'female')
        delta_beta_threshold: Minimum delta_beta value to keep
        num_permutations: Number of permutations for FDR calculation
    
    Returns:
        DataFrame with results
    """
    logger.info(f"Processing {len(data)} samples for target '{target}'")
    group_a_label = format_group_label(group_comparison['group_a'])
    group_b_label = format_group_label(group_comparison['group_b'])
    logger.info(f"Group comparison: {group_a_label} vs {group_b_label}")
    logger.info(f"Using {num_permutations} permutations for FDR calculation")
    
    embeddings = []
    sample_names = []
    for name, arr in data.items():
        sample_names.append(name.split('.')[0])
        embeddings.append(arr)
    embeddings = np.float32(embeddings)
    logger.debug(f"Created embeddings array of shape {embeddings.shape}")
    
    modified_groups = create_modified_groups(
        groups_df, sample_names, group_comparison, target
    )
    logger.debug(f"Created {len(set(modified_groups))} modified groups: {set(modified_groups)}")
    
    tau, tbl, results_df, means_obs, levels = permutation_fdr_delta_beta(
        embeddings, np.array(modified_groups), B=num_permutations
    )
    logger.info(f"FDR analysis complete. Threshold (tau): {tau:.4f}")
    logger.info(f"Found {len(results_df)} features before threshold filtering")
    
    results_df = results_df[results_df.delta_beta > delta_beta_threshold]
    logger.info(f"After threshold filtering (>{delta_beta_threshold}): {len(results_df)} features")
    
    # Handle empty results case
    if len(results_df) == 0:
        logger.warning(f"No features found above threshold {delta_beta_threshold}. Returning empty DataFrame.")
        # Create empty DataFrame with expected columns for downstream processing
        # Include all columns that might be needed by visualize function
        empty_df = pd.DataFrame({
            'feature': pd.Series(dtype='int64'),
            'delta_beta': pd.Series(dtype='float64'),
            'db_obs': pd.Series(dtype='float64'),
            'best_pair': pd.Series(dtype=object),  # Required by visualize function
            'feature_name': pd.Series(dtype=object),
            'chromosome': pd.Series(dtype=object),
            'location': pd.Series(dtype='int64')
        })
        return empty_df, group_comparison
    
    # Use pre-loaded names array instead of loading from file
    results_df["feature_name"] = names[results_df["feature"]]
    
    # Optimization 2: Vectorize string operations instead of using .apply()
    # Original code (commented for reference):
    # results_df["chromosome"] = results_df["feature_name"].apply(
    #     lambda x: x.split('.')[0] + '.' + x.split('.')[1][0]
    # )
    # results_df["location"] = results_df["feature_name"].apply(
    #     lambda x: x.split('.')[1][1:]
    # )
    # Vectorized version (produces identical results):
    splits = results_df["feature_name"].str.split('.', expand=True, n=1)
    # Access columns by name (str.split with expand=True creates columns 0, 1, etc.)
    # Check if splits has the expected columns
    if len(splits.columns) < 2:
        raise ValueError(f"Expected 2 columns from split, got {len(splits.columns)}. Feature names may have unexpected format.")
    results_df["chromosome"] = splits[0].astype(str) + '.' + splits[1].str[0]
    results_df["location"] = splits[1].str[1:]
    
    results_df = results_df.sort_values(['chromosome', 'location'])
    results_df["location"] = results_df["location"].astype(int)
    
    return results_df, group_comparison


def visualize(
    csv: pd.DataFrame,
    group_comparison: dict,
    target: str,
    names: np.ndarray,
    output_path: str,
    exclude_control: bool = True,
    font_sizes: dict = None
):
    """
    Visualize methylation patterns across chromosomes.
    
    Args:
        csv: DataFrame with methylation results
        group_comparison: Dict with group_a and group_b lists
        target: Target subset ('all', 'male', 'female')
        names: Pre-loaded global_set.npy array
        output_path: Path to save the figure
        exclude_control: Whether to exclude control groups from visualization
        font_sizes: Dict with font sizes for different text elements (title, chromosome_label, location_label, legend, empty_plot_message)
    """
    # Set default font sizes if not provided
    if font_sizes is None:
        font_sizes = {
            'title': 14,
            'chromosome_label': 12,
            'location_label': 9,
            'legend': 10,
            'empty_plot_message': 16
        }
    
    logger.info(f"Creating visualization for target '{target}' with {len(csv)} features")
    
    # Handle empty DataFrame case
    if len(csv) == 0:
        logger.warning("No features to visualize. Creating empty plot.")
        plt.figure(figsize=(18, 8))
        plt.text(0.5, 0.5, f'No features found above threshold for {target}', 
                ha='center', va='center', fontsize=font_sizes['empty_plot_message'], transform=plt.gca().transAxes)
        plt.title(f'Methylation Differences - {target} (No features found)', fontsize=font_sizes['title'])
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    if exclude_control:
        csv = csv[~csv.best_pair.str.contains('c')]
        logger.debug(f"After excluding control: {len(csv)} features")
        
        # Check again after filtering
        if len(csv) == 0:
            logger.warning("No features remaining after excluding control. Creating empty plot.")
            plt.figure(figsize=(18, 8))
            plt.text(0.5, 0.5, f'No features found after excluding control for {target}', 
                    ha='center', va='center', fontsize=font_sizes['empty_plot_message'], transform=plt.gca().transAxes)
            plt.title(f'Methylation Differences - {target} (No features found)', fontsize=font_sizes['title'])
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            return
    
    # Optimization 4: Cache chromosome start/stop calculations (already efficient, but optimized string operations)
    # Calculate chromosome boundaries once and reuse
    starts = {}
    stops = {}
    for n in names:
        # Optimize: split once and reuse
        parts = n.split('.')
        chrom = parts[0] + '.' + parts[1][0]
        location_val = int(parts[1][1:])
        starts[chrom] = min(starts.get(chrom, float('inf')), location_val)
        stops[chrom] = max(stops.get(chrom, float('-inf')), location_val)
    
    # Set up the figure with a large size for publication quality
    plt.figure(figsize=(18, 8))
    
    # Get unique chromosomes
    chromosomes = sorted(csv['chromosome'].unique())
    
    # Create a color map for chromosomes using hash of chromosome name
    def get_stable_color(chrom_name):
        hash_val = hash(str(chrom_name)) % 100
        return plt.cm.rainbow(hash_val / 100)
    
    chrom_color_dict = {chrom: get_stable_color(chrom) for chrom in chromosomes}
    
    # Initialize variables to track positions
    current_pos = 0
    chrom_positions = {}
    
    # Get group labels for comparison - find actual column names in the dataframe
    group_a_label = f"group_a_{target}"
    group_b_label = f"group_b_{target}"
    
    # Find the actual column names that match our group labels
    mean_beta_cols = [col for col in csv.columns if col.startswith('mean_beta_')]
    group_a_col = None
    group_b_col = None
    
    for col in mean_beta_cols:
        if group_a_label in col:
            group_a_col = col
        if group_b_label in col:
            group_b_col = col
    
    if group_a_col is None or group_b_col is None:
        raise ValueError(
            f"Could not find mean_beta columns for group comparison. "
            f"Expected columns containing '{group_a_label}' and '{group_b_label}'. "
            f"Found columns: {mean_beta_cols}"
        )
    
    # Plot each chromosome as a horizontal bar
    for chrom in chromosomes:
        # Get data for this chromosome and sort by location
        chrom_data = csv[csv['chromosome'] == chrom].sort_values('location')
        width = 100  # Fixed width for each chromosome
        
        # Plot chromosome bar
        plt.fill_between(
            [current_pos, current_pos + width],
            [-0.5, -0.5], [0.5, 0.5],
            color=chrom_color_dict[chrom], alpha=0.3
        )
        
        # Store chromosome position for later annotation
        chrom_positions[chrom] = (current_pos, width)
        
        # Get chromosome start and stop positions
        chrom_start = starts[chrom]
        chrom_stop = stops[chrom]
        
        # Optimization 3: Vectorize visualization loop instead of using .iterrows()
        # Original code (commented for reference):
        # for _, row in chrom_data.iterrows():
        #     x = current_pos + (row['location'] - chrom_start) / (chrom_stop - chrom_start) * width
        #     mean_beta_a = row[group_a_col]
        #     mean_beta_b = row[group_b_col]
        #     linewidth = abs(mean_beta_a - mean_beta_b) / 10
        #     if mean_beta_a > mean_beta_b:
        #         plt.plot([x, x], [-0.5, 0.5], color='blue', alpha=0.7, linewidth=linewidth)
        #     else:
        #         plt.plot([x, x], [-0.5, 0.5], color='green', alpha=0.7, linewidth=linewidth)
        #     loc_text = str(row['location'] * 1000)
        #     plt.text(x, -0.6, loc_text, rotation=45, horizontalalignment='right', verticalalignment='top', fontsize=6)
        
        # Vectorized version (produces identical results):
        locations = chrom_data['location'].values
        mean_beta_a_values = chrom_data[group_a_col].values
        mean_beta_b_values = chrom_data[group_b_col].values
        
        # Calculate all x positions at once
        x_positions = current_pos + (locations - chrom_start) / (chrom_stop - chrom_start) * width
        linewidths = np.abs(mean_beta_a_values - mean_beta_b_values) / 10
        
        # Plot lines in batches by color to maintain identical visual output
        blue_mask = mean_beta_a_values > mean_beta_b_values
        green_mask = ~blue_mask
        
        # Plot blue lines (maintain order by iterating in original order)
        for idx in range(len(chrom_data)):
            if blue_mask[idx]:
                plt.plot([x_positions[idx], x_positions[idx]], [-0.5, 0.5], 
                        color='blue', alpha=0.7, linewidth=linewidths[idx])
        
        # Plot green lines (maintain order by iterating in original order)
        for idx in range(len(chrom_data)):
            if green_mask[idx]:
                plt.plot([x_positions[idx], x_positions[idx]], [-0.5, 0.5], 
                        color='green', alpha=0.7, linewidth=linewidths[idx])
        
        # Add location text with aggressive thinning to prevent overlap
        # Only show a small subset of labels to keep visualization readable
        if len(chrom_data) > 0:
            # Calculate minimum spacing needed for readable labels (larger spacing)
            min_spacing = width / 15  # Increased from width/50 for much more spacing
            
            # Aggressively thin out labels based on density
            if len(chrom_data) > 200:
                max_labels = 20  # Maximum 20 labels per chromosome
            elif len(chrom_data) > 100:
                max_labels = 25  # Maximum 25 labels
            elif len(chrom_data) > 50:
                max_labels = 30  # Maximum 30 labels
            elif len(chrom_data) > 20:
                max_labels = 15  # Maximum 15 labels for medium density
            else:
                max_labels = len(chrom_data)  # Show all if very few features
            
            label_step = max(1, len(chrom_data) // max_labels) if max_labels < len(chrom_data) else 1
            
            # Track last label position to prevent overlaps
            last_label_x = -float('inf')
            
            # Add location text with improved spacing
            for idx, (_, row) in enumerate(chrom_data.iterrows()):
                # Only show label if it's in our step pattern
                if idx % label_step != 0:
                    continue
                
                # Check if this label would overlap with the previous one
                if abs(x_positions[idx] - last_label_x) < min_spacing:
                    continue
                
                loc_text = str(int(row['location'] * 1000))
                plt.text(
                    x_positions[idx], -0.8, loc_text,  # Moved further down
                    rotation=45,
                    horizontalalignment='right',
                    verticalalignment='top',
                    fontsize=font_sizes['location_label'],
                    alpha=0.8  # Slightly more opaque for better visibility
                )
                last_label_x = x_positions[idx]
        
        current_pos += width + 20  # Space between chromosomes
    
    # Add chromosome labels with improved readability
    for chrom, (pos, width) in chrom_positions.items():
        plt.text(
            pos + width/2, 1.2, chrom,
            horizontalalignment='center',
            verticalalignment='bottom',
            fontsize=font_sizes['chromosome_label'],
            fontweight='bold'
        )
    
    # Customize the plot
    plt.ylim(-3.0, 1.5)  # Even more bottom margin for location labels to prevent overlap
    plt.xlim(-20, current_pos + 20)
    plt.axis('off')
    
    # Create legend labels with proper formatting for control groups
    group_a_label = format_group_label(group_comparison['group_a'])
    group_b_label = format_group_label(group_comparison['group_b'])
    
    # Add legend
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label=f'{group_a_label} {target}'),
        Patch(facecolor='green', alpha=0.7, label=f'{group_b_label} {target}')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=font_sizes['legend'])
    
    # Add title
    plt.title('Methylation Patterns Across Chromosomes', pad=20, fontsize=font_sizes['title'])
    
    # Adjust layout to prevent location labels from being cut off
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])  # Increased bottom margin
    
    # Save figure
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved visualization to {output_path}")


def main(cfg: DictConfig):
    """Main function to run methylation analysis."""
    # Get project root from hydra output directory
    # When using sweeps, we're in outputs/YYYY-MM-DD/HH-MM-SS/run_name/
    # We need to go up to find the project root (where conf/ and methylation/ are)
    hydra_output_dir = Path.cwd()
    
    # Try to find project root by looking for conf/ directory
    # Start from current directory and go up
    project_root = hydra_output_dir
    max_levels = 5  # Safety limit
    for _ in range(max_levels):
        if (project_root / "conf" / "config.yaml").exists():
            break
        project_root = project_root.parent
    else:
        # Fallback: assume we're in outputs/... and go up 3 levels
        project_root = hydra_output_dir.parent.parent.parent
    
    logger.debug(f"Detected project root: {project_root}")
    
    # Resolve paths relative to project root
    feature_vector_path = project_root / cfg.input.feature_vector_path
    groups_path = project_root / cfg.input.groups_path
    global_set_path = project_root / cfg.input.global_set_path
    
    # Use hydra output directory for results
    output_dir = Path(cfg.output.folder) if cfg.output.folder.startswith('/') else hydra_output_dir / cfg.output.folder
    
    logger.info("=" * 80)
    logger.info("Starting methylation analysis")
    logger.info(f"Project root: {project_root}")
    logger.info(f"Output directory: {output_dir}")
    group_a_label = format_group_label(cfg.group_comparison['group_a'])
    group_b_label = format_group_label(cfg.group_comparison['group_b'])
    logger.info(f"Group comparison: {group_a_label} vs {group_b_label}")
    logger.info(f"Targets: {cfg.targets}")
    logger.info("=" * 80)
    
    # Load data
    logger.info(f"Loading feature vectors from {feature_vector_path}")
    data = joblib.load(feature_vector_path)
    logger.info(f"Loaded {len(data)} samples")
    
    logger.info(f"Loading groups from {groups_path}")
    groups_df = pd.read_csv(groups_path, sep="\t")
    logger.info(f"Loaded {len(groups_df)} group entries")
    
    # Load global_set.npy once and reuse (Optimization 1: Cache file loading)
    logger.info(f"Loading global_set from {global_set_path}")
    names = np.load(str(global_set_path))
    logger.info(f"Loaded {len(names)} feature names")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each target
    targets = cfg.targets if isinstance(cfg.targets, list) else [cfg.targets]
    for target in tqdm(targets, desc="Processing targets"):
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing target: {target}")
        logger.info(f"{'='*80}")
        
        # Get differences between groups
        results_df, group_comparison = get_differences_between_groups(
            data=data,
            groups_df=groups_df,
            names=names,  # Pass pre-loaded array instead of path
            group_comparison=cfg.group_comparison,
            target=target,
            delta_beta_threshold=cfg.delta_beta_threshold,
            num_permutations=cfg.num_permutations
        )
        
        # Create output filenames with group comparison info
        # Use simplified format for filenames (replace "c:" with "c" for cleaner filenames)
        def format_filename_group(group_list):
            """Format group list for filename, replacing 'c:3' with 'c3'."""
            parts = []
            for g in group_list:
                g_str = str(g)
                if g_str.startswith("c:"):
                    parts.append(f"c{g_str.split(':')[1]}")
                else:
                    parts.append(str(g))
            return '+'.join(parts)
        
        group_a_str = format_filename_group(group_comparison['group_a'])
        group_b_str = format_filename_group(group_comparison['group_b'])
        filename_suffix = f"g{group_a_str}_vs_g{group_b_str}_{target}"
        
        # Save results
        output_csv_path = output_dir / f"mouse_differences_{filename_suffix}.csv"
        results_df.to_csv(output_csv_path, index=False)
        logger.info(f"Saved results to {output_csv_path}")
        
        # Visualize
        output_fig_path = output_dir / f"methylation_visualization_{filename_suffix}.png"
        visualize(
            csv=results_df,
            group_comparison=cfg.group_comparison,
            target=target,
            names=names,  # Pass pre-loaded array instead of path
            output_path=str(output_fig_path),
            exclude_control=cfg.visualization.exclude_control,
            font_sizes=cfg.visualization.font_sizes if hasattr(cfg.visualization, 'font_sizes') else None
        )
    
    logger.info(f"\n{'='*80}")
    logger.info("Analysis complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"{'='*80}")


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def hydra_main_func(cfg: DictConfig) -> None:
    """Hydra main entry point for methylation analysis."""
    main(cfg)


if __name__ == "__main__":
    hydra_main_func()
