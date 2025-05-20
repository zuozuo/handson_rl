import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import wandb
import argparse

def plot_beta_distributions(alpha_beta_pairs, x_values, project_name, backend):
    """
    Generates Beta distribution data and either logs it to W&B as tables
    or plots it locally.

    Parameters:
    alpha_beta_pairs (list of tuples): A list of (alpha, beta) parameter pairs.
    x_values (numpy.ndarray): X values for PDF calculation (typically [0, 1]).
    project_name (str): W&B project name (if backend='wandb').
    backend (str): 'wandb' or 'local'.
    """
    if backend == "wandb":
        # Construct a descriptive run name
        run_name_parts = [f'a{a}b{b}' for a,b in alpha_beta_pairs[:2]] # Use first two pairs for brevity
        run_name = f"beta_data_log_{'_'.join(run_name_parts)}"
        if len(alpha_beta_pairs) > 2:
            run_name += "_etc"
        
        # Initialize W&B run
        current_run = wandb.init(project=project_name, name=run_name, job_type="data_generation")
        
        if current_run is None:
            print("Error: Failed to initialize W&B run.")
            return # Exit if wandb.init fails

        try:
            print(f"Logging to W&B project: {project_name}, run: {current_run.name}")
            # Log configuration
            current_run.config.update({
                "alpha_beta_pairs": alpha_beta_pairs,
                "backend": backend,
                "project_name_used": project_name,
                "num_x_points": len(x_values) # Log the number of points used for x_values
            })

            pdf_data_for_table = []
            summary_stats_for_table = []

            for a, b in alpha_beta_pairs:
                # Calculate PDF values for current (a,b) pair
                y_values = beta.pdf(x_values, a, b)
                for x_val, pdf_val in zip(x_values, y_values):
                    # Ensure all data for the table are floats for consistency
                    pdf_data_for_table.append([float(a), float(b), float(x_val), float(pdf_val)])

                # Calculate summary statistics
                mean_val = beta.mean(a, b)
                var_val = beta.var(a, b)
                
                mode_val = np.nan # Default mode to NaN
                if a > 1 and b > 1: # Mode is (a-1)/(a+b-2) only if a > 1 and b > 1
                    mode_val = (a - 1) / (a + b - 2)
                # For other cases (uniform, J-shaped, L-shaped, U-shaped), 
                # mode is at boundaries, undefined, or not unique as a single central value.
                # np.nan is a suitable representation for these scenarios in the table.
                
                summary_stats_for_table.append([float(a), float(b), float(mean_val), float(mode_val), float(var_val)])

            # Create W&B Tables
            pdf_table = wandb.Table(data=pdf_data_for_table, columns=["alpha", "beta", "x", "pdf_value"])
            summary_table = wandb.Table(data=summary_stats_for_table, columns=["alpha", "beta", "mean", "mode", "variance"])

            # Log tables to W&B
            current_run.log({
                "Beta PDF Data": pdf_table,
                "Beta Summary Statistics": summary_table
            })
            print("Beta distribution PDF data and summary statistics logged to W&B.")
        
        except Exception as e:
            print(f"An error occurred during W&B logging: {e}")

        finally:
            # Ensure wandb.finish is called to close the run
            current_run.finish()
            print("W&B run finished.")

    elif backend == "local":
        # Local plotting logic remains the same
        fig, ax = plt.subplots(figsize=(12, 8))
        for a, b in alpha_beta_pairs:
            y_values = beta.pdf(x_values, a, b)
            ax.plot(x_values, y_values, label=f'Beta({a}, {b})')

        ax.set_title('Beta Distribution for Different α and β Parameters', fontsize=16)
        ax.set_xlabel('x (Value of the random variable)', fontsize=12)
        ax.set_ylabel('Probability Density Function (PDF)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        
        plt.show() # Display plot locally
        print("Plot displayed locally. No W&B logging.")
        # Explicitly close the figure after showing it locally to free resources
        plt.close(fig) 
    else:
        # Handle unknown backend
        print(f"Error: Unknown backend '{backend}'. Please choose 'wandb' or 'local'.")


if __name__ == '__main__':
    # Update parser description and help string for clarity
    parser = argparse.ArgumentParser(description="Visualize Beta distributions: log data tables to W&B or display plot locally.")
    parser.add_argument(
        "--project_name",
        type=str,
        default="handson_rl-mab-visualizations",
        help="Name of the Weights & Biases project to log to (used if backend is 'wandb')."
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["wandb", "local"], 
        default="local",            
        help="Output backend: 'wandb' to log data tables to W&B, 'local' to display plot locally only."
    )
    args = parser.parse_args()

    x = np.linspace(0, 1, 500) # Define x values for PDF calculation
    
    # Define a list of (alpha, beta) pairs to visualize/log
    parameter_pairs = [
        (1, 1), (2, 2), (5, 5), (2, 5), (5, 2),       # Symmetric and asymmetric unimodal
        (0.5, 0.5), (0.8, 0.8),                       # U-shaped
        (1, 3), (3, 1),                               # L-shaped and J-shaped (mode at boundary)
        (10, 30), (30, 10),                           # More sharply peaked
        (1.5, 1.5), (1.5, 3.5)                        # Additional examples
    ]

    plot_beta_distributions(parameter_pairs, x, args.project_name, args.backend)
