import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import wandb
import argparse

def plot_beta_distributions(alpha_beta_pairs, x_values, project_name, backend):
    """
    Generates a Beta distribution plot and either logs it to W&B as an image
    or displays it locally.

    Parameters:
    alpha_beta_pairs (list of tuples): A list of (alpha, beta) parameter pairs.
    x_values (numpy.ndarray): X values for PDF calculation (typically [0, 1]).
    project_name (str): W&B project name (if backend='wandb').
    backend (str): 'wandb' or 'local'.
    """

    # --- 1. Create the Matplotlib figure ---
    # This figure will be used by both 'wandb' and 'local' backends
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
    # --- Figure creation complete ---

    current_run = None # Initialize current_run to None
    if backend == "wandb":
        # Construct a descriptive run name
        run_name_parts = [f'a{a}b{b}' for a,b in alpha_beta_pairs[:2]]
        run_name = f"beta_plot_{'_'.join(run_name_parts)}" # Descriptive name for plot logging
        if len(alpha_beta_pairs) > 2:
            run_name += "_etc"
        
        # Initialize W&B run
        current_run = wandb.init(project=project_name, name=run_name, job_type="visualization")
        
        if current_run is None:
            print("Error: Failed to initialize W&B run.")
            plt.close(fig) # Close the figure if W&B init fails
            return

        try:
            print(f"Logging to W&B project: {project_name}, run: {current_run.name}")
            # Log configuration
            current_run.config.update({
                "alpha_beta_pairs": alpha_beta_pairs,
                "backend": backend,
                "project_name_used": project_name,
                "num_x_points": len(x_values)
            })

            # --- Log the Matplotlib figure as an image to W&B ---
            current_run.log({"Beta Distributions Plot": wandb.Image(fig)})
            print("Plot image logged to W&B.")
        
        except Exception as e:
            print(f"An error occurred during W&B logging: {e}")

        finally:
            # Ensure wandb.finish is called to close the run
            if current_run: # Check if current_run was successfully initialized
                current_run.finish()
                print("W&B run finished.")
            # Close the figure after logging or if an error occurred,
            # and ensure it's not shown via plt.show() for wandb backend
            plt.close(fig) 

    elif backend == "local":
        plt.show() # Display plot locally
        print("Plot displayed locally. No W&B logging.")
        plt.close(fig) # Explicitly close the figure after showing
    else:
        print(f"Error: Unknown backend '{backend}'. Please choose 'wandb' or 'local'.")
        # Ensure figure is closed if it was created but backend is invalid
        if fig:
             plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize Beta distributions: log plot image to W&B or display plot locally.")
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
        help="Output backend: 'wandb' to log plot image to W&B, 'local' to display plot locally only."
    )
    args = parser.parse_args()

    x = np.linspace(0, 1, 500) 
    
    parameter_pairs = [
        (1, 1), (2, 2), (5, 5), (2, 5), (5, 2),       
        (0.5, 0.5), (0.8, 0.8),                       
        (1, 3), (3, 1),                               
        (10, 30), (30, 10),                           
        (1.5, 1.5), (1.5, 3.5)                        
    ]

    plot_beta_distributions(parameter_pairs, x, args.project_name, args.backend)
