import os
import Progression

# Create a folder to save the images
output_folder = "output_images"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Run the main code 10 times
for run in range(1, 11):
    fig = Progression.run_code()

    # Save the 2x2 grid of plots as a file
    fig.savefig(os.path.join(output_folder, f'run_{run}_ith_level_plots.png'))

    # Close the figure to free up memory
    fig.clf()

    print(f"Run {run} completed and figure saved.")