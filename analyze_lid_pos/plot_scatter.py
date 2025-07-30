import pandas as pd
import matplotlib.pyplot as plt
import os

# Load your dataframe
df = pd.read_excel("wrong_screwed_lid/data/inner_circle_df.xlsx")  # Uncomment if you need to load from Excel

# Assuming you already have a DataFrame `df` with `centre_y` and other metrics like `ellipse_area`, `area_ratio`, etc.
metrics = ['ellipse_area', 'area_ratio', 'ellipse_area2', 'area_ratio2', 'diff', 'angle']

# Directory to save the plots
output_folder = "wrong_screwed_lid/data/"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Plot scatter for each metric against `centre_y`
for metric in metrics:
    plt.figure(figsize=(8, 6))
    plt.scatter(df['centre_y'], df[metric], color='blue', edgecolors='black', alpha=0.5)
    plt.title(f"Scatter Plot of {metric} vs. centre_y")
    plt.xlabel('centre_y')
    plt.ylabel(metric)
    plt.grid(True)

    # Save the plot as an image file
    plot_path = os.path.join(output_folder, f"{metric}_vs_centre_y.png")
    plt.savefig(plot_path)
    plt.close()  # Close the plot to free memory

    print(f"Saved plot for {metric} vs centre_y at: {plot_path}")
