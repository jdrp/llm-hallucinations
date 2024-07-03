import json
import matplotlib.pyplot as plt
import argparse

def plot_histogram(json_file, outfile):
    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract the hallucination_pred scores
    scores = [item['hallucination_pred'] for item in data]

    # Plot the histogram
    plt.figure(figsize=(6, 4))  # Reduced plot size
    plt.hist(scores, bins=10, range=(0, 1), edgecolor='black', color='skyblue')  # Softer color for the bars
    plt.xlabel('Hallucination Prediction Score', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.xlim(0, 1)  # Set the x-axis range exactly from 0 to 1

    # Adjust layout to fix margins
    plt.tight_layout()

    # Save the plot to a file
    plt.savefig(outfile)
    plt.close()
    print(f"Histogram saved to {outfile}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot histogram of hallucination prediction scores from JSON file.')
    parser.add_argument('--data', type=str, required=True, help='Path to the JSON file')
    parser.add_argument('--outfile', type=str, required=True, help='Output file for the histogram plot')

    args = parser.parse_args()
    plot_histogram(args.data, args.outfile)
