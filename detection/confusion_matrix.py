import numpy as np
import matplotlib.pyplot as plt
import argparse


def plot_confusion_matrix(tp, fp, tn, fn, outfile):
    cm = np.array([[tp, fn], [fp, tn]])
    plt.figure(figsize=(3, 3))  # Smaller figure size
    plt.matshow(cm, cmap='Blues', fignum=1)

    for (i, j), val in np.ndenumerate(cm):
        plt.text(j, i, f'{val}', ha='center', va='center', color='black', fontsize=14)

    plt.xlabel('Predicted', fontsize=14, labelpad=10)
    plt.ylabel('True', rotation=90, labelpad=10, fontsize=14)
    plt.xticks([0, 1], ['Hallucinated', 'Factual'], fontsize=12)
    plt.yticks([0, 1], ['Hallucinated', 'Factual'], rotation=90, va='center', fontsize=12)

    plt.gca().xaxis.set_ticks_position('top')
    plt.gca().xaxis.set_label_position('top')

    plt.tight_layout(pad=2.0)
    plt.savefig(outfile, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {outfile}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate and save confusion matrix.')
    parser.add_argument('--tp', type=int, required=True, help='True Positives')
    parser.add_argument('--fp', type=int, required=True, help='False Positives')
    parser.add_argument('--tn', type=int, required=True, help='True Negatives')
    parser.add_argument('--fn', type=int, required=True, help='False Negatives')
    parser.add_argument('--outfile', type=str, required=True, help='Output file for confusion matrix')

    args = parser.parse_args()
    plot_confusion_matrix(args.tp, args.fp, args.tn, args.fn, args.outfile)
