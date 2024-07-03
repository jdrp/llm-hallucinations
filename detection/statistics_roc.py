import json
import argparse
import os
import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc

def get_model_name(filename: str, sep: str = '_'):
    parts = filename.rsplit('.', 1)[0].split(sep)
    return parts[-1]

def get_test_name(filename: str):
    base_name = os.path.basename(filename).rsplit('.', 1)[0]
    if 'nocot' in base_name:
        test_type = 'BCV'
    elif 'cot' in base_name:
        test_type = 'BCV-CoT'
    elif 'sca' in base_name:
        test_type = 'SCA'
    else:
        test_type = 'BCV'

    if 'medical' in base_name:
        role = 'Medical'
    elif 'agnostic' in base_name:
        role = 'Agnostic'
    else:
        role = 'Unknown'

    return f'{test_type} ({role})'

def classify_predictions(pred_data: list[dict], pred_key: str, true_key: str, threshold: float) -> tuple[list, list, list, list]:
    tp = []; tn = []; fp = []; fn = []
    for row in pred_data:
        pred = 1 if row[pred_key] >= threshold else 0
        if pred == row[true_key] == 1:
            tp.append(row)
        elif pred == row[true_key] == 0:
            tn.append(row)
        elif pred == 1 and row[true_key] == 0:
            fp.append(row)
        elif pred == 0 and row[true_key] == 1:
            fn.append(row)
        else:
            print(f'Error on row {row}')
            raise ValueError('Hallucination predictions/ground truth must be 0 or 1')
    return tp, tn, fp, fn

def calculate_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    return {
        'accuracy': (tp + tn) / (tp + tn + fp + fn),
        'precision': (precision := tp / (tp + fp) if (tp + fp) > 0 else 0),
        'sensitivity': (sensitivity := tp / (tp + fn) if (tp + fn) > 0 else 0.),
        'specificity': (specificity := tn / (tn + fp) if (tn + fp) > 0 else 0.),
        'f1': 2 * precision * sensitivity / (precision + sensitivity) if (precision + sensitivity) > 0 else 0.,
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0.,
        'fnr': fn / (fn + tp) if (fn + tp) > 0 else 0.,
        'acc_balanced': (sensitivity + specificity) / 2.
    }

def get_optimal_threshold(fpr, tpr, thresholds):
    # Calculate Youden's J statistic for each threshold
    j_scores = tpr - fpr
    # Get the index of the maximum J score
    optimal_idx = np.argmax(j_scores)
    # Get the optimal threshold
    optimal_threshold = thresholds[optimal_idx]
    return optimal_threshold

model_colors = {
    'llama3': 'blue',
    'llama3-gradient': 'lightblue',
    'nous-hermes2': 'red',
    'gemma': 'plum',
    'mistral': 'orange',
    'llama2': 'lightgreen',
    'llama2-13b': 'green',
    'Random': 'black'  # Adding color for the Random Chance line
}

def get_color_by_test(test_name: str):
    test_type = test_name.split()[0]
    if test_type == 'BCV':
        return 'rgba(255, 0, 0, 1)' if 'Medical' in test_name else 'rgba(255, 0, 0, 0.3)'
    elif test_type == 'BCV-CoT':
        return 'rgba(0, 200, 100, 1)' if 'Medical' in test_name else 'rgba(0, 200, 100, 0.3)'
    elif test_type == 'SCA':
        return 'rgba(0, 0, 255, 1)' if 'Medical' in test_name else 'rgba(0, 0, 255, 0.3)'
    return 'black'

def plot_roc_curves(true_labels_list: list[np.ndarray], predicted_scores_list: list[np.ndarray],
                    names: list[str], colors: list[str], save_path: str, legend_title: str) -> None:
    fig = go.Figure()

    for true_labels, predicted_scores, name, color in zip(true_labels_list, predicted_scores_list, names, colors):
        fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores, drop_intermediate=False)
        roc_auc = auc(fpr, tpr)
        optimal_threshold = get_optimal_threshold(fpr, tpr, thresholds)

        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines+markers',
            name=f'{name}',  # (AUC = {roc_auc:.3f})',
            line=dict(color=color, width=2),  # Thicker lines
            marker=dict(size=4),  # Larger markers
            hovertemplate='<b>Threshold: %{text}</b><br>TPR: %{y}<br>FPR: %{x}',
            text=thresholds
        ))

    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random',
        line=dict(dash='dash', color=model_colors['Random'], width=4)  # Thicker dashed line
    ))

    fig.update_layout(
        font=dict(size=16),  # Increase the font size
        xaxis=dict(
            title='False Positive Rate',
            title_font=dict(size=18),  # Increase the font size for the x-axis title
            tickfont=dict(size=16),  # Increase the font size for the x-axis ticks
            scaleanchor="y",
            scaleratio=1,
            range=[0, 1],  # Ensure square plot and set range for x-axis
            gridcolor='lightgrey',  # Color of the grid lines
            linecolor='black',  # Color of the axis line
            ticks='outside',  # Place ticks outside the plot
            tickcolor='black',  # Color of the ticks
        ),
        yaxis=dict(
            title='True Positive Rate',
            title_font=dict(size=18),  # Increase the font size for the y-axis title
            tickfont=dict(size=16),  # Increase the font size for the y-axis ticks
            constrain='domain',
            range=[0, 1],  # Set range for y-axis
            gridcolor='lightgrey',  # Color of the grid lines
            linecolor='black',  # Color of the axis line
            ticks='outside',  # Place ticks outside the plot
            tickcolor='black',  # Color of the ticks
        ),
        legend=dict(
            x=0.6, y=0.1,
            font=dict(size=16)  # Increase the font size for the legend
        ),
        width=550,  # Set the width of the plot
        height=550,  # Set the height of the plot to make it square
        plot_bgcolor='white',  # Background color of the plot area
        paper_bgcolor='white',  # Background color around the plot
        margin=dict(l=20, r=20, t=20, b=20)  # Set minimal margins
    )

    if save_path:
        fig.write_image(save_path)
        print(f"ROC curve saved to {save_path}")
    else:
        fig.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Hallucination Detection Results")
    parser.add_argument('--data', default='results/', help='Directory with prediction results or single JSON file with predictions and ground truth')
    parser.add_argument('--filter', default=None, nargs='+', help='Filter for filenames')
    parser.add_argument('--predkey', default='hallucination_pred', help='JSON key for the prediction')
    parser.add_argument('--truekey', default='hallucination', help='JSON key for ground truth')
    parser.add_argument('--roc', default=True, action=argparse.BooleanOptionalAction, help='Plot ROC curve')
    parser.add_argument('--legend', default=True, action=argparse.BooleanOptionalAction, help='Show legend')
    parser.add_argument('--outfile', default=None, help='Output file for the ROC curve plot')
    args = parser.parse_args()

    if args.filter:
        if not os.path.isdir(args.data):
            raise ValueError(f'The data argument should be a directory when using filter. Provided: {args.data}')
        datafiles = [os.path.join(args.data, f) for f in os.listdir(args.data) if os.path.isfile(os.path.join(args.data, f)) and all(fil in f for fil in args.filter)]
    else:
        if not os.path.isfile(args.data):
            raise ValueError(f'The data argument should be a file when not using filter. Provided: {args.data}')
        datafiles = [args.data]

    print(datafiles)

    true_labels_list = []
    predicted_scores_list = []
    test_names = []
    model_names = []

    for datafile in datafiles:
        if not os.path.isfile(datafile) or os.path.getsize(datafile) == 0:
            continue
        with open(datafile, 'r') as f:
            pred_data = json.load(f)
        print('-----------------------')
        model_name = get_model_name(datafile)
        test_name = get_test_name(datafile)
        test_names.append(test_name)
        model_names.append(model_name)
        print(f'MODEL: {model_name} TEST: {test_name}')

        true_labels = np.array([row[args.truekey] for row in pred_data])
        predicted_scores = np.array([row[args.predkey] for row in pred_data])
        true_labels_list.append(true_labels)
        predicted_scores_list.append(predicted_scores)

        fpr, tpr, thresholds = roc_curve(true_labels, predicted_scores, drop_intermediate=False)
        optimal_threshold = get_optimal_threshold(fpr, tpr, thresholds)
        print(f'Optimal threshold for {test_name}: {optimal_threshold:.3f}')

        tp, tn, fp, fn = (len(a) for a in classify_predictions(pred_data, args.predkey, args.truekey, optimal_threshold))
        print(f'TP: {tp}\nTN: {tn}\nFP: {fp}\nFN: {fn}')
        metrics = calculate_metrics(tp, tn, fp, fn)
        for k, v in metrics.items():
            print(f'{k}: {v:.3f}')

    unique_test_names = set(test_names)
    unique_model_names = set(model_names)

    if args.roc:
        if len(unique_model_names) == 1:
            legend_title = f'{unique_model_names.pop()} Tests'
            names = test_names
            colors = [get_color_by_test(name) for name in test_names]
        elif len(unique_test_names) == 1:
            legend_title = f'{unique_test_names.pop()} Models'
            names = model_names
            colors = [model_colors.get(name, 'black') for name in model_names]
        else:
            legend_title = 'Models and Tests'
            names = [f'{model} ({test})' for model, test in zip(model_names, test_names)]
            colors = [f'rgb({np.random.randint(0, 255)}, {np.random.randint(0, 255)}, {np.random.randint(0, 255)})' for _ in names]

        plot_roc_curves(true_labels_list, predicted_scores_list, names, colors, args.outfile, legend_title)

if __name__ == '__main__':
    main()
