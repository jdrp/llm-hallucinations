import os
import json
import random
import argparse

def load_json(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def create_combined_medical_test_file(reports_dir, qch_dir, output_filepath, precision):
    combined_data = []

    report_files = [f for f in os.listdir(reports_dir) if f.endswith('.json')]
    for report_file in report_files:
        try:
            report_id = report_file.split('.')[0].replace('report', '')
            report_path = os.path.join(reports_dir, report_file)
            qch_path = os.path.join(qch_dir, f'report{report_id}_qch.json')
            
            report_content = load_json(report_path)
            qa_pairs_content = load_json(qch_path)

            report_clean = (
                f'Chief Complaint: {report_content["chief_complaint"]}\n'
                f'Admit Diagnosis: {report_content["admit_diagnosis"]}\n'
                'Discharge Diagnosis:\n'
            )
            for diagnosis in report_content["discharge_diagnosis"]:
                report_clean += f' - {diagnosis}\n'
            report_clean += f'Report Text: {report_content["report_text"]}\n\n'
            
            for qa in qa_pairs_content:
                if random.random() < precision:
                    chosen_answer = qa['right_answer']
                    hallucination_flag = 0
                else:
                    chosen_answer = qa['hallucinated_answer']
                    hallucination_flag = 1

                combined_data.append({
                    'context': report_clean,
                    'question': qa['question'],
                    'answer': chosen_answer,
                    'hallucination': hallucination_flag
                })
        except TypeError:
            continue

    with open(output_filepath, 'w') as f:
        json.dump(combined_data, f, indent=4)


def main() -> None:
    parser = argparse.ArgumentParser(description='Generate a combined medical test file with a given precision.')
    parser.add_argument('--reports_dir', type=str, required=True, help='Directory containing report JSON files.')
    parser.add_argument('--qch_dir', type=str, required=True, help='Directory containing question_correct_hallucinated JSON files.')
    parser.add_argument('--output_filepath', type=str, required=True, help='Filepath to save the combined medical test JSON file.')
    parser.add_argument('--precision', type=float, required=True, help='Precision value between 0 and 1 for choosing correct answers.')
    args = parser.parse_args()
    
    create_combined_medical_test_file(args.reports_dir, args.qch_dir, args.output_filepath, args.precision)


if __name__ == '__main__':
    main()
