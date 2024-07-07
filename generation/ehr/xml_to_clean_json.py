import os
import xml.etree.ElementTree as ET
import pandas as pd
import json
import re
import boto3


excel_path = '../../../data/icd/CMS32_DESC_LONG_SHORT_DX.xlsx'
reports_folder = 'reports'
output_folder = 'reports_json'

aws_access_key = '********'
aws_secret_key = '********'
aws_region = 'eu-central-1'
s3_bucket = 'tfgdata'

s3 = boto3.client(
    service_name='s3',
    region_name=aws_region,
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key
)

for bucket in s3.list_buckets()['Buckets']:
    print(bucket)


def create_dx_descriptions_dict(excel_path):
    # create dx dictionary mapping icd codes to their meaning
    dx_df = pd.read_excel(excel_path)
    dx_descriptions = pd.Series(dx_df['LONG DESCRIPTION'].values, index=dx_df['DIAGNOSIS CODE']).to_dict()
    return dx_descriptions


def clean_text(text):
    # remove the initial anonymization notice if present
    text = re.sub(r'^\[\s*Report de-identified .+?]\s*\n', '', text, flags=re.DOTALL)
    # TODO **placeholders ???
    # text = re.sub(r'(\*\*(NAME|DATE))\[\w+\]', r'\1', text, flags=re.IGNORECASE)
    # identify the last line of underscores and remove everything after it
    parts = text.rsplit('_', 1)
    if len(parts) > 1:
        text = parts[0]
    # remove lines of underscores
    text = re.sub(r'_+', '', text)
    # all lowercase
    text = text.lower()
    # remove unnecessary whitespace and line breaks
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def translate_report_to_json(reports_folder, output_folder, dx_descriptions):
    # create output folder if needed
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for report_file in os.listdir(reports_folder):
        if report_file.endswith('.xml'):
            file_path = os.path.join(reports_folder, report_file)
            tree = ET.parse(file_path)
            root = tree.getroot()

            # extract relevant content
            chief_complaint = root.find('chief_complaint').text.strip()
            admit_code = root.find('admit_diagnosis').text.replace('.', '')
            admit_diagnosis_translated = dx_descriptions.get(admit_code, "Unknown diagnosis code")
            discharge_codes = root.find('discharge_diagnosis').text.split(',')
            discharge_diagnosis_translated = [
                dx_descriptions.get(code.strip().replace('.', ''), "Unknown diagnosis code") for code in discharge_codes
                if code.strip()]
            report_text = root.find('report_text').text.strip()

            report_data = {
                "chief_complaint": chief_complaint.lower(),
                "admit_diagnosis": admit_diagnosis_translated.lower(),
                "discharge_diagnosis": [s.lower() for s in discharge_diagnosis_translated],
                "report_text": clean_text(report_text)
            }

            # save the report data in json format
            json_file_name = os.path.splitext(report_file)[0] + '.json'
            output_path = os.path.join(output_folder, json_file_name)
            with open(output_path, 'w') as json_file:
                json.dump(report_data, json_file, indent=4)


os.makedirs(output_folder, exist_ok=True)
dx_descriptions = create_dx_descriptions_dict(excel_path)
translate_report_to_json(reports_folder, output_folder, dx_descriptions)