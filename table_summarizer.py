# class to summarize table data and replace them in their respective positions in the final report.

import os
import requests
import pandas as pd
import csv
from io import StringIO

class TableSummarizer:
    def __init__(self, api_token):
        self.api_token = api_token

    # Function to summarize a chunk of table using Mistral LLM
    def summarize_chunk(self, table_chunk_text):
        headers = {
            "Authorization": f"Bearer {self.api_token}"  # Use the API token provided during initialization
        }
        
        prompt = (
            f"Summarize the following table chunk in very short and complete sentences, capturing all key details. "
            f"The summary should only highlight essential information and omit any unnecessary or irrelevant details. "
            f"Ensure that the table or prompt itself is not included in the summary. {table_chunk_text}. The summary is: "
        )

        payload = {
            "inputs": prompt,
        }

        response = requests.post(
            "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.3",
            headers=headers, json=payload
        )

        if response.status_code == 200:
            json_response = response.json()
            if 'generated_text' in json_response[0]:
                raw_summary = json_response[0]['generated_text'].strip()
                clean_summary = raw_summary.replace(prompt, '').replace(table_chunk_text, '').strip()
                return clean_summary
            else:
                print("Error: 'generated_text' not found in the response.")
                return "Could not summarize this chunk."
        else:
            print(f"Error: Received status code {response.status_code}")
            return "Could not summarize this chunk."

    # Function to split a DataFrame into smaller chunks
    def split_dataframe(self, df, chunk_size):
        chunks = [df[i:i + chunk_size] for i in range(0, df.shape[0], chunk_size)]
        return chunks

    # Function to parse the table text from CSV format
    def parse_table_csv(self, csv_text):
        data = StringIO(csv_text)
        reader = csv.reader(data)
        table_data = list(reader)
        if len(table_data) > 1:
            df = pd.DataFrame(table_data[1:], columns=table_data[0])
            return df
        else:
            print("Error: Table data is not structured properly.")
            return pd.DataFrame()

    # Function to process the input file and generate output
    def process_file(self, input_file, output_file):
        input_dir = os.path.dirname(input_file)
        output_dir = os.path.dirname(output_file)
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            output_lines = []

        current_table = []
        in_table = False

        for line in lines:
            if "<table start>" in line:
                in_table = True
                current_table = []
            elif "<table end>" in line and in_table:
                in_table = False
                csv_content = "\n".join(current_table)
                df = self.parse_table_csv(csv_content)
                if not df.empty:
                    chunk_size = 2  # Adjust the number of rows per chunk as needed
                    chunks = self.split_dataframe(df, chunk_size)
                    # Summarize each chunk and combine the summaries
                    table_summary = []
                    for chunk in chunks:
                        chunk_text = chunk.to_string(index=False)
                        chunk_summary = self.summarize_chunk(chunk_text)
                        table_summary.append(chunk_summary)

                    # Combine all the chunk summaries into one
                    final_summary = " ".join(table_summary)
                    output_lines.append(final_summary + '\n')
                else:
                    output_lines.append("Table could not be parsed or was empty.\n")
            elif in_table:
                current_table.append(line.strip())
            else:
                output_lines.append(line)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(output_lines)
            print(f"Output written to: {output_file}")


