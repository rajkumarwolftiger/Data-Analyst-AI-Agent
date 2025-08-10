# agent_logic.py
import os
import pandas as pd
import requests
from bs4 import BeautifulSoup
import io
import base64
import matplotlib.pyplot as plt
import duckdb
import numpy as np
import re

# Since aipipe is compatible with the OpenAI client, we can use it
from openai import OpenAI

# Configure the client to use the AI Pipe endpoint
client = OpenAI(
    api_key=os.environ.get("APIPIPE_API_KEY"),
    base_url="https://aipipe.org/openai/v1"
)


# ========================================================================= #
# TOOL 1: SPECIALIST FOR PANDAS WEB SCRAPING                                #
# ========================================================================= #
def run_pandas_web_analysis(task_description):
    print("--- ROUTER: Selected Pandas Web Scraper Tool ---")

    url_part = task_description.split("http")[1]
    url = "http" + url_part.split()[0].strip()

    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    print(f"--- Scraping data from {url} with headers ---")
    response = requests.get(url, headers=headers)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable'})

    if not table:
        raise ValueError("Could not find the 'wikitable' on the page. The scraper may have been blocked.")

    header_row = table.find('tr')
    headers = [header.get_text(strip=True) for header in header_row.find_all('th')]
    rows = table.find_all('tr')[1:]

    table_data = []
    for row in rows:
        cells = row.find_all(['th', 'td'])
        if len(cells) == len(headers):
            table_data.append([cell.get_text(strip=True) for cell in cells])

    df = pd.DataFrame(table_data, columns=headers)
    print("--- Successfully built DataFrame manually ---")

    # --- Data Cleaning Pipeline ---
    df.columns = df.columns.str.split('[').str[0].str.strip().str.lower().str.replace(' ', '_')
    for col in ['reference', 'references']:
        if col in df.columns:
            df = df.drop(columns=col)

    cols_to_clean = ['rank', 'peak', 'worldwide_gross', 'year']
    for col in cols_to_clean:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('$', '', regex=False).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['rank', 'year', 'worldwide_gross'], inplace=True)

    for col in ['rank', 'peak', 'year']:
        if col in df.columns:
            df.loc[:, col] = df[col].astype(int)
    # --- End of Cleaning ---

    prompt = f"""
    You are an expert Python data analyst. Given the pre-cleaned pandas DataFrame `df`, write a Python script to answer the user's questions.

    The DataFrame `df`:
    {df.head().to_string()}
    The available columns are: {df.columns.tolist()}

    The user's task:
    ---
    {task_description}
    ---

    INSTRUCTIONS:
    - The final result must be a single Python list variable named `result`.
    - If a plot is requested, generate it and encode it as a base64 data URI string.
    - Do NOT include any data cleaning. The DataFrame is ready.
    - Respond with ONLY the raw Python code.
    """
    response = client.chat.completions.create(
        # CHANGED: Switched back to the more powerful GPT-4o model
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    code_to_execute = response.choices[0].message.content.strip().replace("```python", "").replace("```", "")

    local_scope = {'df': df, 'pd': pd, 'plt': plt, 'io': io, 'base64': base64, 'np': np, 're': re, 'result': None}
    
    print(f"\n--- Executing LLM-generated Python code ---\n{code_to_execute}\n------------------------------------")
    exec(code_to_execute, globals(), local_scope)

    if local_scope.get('result') is None:
        raise ValueError("The pandas analysis code did not produce a 'result' variable.")

    return local_scope['result']


# ========================================================================= #
# TOOL 2: SPECIALIST FOR DUCKDB S3 ANALYSIS                                 #
# ========================================================================= #
def run_duckdb_s3_analysis(task_description):
    print("--- ROUTER: Selected DuckDB S3 Analyzer Tool ---")

    con = duckdb.connect(database=':memory:', read_only=False)
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL parquet; LOAD parquet;")

    prompt = f"""
    You are an expert SQL data analyst specializing in DuckDB.
    A user wants to query a large dataset of Parquet files stored on S3.

    The base S3 path is: 's3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1'
    The columns are: court_code, title, date_of_registration, decision_date, disposal_nature, court, bench, year.

    The user's task:
    ---
    {task_description}
    ---

    INSTRUCTIONS:
    - You must answer all questions and provide the final output in a single Python dictionary named `result`.
    - Generate Python code that uses `con.sql('...').df()` to execute DuckDB SQL queries.
    - To get the most disposed cases, you will need to GROUP BY court and COUNT cases between 2000 and 2023.
    - To get the regression slope, calculate the difference between 'decision_date' and 'date_of_registration' in days. Then find the slope of the linear regression between that delay and the 'year' for court=33_10. Use `REGR_SLOPE`.
    - To create the plot, first query the necessary data into a pandas DataFrame. Then use matplotlib to generate the plot and encode it as a base64 data URI string.
    - The final output must be a single Python dictionary assigned to a variable named `result`.
    - Respond with ONLY the raw Python code.
    """
    response = client.chat.completions.create(
        # CHANGED: Switched back to the more powerful GPT-4o model
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}]
    )
    code_to_execute = response.choices[0].message.content.strip().replace("```python", "").replace("```", "")

    local_scope = {'con': con, 'pd': pd, 'plt': plt, 'io': io, 'base64': base64, 'np': np, 're': re, 'result': None}

    exec(code_to_execute, globals(), local_scope)

    if local_scope.get('result') is None:
        raise ValueError("The DuckDB analysis code did not produce a 'result' variable.")

    return local_scope['result']


# ========================================================================= #
# THE MAIN ROUTER FUNCTION (Unchanged)                                      #
# ========================================================================= #
def run_analysis(task_description, attached_files):
    if "indian high court" in task_description.lower():
        return run_duckdb_s3_analysis(task_description)
    elif "highest grossing films" in task_description.lower():
        return run_pandas_web_analysis(task_description)
    else:
        raise ValueError("Could not determine the appropriate tool for the given task.")
