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

# This function does not need changes, it is already robust.
def analyze_dataframe(df, task_description, model_name="mistralai/mistral-7b-instruct"):
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
    - The final result must be a single Python list or dictionary variable named `result` that matches the format requested in the task.
    - If a plot is requested, generate it and encode it as a base64 data URI string.
    - Do NOT include any data cleaning. The DataFrame is ready.
    - Respond with ONLY the raw Python code.
    """
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}]
    )
    code_to_execute = response.choices[0].message.content.strip().replace("```python", "").replace("```", "")

    local_scope = {'df': df, 'pd': pd, 'plt': plt, 'io': io, 'base64': base64, 'np': np, 're': re, 'result': None}

    print(f"\n--- Executing LLM-generated code ---\n{code_to_execute}\n------------------------------------")
    exec(code_to_execute, globals(), local_scope)

    if local_scope.get('result') is None:
        raise ValueError("The analysis code did not produce a 'result' variable.")

    return local_scope['result']


# ========================================================================= #
# --- THE UPDATED FUNCTION WITH THE PRECISION CLEANING PIPELINE ---         #
# ========================================================================= #
def run_pandas_web_analysis(task_description):
    print("--- ROUTER: Selected Pandas Web Scraper Tool ---")
    
    # --- 1. Scraping (Unchanged) ---
    url_part = task_description.split("http")[1]
    url = "http" + url_part.split()[0].strip()
    headers = {'User-Agent': 'Mozilla/5.0'}
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'lxml')
    table = soup.find('table', {'class': 'wikitable'})
    if not table:
        raise ValueError("Could not find the 'wikitable' on the page.")
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

    # --- 2. THE NEW, MORE PRECISE CLEANING PIPELINE ---
    print("--- Starting Final Data Cleaning and Standardization ---")
    
    # A. Clean and standardize column names
    df.columns = df.columns.str.split('[').str[0].str.strip().str.lower().str.replace(' ', '_')
    
    # B. Drop the unneeded 'reference(s)' column
    for col in ['reference', 'references']:
        if col in df.columns:
            df = df.drop(columns=col)
    
    # C. For all columns that should be numeric, apply a precise cleaning process
    cols_to_clean = ['rank', 'peak', 'worldwide_gross', 'year']
    for col in cols_to_clean:
        if col in df.columns:
            # First, ensure the column is a string type for cleaning
            df[col] = df[col].astype(str)
            # Then, explicitly remove dollar signs and commas
            df[col] = df[col].str.replace('$', '', regex=False).str.replace(',', '', regex=False)
            # Finally, convert to numeric, coercing any errors
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # D. Drop any rows that failed to convert properly in key columns
    df.dropna(subset=['rank', 'year', 'worldwide_gross'], inplace=True)
    
    # E. Now that data is guaranteed clean, convert the appropriate columns to integer type
    for col in ['rank', 'peak', 'year']:
        if col in df.columns:
            # Use .loc to avoid SettingWithCopyWarning
            df.loc[:, col] = df[col].astype(int)
    
    # --- End of Cleaning ---

    # 3. Call the analyzer function
    return analyze_dataframe(df, task_description)


# ========================================================================= #
# TOOL 2: SPECIALIST FOR DUCKDB S3 ANALYSIS (Unchanged)                     #
# ========================================================================= #
def run_duckdb_s3_analysis(task_description):
    print("--- ROUTER: Selected DuckDB S3 Analyzer Tool ---")
    con = duckdb.connect(database=':memory:', read_only=False)
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL parquet; LOAD parquet;")
    print("--- DuckDB extensions installed ---")

    # Pass the task to the generic analyzer function
    return analyze_dataframe(None, task_description, model_name="mistralai/mistral-7b-instruct") # We pass None for df, as it's not needed for the SQL prompt


# ========================================================================= #
# THE MAIN ROUTER FUNCTION  (Unchanged)                                     #
# ========================================================================= #
def run_analysis(task_description, attached_files):
    if "indian high court" in task_description.lower():
        return run_duckdb_s3_analysis(task_description)
    elif "highest grossing films" in task_description.lower():
        return run_pandas_web_analysis(task_description)
    else:
        raise ValueError("Could not determine the appropriate tool for the given task.")
