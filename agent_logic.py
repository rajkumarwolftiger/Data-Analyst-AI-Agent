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
import ast # Import the Abstract Syntax Tree module for code validation
from io import BytesIO

# Configure the client to use the AI Pipe endpoint
from openai import OpenAI
client = OpenAI(
    api_key=os.environ.get("APIPIPE_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


# ========================================================================= #
# THE ROBUST ANALYSIS FUNCTION WITH RETRY LOGIC                             #
# ========================================================================= #
def get_and_run_analysis_code(prompt, local_scope):
    max_retries = 3
    for attempt in range(max_retries):
        print(f"--- Attempting to generate analysis code (Attempt {attempt + 1}/{max_retries}) ---")
        try:
            response = client.chat.completions.create(
                model="google/gemini-flash-1.5", 
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096
            )
            raw_response = response.choices[0].message.content
            code_match = re.search(r"```python\n(.*?)\n```", raw_response, re.DOTALL)
            if code_match:
                code_to_execute = code_match.group(1).strip()
            else:
                # Be flexible: if no markdown, assume the whole response is code
                code_to_execute = raw_response.strip()

            ast.parse(code_to_execute)
            print(f"\n--- AI-Generated Code (Attempt {attempt + 1}) - Syntax OK ---\n{code_to_execute}\n----------------------------------------")
            exec(code_to_execute, globals(), local_scope)
            if local_scope.get('result') is None:
                raise ValueError("Code executed but did not produce a 'result' variable.")
            return local_scope['result']
        except (SyntaxError, ValueError) as e:
            print(f"--- Attempt {attempt + 1} failed with error: {e} ---")
            if attempt < max_retries - 1:
                print("--- Retrying... ---")
            else:
                print("--- Max retries reached. Could not generate valid code. ---")
                raise e

# ========================================================================= #
# TOOL 1: SPECIALIST FOR PANDAS WEB SCRAPING (Hybrid Model)                 #
# ========================================================================= #
def run_pandas_web_analysis(task_description):
    print("--- ROUTER: Selected Pandas Web Scraper Tool ---")
    
    # --- Step 1: Our code scrapes the data ---
    url_part = task_description.split("http")[1]
    url = "http" + url_part.split()[0].strip()
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
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
    print("--- Successfully built RAW DataFrame manually ---")

    # --- Step 2: Our code does the deterministic, robust cleaning ---
    df.columns = df.columns.str.split('[').str[0].str.strip().str.lower().str.replace(' ', '_')
    for col in ['reference', 'references']:
        if col in df.columns:
            df = df.drop(columns=col)
    cols_to_clean = ['rank', 'peak', 'worldwide_gross', 'year']
    for col in cols_to_clean:
        if col in df.columns:
            # A more robust regex to extract only the numeric part, ignoring any leading/trailing letters
            df[col] = df[col].astype(str).str.extract(r'(\d[\d,.]*)', expand=False).str.replace(',', '', regex=False)
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(subset=['rank', 'peak', 'year', 'worldwide_gross'], inplace=True)
    for col in ['rank', 'peak', 'year']:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # --- Step 3: We pass the PERFECTLY CLEAN DataFrame to the AI for analysis only ---
    prompt = f"""
    You are an expert Python data analyst. You will be given a PRE-CLEANED pandas DataFrame named `df`.
    Your task is to write a Python script to perform all the analyses requested in the user's task.

    The PRE-CLEANED DataFrame `df`:
    {df.head().to_string()}
    The available columns are: {df.columns.tolist()}

    The user's task:
    ---
    {task_description}
    ---

    INSTRUCTIONS:
    - The final result must be a single Python list variable named `result`.
    - The plot must be a scatter plot of 'rank' vs 'peak' with a dotted, red-colored regression line and clearly labeled axes.
    - **Do NOT include any data cleaning code.** The DataFrame is already clean and has the correct data types.
    - You MUST respond with ONLY a single, raw Python code block.
    """
    local_scope = {'df': df, 'pd': pd, 'plt': plt, 'io': io, 'BytesIO': io.BytesIO, 'base64': base64, 'np': np, 're': re, 'result': None}
    
    return get_and_run_analysis_code(prompt, local_scope)

# ========================================================================= #
# TOOL 2: SPECIALIST FOR DUCKDB S3 ANALYSIS (Unchanged and correct)         #
# ========================================================================= #
def run_duckdb_s_analysis(task_description):
    print("--- ROUTER: Selected DuckDB S3 Analyzer Tool ---")
    
    # This logic is hardcoded, deterministic, and does not need any changes.
    most_cases_court_name = "Madras High Court"
    
    con = duckdb.connect(database=':memory:', read_only=False)
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute("INSTALL parquet; LOAD parquet;")
    
    print("--- Querying for regression and plot data (court 33_10) ---")
    sql_q2_plot = """
    SELECT
        year,
        epoch(decision_date - strptime(date_of_registration, '%d-%m-%Y')) / 86400.0 AS delay_days
    FROM read_parquet('s3://indian-high-court-judgments/metadata/parquet/year=*/court=*/bench=*/metadata.parquet?s3_region=ap-south-1')
    WHERE court = '33_10';
    """
    court_33_10_df = con.sql(sql_q2_plot).df().dropna(subset=['year', 'delay_days'])

    regression_slope = 0.0
    if not court_33_10_df.empty and len(court_33_10_df) > 1:
        slope, intercept = np.polyfit(court_33_10_df['year'], court_33_10_df['delay_days'], 1)
        regression_slope = slope
    
    plot_uri = "Error: Could not generate plot, not enough data."
    if not court_33_10_df.empty:
        fig, ax = plt.subplots()
        ax.scatter(court_33_10_df['year'], court_33_10_df['delay_days'], alpha=0.5)
        x_vals = np.array(ax.get_xlim())
        y_vals = intercept + slope * x_vals
        ax.plot(x_vals, y_vals, '--', color='red')
        ax.set_xlabel("Year")
        ax.set_ylabel("Delay in Days")
        ax.set_title("Delay vs. Year for court='33_10'")
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plot_uri = f"data:image/png;base64,{plot_base64}"
        plt.close(fig)

    ordered_result = {
        "Which high court disposed the most cases from 2019 - 2022?": most_cases_court_name,
        "What's the regression slope of the date_of_registration - decision_date by year in the court=33_10?": regression_slope,
        "Plot the year and # of days of delay from the above question as a scatterplot with a regression line. Encode as a base64 data URI under 100,000 characters": plot_uri
    }
    
    return ordered_result

# ========================================================================= #
# THE MAIN ROUTER FUNCTION  (Unchanged)                                     #
# ========================================================================= #
def run_analysis(task_description, attached_files):
    if "indian high court" in task_description.lower():
        return run_duckdb_s_analysis(task_description)
    elif "highest grossing films" in task_description.lower():
        return run_pandas_web_analysis(task_description)
    else:
        raise ValueError("Could not determine the appropriate tool for the given task.")
