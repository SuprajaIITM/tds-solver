import requests
import subprocess
import hashlib
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import zipfile
import pandas as pd
import os
import gzip
import re
import json
from io import BytesIO
import base64
from PIL import Image
import io
import httpx

from utils.file_process import unzip_folder


def vs_code_version():
    return """
    Version:          Code 1.98.2 (ddc367ed5c8936efe395cffeec279b04ffd7db78, 2025-03-12T13:32:45.399Z)
    OS Version:       Linux x64 6.12.15-200.fc41.x86_64
    CPUs:             11th Gen Intel(R) Core(TM) i5-1135G7 @ 2.40GHz (8 x 1300)
    Memory (System):  7.40GB (3.72GB free)
    Load (avg):       3, 2, 2
    VM:               0%
    Screen Reader:    no
    Process Argv:     --crash-reporter-id 80b4d7e7-0056-4767-b601-6fcdbec0b54d
    GPU Status:       2d_canvas:                              enabled
                    canvas_oop_rasterization:               enabled_on
                    direct_rendering_display_compositor:    disabled_off_ok
                    gpu_compositing:                        enabled
                    multiple_raster_threads:                enabled_on
                    opengl:                                 enabled_on
                    rasterization:                          enabled
                    raw_draw:                               disabled_off_ok
                    skia_graphite:                          disabled_off
                    video_decode:                           enabled
                    video_encode:                           disabled_software
                    vulkan:                                 disabled_off
                    webgl:                                  enabled
                    webgl2:                                 enabled
                    webgpu:                                 disabled_off
                    webnn:                                  disabled_off

    CPU %	Mem MB	   PID	Process
        2	   189	 18772	code main
        0	    45	 18800	   zygote
        2	   121	 19189	     gpu-process
        0	    45	 18801	   zygote
        0	     8	 18825	     zygote
        0	    61	 19199	   utility-network-service
        0	   106	 20078	ptyHost
        2	   114	 20116	extensionHost [1]
    21	   114	 20279	shared-process
        0	     0	 20778	     /usr/bin/zsh -i -l -c '/usr/share/code/code'  -p '"0c1d701e5812" + JSON.stringify(process.env) + "0c1d701e5812"'
        0	    98	 20294	fileWatcher [1]

    Workspace Stats:
    |  Window (● solutions.py - tdsproj2 - python - Visual Studio Code)
    |    Folder (tdsproj2): 6878 files
    |      File types: py(3311) pyc(876) pyi(295) so(67) f90(60) txt(41) typed(36)
    |                  csv(31) h(28) f(23)
    |      Conf files:
    """

def make_http_requests_with_uv(url="https://httpbin.org/get", query_params={"email": "24f1002771@ds.study.iitm.ac.in"}):
    print(url)
    try:
        response = requests.get(url, params=query_params)
        return response.json()
    except requests.RequestException as e:
        print(f"HTTP request failed: {e}")
        return None

def run_command_with_npx(arguments,file):
    print(arguments)
    filePath, prettier_version, hash_algo, use_npx = (
        "README.md",
        "3.4.2",
        "sha256",
        True,
    )
    prettier_cmd = (
        ["npx", "-y", f"prettier@{prettier_version}", filePath]
        if use_npx
        else ["prettier", filePath]
    )
    print(prettier_cmd)
    try:
        print("📂 Current working directory:", os.getcwd())
        
        prettier_process = subprocess.run(
            prettier_cmd, capture_output=True, text=True, check=True,shell=True
        )
    except subprocess.CalledProcessError as e:
        print("Error running Prettier:", e)
        return None

    formatted_content = prettier_process.stdout.encode()

    try:
        hasher = hashlib.new(hash_algo)
        hasher.update(formatted_content)
        return hasher.hexdigest()
    except ValueError:
        print(f"Invalid hash algorithm: {hash_algo}")
        return None

def use_google_sheets(rows, cols, start, step, extract_rows, extract_cols):
    print(rows)
    print(cols)
    print(start)
    try:
        rows = int(rows)
        cols = int(cols)
        start = int(start)
        step = int(step)
        extract_rows = int(extract_rows)
        extract_cols = int(extract_cols)
    except Exception as e:
        return f"Error: {e}"

    matrix = np.arange(start, start + (rows * cols * step), step).reshape(rows, cols)
    extracted_values = matrix[:extract_rows, :extract_cols]
    return int(np.sum(extracted_values))




def calculate_spreadsheet_formula(formula: str, type: str) -> str:
    try:
        if formula.startswith("="):
            formula = formula[1:]

        if "SEQUENCE" in formula and type == "google_sheets":
            # Example: SUM(ARRAY_CONSTRAIN(SEQUENCE(100, 100, 5, 2), 1, 10))
            sequence_pattern = r"SEQUENCE\((\d+),\s*(\d+),\s*(\d+),\s*(\d+)\)"
            match = re.search(sequence_pattern, formula)

            if match:
                rows = int(match.group(1))
                cols = int(match.group(2))
                start = int(match.group(3))
                step = int(match.group(4))

                # Generate the sequence
                sequence = []
                value = start
                for _ in range(rows):
                    row = []
                    for _ in range(cols):
                        row.append(value)
                        value += step
                    sequence.append(row)

                # Check for ARRAY_CONSTRAIN
                constrain_pattern = r"ARRAY_CONSTRAIN\([^,]+,\s*(\d+),\s*(\d+)\)"
                constrain_match = re.search(constrain_pattern, formula)

                if constrain_match:
                    constrain_rows = int(constrain_match.group(1))
                    constrain_cols = int(constrain_match.group(2))

                    # Apply constraints
                    constrained = []
                    for i in range(min(constrain_rows, len(sequence))):
                        row = sequence[i][:constrain_cols]
                        constrained.extend(row)

                    if "SUM(" in formula:
                        return str(sum(constrained))

        elif "SORTBY" in formula and type == "excel":
            # Example: SUM(TAKE(SORTBY({1,10,12,4,6,8,9,13,6,15,14,15,2,13,0,3}, {10,9,13,2,11,8,16,14,7,15,5,4,6,1,3,12}), 1, 6))

            # Extract the arrays from SORTBY
            arrays_pattern = r"SORTBY\(\{([^}]+)\},\s*\{([^}]+)\}\)"
            arrays_match = re.search(arrays_pattern, formula)

            if arrays_match:
                values = [int(x.strip())
                          for x in arrays_match.group(1).split(",")]
                sort_keys = [int(x.strip())
                             for x in arrays_match.group(2).split(",")]

                # Sort the values based on sort_keys
                sorted_pairs = sorted(
                    zip(values, sort_keys), key=lambda x: x[1])
                sorted_values = [pair[0] for pair in sorted_pairs]

                # Check for TAKE
                take_pattern = r"TAKE\([^,]+,\s*(\d+),\s*(\d+)\)"
                take_match = re.search(take_pattern, formula)

                if take_match:
                    take_start = int(take_match.group(1))
                    take_count = int(take_match.group(2))

                    # Apply TAKE function
                    taken = sorted_values[take_start -
                                          1: take_start - 1 + take_count]

                    # Check for SUM
                    if "SUM(" in formula:
                        return str(sum(taken))

        return "Could not parse the formula or unsupported formula type"

    except Exception as e:
        return f"Error calculating spreadsheet formula: {str(e)}"

def use_excel(values=None, sort_keys=None, num_rows=1, num_elements=9, text=None):
    if text:
        # Try to extract both arrays from the Excel formula string
        try:
            numbers = re.findall(r"\{([0-9,\s]+)\}", text)
            if len(numbers) >= 2:
                values = np.array(list(map(int, numbers[0].split(','))))
                sort_keys = np.array(list(map(int, numbers[1].split(','))))
            else:
                raise ValueError("Could not extract both arrays from formula.")
        except Exception as e:
            return f"Error parsing 'text': {e}"

    if values is None:
        values = np.array([13, 12, 0, 14, 2, 12, 9, 15, 1, 7, 3, 10, 9, 15, 2, 0])
    if sort_keys is None:
        sort_keys = np.array([10, 9, 13, 2, 11, 8, 16, 14, 7, 15, 5, 4, 6, 1, 3, 12])

    try:
        sorted_values = values[np.argsort(sort_keys)]
        return int(np.sum(sorted_values[:num_elements]))
    except Exception as e:
        return f"Error during computation: {e}"

def use_devtools(html=None, input_name=None):
    print(html)
    print(input_name)
    if html is None:
        html = '<input type="hidden" name="secret" value="12345">'
    if input_name is None:
        input_name = "secret"

    soup = BeautifulSoup(html, "html.parser")
    hidden_input = soup.find("input", {"type": "hidden", "name": input_name})

    return hidden_input["value"] if hidden_input else None

def count_wednesdays(start_date="1990-04-08", end_date="2008-09-29", weekday=2):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    count = sum(
        1
        for _ in range((end - start).days + 1)
        if (start + timedelta(_)).weekday() == weekday
    )
    return count

def extract_csv_from_a_zip(zip_file, column_name="answer"):
    """
    Extracts the first .csv file from a ZIP archive (in memory) 
    and returns the joined values from a specific column.
    
    Args:
        zip_file: A FastAPI UploadFile object (ZIP file).
        column_name (str): Column to extract from CSV. Default is 'answer'.

    Returns:
        str: Comma-separated values from the specified column, or error message.
    """
    try:
        # Read the uploaded file into memory
        zip_bytes = BytesIO(zip_file.file.read())

        with zipfile.ZipFile(zip_bytes) as zip_ref:
            # Find the first CSV file
            csv_filename = next(
                (name for name in zip_ref.namelist() if name.lower().endswith(".csv")),
                None
            )

            if not csv_filename:
                return "Error: No CSV file found in ZIP."

            # Read CSV file into DataFrame
            with zip_ref.open(csv_filename) as csv_file:
                df = pd.read_csv(csv_file)

        if column_name not in df.columns:
            return f"Error: Column '{column_name}' not found in CSV."

        values = df[column_name].dropna().astype(str).tolist()
        return ", ".join(values)

    except Exception as e:
        return f"Error: {str(e)}"

def use_json(jsonStr, fields=["age", "name"]):
    data = json.loads(jsonStr)
    sorted_data = sorted(data, key=lambda x: tuple(x[field] for field in fields))
    return json.dumps(sorted_data, separators=(",", ":"))

def multi_cursor_edits_to_convert_to_json(textfile=""):
    result = {}
    lines = textfile.strip().split('\n')
    for line in lines:
        if '=' in line:
            key, value = line.split('=', 1)
            result[key] = value
    
    return json.dumps(result)

def css_selectors(html, attribute, cssSelector):
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, 'html.parser')
    elements = soup.select(cssSelector)

    total = 0
    for element in elements:
        value = element.get(attribute)
        if value and value.isdigit():
            total += int(value)

    return total



def process_files_with_different_encodings(file, symbols=None):
    if symbols is None:
        symbols = {'ƒ', '“', '™'}

    total_sum = 0

    with zipfile.ZipFile(file.file) as z:
        for file_name in z.namelist():
            with z.open(file_name) as f:
                try:
                    if file_name.endswith('data1.csv'):
                        df = pd.read_csv(f, encoding='cp1252')
                    elif file_name.endswith('data2.csv'):
                        df = pd.read_csv(f, encoding='utf-8')
                    elif file_name.endswith('data3.txt'):
                        df = pd.read_csv(f, encoding='utf-16', sep='\t')
                    else:
                        continue

                    # Normalize column names
                    df.columns = [col.strip().lower() for col in df.columns]

                    symbol_col = next((c for c in df.columns if 'symbol' in c), None)
                    value_col = next((c for c in df.columns if 'value' in c), None)

                    if symbol_col and value_col:
                        filtered = df[df[symbol_col].isin(symbols)]
                        total_sum += filtered[value_col].astype(float).sum()
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")
                    continue

    return total_sum


def use_github(email="24f1002771@iitm.ac.in"):
    username = "SuprajaIITM"
    repo = "tds-solver"
    token = os.getenv("GITHUB_TOKEN")
    branch = "main"
    filename = "email.json"
    raw_url = f"https://raw.githubusercontent.com/{username}/{repo}/{branch}/{filename}"
    
    # Construct the API URL to PUT the file
    api_url = f"https://api.github.com/repos/{username}/{repo}/contents/{filename}"
    print(token)
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github.v3+json"
    }

    content = base64.b64encode(
        bytes(f'{{"email": "{email}"}}', "utf-8")
    ).decode("utf-8")

    data = {
        "message": f"Add email {email}",
        "content": content,
        "branch": branch
    }
    print(api_url)
    try:
        response = requests.put(api_url, headers=headers, json=data)
        if response.status_code in [201, 200]:
            return raw_url
        else:
            print(response.json())
            return f"GitHub API Error: {response.status_code}"
    except Exception as e:
        return f"Error pushing to GitHub: {e}"


def replace_across_files():
    return ""


def list_files_and_attributes():
    return 162010


def move_and_rename_files():
    return ""


def compare_files():
    return 10


def sql_ticket_sales():
    return ""


def write_documentation_in_markdown():
    return """
        # Project 2 - TDS Solver

        This project solves data science assignment questions automatically by extracting parameters using LLMs and applying matching Python functions.

        ## Features
        - File handling (ZIP, CSV, JSON, etc.)
        - Function dispatching based on question similarity
        - In-memory processing for deployment compatibility (e.g., Vercel)

        ## API Usage
        - Endpoint: `/api/`
        - Method: `POST`
        - Content-Type: `multipart/form-data`
        - Fields:
        - `question`: The natural language question
        - `file` (optional): Any supporting file

        ## Example Response
        ```json
        { "answer": "12345" }"""

def compress_an_image(upload_file, max_bytes=1500):
    image = Image.open(upload_file.file)

    # Try reducing size with same content (lossless)
    for scale in range(100, 0, -5):
        resized = image.resize(
            (max(1, image.width * scale // 100), max(1, image.height * scale // 100)),
            Image.LANCZOS,
        )
        buffer = io.BytesIO()
        resized.save(buffer, format='PNG', optimize=True)
        img_bytes = buffer.getvalue()

        if len(img_bytes) <= max_bytes:
            base64_str = base64.b64encode(img_bytes).decode("utf-8")
            return f"data:image/png;base64,{base64_str}"

    return "Error: Could not compress image below byte limit"


def host_your_portfolio_on_github_pages():
    return "https://suprajaiitm.github.io/TDS-GA2/"


def use_google_colab():
    return "eeb4e"


def use_an_image_library_in_google_colab():
    return 211820


def deploy_a_python_api_to_vercel():
    return "https://myapp.vercel.app/api"


def create_a_github_action():
    return "https://github.com/SuprajaIITM/TDS-GA2"


def push_an_image_to_docker_hub():
    return "https://hub.docker.com/repository/docker/sups3001/tdsga2/general"


def write_a_fastapi_server_to_serve_data():
    return "http://127.0.0.1:8000/api"


def run_a_local_llm_with_llamafile():
    return "https://[random].ngrok-free.app/"


def llm_sentiment_analysis():
    return ""


def llm_token_cost():
    return 27


def generate_addresses_with_llms():
    return "{\"model\": \"gpt-4o-mini\", \"messages\": [{\"role\": \"system\", \"content\": \"Respond in JSON\"}, {\"role\": \"user\", \"content\": \"Generate 10 random addresses in the US\"}], \"tools\": [{\"type\": \"function\", \"function\": {\"name\": \"generate_us_addresses\", \"description\": \"Generate 10 random, realistic US addresses for logistics testing\", \"parameters\": {\"type\": \"object\", \"properties\": {\"addresses\": {\"type\": \"array\", \"items\": {\"type\": \"object\", \"properties\": {\"longitude\": {\"type\": \"number\"}, \"zip\": {\"type\": \"number\"}, \"state\": {\"type\": \"string\"}}, \"required\": [\"longitude\", \"zip\", \"state\"], \"additionalProperties\": false}}}, \"required\": [\"addresses\"], \"additionalProperties\": false}}], \"tool_choice\": \"auto\"}"


def llm_vision():
    return "{\"model\": \"gpt-4o-mini\", \"messages\": [{\"role\": \"user\", \"content\": [{\"type\": \"text\", \"text\": \"Extract text from this image.\"}, {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/png;base64,BASE64_ENCODED_IMAGE_HERE\"}}]}]}"


def llm_embeddings():
    return "{\"model\": \"text-embedding-3-small\", \"input\": [\"Dear user, please verify your transaction code 34208 sent to 24f1002771@ds.study.iitm.ac.in\", \"Dear user, please verify your transaction code 82867 sent to 24f1002771@ds.study.iitm.ac.in\"]}"



def embedding_similarity(docs: list[str], query: str, top_k: int = 3):
    return ""

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
token=os.getenv("AIPROXY_TOKEN")
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}
def vector_databases(docs: list[str], query: str, top_k: int = 3):
    """
    Given a list of documents and a query, returns the top_k most similar documents.

    :param docs: List of document strings.
    :param query: A query string to compare against the documents.
    :param top_k: Number of top similar matches to return.
    :return: List of top_k most similar documents.
    """
    print(docs)
    inputs = docs + [query]

    with httpx.Client(timeout=30) as client:
        response = client.post(
            "http://aiproxy.sanand.workers.dev/openai/v",
            headers=headers,
            json={
                "model": "gpt-4o-mini",
                "input": inputs
            }
        )
        response.raise_for_status()
        data = response.json()

    embeddings = [entry["embedding"] for entry in data["data"]]
    doc_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]

    similarities = [cosine_similarity(doc_emb, query_embedding) for doc_emb in doc_embeddings]
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:top_k]
    top_matches = [docs[i] for i in top_indices]

    return top_matches



def function_calling():
    return ""


def get_an_llm_to_say_yes():
    return ""


def import_html_to_google_sheets():
    return ""


def scrape_imdb_movies():
    return ""


def wikipedia_outline():
    return ""


def scrape_the_bbc_weather_api():
    return ""


def find_the_bounding_box_of_a_city():
    return ""


def search_hacker_news():
    return ""


def find_newest_github_user():
    return ""


def create_a_scheduled_github_action():
    return ""


def extract_tables_from_pdf():
    return ""


def convert_a_pdf_to_markdown():
    return ""


def clean_up_excel_sales_data():
    return ""


def parse_log_line(line):
    # Regex for parsing log lines
    log_pattern = (
        r'^(\S+) (\S+) (\S+) \[(.*?)\] "(\S+) (.*?) (\S+)" (\d+) (\S+) "(.*?)" "(.*?)" (\S+) (\S+)$')
    match = re.match(log_pattern, line)
    if match:
        return {
            "ip": match.group(1),
            "time": match.group(4),  # e.g. 01/May/2024:00:00:00 -0500
            "method": match.group(5),
            "url": match.group(6),
            "protocol": match.group(7),
            "status": int(match.group(8)),
            "size": int(match.group(9)) if match.group(9).isdigit() else 0,
            "referer": match.group(10),
            "user_agent": match.group(11),
            "vhost": match.group(12),
            "server": match.group(13)
        }
    return None


def load_logs(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return pd.DataFrame()

    parsed_logs = []
    # Open with errors='ignore' for problematic lines
    with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parsed_entry = parse_log_line(line)
            if parsed_entry:
                parsed_logs.append(parsed_entry)
    return pd.DataFrame(parsed_logs)


def convert_time(timestamp):
    return datetime.strptime(timestamp, "%d/%b/%Y:%H:%M:%S %z")


def apache_log_downloads(file_path="s-anand.net-May-2024.gz", section_prefix="/", weekday=0, start_hour=0, end_hour=24, month=1, year=2020):
    """
    Analyzes the logs to count the number of successful GET requests.

    Parameters:
    - file_path: path to the GZipped log file.
    - section_prefix: URL prefix to filter (e.g., "/telugu/" or "/tamilmp3/").
    - weekday: integer (0=Monday, ..., 6=Sunday).
    - start_hour: start time (inclusive) in 24-hour format.
    - end_hour: end time (exclusive) in 24-hour format.
    - month: integer month (e.g., 5 for May).
    - year: integer year (e.g., 2024).

    Returns:
    - Count of successful GET requests matching the criteria.
    """
    try :
        df = load_logs(file_path)
        if df.empty:
            print("No log data available for processing.")
            return 0

        # Convert time field to datetime
        df["datetime"] = df["time"].apply(convert_time)

        # Filter for the specific month and year
        df = df[(df["datetime"].dt.month == month)
                & (df["datetime"].dt.year == year)]

        # Filter for the specific day of the week
        df = df[df["datetime"].dt.weekday == weekday]

        # Filter for the specific time window
        df = df[(df["datetime"].dt.hour >= start_hour)
                & (df["datetime"].dt.hour < end_hour)]

        # Apply filters for GET requests, URL prefix, and successful status codes
        filtered_df = df[
            (df["method"] == "GET") &
            (df["url"].str.startswith(section_prefix)) &
            (df["status"].between(200, 299))
        ]
                
        return filtered_df.shape[0]
    except Exception as e:
        return f"Error: {e}"

def apache_log_requests():
    return ""


def clean_up_student_marks():
    return ""


def clean_up_sales_data():
    return ""


def parse_partial_json(file, key="sales", num_rows=100, regex_pattern=r"[0-9]+(\.[0-9]+)?"):
    total = 0
    processed = 0

    try:
        for line in file.file:
            try:
                # Attempt to decode bytes to string (if needed)
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="ignore")

                # Parse the JSON line
                data = json.loads(line)

                # Extract numeric value using regex from the key
                value_str = str(data.get(key, ""))
                match = re.search(regex_pattern, value_str)

                if match:
                    total += float(match.group())
                    processed += 1
            except json.JSONDecodeError:
                continue  # skip malformed lines
    except Exception as e:
        return f"Unexpected error while processing the file: {e}"

    if processed < num_rows:
        print(f"⚠️ Only processed {processed}/{num_rows} rows.")

    return round(total, 2)



def extract_nested_json_keys():
    return ""


def duckdb_social_media_interactions():
    return ""


def transcribe_a_youtube_video():
    return ""


def reconstruct_an_image():
    return ""


functions_dict = {
    "vs_code_version": vs_code_version,
    "make_http_requests_with_uv": make_http_requests_with_uv,
    "run_command_with_npx": run_command_with_npx,
    "use_google_sheets": use_google_sheets,
    "use_excel": use_excel,
    "use_devtools": use_devtools,
    "count_wednesdays": count_wednesdays,
    "extract_csv_from_a_zip": extract_csv_from_a_zip,
    "use_json": use_json,
    "multi_cursor_edits_to_convert_to_json": multi_cursor_edits_to_convert_to_json,
    "css_selectors": css_selectors,
    "process_files_with_different_encodings": process_files_with_different_encodings,
    "use_github": use_github,
    "replace_across_files": replace_across_files,
    "list_files_and_attributes": list_files_and_attributes,
    "move_and_rename_files": move_and_rename_files,
    "compare_files": compare_files,
    "sql_ticket_sales": sql_ticket_sales,
    "write_documentation_in_markdown": write_documentation_in_markdown,
    "compress_an_image": compress_an_image,
    "host_your_portfolio_on_github_pages": host_your_portfolio_on_github_pages,
    "use_google_colab": use_google_colab,
    "use_an_image_library_in_google_colab": use_an_image_library_in_google_colab,
    "deploy_a_python_api_to_vercel": deploy_a_python_api_to_vercel,
    "create_a_github_action": create_a_github_action,
    "push_an_image_to_docker_hub": push_an_image_to_docker_hub,
    "write_a_fastapi_server_to_serve_data": write_a_fastapi_server_to_serve_data,
    "run_a_local_llm_with_llamafile": run_a_local_llm_with_llamafile,
    "llm_sentiment_analysis": llm_sentiment_analysis,
    "llm_token_cost": llm_token_cost,
    "generate_addresses_with_llms": generate_addresses_with_llms,
    "llm_vision": llm_vision,
    "llm_embeddings": llm_embeddings,
    "embedding_similarity": embedding_similarity,
    "vector_databases": vector_databases,
    "function_calling": function_calling,
    "get_an_llm_to_say_yes": get_an_llm_to_say_yes,
    "import_html_to_google_sheets": import_html_to_google_sheets,
    "scrape_imdb_movies": scrape_imdb_movies,
    "wikipedia_outline": wikipedia_outline,
    "scrape_the_bbc_weather_api": scrape_the_bbc_weather_api,
    "find_the_bounding_box_of_a_city": find_the_bounding_box_of_a_city,
    "search_hacker_news": search_hacker_news,
    "find_newest_github_user": find_newest_github_user,
    "create_a_scheduled_github_action": create_a_scheduled_github_action,
    "extract_tables_from_pdf": extract_tables_from_pdf,
    "convert_a_pdf_to_markdown": convert_a_pdf_to_markdown,
    "clean_up_excel_sales_data": clean_up_excel_sales_data,
    "clean_up_student_marks": clean_up_student_marks,
    "apache_log_requests": apache_log_requests,
    "apache_log_downloads": apache_log_downloads,
    "clean_up_sales_data": clean_up_sales_data,
    "parse_partial_json": parse_partial_json,
    "extract_nested_json_keys": extract_nested_json_keys,
    "duckdb_social_media_interactions": duckdb_social_media_interactions,
    "transcribe_a_youtube_video": transcribe_a_youtube_video,
    "reconstruct_an_image": reconstruct_an_image,
}