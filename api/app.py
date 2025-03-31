from fastapi import FastAPI, UploadFile, Form, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
import io
import zipfile
import tempfile
import datetime
import json
import traceback


from utils.file_process import unzip_folder
from utils.question_matching import find_similar_question
from utils.function_definitions_llm import function_definitions_objects_llm
from utils.openai_api import extract_parameters
from utils.solution_functions import functions_dict
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()

# Allow CORS if needed (optional)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI API setup
openai.api_key = os.getenv("AIPROXY_TOKEN")
openai.api_base = "http://aiproxy.sanand.workers.dev/openai/v1"

# Logging setup
LOG_PATH = "logs.txt"
def log_request(question, file_list, answer):
    with open(LOG_PATH, "a") as log:
        timestamp = datetime.datetime.now().isoformat()
        log.write(json.dumps({
            "timestamp": timestamp,
            "question": question,
            "files": file_list,
            "answer": answer
        }) + "\n")

@app.post("/api/")
async def process_file(question: str = Form(...), file: UploadFile = None):
    try:
        # Match to known question type and get relevant function
        matched_function, matched_description = find_similar_question(question)
        solution_function = functions_dict.get(
            matched_function,
            lambda *args, **kwargs: "No matching function found"
        )
        # Extract parameters from question via LLM
        parameters = extract_parameters(
            str(question),
            function_definitions_llm=function_definitions_objects_llm[matched_function],
        )
        # Handle file logic
        print(solution_function)
        if file:
            if file.filename.endswith(".zip"):
                if isinstance(parameters, dict):
                    answer = solution_function(file, **parameters)
                elif isinstance(parameters, (list, tuple)):
                    answer = solution_function(file, *parameters)
                else:
                    answer = solution_function(file)
            else:
                if isinstance(parameters, dict):
                    answer = solution_function(file, **parameters)
                elif isinstance(parameters, (list, tuple)):
                    answer = solution_function(file, *parameters)
                else:
                    answer = solution_function(file)
        else:
            if isinstance(parameters, dict):
                answer = solution_function(**parameters)
            elif isinstance(parameters, (list, tuple)):
                answer = solution_function(*parameters)
            else:
                answer = solution_function()




        return JSONResponse(content={"answer": answer})

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(content={"answer": f"Error: {str(e)}"}, status_code=500)


#curl -X POST "http://127.0.0.1:8000/api/" -H "Content-Type: multipart/form-data" -F "question=Let's make sure you know how to use npx and prettier.\n    Download . In the directory where you downloaded it, make sure it is called README.md, and run npx -y prettier@3.4.2 README.md | sha256sum.\n    What is the output of the command?" -F "file=@README.md"
#curl -X POST "http://127.0.0.1:8000/api/" -H "Content-Type: multipart/form-data" -F "question=Install and run Visual Studio Code. In your Terminal (or Command Prompt), type code -s and press Enter. Copy and paste the entire output below.\n    What is the output of code -s?"
#curl -X POST "http://127.0.0.1:8000/api/" -H "Content-Type: multipart/form-data" -F "question=Running uv run --with httpie -- https [URL] installs the Python package httpie and sends a HTTPS request to the URL.\n    Send a HTTPS request to https://httpbin.org/get with the URL encoded parameter email set to 24f1002771@ds.study.iitm.ac.in\n    What is the JSON output of the command? (Paste only the JSON body, not the headers)"
#curl -X POST "http://127.0.0.1:8000/api/" -H "Content-Type: multipart/form-data" -F "question=Let's make sure you can write formulas in Google Sheets. Type this formula into Google Sheets. (It won't work in Excel)\n    =SUM(ARRAY_CONSTRAIN(SEQUENCE(100, 100, 15, 12), 1, 10))\n    What is the result?"
#curl -X POST "http://127.0.0.1:8000/api/" -H "Content-Type: multipart/form-data" -F "question=Just above this paragraph, there's a hidden input with a secret value.\n    What is the value in the hidden input?"
#curl -X POST "http://127.0.0.1:8000/api/" -H "Content-Type: multipart/form-data" -F "question=Download and unzip file  which has a single extract.csv file inside.\n    What is the value in the \"answer\" column of the CSV file?" -F "file=q-extract-csv-zip.zip"
#curl -X POST "http://127.0.0.1:8000/api/" -H "Content-Type: multipart/form-data" -F "question=Let's make sure you know how to use JSON. Sort this JSON array of objects by the value of the age field. In case of a tie, sort by the name field. Paste the resulting JSON below without any spaces or newlines.\n    [{\"name\":\"Alice\",\"age\":60},{\"name\":\"Bob\",\"age\":77},{\"name\":\"Charlie\",\"age\":55},{\"name\":\"David\",\"age\":15},{\"name\":\"Emma\",\"age\":2},{\"name\":\"Frank\",\"age\":19},{\"name\":\"Grace\",\"age\":97},{\"name\":\"Henry\",\"age\":67},{\"name\":\"Ivy\",\"age\":52},{\"name\":\"Jack\",\"age\":59},{\"name\":\"Karen\",\"age\":91},{\"name\":\"Liam\",\"age\":76},{\"name\":\"Mary\",\"age\":16},{\"name\":\"Nora\",\"age\":34},{\"name\":\"Oscar\",\"age\":0},{\"name\":\"Paul\",\"age\":30}]\n    Sorted JSON:
#curl -X POST "http://127.0.0.1:8000/api/" -H "Content-Type: multipart/form-data" -F "question=Download  and use multi-cursors and convert it into a single JSON object, where key=value pairs are converted into {key: value, key: value, ...}.\n    What's the result when you paste the JSON at tools-in-data-science.pages.dev/jsonhash and click the Hash button?"
#curl -X POST "http://127.0.0.1:8000/api/" -H "Content-Type: multipart/form-data" -F "question=Download and process the files in  which contains three files with different encodings:\n    data1.csv: CSV file encoded in CP-1252\n    data2.csv: CSV file encoded in UTF-8\n    data3.txt: Tab-separated file encoded in UTF-16\n    Each file has 2 columns: symbol and value. Sum up all the values where the symbol matches \u201a OR \u02c6 OR \u2021 across all three files.\n    What is the sum of all values associated with these symbols" -F file="@q-unicode-data.zip"
#curl -X POST "http://127.0.0.1:8000/api/" -H "Content-Type: multipart/form-data" -F "question=Let's make sure you know how to use GitHub. Create a GitHub account if you don't have one. Create a new public repository. Commit a single JSON file called email.json with the value {\"email\": \"24f1002771@ds.study.iitm.ac.in\"} and push it."
#curl -X POST "http://127.0.0.1:8000/api/" -H "Content-Type: multipart/form-data" -F "question=Case Study: Recovering Sales Data for ReceiptRevive Analytics\nReceiptRevive Analytics is a data recovery and business intelligence firm specializing in processing legacy sales data from paper receipts. Many of the client companies have archives of receipts from past years, which have been digitized using OCR (Optical Character Recognition) techniques. However, due to the condition of some receipts (e.g., torn, faded, or partially damaged), the OCR process sometimes produces incomplete JSON data. These imperfections can lead to truncated fields or missing values, which complicates the process of data aggregation and analysis.\n\nOne of ReceiptRevive’s major clients, RetailFlow Inc., operates numerous brick-and-mortar stores and has an extensive archive of old receipts. RetailFlow Inc. needs to recover total sales information from a subset of these digitized receipts to analyze historical sales performance. The provided JSON data contains 100 rows, with each row representing a sales entry. Each entry is expected to include four keys:\n\ncity: The city where the sale was made.\nproduct: The product that was sold.\nsales: The number of units sold (or sales revenue).\nid: A unique identifier for the receipt.\nDue to damage to some receipts during the digitization process, the JSON entries are truncated at the end, and the id field is missing. Despite this, RetailFlow Inc. is primarily interested in the aggregate sales value.\n\nYour Task\nAs a data recovery analyst at ReceiptRevive Analytics, your task is to develop a program that will:\n\nParse the Sales Data:\nRead the provided JSON file containing 100 rows of sales data. Despite the truncated data (specifically the missing id), you must accurately extract the sales figures from each row.\n\nData Validation and Cleanup:\nEnsure that the data is properly handled even if some fields are incomplete. Since the id is missing for some entries, your focus will be solely on the sales values.\n\nCalculate Total Sales:\nSum the sales values across all 100 rows to provide a single aggregate figure that represents the total sales recorded.\n\nBy successfully recovering and aggregating the sales data, ReceiptRevive Analytics will enable RetailFlow Inc. to:\nReconstruct Historical Sales Data: Gain insights into past sales performance even when original receipts are damaged.\nInform Business Decisions: Use the recovered data to understand sales trends, adjust inventory, and plan future promotions.\nEnhance Data Recovery Processes: Improve methods for handling imperfect OCR data, reducing future data loss and increasing data accuracy.\nBuild Client Trust: Demonstrate the ability to extract valuable insights from challenging datasets, thereby reinforcing client confidence in ReceiptRevive's services.\n\nDownload the data from q-parse-partial-json.jsonl\n\nWhat is the total sales value?" -F "file=@q-parse-partial-json.jsonl"
#curl -X POST "http://127.0.0.1:8000/api/" -H "Content-Type: multipart/form-data" -F "question=Log Analysis for DataSure Technologies\n    DataSure Technologies is a leading provider of IT infrastructure and software solutions, known for its robust systems and proactive maintenance practices. As part of their service offerings, DataSure collects extensive logs from thousands of servers and applications worldwide. These logs, stored in JSON format, are rich with information about system performance, error events, and user interactions. However, the logs are complex and deeply nested, which can make it challenging to quickly identify recurring issues or anomalous behavior.\n\n    Recently, DataSure's operations team observed an increase in system alerts and minor anomalies reported by their monitoring tools. To diagnose these issues more effectively, the team needs to perform a detailed analysis of the log files. One critical step is to count how often a specific key (e.g., \"errorCode\", \"criticalFlag\", or any other operational parameter represented by K) appears in the log entries.\n\n    Key considerations include:\n\n    Complex Structure: The log files are large and nested, with multiple levels of objects and arrays. The target key may appear at various depths.\n    Key vs. Value: The key may also appear as a value within the logs, but only occurrences where it is a key should be counted.\n    Operational Impact: Identifying the frequency of this key can help pinpoint common issues, guide system improvements, and inform maintenance strategies.\n    Your Task\n    As a data analyst at DataSure Technologies, you have been tasked with developing a script that processes a large JSON log file and counts the number of times a specific key, represented by the placeholder K, appears in the JSON structure. Your solution must:\n\n    Parse the Large, Nested JSON: Efficiently traverse the JSON structure regardless of its complexity.\n    Count Key Occurrences: Increment a count only when K is used as a key in the JSON object (ignoring occurrences of K as a value).\n    Return the Count: Output the total number of occurrences, which will be used by the operations team to assess the prevalence of particular system events or errors.\n    By accurately counting the occurrences of a specific key in the log files, DataSure Technologies can:\n\n    Diagnose Issues: Quickly determine the frequency of error events or specific system flags that may indicate recurring problems.\n    Prioritize Maintenance: Focus resources on addressing the most frequent issues as identified by the key count.\n    Enhance Monitoring: Improve automated monitoring systems by correlating key occurrence data with system performance metrics.\n    Inform Decision-Making: Provide data-driven insights that support strategic planning for system upgrades and operational improvements.\n    Download the data from q-extract-nested-json-keys.json\n\n    How many times does K appear as a key?" -F "file=@q-extract-nested-json-keys.json"
#