from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
import os
import tempfile
import zipfile
import openai
import json

app = FastAPI()
token = os.getenv("AIPROXY_TOKEN")
openai.api_base = "http://aiproxy.sanand.workers.dev/openai/v1"
openai.api_key = token

@app.post("/api/")
async def solve(question: str = Form(...), file: UploadFile = None):
    file_content = ""

    if file and file.filename.endswith(".zip"):
        temp_dir = tempfile.gettempdir()
        temp_zip_path = os.path.join(temp_dir, file.filename)
        with open(temp_zip_path, "wb") as f:
            f.write(await file.read())

        extract_dir = os.path.join(temp_dir, "unzipped")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(temp_zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        subfolders = [f.path for f in os.scandir(extract_dir) if f.is_dir()]
        if not subfolders:
            return JSONResponse(content={"answer": "No folder found inside zip"}, status_code=400)

        abcd_dir = subfolders[0]
        csv_path = os.path.join(abcd_dir, "extract.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                file_content = f.read()

    # 🔥 Construct GPT prompt
    full_prompt = f"""You are solving a graded data science assignment question. 
Question: {question}
"""

    if file_content:
        full_prompt += f"\nThe contents of extract.csv are:\n{file_content}"

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant solving IITM data science graded questions. Only return the final answer as a string, nothing else."},
                {"role": "user", "content": full_prompt}
            ]
        )
        raw_output = response["choices"][0]["message"]["content"].strip()
        # Remove extra markdown-style formatting (```json ... ```)
        try:
            parsed = json.loads(raw_output)
            if isinstance(parsed, str):
                final_answer = parsed
            else:
                final_answer = raw_output  # e.g. already JSON
        except:
            final_answer = raw_output  # fallback

        return JSONResponse(content={"answer": final_answer})
    except Exception as e:
        return JSONResponse(content={"answer": f"Error: {str(e)}"}, status_code=500)

#curl -X POST "http://127.0.0.1/api/" -H "Content-Type: multipart/form-data" -F "question=Install and run Visual Studio Code. In your Terminal (or Command Prompt), type code -s and press Enter. Copy and paste the entire output below. What is the output of code -s?"
#curl -X POST "http://127.0.0.1:8000/api/" -H "Content-Type: multipart/form-data" -F "question=Running uv run --with httpie -- https [URL] installs the Python package httpie and sends a HTTPS request to the URL.Send a HTTPS request to https://httpbin.org/get with the URL encoded parameter email set to 24f1002771@ds.study.iitm.ac.in.What is the JSON output of the command? (Paste only the JSON body, not the headers)"
#curl -X POST "http://127.0.0.1:8000/api/" -H "Content-Type: multipart/form-data" -F "question=Download q-move-rename-files.zip and extract it. Use mv to move all files under folders into an empty folder. Then rename all files replacing each digit with the next. 1 becomes 2, 9 becomes 0, a1b9c.txt becomes a2b0c.txt.What does running grep . * | LC_ALL=C sort | sha256sum in bash on that folder show?"" -F "file=@q-move-rename-files.zip"

