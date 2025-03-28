from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from model.solver import solve_question
from mangum import Mangum

app = FastAPI()

@app.post("/")
async def solve(question: str = Form(...), file: UploadFile = None):
    temp_path = None
    if file:
        temp_path = f"/tmp/{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(file.file.read())
    answer = solve_question(question, temp_path)
    return JSONResponse(content={"answer": answer})

# 👇👇 This is what Vercel looks for
handler = Mangum(app)
