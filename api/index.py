from fastapi import FastAPI, Form
from fastapi.responses import JSONResponse
from mangum import Mangum

app = FastAPI()

@app.post("/")
async def solve(question: str = Form(...)):
    return JSONResponse(content={"answer": "You asked: " + question})

handler = Mangum(app)
