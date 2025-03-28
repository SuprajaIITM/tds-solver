from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on Vercel!"}

@app.get("/greet")
def greet(name: str = "World"):
    return JSONResponse(content={"greeting": f"Hello, {name}!"})
