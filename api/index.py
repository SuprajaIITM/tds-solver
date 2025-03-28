from fastapi import FastAPI
from mangum import Mangum

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on Vercel!"}

@app.get("/greet")
def greet(name: str = "World"):
    return {"greeting": f"Hello, {name}!"}

handler = Mangum(app)
