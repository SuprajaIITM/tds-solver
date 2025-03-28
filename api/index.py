from fastapi import FastAPI
from fastapi.responses import JSONResponse
from vercel_fastapi import VercelFastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello from FastAPI on Vercel!"}

@app.get("/greet")
def greet(name: str = "World"):
    return JSONResponse(content={"greeting": f"Hello, {name}!"})

# This wraps it so Vercel can understand it
app = VercelFastAPI(app)
