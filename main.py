from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "WORKING SUCCESSFULLY 🚀"}

@app.post("/reset")
def reset():
    return {"status": "reset ok"}

@app.post("/step")
def step():
    return {"status": "step ok"}
