from fastapi import FastAPI

app = FastAPI(title="eBay Price Prediction API")


@app.get("/health")
def health():
    return {"status": "ok"}
