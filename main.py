import os

import httpx
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_SECRET_KEY = os.getenv("API_SECRET_KEY")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:1b")

SYSTEM_PROMPT_TEMPLATE = """You are an address extraction tool for Colombian package delivery labels.
The input is raw OCR text scanned from a physical package label.
It contains noise: barcodes, weights, tracking numbers, sender info, reference notes.

Your task: extract ONLY the recipient delivery address.

Rules:
0. The delivery addresses in this route are located in {city}, Colombia. Use this as context when identifying the recipient address.
1. The delivery address always follows the keyword "Destino:" if present in the text.
2. Stop extracting when you encounter any of these keywords: Referencia:, Origen:, EAD:, FACTURA, PESO:, PO:, TN:, ST:, Intento
3. Remove from the address: postal codes (C.P. XXXXX or CP XXXXX), phone numbers (Tel: or Telefono:), the word Colombia (it will be added later)
4. Output ONLY the clean address string. No explanation. No extra text. No quotes. No punctuation at start or end.
5. If no address can be found or identified, output exactly the word: ADDRESS_NOT_FOUND

Valid Colombian address format examples:
- Carrera 18b # 32 - 06 Sur, Quiroga Central, {city}
- Calle 10 # 43E - 31, El Poblado, {city}
- Av Carrera 30 # 45 - 10, Teusaquillo, {city}
- KR 18b # 32 - 06 SUR, Rafael Uribe Uribe, {city}
- Transversal 8 # 15 - 30, Chapinero, {city}"""

app = FastAPI(title="Router App - Address Extractor")


class ExtractRequest(BaseModel):
    ocr_text: str
    city: str


class ExtractResponse(BaseModel):
    address: str
    success: bool


@app.get("/health")
async def health():
    return {"status": "ok", "model": OLLAMA_MODEL}


@app.post("/extract-address", response_model=ExtractResponse)
async def extract_address(
    request: ExtractRequest,
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    if x_api_key != API_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if not request.ocr_text or not request.ocr_text.strip():
        raise HTTPException(status_code=400, detail="ocr_text is empty")

    city = request.city.strip() or "Colombia"
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(city=city)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "system": system_prompt,
                    "prompt": request.ocr_text.strip(),
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 60,
                    },
                },
            )
        response.raise_for_status()
        result = response.json()
        extracted = result.get("response", "").strip()
        logger.info(f"Request received — text length: {len(request.ocr_text)} chars")
        logger.info(f"Route city used: {city}")
        logger.info(f"AI response: {extracted}")
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Model inference timed out")
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Ollama error: {str(e)}")

    if not extracted or extracted == "ADDRESS_NOT_FOUND":
        return ExtractResponse(address="", success=False)

    return ExtractResponse(address=extracted, success=True)
