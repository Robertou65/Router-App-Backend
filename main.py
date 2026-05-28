import os
from collections import OrderedDict

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
GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")

GEOCODE_CACHE_MAX = 200
geocode_cache = OrderedDict()

SYSTEM_PROMPT_TEMPLATE = """You are an address extraction tool for Colombian package delivery labels.
The input is raw OCR text scanned from a physical package label.
It contains noise: barcodes, weights, tracking numbers, sender info, reference notes.

Your task: extract ONLY the recipient delivery address.

Rules:
0. The delivery addresses in this route are located in {city}, Colombia. Use this as context when identifying the recipient address.
1. The delivery address always follows the keyword "Destino:" if present in the text.
2. Stop extracting when you encounter any of these keywords: Referencia:, Origen:, EAD:, FACTURA, PESO:, PO:, TN:, ST:, Intento
3. Always append the city and country at the end of the extracted address in the format: <street address>, <city>, Colombia. Example: Calle 26 # 51 - 20, Chapinero, {city}, Colombia. Remove postal codes (C.P. XXXXX or CP XXXXX) and phone numbers (Tel: or Telefono:) before appending city and country.
4. Output ONLY the clean address string. No explanation. No extra text. No quotes. No punctuation at start or end.
5. If no address can be found or identified, output exactly the word: ADDRESS_NOT_FOUND

Valid Colombian address format examples:
- Carrera 18b # 32 - 06 Sur, Quiroga Central, {city}, Colombia
- Calle 10 # 43E - 31, El Poblado, {city}, Colombia
- Av Carrera 30 # 45 - 10, Teusaquillo, {city}, Colombia
- KR 18b # 32 - 06 SUR, Rafael Uribe Uribe, {city}, Colombia
- Transversal 8 # 15 - 30, Chapinero, {city}, Colombia"""

app = FastAPI(title="Router App - Address Extractor")


class ExtractRequest(BaseModel):
    ocr_text: str
    city: str


class ExtractResponse(BaseModel):
    address: str
    lat: float
    lng: float
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

    if not GOOGLE_MAPS_API_KEY:
        raise HTTPException(status_code=500, detail="Google Maps API key not configured")

    city = request.city.strip() or "Colombia"
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(city=city)

    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "system": system_prompt,
                    "prompt": request.ocr_text.strip(),
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 40,
                        "top_k": 1,
                    },
                },
            )
            response.raise_for_status()
        except httpx.TimeoutException:
            raise HTTPException(status_code=504, detail="Model inference timed out")
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Ollama error: {str(e)}")

        result = response.json()
        extracted = result.get("response", "").strip()
        logger.info(f"Request received — text length: {len(request.ocr_text)} chars")
        logger.info(f"Route city used: {city}")
        logger.info(f"AI response: {extracted}")

        if not extracted or extracted == "ADDRESS_NOT_FOUND":
            return ExtractResponse(address="", lat=0.0, lng=0.0, success=False)

        normalized_address = extracted.strip().lower()
        cached = geocode_cache.get(normalized_address)
        if cached:
            lat, lng = cached
            logger.info(f"Geocoding result: cache hit lat={lat}, lng={lng}")
            return ExtractResponse(address=extracted, lat=lat, lng=lng, success=True)

        try:
            geo_response = await client.get(
                "https://maps.googleapis.com/maps/api/geocode/json",
                params={
                    "address": extracted,
                    "key": GOOGLE_MAPS_API_KEY,
                },
            )
            geo_response.raise_for_status()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Geocoding error: {str(e)}")

        geo_data = geo_response.json()
        geo_status = geo_data.get("status")
        geo_results = geo_data.get("results") or []
        if geo_status == "ZERO_RESULTS" or not geo_results:
            logger.info(f"Geocoding result: {geo_status or 'EMPTY_RESULTS'}")
            return ExtractResponse(address=extracted, lat=0.0, lng=0.0, success=False)
        if geo_status != "OK":
            raise HTTPException(status_code=502, detail=f"Geocoding error: {geo_status}")

        location = geo_results[0].get("geometry", {}).get("location", {})
        lat = location.get("lat")
        lng = location.get("lng")
        if lat is None or lng is None:
            logger.info("Geocoding result: missing coordinates")
            return ExtractResponse(address=extracted, lat=0.0, lng=0.0, success=False)

        geocode_cache[normalized_address] = (lat, lng)
        if len(geocode_cache) > GEOCODE_CACHE_MAX:
            geocode_cache.popitem(last=False)

        logger.info(f"Geocoding result: lat={lat}, lng={lng}")
        return ExtractResponse(address=extracted, lat=lat, lng=lng, success=True)
