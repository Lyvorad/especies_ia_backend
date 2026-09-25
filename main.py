import os
import io
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv

# Cargar variables de entorno desde .env si existe localmente
load_dotenv()

app = FastAPI(
    title="API Botánica de Reconocimiento de Especies",
    version="1.0.0"
)

# Configurar CORS para permitir peticiones desde cualquier frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. Definición del Esquema JSON con Pydantic ---
class EspecieItem(BaseModel):
    nombre_comun: str = Field(description="Nombre local o común tal como aparece en la base de datos")
    especie: str = Field(description="Nombre científico completo tal como aparece en la base de datos")
    especie_deletreada: str = Field(description="Nombre científico con cada letra separada por guiones para lectura TTS, ej: S-w-i-e-t-e-n-i-a")
    familia: str = Field(description="Familia botánica tal como aparece en la base de datos")
    familia_deletreada: str = Field(description="Nombre de la familia con cada letra separada por guiones, ej: M-E-L-I-A-C-E-A-E")

class RespuestaAnalisis(BaseModel):
    resultados: list[EspecieItem] = Field(description="Lista de todas las especies identificadas en la imagen")


# --- 2. Cargar Base de Datos Local ---
PATH_ESPECIES = os.path.join(os.path.dirname(__file__), "data", "especies.txt")

try:
    with open(PATH_ESPECIES, "r", encoding="utf-8") as f:
        BASE_DATOS_ESPECIES = f.read()
except FileNotFoundError:
    BASE_DATOS_ESPECIES = ""
    print("WARNING: No se encontró el archivo data/especies.txt")


# --- 3. Inicializar Cliente de Gemini ---
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("WARNING: GEMINI_API_KEY no encontrada en las variables de entorno.")

client = genai.Client(api_key=api_key)


# --- 4. Prompt de Sistema ---
PROMPT_SISTEMA = f"""
Eres un asistente botánico experto encargado de analizar imágenes de exámenes, fichas o muestras botánicas.
Tu objetivo es identificar las especies forestales presentes en la imagen y cotejarlas de forma EXACTA con la BASE DE DATOS DE REFERENCIA provista.

REGLAS DE ORO:
1. Analiza cuidadosamente la imagen enviada por el usuario.
2. Identifica TODAS las especies presentes en la imagen (pueden ser 1, 2, 5 o más). Incluye cada una en la lista de resultados.
3. Para cada especie detectada, busca la coincidencia correspondiente en la BASE DE DATOS DE REFERENCIA.
4. Extrae los valores EXACTOS de 'Nombre local', 'Especie' y 'Familia'. Respeta al 100% la ortografía, mayúsculas, minúsculas, puntos, comas, paréntesis y acentos según figuran en la base de datos.
5. Para los campos `especie_deletreada` y `familia_deletreada`, toma el texto extraído y separa absolutamente CADA LETRA con un guion corto '-'. Las palabras dentro del texto se separan con espacio. Ejemplos:
   - "Swietenia macrophylla King" -> "S-w-i-e-t-e-n-i-a m-a-c-r-o-p-h-y-l-l-a K-i-n-g"
   - "MELIACEAE" -> "M-E-L-I-A-C-E-A-E"

BASE DE DATOS DE REFERENCIA:
{BASE_DATOS_ESPECIES}
"""


# --- 5. Endpoints de la API ---
@app.get("/")
def health_check():
    return {"status": "ok", "message": "API Botánica activa y lista"}


@app.post("/analizar-examen/", response_model=RespuestaAnalisis)
async def analizar_examen(file: UploadFile = File(...)):
    # Validar que se envíe una imagen
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="El archivo enviado debe ser una imagen (PNG, JPG, JPEG, WEBP).")

    try:
        # Leer el contenido de la imagen
        contents = await file.read()
        imagen = Image.open(io.BytesIO(contents))

        # Petición a Gemini probando modelos vigentes con fallback
        model_candidates = ['gemini-flash-latest', 'gemini-3.8-flash', 'gemini-2.5-flash', 'gemini-2.5-pro']
        response = None
        last_exception = None

        for model_name in model_candidates:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        imagen,
                        PROMPT_SISTEMA,
                        "Analiza la imagen enviada y responde completando los datos estrictos de la base de datos."
                    ],
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=RespuestaAnalisis,
                        temperature=0.0  # Temperatura 0 para asegurar cero alucinación
                    )
                )
                if response and response.parsed:
                    break
            except Exception as err:
                print(f"Intento fallido con modelo {model_name}: {err}")
                last_exception = err

        if not response or not response.parsed:
            if last_exception:
                raise last_exception
            else:
                raise Exception("No se pudo obtener respuesta válida de Gemini.")

        # Retornar el JSON validado directamente
        return response.parsed

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error procesando la imagen con IA: {str(e)}")






    