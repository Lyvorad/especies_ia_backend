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
    nombre_comun: str = Field(description="Nombre(s) local(es) o común(es) tal como aparece(n) en la base de datos (puede incluir varios separados por coma)")
    nombres_comunes_lista: list[str] = Field(description="Lista de cada nombre común por separado. Si hay varios separados por comas, cada uno es un elemento. Ej: ['aguano', 'caoba'] o ['cedro rojo']")
    especie: str = Field(description="Nombre científico completo tal como aparece en la base de datos")
    especie_deletreada: str = Field(description="Nombre científico deletreado letra por letra con ', ' entre letras. Usa 'espacio' entre palabras SOLO si el nombre tiene 3 o más palabras. Ej (3 palabras): 'S, w, i, e, t, e, n, i, a, espacio, m, a, c, r, o, p, h, y, l, l, a, espacio, K, i, n, g'. Ej (1 palabra): 'M, E, L, I, A, C, E, A, E'")
    familia: str = Field(description="Familia botánica tal como aparece en la base de datos")
    familia_deletreada: str = Field(description="Familia deletreada letra por letra con ', ' entre letras. Usa 'espacio' entre palabras SOLO si la familia tiene 3 o más palabras.")
    campo_a_completar: str = Field(description="Campo que estaba en blanco en el examen/ficha. Valores: 'nombre_comun', 'especie', 'familia', o 'ninguno'.")
    campo_a_completar_deletreado: str = Field(description=(
        "Deletreo del valor del campo que faltaba completar. "
        "REGLAS DE DELETREO: separa letras con ', '. Usa la palabra 'espacio' entre palabras SOLO si ese nombre tiene 3 o más palabras. "
        "Si campo_a_completar='nombre_comun' y hay varios nombres, deletrea cada nombre por separado y sepáralos con ' | '. "
        "Ej nombre_comun con 2 nombres de 1 palabra: 'a, g, u, a, n, o | c, a, o, b, a'. "
        "Ej nombre_comun con 1 nombre de 3 palabras: 'p, a, l, o, espacio, d, e, espacio, r, o, s, a'. "
        "Ej especie (3 palabras): 'S, w, i, e, t, e, n, i, a, espacio, m, a, c, r, o, p, h, y, l, l, a, espacio, K, i, n, g'. "
        "Si campo_a_completar='ninguno', pon cadena vacía."
    ))

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
5. En `nombres_comunes_lista`: si el nombre común tiene varios nombres separados por coma (ej: "aguano, caoba"), pon cada uno como elemento separado: ["aguano", "caoba"]. Si es uno solo, pon solo ese elemento: ["cedro rojo"].
6. REGLA DE DELETREO (aplica a especie_deletreada, familia_deletreada y campo_a_completar_deletreado):
   - Separa cada letra con coma y espacio: ', '
   - Usa la palabra 'espacio' entre palabras SOLAMENTE si ese nombre/valor tiene 3 O MÁS palabras.
   - Si tiene 1 o 2 palabras: NO uses 'espacio', deletrea todas las letras seguidas (separadas por comas).
   - Ejemplos:
     * "aguano" (1 palabra) → "a, g, u, a, n, o"
     * "caoba" (1 palabra) → "c, a, o, b, a"
     * "MELIACEAE" (1 palabra) → "M, E, L, I, A, C, E, A, E"
     * "Cedrus libani" (2 palabras) → "C, e, d, r, u, s, l, i, b, a, n, i"  ← SIN espacio
     * "Swietenia macrophylla King" (3 palabras) → "S, w, i, e, t, e, n, i, a, espacio, m, a, c, r, o, p, h, y, l, l, a, espacio, K, i, n, g"
     * "palo de rosa" (3 palabras) → "p, a, l, o, espacio, d, e, espacio, r, o, s, a"
7. DETECTA QUÉ CAMPO ESTABA EN BLANCO en el examen o ficha:
   - Nombre común vacío → campo_a_completar = "nombre_comun"
   - Especie/nombre científico vacío → campo_a_completar = "especie"
   - Familia vacía → campo_a_completar = "familia"
   - Ninguno vacío → campo_a_completar = "ninguno"
8. En `campo_a_completar_deletreado`:
   - Si es "nombre_comun" con varios nombres: deletrea cada nombre por separado aplicando la regla 6, y sepáralos con ' | ' (pipe con espacios).
     Ej "aguano, caoba": "a, g, u, a, n, o | c, a, o, b, a"
   - Si es "especie" o "familia": aplica directamente la regla 6 al valor completo.
   - Si es "ninguno": pon cadena vacía "".

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

        # Petición a Gemini probando modelos 3.x vigentes con fallback
        model_candidates = ['gemini-3.6-flash', 'gemini-3.1-flash-lite', 'gemini-flash-lite-latest', 'gemini-3.5-flash-lite']
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






    