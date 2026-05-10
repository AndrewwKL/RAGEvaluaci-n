"""
RAG Pipeline con Evaluación RAGAS
==================================
Taller: Medir el desempeño de un pipeline RAG usando RAGAS.

Documentos usados:
  - reglamento_academico.pdf
  - programas_asignaturas.pdf

Requisitos:
  pip install langchain langchain-community langchain-google-genai langchain-chroma
  pip install chromadb pypdf python-dotenv scikit-learn ragas

Configuración:
  Crear archivo .env con:  GOOGLE_API_KEY=tu_api_key_aqui
"""

import os
import shutil
import time
import pandas as pd
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# 0. CONFIGURACIÓN
# ─────────────────────────────────────────────
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise EnvironmentError("[ERROR] GOOGLE_API_KEY no encontrada. Crea un archivo .env con tu clave.")
print(f"[OK] API Key cargada (termina en ...{API_KEY[-6:]})")

# ─────────────────────────────────────────────
# PARÁMETROS DEL PIPELINE
# ─────────────────────────────────────────────
PDF_DIR          = "pdfs"          # Carpeta con los PDFs
CHUNK_SIZE       = 500             # Caracteres por fragmento
CHUNK_OVERLAP    = 50              # Solapamiento entre fragmentos
K_RETRIEVED      = 5              # Fragmentos recuperados por consulta
EMBEDDING_MODEL  = "gemini-embedding-001"
LLM_MODEL        = "gemini-2.0-flash"
PERSIST_DIR      = "./chroma_db_taller"

print("\n=== PARÁMETROS DEL PIPELINE ===")
print(f"  Documentos:        {PDF_DIR}/")
print(f"  Modelo embeddings: {EMBEDDING_MODEL}")
print(f"  chunk_size:        {CHUNK_SIZE}  |  chunk_overlap: {CHUNK_OVERLAP}")
print(f"  k (chunks ret.):   {K_RETRIEVED}")
print(f"  LLM generador:     {LLM_MODEL}")
print(f"  LLM juez (RAGAS):  {LLM_MODEL}")

# ─────────────────────────────────────────────
# PASO 1 — CARGA DE DOCUMENTOS PDF
# ─────────────────────────────────────────────
from langchain_community.document_loaders import PyPDFLoader

print("\n=== PASO 1: Carga de documentos ===")
pdf_files = sorted([f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")])
print(f"  PDFs encontrados: {len(pdf_files)}")
for f in pdf_files:
    kb = os.path.getsize(os.path.join(PDF_DIR, f)) // 1024
    print(f"    - {f}  ({kb} KB)")

documents = []
for pdf_file in pdf_files:
    path = os.path.join(PDF_DIR, pdf_file)
    loader = PyPDFLoader(path)
    pages = loader.load()
    documents.extend(pages)
    print(f"  [OK] {pdf_file}: {len(pages)} páginas")

print(f"\n  Total páginas cargadas: {len(documents)}")

# ─────────────────────────────────────────────
# PASO 2 — CHUNKING
# ─────────────────────────────────────────────
from langchain_text_splitters import RecursiveCharacterTextSplitter

print("\n=== PASO 2: Chunking ===")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ".", " "],
)
chunks = text_splitter.split_documents(documents)
print(f"  Páginas:   {len(documents)}")
print(f"  Chunks:    {len(chunks)}  (factor {len(chunks)/len(documents):.1f}x)")

# ─────────────────────────────────────────────
# PASO 3 — EMBEDDINGS
# ─────────────────────────────────────────────
from langchain_google_genai import GoogleGenerativeAIEmbeddings

print("\n=== PASO 3: Embeddings ===")
embeddings_model = GoogleGenerativeAIEmbeddings(
    model=EMBEDDING_MODEL,
    google_api_key=API_KEY,
)
sample_vec = embeddings_model.embed_query(chunks[0].page_content)
print(f"  Modelo:     {EMBEDDING_MODEL}")
print(f"  Dimensión:  {len(sample_vec)}")

# ─────────────────────────────────────────────
# PASO 4 — BASE VECTORIAL (ChromaDB)
# ─────────────────────────────────────────────
from langchain_chroma import Chroma

print("\n=== PASO 4: Base vectorial ChromaDB ===")
if os.path.exists(PERSIST_DIR):
    shutil.rmtree(PERSIST_DIR)
    print(f"  Base anterior eliminada: {PERSIST_DIR}")

print(f"  Indexando {len(chunks)} fragmentos...")
vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    persist_directory=PERSIST_DIR,
    collection_name="taller_rag",
    collection_metadata={"hnsw:space": "cosine"},
)
total = vector_store._collection.count()
print(f"  [OK] {total} fragmentos indexados en ChromaDB")

# ─────────────────────────────────────────────
# PASO 5-7 — PIPELINE RAG COMPLETO
# ─────────────────────────────────────────────
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

print("\n=== PASO 5-7: Pipeline RAG ===")

llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL,
    google_api_key=API_KEY,
    temperature=0.1,
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": K_RETRIEVED},
)

PROMPT_TEMPLATE = """Eres un asistente académico especializado en reglamentos y programas de asignaturas.
Responde la pregunta usando ÚNICAMENTE la información del contexto proporcionado.
Si la respuesta no está en el contexto, responde exactamente: "No encontré información sobre esto en el documento."

Contexto:
{context}

Pregunta: {question}

Respuesta:"""

prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)


def rag_pipeline(pregunta: str, k: int = K_RETRIEVED, verbose: bool = False) -> dict:
    """Ejecuta el pipeline RAG completo y retorna un diccionario con pregunta,
    fragmentos recuperados, contexto ensamblado y respuesta generada."""
    docs = retriever.invoke(pregunta)

    contexto = "\n\n---\n\n".join(
        f"[Fragmento {i+1} — Pág. {doc.metadata.get('page', '?')}]\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

    prompt = prompt_template.invoke({"context": contexto, "question": pregunta})
    respuesta_llm = llm.invoke(prompt)

    texto = (
        respuesta_llm.content[0].get("text", "")
        if isinstance(respuesta_llm.content, list)
        else respuesta_llm.content
    )

    if verbose:
        print(f"\n  ❓ {pregunta}")
        for i, d in enumerate(docs, 1):
            fuente = os.path.basename(d.metadata.get("source", "?"))
            pagina = d.metadata.get("page", "?")
            print(f"     [{i}] {fuente} Pág.{pagina}: {d.page_content[:80]}...")
        print(f"  🤖 {texto}\n")

    return {
        "pregunta": pregunta,
        "fragmentos": docs,
        "contexto": contexto,
        "respuesta": texto,
    }


# ─────────────────────────────────────────────
# PASO 8 — EVALUACIÓN CON RAGAS
# ─────────────────────────────────────────────
from ragas import evaluate, EvaluationDataset
from ragas.metrics.collections import faithfulness, answer_relevancy, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

print("\n=== PASO 8: Evaluación RAGAS ===")

# ── Casos de prueba (8 preguntas, 2 de cada tipo) ──────────────────────────
#
# Tipo A — Respuesta textualmente en el documento (RAG debe encontrarla exacta)
# Tipo B — Vocabulario diferente al del documento (prueba embeddings semánticos)
# Tipo C — Requiere combinar información de varios chunks
# Tipo D — El sistema NO debería tener la respuesta (detecta alucinaciones)
#
muestras_evaluacion = [
    # ── Tipo A: Respuesta textual en el documento ─────────────────────
    {
        "user_input":  "¿Cuántos créditos tiene la asignatura de Pre-Cálculo?",
        "reference":   "Pre-Cálculo tiene 3 créditos y se dicta en el Semestre 1.",
        "tipo":        "A - Textual en documento",
    },
    {
        "user_input":  "¿Cuál es el promedio mínimo para acceder a un examen extraordinario?",
        "reference":   "El promedio mínimo requerido es 2.8 sobre 5.0.",
        "tipo":        "A - Textual en documento",
    },
    # ── Tipo B: Vocabulario diferente — prueba semántica ──────────────
    {
        "user_input":  "¿Qué ayuda económica recibe un alumno con notas sobresalientes?",
        "reference":   "Los estudiantes con promedio superior a 4.5 reciben un incentivo económico del 30% del valor de la matrícula.",
        "tipo":        "B - Vocabulario diferente",
    },
    {
        "user_input":  "¿Qué pasa si un estudiante deja de estudiar sin avisar a la institución?",
        "reference":   "Recibe una sanción administrativa: pérdida del derecho a certificados académicos por dos semestres.",
        "tipo":        "B - Vocabulario diferente",
    },
    # ── Tipo C: Combinar información de varios chunks ──────────────────
    {
        "user_input":  "¿Qué materias se dictan en el Semestre 1 y cuántos créditos tienen en total?",
        "reference":   "En Semestre 1 se dictan Pre-Cálculo (3), Cálculo Diferencial (4), Lógica Matemática (3) y Habilidades Comunicativas (2), sumando 12 créditos.",
        "tipo":        "C - Combinar chunks",
    },
    {
        "user_input":  "¿Cuáles son las causas válidas para solicitar un examen extraordinario y qué promedio mínimo se necesita?",
        "reference":   "Enfermedad grave comprobada, duelo familiar de primer grado o calamidad doméstica. El promedio mínimo es 2.8.",
        "tipo":        "C - Combinar chunks",
    },
    # ── Tipo D: El sistema NO tiene la respuesta (detecta alucinaciones) ─
    {
        "user_input":  "¿Cuál es el nombre del rector de la universidad y cuándo fue elegido?",
        "reference":   "No encontré información sobre esto en el documento.",
        "tipo":        "D - No está en documento",
    },
    {
        "user_input":  "¿Cuánto cuesta la matrícula exacta en pesos colombianos para el semestre 2024-2?",
        "reference":   "No encontré información sobre esto en el documento.",
        "tipo":        "D - No está en documento",
    },
]

# Ejecutar RAG para cada muestra
registros = []
print(f"\n  Ejecutando RAG para {len(muestras_evaluacion)} preguntas...\n")
for i, m in enumerate(muestras_evaluacion):
    if i > 0:
        print("  ⏳ Esperando 15s para no exceder el rate limit de la API gratuita...")
        time.sleep(15)
    resultado = rag_pipeline(m["user_input"], verbose=False)
    registros.append({
        "user_input":         m["user_input"],
        "retrieved_contexts": [doc.page_content for doc in resultado["fragmentos"]],
        "response":           resultado["respuesta"],
        "reference":          m["reference"],
    })
    print(f"  [OK] [{m['tipo']}]")
    print(f"       ❓ {m['user_input']}")
    print(f"       🤖 {resultado['respuesta'][:100]}...\n")

# Evaluar con RAGAS
llm_juez        = LangchainLLMWrapper(llm)
embeddings_juez = LangchainEmbeddingsWrapper(embeddings_model)
dataset         = EvaluationDataset.from_list(registros)

print("  ⏳ Esperando 30s antes de la evaluación RAGAS para evitar rate limit...")
time.sleep(30)
print("  Evaluando con RAGAS (consume tokens adicionales)...")
resultados_ragas = evaluate(
    dataset=dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    llm=llm_juez,
    embeddings=embeddings_juez,
)

# ─────────────────────────────────────────────
# RESULTADOS FINALES
# ─────────────────────────────────────────────
df = resultados_ragas.to_pandas()

cols_score = ["faithfulness", "answer_relevancy", "context_precision"]
cols_tipo  = ["user_input"] + [c for c in cols_score if c in df.columns]

print("\n" + "="*80)
print("TABLA DE RESULTADOS RAGAS")
print("="*80)

# Añadir tipo de pregunta
tipos = [m["tipo"] for m in muestras_evaluacion]
df["tipo"] = tipos

tabla = df[["user_input", "tipo"] + [c for c in cols_score if c in df.columns]]
pd.set_option("display.max_colwidth", 55)
pd.set_option("display.width", 120)
print(tabla.to_string(index=False))

print("\n" + "-"*40)
print("Promedios globales:")
for col in cols_score:
    if col in df.columns:
        score = df[col].mean()
        interpretacion = (
            "Excelente"  if score > 0.9 else
            "Bueno"      if score > 0.7 else
            "Aceptable"  if score > 0.5 else
            "Problemático"
        )
        print(f"  {col:<25} {score:.4f}  →  {interpretacion}")

print("\n[OK] Evaluación completada.")
print(f"     Base vectorial en: {PERSIST_DIR}/")
