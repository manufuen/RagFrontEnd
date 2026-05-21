import os
import requests
import streamlit as st
from dotenv import load_dotenv


load_dotenv()

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")


st.set_page_config(
    page_title="RAG Chat",
    page_icon="📚",
    layout="wide"
)


if "messages" not in st.session_state:
    st.session_state.messages = []

if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL


def get_backend_url() -> str:
    return st.session_state.backend_url.rstrip("/")


def check_backend_health():
    response = requests.get(f"{get_backend_url()}/", timeout=10)
    response.raise_for_status()
    return response.json()


def get_elasticsearch_health():
    response = requests.get(f"{get_backend_url()}/health/elasticsearch", timeout=10)
    response.raise_for_status()
    return response.json()


def get_indices():
    response = requests.get(f"{get_backend_url()}/indices", timeout=10)
    response.raise_for_status()
    return response.json()


def upload_document(uploaded_file):
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }

    response = requests.post(
        f"{get_backend_url()}/upload",
        files=files,
        timeout=300,
    )

    response.raise_for_status()
    return response.json()


def ask_question(question: str):
    response = requests.post(
        f"{get_backend_url()}/chat",
        json={"question": question},
        timeout=300,
    )

    response.raise_for_status()
    return response.json()


def render_sources(sources):
    if not sources:
        st.info("No se recuperaron fuentes.")
        return

    clean_sources = []

    for source in sources:
        clean_sources.append({
            "documento": source.get("nombre_documento"),
            "chunk": source.get("chunk_id"),
            "tema": source.get("tema"),
            "score": round(source.get("score", 0), 4) if source.get("score") is not None else None,
            "bm25": round(source.get("bm25_score", 0), 4) if source.get("bm25_score") is not None else None,
            "vector": round(source.get("vector_score", 0), 4) if source.get("vector_score") is not None else None,
            "preview": " ".join(source.get("texto", "").split())[:250],
        })

    st.dataframe(clean_sources, use_container_width=True)


st.title("📚 Chat de RAG con búsqueda híbrida")
st.caption("Sube documentos, clasifícalos por temática y pregunta sobre ellos usando búsqueda híbrida BM25 + embeddings.")


with st.sidebar:
    st.header("⚙️ Configuración")

    st.session_state.backend_url = st.text_input(
        "Backend URL",
        value=st.session_state.backend_url,
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Probar backend"):
            try:
                health = check_backend_health()
                st.success("Backend conectado")
                st.json(health)
            except Exception as e:
                st.error(f"No se pudo conectar con el backend: {e}")

    with col2:
        if st.button("Probar Elastic"):
            try:
                health = get_elasticsearch_health()
                st.success("Elasticsearch conectado")
                st.json(health)
            except Exception as e:
                st.error(f"No se pudo conectar con Elasticsearch: {e}")

    st.divider()

    st.header("📤 Ingestar documento")

    uploaded_file = st.file_uploader(
        "Sube un PDF, DOCX o TXT",
        type=["pdf", "docx", "txt"],
    )

    if uploaded_file is not None:
        st.write(f"Archivo seleccionado: `{uploaded_file.name}`")

        if st.button("Ingestar documento", type="primary"):
            with st.spinner("Procesando documento..."):
                try:
                    result = upload_document(uploaded_file)
                    details = result.get("details", {})

                    st.success("Documento ingestado correctamente")

                    st.write("**Tema detectado:**", details.get("tema"))
                    st.write("**Índice Elastic:**", details.get("indice_elastic"))
                    st.write("**Chunks generados:**", details.get("chunks"))
                    st.write("**Documento ID:**", details.get("documento_id"))

                    if details.get("keywords"):
                        st.write("**Keywords:**", ", ".join(details.get("keywords", [])))

                except Exception as e:
                    st.error(f"Error al ingestar documento: {e}")

    st.divider()

    st.header("🗂️ Índices")

    if st.button("Ver índices RAG"):
        try:
            indices_data = get_indices()
            indices = indices_data.get("indices", [])

            if indices:
                st.write(indices)
            else:
                st.info("Todavía no hay índices RAG.")
        except Exception as e:
            st.error(f"Error consultando índices: {e}")

    st.divider()

    if st.button("Limpiar historial de chat"):
        st.session_state.messages = []
        st.rerun()


st.subheader("💬 Chat")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant" and message.get("metadata"):
            with st.expander("Ver detalles de búsqueda"):
                metadata = message["metadata"]

                st.write("**Tema detectado:**", metadata.get("tema"))
                st.write("**Índice consultado:**", metadata.get("index_name"))
                st.write("**Encontrado:**", metadata.get("found"))
                st.write("**Motivo:**", metadata.get("reason"))

                render_sources(metadata.get("sources", []))


question = st.chat_input("Pregunta algo sobre los documentos ingestados...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en Elasticsearch..."):
            try:
                result = ask_question(question)

                answer = result.get("answer", "No se recibió respuesta.")
                st.markdown(answer)

                with st.expander("Ver detalles de búsqueda"):
                    st.write("**Tema detectado:**", result.get("tema"))
                    st.write("**Índice consultado:**", result.get("index_name"))
                    st.write("**Encontrado:**", result.get("found"))
                    st.write("**Motivo:**", result.get("reason"))

                    render_sources(result.get("sources", []))

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "metadata": result,
                })

            except Exception as e:
                error_message = f"Error al consultar el backend: {e}"
                st.error(error_message)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message,
                })