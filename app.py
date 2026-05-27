import os # librería para manejar variables de entorno y rutas
import requests # libería para hacer peticiones HTTP al backend
import streamlit as st # librería para crear la interfaz de usuario de la aplicación
from dotenv import load_dotenv # librería para cargar variables de entorno desde un archivo .env


# =============================================================================
# Configuración inicial
# =============================================================================
load_dotenv()

DEFAULT_BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="📚",
    layout="wide",
)


# =============================================================================
# Estado de sesión
# =============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "backend_url" not in st.session_state:
    st.session_state.backend_url = DEFAULT_BACKEND_URL

if "backend_status" not in st.session_state:
    st.session_state.backend_status = "Sin probar"

if "elastic_status" not in st.session_state:
    st.session_state.elastic_status = "Sin probar"

if "last_indices" not in st.session_state:
    st.session_state.last_indices = []

if "last_documents" not in st.session_state:
    st.session_state.last_documents = []

if "confirm_delete_all_topics" not in st.session_state:
    st.session_state.confirm_delete_all_topics = False

if "confirm_delete_all_documents" not in st.session_state:
    st.session_state.confirm_delete_all_documents = False


# =============================================================================
# Cliente HTTP del backend
# =============================================================================
def get_backend_url() -> str:
    # Asegura que la URL del backend no termine con una barra para evitar problemas al concatenar endpoints
    return st.session_state.backend_url.rstrip("/")


def request_json(method: str, endpoint: str, **kwargs):
    # Realiza una petición HTTP al backend y devuelve la respuesta JSON. 
    response = requests.request(
        method=method,
        url=f"{get_backend_url()}{endpoint}",
        timeout=kwargs.pop("timeout", 30),
        **kwargs,
    )
    response.raise_for_status()

    if not response.content:
        return {}

    return response.json()


''' 
Endpoints que el frontend espera que el backend implemente. Cada función hace una petición HTTP al endpoint correspondiente y devuelve la respuesta JSON.
'''
def check_backend_health():
    return request_json("GET", "/", timeout=10)


def get_elasticsearch_health():
    return request_json("GET", "/health/elasticsearch", timeout=10)


def get_indices():
    return request_json("GET", "/indices", timeout=10)


def delete_index(index_name: str):
    return request_json("DELETE", f"/indices/{index_name}", timeout=60)


def delete_all_indices():
    return request_json("DELETE", "/indices", timeout=120)


def get_documents():
    return request_json("GET", "/documents", timeout=30)


def delete_document(document_id: str):
    return request_json("DELETE", f"/documents/{document_id}", timeout=60)


def delete_all_documents():
    return request_json("DELETE", "/documents", timeout=120)


def upload_document(uploaded_file):
    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            uploaded_file.type or "application/octet-stream",
        )
    }

    return request_json(
        "POST",
        "/upload",
        files=files,
        timeout=300,
    )


def ask_question(question: str):
    return request_json(
        "POST",
        "/chat",
        json={"question": question},
        timeout=300,
    )


# =============================================================================
# Helpers para temáticas
# =============================================================================
def extract_index_name(index_item):
    # Devuelve exactamente el nombre real del índice: por ejemplo, rag_musica.
    if isinstance(index_item, str):
        return index_item

    if isinstance(index_item, dict):
        return (
            index_item.get("index_name")
            or index_item.get("indice_elastic")
            or index_item.get("indice")
            or index_item.get("name")
            or index_item.get("index")
            or "-"
        )

    return str(index_item)


def load_indices_into_state():
    # Carga las temáticas (índices) desde el backend y las guarda en el estado de sesión para mostrarlas en la pestaña de temáticas.
    indices_data = get_indices()
    st.session_state.last_indices = indices_data.get("indices", [])


def extract_document_id(document_item):
    # Devuelve el ID del documento asignado por el backend.
    if isinstance(document_item, str):
        return document_item

    if isinstance(document_item, dict):
        return (
            document_item.get("documento_id")
            or document_item.get("document_id")
            or document_item.get("id")
            or document_item.get("nombre_documento")
            or document_item.get("filename")
            or document_item.get("name")
            or "-"
        )

    return str(document_item)


def extract_document_name(document_item):
    # Devuelve el nombre del documento asignado por el backend.
    if isinstance(document_item, str):
        return document_item

    if isinstance(document_item, dict):
        return (
            document_item.get("nombre_documento")
            or document_item.get("filename")
            or document_item.get("file_name")
            or document_item.get("name")
            or document_item.get("title")
            or extract_document_id(document_item)
        )

    return str(document_item)


def load_documents_into_state():
    # Carga los documentos ingestados desde el backend y los guarda en el estado de sesión para mostrarlos en la pestaña de documentos.
    documents_data = get_documents()
    st.session_state.last_documents = (
        documents_data.get("documents")
        or documents_data.get("documentos")
        or documents_data.get("items")
        or []
    )


# =============================================================================
# Componentes visuales reutilizables
# =============================================================================
def render_chunks(chunks):
    # Renderiza la lista de chunks recuperados en la traza de búsqueda. 
    if not chunks:
        st.info("No se recuperaron chunks. El asistente no debería inventar una respuesta.")
        return

    st.markdown("#### 📚 Chunks recuperados")

    for i, chunk_data in enumerate(chunks, start=1):
        documento = chunk_data.get("nombre_documento", "Documento desconocido")
        tema = chunk_data.get("tema", "-")
        chunk_id = chunk_data.get("chunk_id", "-")
        score = chunk_data.get("score")
        bm25 = chunk_data.get("bm25_score")
        vector = chunk_data.get("vector_score")
        texto = " ".join(chunk_data.get("texto", "").split())

        with st.container(border=True):
            st.markdown(f"**Chunk {i}: {documento}**")
            st.caption(f"Temática: {tema} · Chunk ID: {chunk_id}")

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Score",
                round(score, 4) if score is not None else "-",
            )
            col2.metric(
                "BM25",
                round(bm25, 4) if bm25 is not None else "-",
            )
            col3.metric(
                "Vector",
                round(vector, 4) if vector is not None else "-",
            )

            if texto:
                st.markdown(f"> {texto[:700]}...")

    with st.expander("Vista técnica de chunks"):
        rows = []

        for chunk_data in chunks:
            rows.append(
                {
                    "documento": chunk_data.get("nombre_documento"),
                    "chunk": chunk_data.get("chunk_id"),
                    "tematica": chunk_data.get("tema"),
                    "score": round(chunk_data.get("score", 0), 4)
                    if chunk_data.get("score") is not None
                    else None,
                    "bm25": round(chunk_data.get("bm25_score", 0), 4)
                    if chunk_data.get("bm25_score") is not None
                    else None,
                    "vector": round(chunk_data.get("vector_score", 0), 4)
                    if chunk_data.get("vector_score") is not None
                    else None,
                    "preview": " ".join(chunk_data.get("texto", "").split())[:250],
                }
            )

        st.dataframe(rows, use_container_width=True, hide_index=True)


def render_search_trace(data):
    # Renderiza la traza de búsqueda RAG recibida del backend.
    selected_document = data.get("selected_document")
    chunks = data.get("sources", [])

    st.markdown("#### 🔎 Traza RAG")

    st.write("✅ Pregunta recibida")
    st.write(f"🏷️ Temática detectada: `{data.get('tema', '-')}`")
    st.write(f"🗂️ Índice consultado: `{data.get('index_name', '-')}`")
    st.write(f"📚 Chunks recuperados: `{len(chunks)}`")

    if data.get("found"):
        st.success("Se encontró evidencia en los documentos ingestados.")
    else:
        st.warning("No se encontró evidencia suficiente en los documentos ingestados.")

    if data.get("reason"):
        st.caption(f"Motivo: {data.get('reason')}")

    if selected_document:
        st.markdown("#### Documento seleccionado")
        st.write("**Nombre:**", selected_document.get("nombre_documento"))
        st.write("**Documento ID:**", selected_document.get("documento_id"))


def render_ingestion_result(result):
    # Renderiza el resultado de la ingesta de un documento. 
    details = result.get("details", {})

    if result.get("status") == "duplicated":
        st.warning("Este documento ya estaba ingestado.")
    else:
        st.success("Documento ingestado correctamente.")

    col1, col2, col3 = st.columns(3)

    col1.metric("Tema detectado", details.get("tema", "-"))
    col2.metric("Chunks", details.get("chunks", "-"))
    col3.metric("Índice Elastic", details.get("indice_elastic", "-"))

    with st.expander("Detalles del documento"):
        st.write("**Documento:**", details.get("nombre_documento", "-"))
        st.write("**Documento ID:**", details.get("documento_id", "-"))
        st.write("**Fecha de ingesta:**", details.get("fecha_ingesta", "-"))

        keywords = details.get("keywords") or []

        if keywords:
            st.markdown("**Keywords:**")
            st.write(" ".join([f"`{keyword}`" for keyword in keywords]))


def build_ingestion_summary(upload_results):
    # Construye una tabla resumen con el resultado de la ingesta de múltiples documentos. Cada fila representa un documento y muestra su estado, temática detectada, índice asignado, número de chunks generados y detalles adicionales como el ID del documento o errores.
    rows = []

    for item in upload_results:
        file_name = item["file_name"]
        status = item["status"]

        if status == "error":
            rows.append(
                {
                    "archivo": file_name,
                    "estado": "error",
                    "tema": "-",
                    "índice": "-",
                    "chunks": "-",
                    "detalle": item.get("error", "-"),
                }
            )
            continue

        result = item.get("result", {})
        details = result.get("details", {})

        rows.append(
            {
                "archivo": file_name,
                "estado": result.get("status", "ingested"),
                "tema": details.get("tema", "-"),
                "índice": details.get("indice_elastic", "-"),
                "chunks": details.get("chunks", "-"),
                "detalle": details.get("documento_id", "-"),
            }
        )

    return rows


def render_bulk_ingestion_summary(upload_results):
    # Renderiza un resumen visual de la ingesta masiva de documentos, mostrando métricas clave como el total de archivos procesados, cuántos fueron correctos, cuántos estaban duplicados y cuántos tuvieron errores. También muestra una tabla con el detalle de cada archivo.
    total = len(upload_results)
    errors = sum(1 for item in upload_results if item["status"] == "error")
    successful = total - errors
    duplicated = sum(
        1
        for item in upload_results
        if item["status"] == "success"
        and item.get("result", {}).get("status") == "duplicated"
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Archivos procesados", total)
    col2.metric("Correctos", successful)
    col3.metric("Duplicados", duplicated)
    col4.metric("Errores", errors)

    summary_rows = build_ingestion_summary(upload_results)
    st.dataframe(summary_rows, use_container_width=True, hide_index=True)


def render_sidebar():
    # Renderiza la barra lateral con la configuración de la URL del backend, botones para probar la conexión al backend y a Elasticsearch, y opciones para limpiar el historial de chat.
    with st.sidebar:
        st.header("⚙️ Configuración")

        st.session_state.backend_url = st.text_input(
            "Backend URL",
            value=st.session_state.backend_url,
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Probar backend", use_container_width=True):
                try:
                    check_backend_health()
                    st.session_state.backend_status = "Online"
                    st.success("Backend conectado")
                except Exception as exc:
                    st.session_state.backend_status = "Error"
                    st.error(f"Backend no disponible: {exc}")

        with col2:
            if st.button("Probar Elastic", use_container_width=True):
                try:
                    get_elasticsearch_health()
                    st.session_state.elastic_status = "Online"
                    st.success("Elastic conectado")
                except Exception as exc:
                    st.session_state.elastic_status = "Error"
                    st.error(f"Elastic no disponible: {exc}")

        st.divider()

        st.info(
            "Modo estricto: el asistente debe responder solo con información "
            "contenida en los documentos ingestados."
        )

        if st.button("Limpiar historial", use_container_width=True):
            st.session_state.messages = []
            st.rerun()


def render_chat_tab():
    # Renderiza la pestaña de chat, mostrando el historial de mensajes y la traza de búsqueda para cada respuesta del asistente. Permite al usuario ingresar una pregunta y muestra la respuesta generada por el backend junto con la evidencia recuperada.
    st.info(
        "Lanza una pregunta para buscar información dentro de tu biblioteca de documentos ingestados."
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("metadata"):
                with st.expander("Ver traza de búsqueda"):
                    render_search_trace(message["metadata"])

                with st.expander("Ver chunks recuperados"):
                    render_chunks(message["metadata"].get("sources", []))

    question = st.chat_input("Pregunta algo sobre los documentos ingestados...")

    if question:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Buscando evidencia en la biblioteca de documentos..."):
                    result = ask_question(question)

                answer = result.get("answer", "No se recibió respuesta.")
                st.markdown(answer)

                if result.get("found"):
                    st.success("Respuesta basada en documentos recuperados.")
                else:
                    st.warning("No se encontró evidencia suficiente en los documentos.")

                with st.expander("Ver traza de búsqueda"):
                    render_search_trace(result)

                with st.expander("Ver chunks recuperados"):
                    render_chunks(result.get("sources", []))

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "metadata": result,
                    }
                )

            except Exception as exc:
                error_message = f"Error al consultar el backend: {exc}"
                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                    }
                )


def render_ingestion_tab():
    # Renderiza la pestaña de ingesta, permitiendo al usuario subir uno o varios documentos para incorporarlos a la biblioteca. Muestra el progreso de la ingesta, el resultado de cada archivo y un resumen al finalizar.
    st.subheader("📤 Ingestar documentos")

    st.write(
        "Sube uno o varios archivos PDF, DOCX o TXT. Pueden ser de distintos "
        "formatos. El frontend los enviará al backend uno por uno para que se "
        "lean, clasifiquen, dividan en chunks, se generen embeddings y se guarden "
        "en Elasticsearch."
    )

    uploaded_files = st.file_uploader(
        "Documentos",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if not uploaded_files:
        st.info("Selecciona uno o varios documentos para iniciar la ingesta.")
        return

    st.markdown("#### Archivos seleccionados")

    selected_rows = [
        {
            "archivo": uploaded_file.name,
            "tipo": uploaded_file.type or "desconocido",
            "tamaño_kb": round(len(uploaded_file.getvalue()) / 1024, 2),
        }
        for uploaded_file in uploaded_files
    ]

    st.dataframe(selected_rows, use_container_width=True, hide_index=True)

    if st.button("Ingestar documentos", type="primary"):
        upload_results = []
        progress = st.progress(0)
        total_files = len(uploaded_files)

        with st.status("Procesando documentos...", expanded=True) as status:
            for index, uploaded_file in enumerate(uploaded_files, start=1):
                st.write(f"📄 Procesando `{uploaded_file.name}` ({index}/{total_files})")
                st.write("🏷️ Clasificando temática")
                st.write("✂️ Dividiendo en chunks")
                st.write("🧠 Generando embeddings")
                st.write("📦 Guardando en Elasticsearch")

                try:
                    result = upload_document(uploaded_file)

                    upload_results.append(
                        {
                            "file_name": uploaded_file.name,
                            "status": "success",
                            "result": result,
                        }
                    )

                    if result.get("status") == "duplicated":
                        st.warning(f"`{uploaded_file.name}` ya estaba ingestado.")
                    else:
                        st.success(f"`{uploaded_file.name}` ingestado correctamente.")

                except Exception as exc:
                    upload_results.append(
                        {
                            "file_name": uploaded_file.name,
                            "status": "error",
                            "error": str(exc),
                        }
                    )
                    st.error(f"Error al ingestar `{uploaded_file.name}`: {exc}")

                progress.progress(index / total_files)

            errors = sum(1 for item in upload_results if item["status"] == "error")

            if errors:
                status.update(
                    label=f"Ingesta finalizada con {errors} error(es)",
                    state="error",
                )
            else:
                status.update(
                    label="Ingesta finalizada correctamente",
                    state="complete",
                )

        st.markdown("### Resumen de ingesta")
        render_bulk_ingestion_summary(upload_results)

        with st.expander("Detalles por archivo"):
            for item in upload_results:
                with st.container(border=True):
                    st.markdown(f"#### {item['file_name']}")

                    if item["status"] == "error":
                        st.error(item.get("error", "Error desconocido"))
                    else:
                        render_ingestion_result(item["result"])


def render_topics_table(indices):
    # Renderiza la tabla de temáticas con buscador.
    st.markdown("#### Temáticas almacenadas")

    search_term = st.text_input(
        "Buscar temática",
        placeholder="Ej: biologia, quimica, medicina, astronomia...",
        key="topics_search",
    ).strip().lower()

    if not indices:
        st.info("Todavía no hay temáticas cargadas o no se han consultado en esta sesión.")
        return

    filtered_indices = []

    for index_item in indices:
        index_name = extract_index_name(index_item)
        clean_topic_name = index_name.replace("rag_", "", 1)

        searchable_text = f"{index_name} {clean_topic_name}".lower()

        if not search_term or search_term in searchable_text:
            filtered_indices.append(index_item)

    if not filtered_indices:
        st.warning("No se encontraron temáticas que coincidan con la búsqueda.")
        return

    st.caption(f"Mostrando {len(filtered_indices)} de {len(indices)} temáticas.")

    header_col1, header_col2 = st.columns([6, 1.5])
    header_col1.markdown("**Temática**")
    header_col2.markdown("**Eliminar**")

    for index_item in filtered_indices:
        index_name = extract_index_name(index_item)

        col1, col2 = st.columns([6, 1.5])

        col1.markdown(f"`{index_name}`")

        if col2.button("Eliminar", key=f"delete_{index_name}", use_container_width=True):
            try:
                delete_index(index_name)
                st.success(f"Temática `{index_name}` eliminada correctamente.")
                load_indices_into_state()
                st.rerun()
            except Exception as exc:
                st.error(
                    f"No se pudo eliminar la temática `{index_name}`. "
                    f"Comprueba que el backend tenga el endpoint DELETE /indices/{{index_name}}. "
                    f"Detalle: {exc}"
                )


def render_tematics_tab():
    # Renderiza la pestaña de temáticas, permitiendo al usuario consultar las temáticas (índices) almacenadas en el backend, ver el número de documentos asociados a cada una y eliminar las temáticas que ya no necesite.
    st.subheader("🗂️ Temáticas")

    st.write(
        "Consulta las temáticas almacenadas, el número de documentos asociados "
        "y elimina las que ya no necesites."
    )

    if st.button("Actualizar temáticas", type="primary"):
        try:
            load_indices_into_state()

            if st.session_state.last_indices:
                st.success("Temáticas actualizadas correctamente.")
            else:
                st.info("Todavía no hay temáticas cargadas.")
        except Exception as exc:
            st.error(f"Error consultando temáticas: {exc}")

    render_topics_table(st.session_state.last_indices)

    st.divider()

    if not st.session_state.confirm_delete_all_topics:
        if st.button("Eliminar todas las temáticas", type="secondary"):
            st.session_state.confirm_delete_all_topics = True
            st.rerun()
    else:
        st.warning("¿Estás seguro de borrar todas las temáticas? Esta acción eliminará todos los índices RAG.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Sí, borrar todas", type="primary", use_container_width=True):
                try:
                    delete_all_indices()
                    st.session_state.last_indices = []
                    st.session_state.confirm_delete_all_topics = False
                    st.success("Todas las temáticas han sido eliminadas correctamente.")
                    st.rerun()
                except Exception as exc:
                    st.error(
                        "No se pudieron eliminar todas las temáticas. "
                        "Comprueba que el backend tenga el endpoint DELETE /indices. "
                        f"Detalle: {exc}"
                    )

        with col2:
            if st.button("Cancelar", use_container_width=True):
                st.session_state.confirm_delete_all_topics = False
                st.rerun()


def render_documents_table(documents):
    # Renderiza la tabla de documentos ingestados con buscador.
    st.markdown("#### Documentos incorporados")

    search_term = st.text_input(
        "Buscar documento",
        placeholder="Ej: Biologia.pdf, quimica, astronomia, documento_id...",
        key="documents_search",
    ).strip().lower()

    if not documents:
        st.info("Todavía no hay documentos cargados o no se han consultado en esta sesión.")
        return

    filtered_documents = []

    for document_item in documents:
        document_id = extract_document_id(document_item)
        document_name = extract_document_name(document_item)

        if isinstance(document_item, dict):
            topic = document_item.get("tema", "")
            index_name = document_item.get("index_name", "")
            fecha_ingesta = document_item.get("fecha_ingesta", "")
        else:
            topic = ""
            index_name = ""
            fecha_ingesta = ""

        searchable_text = (
            f"{document_id} {document_name} {topic} {index_name} {fecha_ingesta}"
        ).lower()

        if not search_term or search_term in searchable_text:
            filtered_documents.append(document_item)

    if not filtered_documents:
        st.warning("No se encontraron documentos que coincidan con la búsqueda.")
        return

    st.caption(f"Mostrando {len(filtered_documents)} de {len(documents)} documentos.")

    header_col1, header_col2, header_col3, header_col4 = st.columns([4, 2, 2, 1.5])
    header_col1.markdown("**Documento**")
    header_col2.markdown("**Temática**")
    header_col3.markdown("**Chunks**")
    header_col4.markdown("**Eliminar**")

    for document_item in filtered_documents:
        document_id = extract_document_id(document_item)
        document_name = extract_document_name(document_item)

        if isinstance(document_item, dict):
            topic = document_item.get("tema", "-")
            chunks = document_item.get("chunks", "-")
        else:
            topic = "-"
            chunks = "-"

        col1, col2, col3, col4 = st.columns([4, 2, 2, 1.5])

        document_url = f"{get_backend_url()}/documents/{document_id}/file"

        col1.markdown(f"[{document_name}]({document_url})")
        col2.markdown(f"`{topic}`")
        col3.markdown(f"`{chunks}`")

        if col4.button("Eliminar", key=f"delete_document_{document_id}", use_container_width=True):
            try:
                delete_document(document_id)
                st.success(f"Documento `{document_name}` eliminado correctamente.")
                load_documents_into_state()
                st.rerun()
            except Exception as exc:
                st.error(
                    f"No se pudo eliminar el documento `{document_name}`. "
                    f"Comprueba que el backend tenga el endpoint DELETE /documents/{{document_id}}. "
                    f"Detalle: {exc}"
                )


def render_documents_tab():
    # Renderiza la pestaña de documentos ingestados, permitiendo al usuario consultar los documentos incorporados en la biblioteca y eliminar los que ya no necesite.
    st.subheader("📄 Documentos ingestados")

    st.write(
        "Consulta los documentos incorporados en la biblioteca y elimina los que ya no necesites."
    )

    if st.button("Actualizar documentos", type="primary"):
        try:
            load_documents_into_state()

            if st.session_state.last_documents:
                st.success("Documentos actualizados correctamente.")
            else:
                st.info("Todavía no hay documentos cargados.")
        except Exception as exc:
            st.error(
                "Error consultando documentos. "
                "Comprueba que el backend tenga el endpoint GET /documents. "
                f"Detalle: {exc}"
            )

    render_documents_table(st.session_state.last_documents)

    st.divider()

    if not st.session_state.confirm_delete_all_documents:
        if st.button("Eliminar todos los documentos", type="secondary"):
            st.session_state.confirm_delete_all_documents = True
            st.rerun()
    else:
        st.warning(
            "¿Estás seguro de borrar todos los documentos? "
            "Esta acción eliminará los documentos ingestados y sus chunks asociados."
        )

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Sí, borrar todos", type="primary", use_container_width=True):
                try:
                    delete_all_documents()
                    st.session_state.last_documents = []
                    st.session_state.confirm_delete_all_documents = False
                    st.success("Todos los documentos han sido eliminados correctamente.")
                    st.rerun()
                except Exception as exc:
                    st.error(
                        "No se pudieron eliminar todos los documentos. "
                        "Comprueba que el backend tenga el endpoint DELETE /documents. "
                        f"Detalle: {exc}"
                    )

        with col2:
            if st.button("Cancelar", use_container_width=True):
                st.session_state.confirm_delete_all_documents = False
                st.rerun()


# =============================================================================
# App principal
# =============================================================================
render_sidebar()

st.title("📚 RAG Assistant")
st.caption(
    "Chat documental con clasificación temática, búsqueda híbrida y respuestas "
    "basadas solo en fuentes ingestadas."
)

st.divider()

tab_chat, tab_ingestion, tab_tematics, tab_documents = st.tabs(
    ["💬 Chat RAG", "📤 Ingesta", "🗂️ Temáticas", "📄 Documentos ingestados"]
)

with tab_chat:
    render_chat_tab()

with tab_ingestion:
    render_ingestion_tab()

with tab_tematics:
    render_tematics_tab()

with tab_documents:
    render_documents_tab()
