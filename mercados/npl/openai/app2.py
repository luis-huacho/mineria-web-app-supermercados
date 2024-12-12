import streamlit as st
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import Utils as u

# Configuración de la página
st.set_page_config(page_title="Buscador de Productos", layout="wide")
st.title("Buscador de Productos Relacionados")

# Configurar el modelo de OpenAI
model = st.selectbox("Selecciona el modelo de OpenAI:", ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"], index=0)

# Selector para el valor de k
k_value = st.slider("Selecciona el número de productos relacionados (k):", min_value=1, max_value=5, value=3)

# Botón para reiniciar la consulta
if st.button("Reiniciar"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.session_state.clear()

# Mostrar historial de conversación
if "historial" not in st.session_state:
    st.session_state["historial"] = []

# Entrada de texto para la consulta del usuario
query_texto = st.text_input("Ingresa tu consulta (por ejemplo, 'aceite de cocina'):")

if query_texto:
    with st.spinner("Buscando productos relacionados y generando respuesta..."):
        # Generar el embedding para el texto ingresado
        query_embedding = u.get_embedding(query_texto)

        # Consultar productos relacionados usando ChromaDB
        productos_relacionados = u.query_products_chromadb(query_embedding=query_embedding, k=k_value)

        # Formatear la respuesta con OpenAI
        formatted_response = u.format_response_with_openai(productos_relacionados, u.gpt_client, model)

        # Guardar la consulta y respuesta en el historial
        st.session_state["historial"].append({"query": query_texto, "related_products": productos_relacionados, "response": formatted_response})

        # Dividir la pantalla en dos columnas
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Productos relacionados (sin formatear):")
            st.write(productos_relacionados)

        with col2:
            st.subheader("Respuesta generada:")
            for idx, response in enumerate(formatted_response, 1):
                st.markdown(f"### Producto {idx}")
                st.write(response)
                st.markdown("---")

# Mostrar historial de conversación
if st.session_state["historial"]:
    st.subheader("Historial de conversación:")
    for idx, item in enumerate(st.session_state["historial"], 1):
        st.markdown(f"### Interacción {idx}")
        st.markdown(f"**Consulta:** {item['query']}")
        st.markdown(f"**Productos relacionados:** {item['related_products']}")
        st.markdown("**Respuestas formateadas:**")
        for response_idx, response in enumerate(item['response'], 1):
            st.markdown(f"- Producto {response_idx}: {response}")
        st.markdown("---")
else:
    st.info("Por favor, inicia una consulta para comenzar la conversación.")
