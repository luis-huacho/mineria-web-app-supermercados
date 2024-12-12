import streamlit as st
import Utils as u

# Configuración de la página
st.set_page_config(page_title="Buscador semántico de productos utilizando una arquitectura RAG", layout="wide")
st.title("Buscador semántico de productos utilizando una arquitectura RAG")

# Agregar un selector de modelos
models = st.multiselect("Selecciona dos modelos de OpenAI:", ["gpt-3.5-turbo", "gpt-4", "gpt-4o", "gpt-4o-mini"], default=["gpt-3.5-turbo", "gpt-4"])

# Selector para el valor de k
k_value = st.slider("Selecciona el número de productos relacionados (k):", min_value=1, max_value=5, value=3)

if len(models) != 2:
    st.warning("Por favor, selecciona exactamente dos modelos.")
else:
    # Botón para reiniciar la consulta
    if st.button("Reiniciar"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.session_state.clear()

    # Entrada de texto para la consulta del usuario
    query_texto = st.text_input("Ingresa tu consulta (por ejemplo, 'aceite de cocina'):")

    if query_texto:
        with st.spinner("Buscando productos relacionados..."):
            # Generar el embedding para el texto ingresado
            query_embedding = u.get_embedding(query_texto)

            # Consultar productos relacionados usando ChromaDB
            productos_relacionados = u.query_products_chromadb(query_embedding=query_embedding, k=k_value)

            # Formatear la respuesta con OpenAI para ambos modelos
            formatted_response_model1 = u.format_response_with_openai(productos_relacionados, u.gpt_client, models[0])
            formatted_response_model2 = u.format_response_with_openai(productos_relacionados, u.gpt_client, models[1])

            # Dividir la pantalla en tres columnas
            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Productos relacionados (sin formatear):")
                st.write(productos_relacionados)

            with col2:
                st.subheader(f"Respuestas formateadas ({models[0]}):")
                for idx, response in enumerate(formatted_response_model1, 1):
                    st.markdown(f"### Producto {idx}")
                    st.write(response)
                    st.markdown("---")

            with col3:
                st.subheader(f"Respuestas formateadas ({models[1]}):")
                for idx, response in enumerate(formatted_response_model2, 1):
                    st.markdown(f"### Producto {idx}")
                    st.write(response)
                    st.markdown("---")
    else:
        st.info("Por favor, ingresa una consulta para empezar.")
