import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import time
import numpy as np
import openai

# Cargar variables de entorno desde el archivo .env
# Esto permite mantener las claves API y configuraciones sensibles fuera del código.
load_dotenv()

# Obtener el valor de OPENAI_API_KEY desde el archivo .env
api_key = os.getenv("OPENAI_API_KEY")

# Inicializar el cliente OpenAI con la clave API obtenida.
gpt_client = OpenAI(
    api_key=api_key,
)

from chromadb import PersistentClient
# Inicializar un cliente persistente para interactuar con ChromaDB.
client = PersistentClient(path="./chroma_db")
# Obtener o crear una colección llamada "productos" para almacenar y consultar datos.
collection = client.get_or_create_collection(name="productos")

# Función para obtener embeddings de texto usando el modelo "text-embedding-ada-002".
def get_embedding(text, model="text-embedding-ada-002"):
    # Normalizar el texto eliminando saltos de línea.
    text = text.replace("\n", " ")
    # Generar el embedding mediante OpenAI API.
    response = gpt_client.embeddings.create(input=[text], model=model).data[0].embedding
    return response

# Generar una respuesta basada en una consulta y productos relacionados.
def generate_response(query, productos_relacionados):
    """
    Genera una respuesta estructurada basada en una consulta del usuario y una lista de productos relacionados.
    """
    response = f"Query: {query}\n\nRelated Products:\n"
    # Iterar sobre los productos relacionados para formatear la respuesta.
    for producto in productos_relacionados:
        # Asegúrate de que las claves coincidan con las que se usaron en productos_relacionados.
        response += (
            f"Producto: {producto.get('producto', 'N/A')}\n"
            f"Descripcion: {producto.get('producto', 'N/A')}\n"
            f"Precio: {producto.get('precio', 'N/A')}\n"
            f"Enlace: {producto.get('url', 'N/A')}\n"
            f"Market: {producto.get('market', 'N/A')}\n"
            f"Categorias: {producto.get('categorias', 'N/A')}\n\n"
        )
    return response

# Consultar ChromaDB para encontrar los productos más similares según un embedding de consulta.
def query_products_chromadb(query_embedding, k=3):
    """
    Busca productos en ChromaDB que sean más similares al embedding de consulta.
    """
    # Realizar la consulta en ChromaDB, especificando el número de resultados y metadatos deseados.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["metadatas", "distances"]
    )

    # Formatear los resultados para retornar una lista de productos.
    productos = [
        {
            "producto": meta["producto"],
            "descripcion": meta["descripcion"],
            "precio": meta["precio"],
            "market": meta["market"],
            "url": meta["url"],
            "categorias": meta["categorias"]
        }
        for meta in results["metadatas"][0]  # Acceder al primer resultado de metadatos.
    ]

    # Retornar productos y, opcionalmente, las distancias de similitud si están disponibles.
    return {
        "productos": productos,
        "distances": results["distances"][0] if "distances" in results else None
    }

# Formatear la respuesta generada por OpenAI para los productos obtenidos.
def format_response_with_openai(api_response, client=gpt_client, model="gpt-3.5-turbo"):
    """
    Formatea los productos devueltos por ChromaDB en respuestas naturales usando OpenAI GPT.
    """
    # Extraer los productos de la respuesta de ChromaDB.
    productos = api_response['productos']
    formatted_documents = []

    # Iterar sobre los productos para generar respuestas formateadas.
    for producto in productos:
        # Crear un texto estructurado con la información del producto.
        producto_text = f"""
        Producto: {producto['producto']}
        Descripción: {producto['descripcion']}
        Precio: S/ {producto['precio']}
        Market: {producto['market']}
        Categorías: {producto['categorias']}
        URL: {producto['url']}
        """

        # Prompt para transformar la información del producto en texto amigable y profesional.
        prompt = f"""
        Se detalla uno o mas productos en formato JSON, el cual tiene las etiquetas: producto, descripción, precio, market, categorías y url.
        Con el valor del "producto" y la "descripción" desarrolla la presentación del producto. Esta presentación debe iniciar con
        el nombre del producto, luego con sus características principales y finalmente con el precio. Asimismo, menciona al "market" 
        al cual pertenece.  
        La información debe ser jovial pero objetiva. No debe contener frases que promuevan la compra solo brindar información.
        Siempre al final de cada presentación se debe incluir la URL del producto, la URL debe mostrarse con el texto: "Ir al producto" , 
        enlazando con el valor de la URL, en una línea debajo del texto. 

        {producto_text}
        """

        try:
            # Generar la respuesta usando OpenAI GPT-3.5-Turbo.
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Asumir que eres un experto en productos de marketplaces, presentas el producto al usuario usando lenguaje general."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=400
            )
            # Extraer el contenido generado.
            generated_text = response.choices[0].message.content.strip()
            formatted_documents.append(generated_text)
        except Exception as e:
            # Manejar errores en el procesamiento de productos.
            formatted_documents.append(f"Error procesando el documento: {str(e)}")

    return formatted_documents
