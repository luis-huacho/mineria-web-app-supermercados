import Utils as u

query_texto = "aceite de cocina"
query_embedding = u.get_embedding(query_texto)
print(query_embedding)

productos_relacionados = u.query_products_chromadb(query_embedding=query_embedding, k=3)
print(productos_relacionados)

formatted_response = u.format_response_with_openai(productos_relacionados, u.gpt_client)
# Imprime el resultado formateado
print("\nRespuestas formateadas:")
for idx, response in enumerate(formatted_response, 1):
    print(f"\nProducto {idx}:")
    print(response)
    print("-" * 50)