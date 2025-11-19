import requests

class LlamaClient:
    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url

    def get_models(self):
        response = requests.get(f"{self.base_url}/models")
        return response.json()

    def get_files_in_vector_db(self):
        response = requests.get(f"{self.base_url}/files-in-vector-db")
        return response.json()

    def ingest_documents(self, documents: list[dict]):
        response = requests.post(f"{self.base_url}/ingest", json=documents)
        return response.json()

    def chat(self, messages: list[dict]):
        response = requests.post(f"{self.base_url}/chat", json=messages)
        return response.json()

    def query_vector_db(self, query: str):
        response = requests.post(f"{self.base_url}/query-vector-db", json=query)
        return response.json()