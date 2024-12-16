from langchain_chroma import Chroma

def create_vector_db(chunks, embedding_model, persist_directory="db/"):
    """Create and persist a vector database from text chunks."""
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    return vectorstore

def load_vector_db(persist_directory="db/"):
    """Load an existing vector database."""
    return Chroma(persist_directory=persist_directory)
