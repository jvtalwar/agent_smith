from openai import OpenAI
from .settings import settings
import numpy as np
#from sentence_transformers import SentenceTransformer

client = OpenAI(api_key = settings.openai_api_key)

def _get_openai_embedding(text: str):
    '''generate embeddings for text'''
    resp = client.embeddings.create(input=[text], model=settings.embedding_model)
    return resp.data[0].embedding


def _calc_dot_product(query: list[float], key: list[float]) -> float:
    """
    Compute the dot product between two embedding vectors stored as Python lists.

    Parameters
    ----------
    query : list[float]
        First embedding vector.
    key : list[float]
        Second embedding vector.

    Returns
    -------
    float
        The dot product (query ⋅ key).

    Raises
    ------
    ValueError
        If the vectors have different lengths.
    """
    # Convert lists to NumPy arrays (float32 saves memory and matches embedding precision)
    q = np.array(query, dtype=np.float32)
    k = np.array(key, dtype=np.float32)

    # Check length equality
    if q.shape != k.shape:
        raise ValueError(f"Shape mismatch: {q.shape} vs {k.shape}")

    # Compute dot product efficiently
    return float(np.dot(q, k))






'''
def _get_sentence_transformers_embedding(text):
    if SentenceTransformer is None:
        raise ImportError("sentence_transformers not installed")
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        return model.encode(text).tolist()
    except Exception as e:
        logging.warning(f"SentenceTransformers embedding failed: {e}")
        return None
'''