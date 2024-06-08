# from langchain_openai import OpenAIEmbeddings
# from dotenv import load_dotenv
# import getpass
# import os

# load_dotenv()

# api_key = os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = api_key
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings


# func that returns an embedding function to be used in 2 different places
# create database, and when we want to query
def get_embedding_function():
    """
    This function creates a single instance of BedrockEmbeddings
    and returns it as an embedding function. This allows for code
    reuse and avoids creating unnecessary connections.
    """
    embeddings = SpacyEmbeddings(model_name="en_core_web_sm")
    # OpenAIEmbeddings(model="text-embedding-3-large")

    # BedrockEmbeddings(
    #     credentials_profile_name="bedrock-admin", region_name="us-east-1"
    # )
    return embeddings
