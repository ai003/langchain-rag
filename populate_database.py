from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
import argparse
import os
import shutil

DATA_PATH = "data"
CHROMA_PATH = "chroma"  # vector store for embeddings

def main():
     # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database() #look into and test clearing function

    #create or update the data store
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)



# use langcahin pdf to load pdfs with page contents and meta data
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

# split data into chuncks recursively (text splitter), -- builds smaller json chunks


def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

# chroma db for adding or creating db with docs


def add_to_chroma(chunks: list[Document]):
    # load the existing database
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    chunks_with_ids = calculate_chunk_ids(chunks)

    # add or update the docs
    existing_items = db.get(include=[])  # id's included by default
    existing_ids = set(existing_items["ids"])  # turn all id's into set
    print(f"Number of existing documents in DB: ")  # print count

    # only add new documents to db that don't exists
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:  # check if new ones being added
            new_chunks.append(chunk)  # append new here

    if len(new_chunks):  # only if there are new docs to add
        print(f"Adding new documents: {len(new_chunks)}")
        # get new chunk id's then add them to db
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist() deprecieted method
    else:
        print("No new documents to add")


def calculate_chunk_ids(chunks):
    # vars for page indexes
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        # if page id is the same as the last one, increment the index
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        # add chunk index to meta-data
        chunk.metadata["id"] = chunk_id
    return chunks

#clearing function
def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


# test chunk of
# documents = load_documents()
# chunks = split_documents(documents)
# print(chunks[0])

if __name__ == "__main__":
    main()