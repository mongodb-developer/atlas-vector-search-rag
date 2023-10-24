# Atlas Vector Search with RAG

The following steps demonstrate how to use Atlas Vector Search with Retrieval-Augmented Generation (RAG) architecture to build Question Answering application for your data. We use the LangChain framework, OpenAI models, as well as Gradio in conjunction with Atlas Vector Search in a RAG architecture, to create this app.


## Setting up the Environment

1. Install the following packages:
```
pip3 install langchain pymongo bs4 openai tiktoken gradio requests lxml argparse unstructured
```
2. Create OpenAI API Key from [here](https://platform.openai.com/account/api-keys). Note that this requires a paid account with OpenAI, with enough credits. OpenAI API requests stop working if credit balance reaches `$0`.

3. Save the OpenAI API key and the MongoDB URI in the `key_param.py` file, like this:
```
openai_api_key = "ENTER_OPENAI_API_KEY_HERE"
MONGO_URI = "ENTER_MONGODB_URI_HERE"
```
4. Create two python scripts:
   - **load_data.py**: This script will be used to load your documents and ingest the text and vector embeddings, in a MongoDB collection.
   - **extract_information.py**: This script will generate the user interface and will allow you to perform question-answering against your data, using Atlas Vector Search and OpenAI.

**Note:** In this demo, I've used:
   - DB Name: `langchain_demo`
   - Collection Name: `collection_of_text_blobs`
   - The text files that I am using as my source data are saved in a directory named `sample_files`.

## Python script for loading the data (load_data.py)

```python
from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
import key_param

# Set the MongoDB URI, DB, Collection Names

client = MongoClient(key_param.MONGO_URI)
dbName = "langchain_demo"
collectionName = "collection_of_text_blobs"
collection = client[dbName][collectionName]

# Initialize the DirectoryLoader
loader = DirectoryLoader( './sample_files', glob="./*.txt", show_progress=True)
data = loader.load()

# Define the OpenAI Embedding Model we want to use for the source data
# The embedding model is different from the language generation model
embeddings = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)

# Initialize the VectorStore, and
# vectorise the text from the documents using the specified embedding model, and insert them into the specified MongoDB collection
vectorStore = MongoDBAtlasVectorSearch.from_documents( data, embeddings, collection=collection )
```
## Atlas Search Index

Create the following Atlas Search index named `default` on the collection, with `knnVector` datatype mapping for the `embedding` field, with this definition:

```json
{
  "mappings": {
    "dynamic": true,
    "fields": {
      "embedding": {
        "type": "knnVector",
        "dimensions": 1536,
        "similarity": "cosine"
      }
    }
  }
}
```

## Python script for the Question Answering App (extract_information.py)

```python
from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import gradio as gr
from gradio.themes.base import Base
import key_param

client = MongoClient(key_param.MONGO_URI)
dbName = "langchain_demo"
collectionName = "collection_of_text_blobs"
collection = client[dbName][collectionName]

# Define the text embedding model
 
embeddings = OpenAIEmbeddings(openai_api_key=key_param.openai_api_key)

# Initialize the Vector Store

vectorStore = MongoDBAtlasVectorSearch( collection, embeddings )

def query_data(query):
    # Convert question to vector using OpenAI embeddings
    # Perform Atlas Vector Search using Langchain's vectorStore
    # similarity_search returns MongoDB documents most similar to the query    

    docs = vectorStore.similarity_search(query, K=1)
    as_output = docs[0].page_content

    # Leveraging Atlas Vector Search paired with Langchain's QARetriever

    # Define the LLM that we want to use -- note that this is the Language Generation Model and NOT an Embedding Model
    # If it's not specified (for example like in the code below),
    # then the default OpenAI model used in LangChain is OpenAI GPT-3.5-turbo, as of August 30, 2023
    
    llm = OpenAI(openai_api_key=key_param.openai_api_key, temperature=0)


    # Get VectorStoreRetriever: Specifically, Retriever for MongoDB VectorStore.
    # Implements _get_relevant_documents which retrieves documents relevant to a query.
    retriever = vectorStore.as_retriever()

    # Load "stuff" documents chain. Stuff documents chain takes a list of documents,
    # inserts them all into a prompt and passes that prompt to an LLM.

    qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

    # Execute the chain

    retriever_output = qa.run(query)

# Return Atlas Vector Search output, and output generated using RAG Architecture
return as_output, retriever_output

# Create a web interface for the app, using Gradio

with gr.Blocks(theme=Base(), title="Question Answering App using Vector Search + RAG") as demo:
    gr.Markdown(
        """
        # Question Answering App using Atlas Vector Search + RAG Architecture
        """)
    textbox = gr.Textbox(label="Enter your Question:")
    with gr.Row():
        button = gr.Button("Submit", variant="primary")
    with gr.Column():
        output1 = gr.Textbox(lines=1, max_lines=10, label="Output with just Atlas Vector Search (returns text field as is):")
        output2 = gr.Textbox(lines=1, max_lines=10, label="Output generated by chaining Atlas Vector Search to Langchain's RetrieverQA + OpenAI LLM:")

# Call query_data function upon clicking the Submit button

    button.click(query_data, textbox, outputs=[output1, output2])

demo.launch()
```
