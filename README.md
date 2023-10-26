# Atlas Vector Search with RAG

The Python scripts in this repo use Atlas Vector Search with Retrieval-Augmented Generation (RAG) architecture to build a Question Answering application. They use the LangChain framework, OpenAI models, as well as Gradio in conjunction with Atlas Vector Search in a RAG architecture, to create this app.


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
4. Use the following two python scripts:
   - **load_data.py**: This script will be used to load your documents and ingest the text and vector embeddings, in a MongoDB collection.
   - **extract_information.py**: This script will generate the user interface and will allow you to perform question-answering against your data, using Atlas Vector Search and OpenAI.

**Note:** In this demo, I've used:
   - DB Name: `langchain_demo`
   - Collection Name: `collection_of_text_blobs`
   - The text files that I am using as my source data are saved in a directory named `sample_files`.

## Main Components

| LangChain                                                                                                                  | OpenAI                                                                                                                           | Atlas Vector Search                                                                                                  | Gradio                                                     |
|----------------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------|
| [**DirectoryLoader**](https://api.python.langchain.com/en/latest/document_loaders/langchain.document_loaders.unstructured.UnstructuredFileLoader.html): <br> - All documents from a directory <br> - Split and load <br> - Uses the [Unstructured](https://python.langchain.com/docs/integrations/document_loaders/unstructured_file.html) package | **Embedding Model**: <br> - [text-embedding-ada-002](https://openai.com/blog/new-and-improved-embedding-model) <br> - Text → Vector embeddings <br> - 1536 dimensions           | [**Vector Store**](https://www.mongodb.com/docs/atlas/atlas-vector-search/vector-search-stage/)                             | [UI](https://www.gradio.app/) for LLM app <br> - Open-source Python library <br> - Allows to quickly create user interfaces for ML models |
| [**RetrievalQA**](https://api.python.langchain.com/en/latest/chains/langchain.chains.retrieval_qa.base.BaseRetrievalQA.html?highlight=retrievalqa#langchain.chains.retrieval_qa.base.BaseRetrievalQA): <br> - Retriever <br> - Question-answering chain                       | **Language model**: <br> - [gpt-3.5-turbo](https://platform.openai.com/docs/models/gpt-3-5) <br> - Understands and generates natural language <br> - Generates text, answers, translations, etc.                                       |                                                                                                                           |                                                            |
| [**MongoDBAtlasVectorSearch**](https://api.python.langchain.com/en/latest/vectorstores/langchain.vectorstores.mongodb_atlas.MongoDBAtlasVectorSearch.html): <br> - Wrapper around Atlas Vector Search <br> - Easily create and store embeddings in MongoDB collections <br> - Perform KNN Search using Atlas Vector Search          |                                                                                                                                                                                      |                                                                                                                           |                                                            |
