# This code creates a question-answering system that can extract information from uploaded PDF documents. <br>Let's break it down section by section:
```
uploaded = files.upload()
pdf_name = list(uploaded.keys())[0]

loader = PyPDFLoader(pdf_name)
documents = loader.load()

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separator="\n"
)
texts = text_splitter.split_documents(documents)
```
### 1. PDF Upload and Processing
- File Upload: Uses Google Colab's files.upload() to let users upload a PDF file

- PDF Loading: PyPDFLoader loads the PDF and extracts its content

- Text Splitting:

    * Divides the document into chunks of 1000 characters each

    * Maintains 200-character overlaps between chunks to preserve context

    * Splits at newline characters to keep logical text groupings
### 2. Embeddings and Vector Store Creation
```
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'}
)

db = FAISS.from_documents(texts, embedding_model)
retriever = db.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1}
)
```
* Embeddings Model: Uses the MiniLM-L6-v2 sentence transformer model to convert text into numerical vectors

* Vector Database:

    * FAISS (Facebook AI Similarity Search) efficiently stores and searches document embeddings

    * Creates a retriever that uses Maximal Marginal Relevance (MMR) to find the most relevant document chunk

    * Returns only the top 1 most relevant result (k=1)
### 3. Language Model Pipeline Setup
```
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.3,
    do_sample=False,
    device="cpu",
    pad_token_id=tokenizer.eos_token_id
)

llm = HuggingFacePipeline(pipeline=pipe)
```
* Model Selection: Uses GPT-2, a smaller version of OpenAI's language models

* Pipeline Configuration:

    * Limits responses to 100 new tokens

    * Sets temperature to 0.3 for slightly deterministic but not completely predictable outputs

    *Disables sampling for more focused answers

    * Runs on CPU (no GPU required)

    * Uses the end-of-sequence token for padding
### 4. QA Chain with Custom Prompt
```
prompt_template = """Use the following context to answer the question. 
If you don't know the answer, say you don't know. Keep the answer concise.

Context: {context}
Question: {question}
Answer:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)
```
* Prompt Template: Guides the LLM to:

   * Use only the provided context
   * Admit when it doesn't know the answer
   * Keep answers brief

* QA Chain:

   * Uses "stuff" chain type (simplest approach that stuffs all documents into the prompt)
   * Combines the retriever and LLM
   * Returns source documents for verification
### 5. Query Function
```
def ask_question(question):
    result = qa_chain({"query": question})
    print("\nQuestion:", question)
    print("Answer:", result["result"])
    print("\nSource Page:", result['source_documents'][0].metadata['page'] + 1)
    print("Relevant Text:", result['source_documents'][0].page_content[:200] + "...")
```
* Takes a question as input
* Runs the QA chain to get an answer
* Prints:

    * The original question
    * The generated answer
    * The PDF page number where the answer was found (with +1 adjustment since pages start at 0)
    * The first 200 characters of the relevant text for verification
# Architecture using mermaid:

```
graph LR
A[PDF] --> B[Text Chunks]
B --> C[Vector Store]
D[Question] --> C
C --> E[Relevant Context]
E --> F[LLM]
F --> G[Answer]
```
