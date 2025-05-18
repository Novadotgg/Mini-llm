### This code creates a question-answering system that can extract information from uploaded PDF documents. <br>Let's break it down section by section:
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
File Upload: Uses Google Colab's files.upload() to let users upload a PDF file

PDF Loading: PyPDFLoader loads the PDF and extracts its content

Text Splitting:
