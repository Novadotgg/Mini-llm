```mermaid
graph LR
A[PDF] --> B[Text Chunks]
B --> C[Vector Store]
D[Question] --> C
C --> E[Relevant Context]
E --> F[LLM]
F --> G[Answer]
```
