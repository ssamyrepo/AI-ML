
### **1. Introduction to Generative AI**
- **Generative AI** is a trending technology used across industries (data science, analytics, engineering, etc.).
- It helps solve complex business problems and improve existing infrastructure.
- **Prompt Engineering** is crucial—**prompt** (input) and **completion** (output) are fundamental concepts.

---

### **2. Prompt Engineering & Structure**
- **Prompt Structure** can include:
  - **Instructions** (e.g., "Answer in one sentence").
  - **Conditional Prompts** (e.g., "Reply only if 'Jal' is in the input").
  - **In-Context Learning (ICL)**: Providing examples to guide the model’s response.
    - **Zero-shot**: No examples.
    - **One-shot**: One example.
    - **Few-shot**: Multiple examples.

---

### **3. Tokens & Tokenization**
- **Tokens** are the smallest units of text processed by AI models.
- Tokenization splits text into chunks (not always word-by-word; subword tokens exist).
- Example: OpenAI’s tokenizer shows how words are split (e.g., "tokenization" → `["token", "ization"]`).

---

### **4. How Generative AI Models Work**
- **Process Flow**:
  1. **Tokenization**: Convert input text into tokens (numeric IDs).
  2. **Context Window**: Model considers recent tokens (short-term memory).
  3. **Foundation Model**: Predicts next tokens based on probabilities.
  4. **Max Tokens**: Limits response length.
  5. **Detokenization**: Converts token IDs back to text.
- **Context Window**:
  - Sliding window of past tokens (e.g., 8,000 tokens for AWS Titan).
  - Exceeding the window truncates older tokens.

---

### **5. Sampling Methods for Token Prediction**
- **Greedy Sampling**: Picks the token with the highest probability (less creative, repetitive).
- **Random Sampling**: Introduces randomness for creativity.
  - **Top-K**: Randomly selects from top `K` probable tokens.
  - **Top-P (Nucleus Sampling)**: Randomly selects from tokens whose cumulative probability ≤ `P`.

---

### **6. Temperature Parameter**
- Controls randomness in outputs:
  - **Low Temperature**: Focused, deterministic responses (high-probability tokens).
  - **High Temperature**: Diverse, creative responses (more uniform probabilities).
- **Best Practice**: Low temperature + high top-P for balanced responses.

---

### **7. AWS Bedrock Overview**
- AWS service for building generative AI applications using foundation models (e.g., Amazon Titan, AI21 Labs, Meta).
- **Steps to Use**:
  1. **Request Model Access** in AWS Bedrock console.
  2. **Choose a Model** (e.g., Titan Text G1 for text generation).
  3. **Invoke Model** using `boto3` (AWS SDK for Python).

---

### **8. Embeddings & Vector Representations**
- **Embeddings** are numerical vectors representing text/images/audio.
- **Cosine Similarity** measures similarity between vectors (used in semantic search).
- **AWS Titan Embeddings**:
  - Converts text to 1,536-dimensional vectors.
  - Used for similarity search (e.g., finding related documents).

---

### **9. Semantic Search with Vector Databases**
- **Traditional Search**: Keyword matching (e.g., SQL `LIKE`).
- **Semantic Search**: Understands context (e.g., "tallest mountain" → "highest peak").
- **Vector Databases** (e.g., Pinecone):
  - Store embeddings for fast similarity searches.
  - Query with a vector to find closest matches.

---

### **10. Multimodal Embeddings**
- Handles **text + images** in the same vector space.
- Example: Search for "red bag" → returns red bag images even if query is text.
- **AWS Titan Multimodal Embeddings**:
  - Supports images/text (output: 1,024-dimensional vectors).

---

### **11. RAG (Retrieval-Augmented Generation)**
- Solves the problem of LLMs lacking knowledge of private/real-time data.
- **Steps**:
  1. **Chunk & Embed**: Split documents into chunks → convert to vectors.
  2. **Store in Vector DB** (e.g., Pinecone).
  3. **Retrieve Relevant Chunks** for a query using cosine similarity.
  4. **Augment Prompt**: Pass chunks + query to LLM for context-aware answers.
- **Example**: Ask "How many years did I work at XYZ?" → RAG fetches relevant text chunks → LLM answers accurately.

---

### **12. Practical Implementation**
- **Code Workflow**:
  1. **Load & Chunk Text** (e.g., using `LangChain`).
  2. **Generate Embeddings** (AWS Titan).
  3. **Store in Pinecone**.
  4. **Query**: Convert query to vector → fetch top matches → pass to LLM.
- **Tools Used**:
  - AWS Bedrock (`boto3`).
  - Pinecone (vector DB).
  - LangChain (chunking/prompt templates).

---

### **13. Key AWS Bedrock Models**
- **Text Generation**: `amazon.titan-text-express-v1` (8K token context).
- **Embeddings**: `amazon.titan-embed-text-v1` (1,536-D vectors).
- **Multimodal**: `amazon.titan-embed-image-v1` (text + images).

---

### **14. Best Practices**
- **Low Temperature + High Top-P**: For precise yet creative responses.
- **Chunk Overlap**: Ensures context continuity in RAG.
- **Monitor Context Window**: Avoid truncation of key info.

---

### **15. Applications**
- Chatbots, document Q&A, image search, personalized recommendations.

---




