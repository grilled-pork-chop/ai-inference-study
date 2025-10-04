# Model Domains

## Overview

AI models are designed for specific **domains** — specialized areas where they excel at solving particular types of problems. Understanding these domains helps you choose the right model for your use case.

!!! tip "Think of domains as specializations"
    Just like doctors specialize (cardiologist, neurologist), AI models specialize in different types of data and tasks.

**Common domains**:

| Domain                          | Input Type                     | Output Type                        | Example Use Cases                                     |
| ------------------------------- | ------------------------------ | ---------------------------------- | ----------------------------------------------------- |
| **Computer Vision**             | Images, videos                 | Classifications, objects, masks    | Face recognition, defect detection                    |
| **Natural Language Processing** | Text                           | Text, classifications              | Chatbots, sentiment analysis                          |
| **Speech & Audio**              | Audio waveforms                | Text, classifications              | Voice assistants, music genre detection               |
| **Embeddings**                  | Text, images, audio            | Dense vectors (numerical)          | Semantic search, similarity matching, recommendations |
| **Large Language Models**       | Text (conversations)           | Generated text                     | Chat assistants, code generation, reasoning           |
| **Multimodal**                  | Text + images + audio          | Text, images, answers              | Visual Q&A, image captioning, video understanding     |
| **Time Series**                 | Sequential numbers             | Predictions, anomalies             | Stock prediction, sensor monitoring                   |
| **Tabular**                     | Structured data (rows/columns) | Classifications, values            | Fraud detection, churn prediction                     |
| **Generative**                  | Text prompts, images           | New content (images, audio, video) | Art generation, synthetic data, content creation      |

---

## Computer Vision

**What it is**: Models that understand and process visual information from images and videos.

**Input formats**:

- **Images**: JPG, PNG, TIFF (224×224, 512×512, 1024×1024 pixels typical)
- **Videos**: MP4, AVI (processed as frame sequences)
- **Format**: RGB (3 channels) or grayscale (1 channel)
- **Preprocessing**: Resize, normalize (0-1 or -1 to 1 range)

### Common Tasks

=== "Image Classification"

    **What**: Assign a single label to an entire image.  

    **Input**: Image (224×224×3)  
    **Output**: Class label + confidence score  

    !!! example "Cat photo example"
        ```
        Input: [cat photo]
        Output: {"class": "cat", "confidence": 0.95}
        ```

    **Common models**:

    - **ResNet** (50, 101, 152 layers) — 25–100MB  
    - **EfficientNet** — 20–80MB  
    - **MobileNet** — 5–20MB  
    - **ViT** — 300MB–2GB  

    !!! info "Use cases"
        - Product categorization  
        - Quality control  
        - Medical image diagnosis  
        - Content moderation  

=== "Object Detection"
    **What**: Find and locate multiple objects in an image.  

    **Input**: Image (any size)  
    **Output**: Bounding boxes + class labels + confidence scores  

    !!! example "Street scene example"
        ```
        Input: [street scene photo]
        Output: [
          {"class": "car", "bbox": [100,150,300,400], "confidence":0.92},
          {"class": "person", "bbox": [450,200,550,500], "confidence":0.88},
          {"class": "traffic_light", "bbox": [200,50,250,150], "confidence":0.91}
        ]
        ```

    **Common models**:  
    - **YOLO (v5, v8, v10)** — 10–300MB  
    - **Faster R-CNN** — 100–500MB  
    - **RetinaNet** — 100–200MB  
    - **DETR** — 150–400MB  

    !!! info "Use cases"
        - Autonomous vehicles  
        - Surveillance  
        - Retail  
        - Agriculture  

=== "Semantic Segmentation"
    **What**: Classify every pixel in an image.  

    **Input**: Image (512×512 typical)  
    **Output**: Mask (same size as input)  

    !!! example "Street scene mask"
        ```
        Input: [street scene image]
        Output: 
          Pixels 1–1000: "road"
          Pixels 1001–2000: "sidewalk"
          Pixels 2001–3000: "building"
        ```

    **Common models**:  
    - **U-Net** — 50–200MB  
    - **DeepLab** — 100–300MB  
    - **Mask R-CNN** — 200–500MB  
    - **SegFormer** — 50–500MB  

    !!! info "Use cases"
        - Medical imaging  
        - Autonomous driving  
        - Satellite imagery  
        - Photo editing  

=== "Face Recognition & Analysis"

    **What**: Identify or analyze faces.  

    **Tasks**: Detection, Recognition, Analysis  
        
    !!! example "Face example"
        ```
        Input: [group photo with 3 people]
        Output: [
          {"id": "person_1", "embedding": [0.12, 0.05, ..., 0.98]},
          {"id": "person_2", "embedding": [0.42, 0.33, ..., 0.11]},
          {"id": "person_3", "embedding": [0.91, 0.04, ..., 0.55]}
        ]
        ``` 

    **Common models**:  
    - **FaceNet** — 100MB  
    - **ArcFace** — 150–300MB  
    - **MTCNN** — 5–20MB  
    - **RetinaFace** — 30MB  

    !!! info "Use cases"
        - Security  
        - Photo organization  
        - Attendance systems  
        - Customer analytics  

=== "Image Generation"

    **What**: Create new images from text or other images.  

    **Input**: Text prompt or image  
    **Output**: Generated image  

    **Common models**:  
    - **Stable Diffusion** — 2–8GB  
    - **DALL-E** — proprietary  
    - **Midjourney** — API only  
    - **GANs** — 50MB–2GB  

    !!! example "Prompt-to-image"
        ```
        Input: "A cat wearing sunglasses on the beach"
        Output: [Generated image]
        ```

    !!! info "Use cases"
        - Creative content  
        - Product visualization  
        - Data augmentation  
        - Image editing  

### Computer Vision Model Sizes

| Model Type           | Small    | Medium    | Large     |
| -------------------- | -------- | --------- | --------- |
| **Classification**   | 5-20MB   | 50-150MB  | 200MB-2GB |
| **Object Detection** | 10-50MB  | 100-300MB | 500MB-2GB |
| **Segmentation**     | 50-100MB | 200-500MB | 1-4GB     |
| **Generation**       | N/A      | 2-4GB     | 6-12GB    |

## Natural Language Processing (NLP)

**What it is**: Models that understand, process, and generate human language.

**Input formats**:
- **Text**: Plain text, sentences, documents
- **Encoding**: UTF-8 strings
- **Preprocessing**: Tokenization (split into words/subwords)
- **Maximum length**: Usually 512-4096 tokens

### Common Tasks

=== "Text Classification"

    **What**: Categorize text into predefined classes.  

    **Input**: Text (sentence or document)  
    **Output**: Class label + confidence score  

    !!! example "Sentiment example"
        ```
        Input: "This product is amazing! Best purchase ever."
        Output: {"sentiment": "positive", "confidence": 0.94}
        ```

    **Common models**:  
    - **BERT (base/large)** — 110–440MB  
    - **DistilBERT** — 66MB  
    - **RoBERTa** — 125–500MB  
    - **XLNet** — 500MB–2GB  

    !!! info "Use cases"
        - Sentiment analysis  
        - Spam detection  
        - Content categorization  
        - Intent classification (chatbots)  

=== "Named Entity Recognition (NER)"

    **What**: Extract and classify named entities (people, places, organizations).  

    **Input**: Text  
    **Output**: Entities with types and positions

    !!! example "NER example"
        ```
        Input: "Apple Inc. was founded by Steve Jobs in Cupertino."
        Output: [
          {"text": "Apple Inc.", "type": "ORGANIZATION", "start":0, "end":10},
          {"text": "Steve Jobs", "type": "PERSON", "start":27, "end":37},
          {"text": "Cupertino", "type": "LOCATION", "start":41, "end":50}
        ]
        ```

    **Common models**:  
    - **spaCy NER** — 15–50MB  
    - **BERT-NER** — 110–440MB  
    - **Flair** — 50–200MB  

    !!! info "Use cases"
        - Information extraction  
        - Document processing  
        - Customer data parsing  
        - Compliance (redaction)  

=== "Question Answering"

    **What**: Answer questions based on a given context.  

    **Input**: Question + context passage  
    **Output**: Answer span + confidence  

    !!! example "QA example"
        ```
        Context: "The Eiffel Tower is located in Paris, France. It was built in 1889."
        Question: "When was the Eiffel Tower built?"
        Output: {"answer": "1889", "confidence": 0.96, "start":65, "end":69}
        ```

    **Common models**:  
    - **BERT-QA** — 110–440MB  
    - **RoBERTa-QA** — 125–500MB  
    - **ELECTRA** — 50–300MB  
    - **T5** — 250MB–3GB  

    !!! info "Use cases"
        - Customer support  
        - Document search  
        - Knowledge bases  
        - Educational tools  

=== "Machine Translation"

    **What**: Translate text between languages.  

    **Input**: Text in source language  
    **Output**: Text in target language  

    !!! example "Translation example"
        ```
        Input: "Hello, how are you?" (English)
        Output: "Bonjour, comment allez-vous ?" (French)
        ```

    **Common models**:  
    - **MarianMT** — 300MB per language pair  
    - **mBART** — 600MB–2GB  
    - **T5** — 250MB–3GB  
    - **NLLB** — 600MB–10GB  

    !!! info "Use cases"
        - Content localization  
        - Real-time chat translation  
        - Document translation  
        - Subtitles  

=== "Text Summarization"

    **What**: Create concise summaries of longer texts.  

    **Input**: Long document (hundreds to thousands of words)  
    **Output**: Short summary (50–200 words typical)  

    !!! example "Summarization example"
        ```
        Input: [2000-word article about climate change]
        Output: "Climate change poses significant risks to global ecosystems. Rising temperatures cause ice melts and extreme weather events. International cooperation is needed to reduce emissions."
        ```

    **Common models**:  
    - **BART** — 400MB–1.6GB  
    - **T5** — 250MB–3GB  
    - **Pegasus** — 500MB–2GB  
    - **DistilBART** — 300MB  

    !!! info "Use cases"
        - News aggregation  
        - Document processing  
        - Email summarization  
        - Meeting notes  

=== "Text Generation (Non-LLM)"

    **What**: Generate text based on prompts (smaller models).  

    **Input**: Text prompt or context  
    **Output**: Generated continuation  

    !!! example "Text generation example"
        ```
        Input: "Once upon a time in a magical forest,"
        Output: "a group of animals discovered a hidden treasure that changed their lives forever."
        ```

    **Common models**:  
    - **GPT-2** — 500MB–6GB  
    - **DistilGPT-2** — 350MB  
    - **T5** — 250MB–3GB  

    !!! info "Use cases"
        - Autocomplete  
        - Creative writing assistance  
        - Data augmentation  
        - Simple chatbots  

### NLP Model Sizes

| Model Type         | Small     | Medium    | Large     |
| ------------------ | --------- | --------- | --------- |
| **Classification** | 15–100MB  | 110–500MB | 1–4GB     |
| **NER/Tagging**    | 15–100MB  | 110–440MB | 500MB–2GB |
| **Generation**     | 300MB–1GB | 1–4GB     | 6–12GB    |
| **Translation**    | 300–600MB | 1–3GB     | 5–10GB    |

## Large Language Models (LLMs)

**What it is**: Very large models (billions of parameters) trained on massive text corpora to understand and generate human-like text.  

**Key differences from NLP models**:
- **Scale**: 1B–100B+ parameters vs. 100M–1B  
- **Generalist**: Can handle many tasks without fine-tuning  
- **Few-shot learning**: Learn from examples in the prompt  
- **Conversational**: Designed for dialogue  

### Common Tasks

=== "Chat & Conversation"

    **What**: Interactive dialogue with context memory.  

    **Input**: Conversation history + new message  
    **Output**: Contextual response  

    !!! example "Conversation example"
        ```
        User: "What's the capital of France?"
        Assistant: "The capital of France is Paris."
        User: "What's its population?"
        Assistant: "Paris has approximately 2.1 million people in the city proper, and about 12 million in the metropolitan area."
        ```

    **Common models**:  
    - **LLaMA-2 (7B, 13B, 70B)** — Open source, 4–140GB  
    - **Mistral (7B, 8×7B)** — High quality, 4–90GB  
    - **GPT-3.5/4** — OpenAI, API only  
    - **Claude (Sonnet, Opus)** — Anthropic, API only  
    - **Gemini** — Google, API only  

    !!! info "Use cases"
        - Customer support chatbots  
        - Virtual assistants  
        - Tutoring systems  
        - Interactive help systems  

=== "Code Generation"

    **What**: Generate, explain, or debug code.  

    **Input**: Natural language description or code  
    **Output**: Code in various languages  

    !!! example "Code generation example"
        ```
        Input: "Write a Python function to calculate factorial recursively"
        Output:
        def factorial(n):
            if n == 0 or n == 1:
                return 1
            return n * factorial(n - 1)
        ```

    **Common models**:  
    - **CodeLLaMA (7B, 13B, 34B)** — 4–68GB  
    - **StarCoder (15B)** — 30GB  
    - **GPT-4** — API only  
    - **Codex** — GitHub Copilot backend  

    !!! info "Use cases"
        - Code completion (IDEs)  
        - Code review assistance  
        - Documentation generation  
        - Bug fixing suggestions  

=== "Instruction Following"

    **What**: Execute complex instructions and tasks.  

    **Input**: Detailed instructions  
    **Output**: Structured response following instructions  

    !!! example "Instruction example"
        ```
        Input: "Analyze this customer review and extract: sentiment, main complaint, product mentioned, and suggest a response. 
        Review: 'The laptop is great but the delivery was delayed by 3 weeks.'"
        Output:
        {
          "sentiment": "mixed",
          "main_complaint": "delivery delay",
          "product": "laptop",
          "suggested_response": "We apologize for the delivery delay. We're glad you're enjoying the laptop. Please contact our support team for compensation options."
        }
        ```

    **Common models**:  
    - **Instruction-tuned LLaMA** — 4–140GB  
    - **Alpaca** — 4–26GB  
    - **Vicuna** — 4–26GB  
    - **GPT-4** — API  

    !!! info "Use cases"
        - Data extraction  
        - Report generation  
        - Workflow automation  
        - Complex reasoning tasks  

=== "Retrieval-Augmented Generation (RAG)"

    **What**: Combine LLM with external knowledge retrieval.  

    **Architecture**:  
        User Query → Retrieve relevant documents → LLM (query + documents) → Grounded response  

    !!! example "RAG example"
        ```
        Query: "What are our Q4 sales numbers?"
        Retrieved: [Internal report with sales data]
        LLM Response: "Based on the Q4 report, sales were $5.2M, up 12% from Q3..."
        ```

    **Components**:  
    - **LLM**: Any chat model  
    - **Vector DB**: Store document embeddings  
    - **Retriever**: Find relevant documents

    !!! info "Use cases"
        - Enterprise knowledge bases  
        - Documentation Q&A  
        - Legal/medical document search  
        - Customer support with company data  

### LLM Model Sizes & Hardware

| Model Size | Parameters | VRAM (FP16) | VRAM (INT4) | Hardware Recommendation |
| ---------- | ---------- | ----------- | ----------- | ----------------------- |
| Small      | 1–3B       | 2–6GB       | 1–2GB       | Consumer GPU, CPU       |
| Medium     | 7–13B      | 14–26GB     | 4–7GB       | RTX 4090, M1 Max, CPU   |
| Large      | 30–40B     | 60–80GB     | 15–20GB     | A100, Multi-GPU         |
| Huge       | 70B+       | 140GB+      | 35GB+       | Multi-A100, Clusters    |

## Speech & Audio

**What it is**: Models that process, understand, and generate audio signals.  

**Input formats**:
- Audio files: WAV, MP3, FLAC  
- Sample rate: 16kHz–48kHz typical  
- Format: Raw waveform or spectrograms  
- Duration: Seconds to hours  

### Common Tasks

=== "Speech Recognition (ASR)"

    **What**: Convert spoken audio to text.  

    **Input**: Audio file or stream  
    **Output**: Transcribed text + timestamps  

    !!! example "Short audio transcription"
        ```
        Input: [5-second audio clip: "Hello, how can I help you?"]
        Output: {
          "text": "Hello, how can I help you?",
          "segments": [
            {"text": "Hello", "start": 0.0, "end": 0.5},
            {"text": "how can I help you", "start": 0.6, "end": 2.1}
          ]
        }
        ```

    **Common models**:  
    - **Whisper (OpenAI)** — 75MB–3GB, multilingual  
    - **Wav2Vec 2.0** — 300MB–1GB, self-supervised  
    - **Conformer** — 100–500MB, state-of-the-art  
    - **DeepSpeech** — 200MB, Mozilla  

    !!! info "Whisper Variants"
        - Whisper Tiny: 39M params, 75MB — Fast, mobile  
        - Whisper Base: 74M params, 150MB — Balanced  
        - Whisper Small: 244M params, 500MB — Good quality  
        - Whisper Medium: 769M params, 1.5GB — High quality  
        - Whisper Large: 1.5B params, 3GB — Best quality  

    !!! info "Use cases"
        - Voice assistants  
        - Meeting transcription  
        - Subtitles/captions  
        - Voice commands  
        - Medical dictation  

=== "Text-to-Speech (TTS)"

    **What**: Generate natural-sounding speech from text.  

    **Input**: Text  
    **Output**: Audio waveform  

    !!! example "TTS example"
        ```
        Input: "Welcome to our service. How can I assist you today?"
        Output: [Audio file with natural speech]
        ```

    **Common models**:  
    - **Tacotron 2** — 100–300MB, neural TTS  
    - **FastSpeech 2** — 50–150MB, fast inference  
    - **VITS** — 100–400MB, end-to-end  
    - **Bark** — 1–8GB, generative audio  

    !!! info "Quality factors"
        - Naturalness: Human-likeness  
        - Intelligibility: Clarity of words  
        - Prosody: Rhythm, intonation, emphasis  
        - Voice cloning: Match specific voices  

    !!! info "Use cases"
        - Audiobook narration  
        - Voice assistants  
        - Accessibility (screen readers)  
        - IVR systems  
        - Navigation systems  

=== "Speaker Recognition"

    **What**: Identify or verify who is speaking.  

    **Types**: Identification (1-to-N), Verification (1-to-1)  

    **Input**: Audio sample  
    **Output**: Speaker ID or similarity score  

    !!! example "Speaker verification"
        ```
        Input: [Audio clip of voice]
        Output: {
          "speaker_id": "user_12345",
          "confidence": 0.94,
          "embedding": [512-dimensional vector]
        }
        ```

    **Common models**:  
    - **X-vector** — 50–200MB, speaker embeddings  
    - **ECAPA-TDNN** — 20–100MB, state-of-the-art  
    - **Resemblyzer** — 50MB, voice cloning  

    !!! info "Use cases"
        - Voice authentication  
        - Forensics  
        - Call center analytics  
        - Multi-speaker diarization  

=== "Audio Classification"

    **What**: Categorize audio into classes.  

    **Input**: Audio clip  
    **Output**: Class label + confidence  

    !!! example "Environmental sound classification"
        ```
        Input: [Audio clip with background noise]
        Output: {"class": "dog_bark", "confidence": 0.89}
        ```

    **Common categories**:  
    - Environmental sounds (traffic, nature)  
    - Music genres  
    - Emotions in speech  
    - Audio events (glass breaking, alarms)  

    **Common models**:  
    - **YAMNet** — 20MB, audio event detection  
    - **PANNs** — 80MB, pretrained audio networks  
    - **AST** — 90MB, audio spectrogram transformer  

    !!! info "Use cases"
        - Security (gunshot detection)  
        - Smart home (sound recognition)  
        - Wildlife monitoring  
        - Music recommendation  

=== "Audio Enhancement"

    **What**: Improve audio quality.  

    **Tasks**: Noise reduction, speech enhancement, echo cancellation, dereverberation  

    **Input**: Noisy/distorted audio  
    **Output**: Enhanced audio  

    !!! example "Noise reduction"
        ```
        Input: [Noisy voice recording]
        Output: [Cleaned/enhanced audio]
        ```

    **Common models**:  
    - **Demucs** — 300MB, source separation  
    - **Conv-TasNet** — 50–100MB, speech enhancement  
    - **FullSubNet** — 50MB, noise suppression  

    !!! info "Use cases"
        - Call quality improvement  
        - Podcast editing  
        - Hearing aids  
        - Video production  

### Audio Model Sizes

| Task                 | Small    | Medium    | Large     |
| -------------------- | -------- | --------- | --------- |
| Speech Recognition   | 75–150MB | 500MB–1GB | 2–3GB     |
| Text-to-Speech       | 50–150MB | 200–400MB | 1–8GB     |
| Audio Classification | 20–50MB  | 80–200MB  | 300MB–1GB |
| Audio Enhancement    | 50–100MB | 200–500MB | 1–2GB     |

## Embeddings

**What it is**: Models that convert data (text, images, audio) into dense vector representations that capture semantic meaning.  

**Key concept**: Similar items have similar vectors (measured by cosine similarity or distance)  

**Output format**: Fixed-size vector (64–4096 dimensions typical)  

!!! info "Why Embeddings Matter"
    - Traditional approach:
    "cat" and "dog" are completely different strings → no notion of similarity
    - Embedding approach:
    "cat" → [0.2, 0.8, 0.1, ..., 0.3] (384 dims)
    "dog" → [0.3, 0.7, 0.2, ..., 0.4] (384 dims)
    Similarity: 0.87 (very similar!)

    "car" → [0.9, 0.1, 0.8, ..., 0.2]
    Similarity to "cat": 0.12 (not similar)

### Common Tasks

=== "Text Embeddings"

    **What**: Convert text to vectors for semantic search, clustering, and retrieval.

    **Input**: Text (sentence, paragraph, document)  
    **Output**: Dense vector (384–4096 dimensions)

    !!! example "Text Embedding Example"
        ```
        Input: "The quick brown fox jumps over the lazy dog"
        Output: [0.023, -0.145, 0.389, ..., 0.012] (384 dims)
        ```

    **Common models**:
    - **Sentence-BERT (SBERT)** — 80–420MB, 384–768 dims
    - **all-MiniLM-L6-v2** — Fast, small, 80MB, 384 dims
    - **all-mpnet-base-v2** — Balanced, 420MB, 768 dims
    - **E5** — Multilingual, 400MB–1.2GB, 768–1024 dims
    - **OpenAI ada-002** — API only, 1536 dims
    - **Cohere embed** — API only, 1024–4096 dims

    !!! info "Use Cases"
        - Semantic search  
        - Recommendation systems  
        - Duplicate detection  
        - Clustering documents  
        - RAG (retrieval-augmented generation)

=== "Image Embeddings"

    **What**: Convert images to vectors for visual similarity.

    **Input**: Image  
    **Output**: Dense vector (512–2048 dimensions)

    !!! example "Image Embedding Example"
        ```
        Input: [Photo of a golden retriever]
        Output: [0.145, 0.023, -0.389, ..., 0.891] (512 dims)
        ```

    **Common models**:
    - **CLIP** — Text+image, 350MB–1.7GB, 512–768 dims
    - **ResNet (feature extraction)** — 25–100MB, 2048 dims
    - **EfficientNet (feature extraction)** — 20–80MB, 1280–2560 dims
    - **DINOv2** — Self-supervised, 300MB–1GB, 384–1024 dims

    !!! info "Use Cases"
        - Reverse image search  
        - Visual recommendation  
        - Image deduplication  
        - Content-based image retrieval  
        - Face verification

=== "Multimodal Embeddings"

    **What**: Joint embedding space for text and images, enabling cross-modal similarity.

    **Input**: Text + Image  
    **Output**: Dense vector (512–1024 dimensions)

    !!! example "Multimodal Embedding Example"
        ```
        Text: "a photo of a cat" → [0.2, 0.8, ...]
        Image: [photo of a cat] → [0.21, 0.79, ...]
        Similarity: 0.92
        ```

    **Common models**:
    - **CLIP (OpenAI)** — 350MB–1.7GB
    - **ALIGN (Google)** — 600MB
    - **Florence (Microsoft)** — 1GB
    - **BLIP** — 500MB–2GB

    !!! info "Use Cases"
        - Text-to-image search  
        - Image captioning  
        - Visual question answering  
        - Cross-modal retrieval  
        - Zero-shot image classification

=== "Audio Embeddings"

    **What**: Convert audio to vectors for similarity and retrieval.

    **Input**: Audio clip  
    **Output**: Dense vector (128–1024 dimensions)

    !!! example "Audio Embedding Example"
        ```
        Input: [5-second audio clip of a guitar riff]
        Output: [0.023, 0.312, -0.145, ..., 0.401] (256 dims)
        ```

    **Common models**:
    - **Wav2Vec 2.0** — Self-supervised, 300MB–1GB
    - **HuBERT** — Audio representation, 100–400MB
    - **CLAP** — Audio-text, 500MB

    !!! info "Use Cases"
        - Music similarity  
        - Audio search  
        - Speaker verification  
        - Sound event detection

### Embedding Dimensions

| Dimension        | Characteristics                           |
| ---------------- | ----------------------------------------- |
| Low (64–256)     | Fast search, less storage, lower quality  |
| Medium (384–768) | Balanced speed & quality, most common     |
| High (1024–4096) | Best quality, slower search, more storage |

## Multimodal Models

**What it is**: Models that can process and understand multiple types of data simultaneously (text, images, audio, video).  

**Key innovation**: Unified understanding across modalities  

### Common Tasks

=== "Image Captioning"

    **What**: Generate text descriptions of images.  

    **Input**: Image  
    **Output**: Natural language caption  

    !!! example "Beach sunset caption"
        ```
        Input: [Photo of a beach at sunset]
        Output: "A beautiful sunset over the ocean with waves crashing on the shore"
        ```

    **Common models**:  
    - **BLIP/BLIP-2** — 500MB–2GB  
    - **GIT** — 700MB  
    - **FLAMINGO** — 10GB+, few-shot learning  
    - **GPT-4 Vision** — API only  

    !!! info "Use cases"
        - Accessibility (alt text generation)  
        - Content moderation  
        - Image search indexing  
        - Social media automation  

=== "Visual Question Answering (VQA)"

    **What**: Answer questions about images.  

    **Input**: Image + question  
    **Output**: Answer  

    !!! example "Kitchen VQA"
        ```
        Input: 
          Image: [Kitchen with red apples on counter]
          Question: "What color are the apples?"
        Output: "Red"

        Question: "How many apples are there?"
        Output: "Three apples"
        ```

    **Common models**:  
    - **BLIP-VQA** — 500MB–2GB  
    - **ViLT** — 400MB, Vision-Language Transformer  
    - **Flamingo** — 10GB+, few-shot  
    - **GPT-4V** — API only, best quality  

    !!! info "Use cases"
        - Interactive image exploration  
        - Educational tools  
        - Accessibility  
        - E-commerce (product Q&A)  

=== "Image-Text Retrieval"

    **What**: Find images from text or text from images.  

    **Input**: Text query OR image query  
    **Output**: Ranked list of images OR text  

    !!! example "Text → Image retrieval"
        ```
        Query: "red sports car at sunset"
        Results: [image1, image2, image3] ranked by relevance
        ```

    !!! example "Image → Text retrieval"
        ```
        Query: [Photo of the Eiffel Tower]
        Results: ["Eiffel Tower in Paris", "French landmark", ...]
        ```

    **Common models**:  
    - **CLIP** — 350MB–1.7GB  
    - **ALIGN** — 600MB  
    - **Florence** — 1GB  

    !!! info "Use cases"
        - Stock photo search  
        - E-commerce visual search  
        - Content discovery  
        - Brand monitoring  

=== "Video Understanding"

    **What**: Analyze and understand video content.  

    **Tasks**: Action recognition, video captioning, video Q&A, temporal event detection  

    **Input**: Video (sequence of frames)  
    **Output**: Labels, captions, or answers  

    !!! example "Basketball video"
        ```
        Input: [Video of person playing basketball]
        Output: {
          "action": "shooting basketball",
          "caption": "A person shoots a three-pointer on an outdoor court",
          "objects": ["person", "basketball", "hoop", "court"],
          "timestamps": [{"action": "dribbling", "start": 0, "end": 3}, ...]
        }
        ```

    **Common models**:  
    - **VideoMAE** — 500MB–2GB, self-supervised  
    - **TimeSformer** — 400MB–1GB, attention-based  
    - **Video-CLIP** — 700MB–2GB, multimodal  
    - **Flamingo** — 10GB+, video Q&A  

    !!! info "Challenges"
        - Temporal modeling (understand sequences)  
        - Computational cost (many frames)  
        - Memory requirements  

=== "Audio-Visual Learning"

    **What**: Joint understanding of audio and visual content.  

    **Input**: Video with audio  
    **Output**: Synchronized understanding  

    !!! example "Concert video"
        ```
        Input: [Video of concert]
        Output: {
          "visual": "band playing on stage",
          "audio": "rock music",
          "synchronization": "guitar solo matches guitarist's movements",
          "speaker": "lead singer is singing"
        }
        ```

    !!! info "Use cases"
        - Video captioning with sound context  
        - Speaker detection in videos  
        - Audio-visual source separation  
        - Lip-sync verification  

    **Common models**:  
    - **Audio-Visual Transformer** — 1–4GB  
    - **Hearing through Video** — 500MB  
    - **AViT** — 800MB  

### Multimodal Model Sizes

| Task                  | Small     | Medium | Large   |
| --------------------- | --------- | ------ | ------- |
| Image Captioning      | 500MB–1GB | 1–3GB  | 5–15GB  |
| VQA                   | 400MB–1GB | 1–3GB  | 10GB+   |
| Video Understanding   | 500MB–1GB | 1–4GB  | 10–50GB |
| Audio-Visual Learning | 500MB–2GB | 2–6GB  | 10GB+   |

## Time Series

**What it is**: Models that analyze sequential data points indexed in time order.  

**Input formats**:
- **Univariate**: Single variable over time (e.g., stock price)  
- **Multivariate**: Multiple variables over time (e.g., temperature + humidity + pressure)  
- **Irregular**: Non-uniform time intervals  
- **Regular**: Fixed time intervals (hourly, daily, etc.)

### Common Tasks

=== "Forecasting"

    **What**: Predict future values based on historical data.  

    **Input**: Historical time series  
    **Output**: Future predictions  

    !!! example "Sales forecast"
        ```
        Input: [Daily sales for past 365 days]
        Historical: [100, 105, 98, 110, ...]
        Output: [Predicted sales for next 30 days]
        Forecast: [115, 118, 120, ...]
        ```

    **Common models**:  
    - **LSTM/GRU** — 10–100MB, recurrent networks  
    - **Transformer (Temporal)** — 50–500MB  
    - **N-BEATS** — 20–100MB, pure DL forecasting  
    - **Prophet** — <1MB, statistical  
    - **TimeGPT** — Foundation model, API  

    !!! info "Traditional methods"
        - ARIMA — <1MB  
        - Exponential Smoothing — <1MB  
        - XGBoost (with features) — 5–50MB  

    !!! info "Use cases"
        - Sales forecasting  
        - Demand prediction  
        - Stock price prediction  
        - Energy load forecasting  
        - Weather prediction  

=== "Anomaly Detection"

    **What**: Identify unusual patterns or outliers in time series.  

    **Input**: Time series data  
    **Output**: Anomaly scores or flags  

    !!! example "CPU usage anomaly"
        ```
        Input: [Server CPU usage over time]
        Normal: [20%, 25%, 22%, 23%, ...]
        Anomaly detected: [20%, 25%, 95%, 23%, ...]
        Output: {
          "anomalies": [{"timestamp": "2024-01-15 14:23", "score": 0.95}],
          "severity": "high"
        }
        ```

    **Common models**:  
    - **Autoencoder** — 5–50MB  
    - **LSTM-Autoencoder** — 10–100MB  
    - **Isolation Forest** — 1–20MB  
    - **One-Class SVM** — <5MB  

    !!! info "Use cases"
        - Fraud detection  
        - System monitoring (DevOps)  
        - Predictive maintenance  
        - Network intrusion detection  
        - Health monitoring  

=== "Classification"

    **What**: Categorize time series into predefined classes.  

    **Input**: Time series sequence  
    **Output**: Class label  

    !!! example "ECG classification"
        ```
        Input: [ECG sensor readings over 10 seconds]
        Output: {"rhythm": "normal_sinus", "confidence": 0.94}

        Input: [Accelerometer data from smartphone]
        Output: {"activity": "walking", "confidence": 0.88}
        ```

    **Common models**:  
    - **1D CNN** — 5–50MB  
    - **LSTM/GRU** — 10–100MB  
    - **InceptionTime** — 50–200MB  
    - **Rocket** — 10–50MB  

    !!! info "Use cases"
        - Human activity recognition  
        - Medical diagnosis (ECG, EEG)  
        - Predictive maintenance  
        - Gesture recognition  
        - Financial pattern recognition  

=== "Segmentation"

    **What**: Divide time series into meaningful segments.  

    **Input**: Long time series  
    **Output**: Segment boundaries + labels  

    !!! example "User activity segmentation"
        ```
        Input: [24 hours of user activity data]
        Output: [
          {"segment": "sleeping", "start": "00:00", "end": "07:00"},
          {"segment": "commuting", "start": "07:00", "end": "08:30"},
          {"segment": "working", "start": "08:30", "end": "17:00"},
          ...
        ]
        ```

    **Common approaches**:  
    - **Hidden Markov Models (HMM)** — <5MB  
    - **LSTM-based** — 20–100MB  
    - **Change point detection** — Statistical  

    !!! info "Use cases"
        - Sleep stage detection  
        - Manufacturing process monitoring  
        - Financial market regime detection  
        - Speech diarization  

### Time Series Model Sizes

| Model Type                   | Small    | Medium    | Large     |
| ---------------------------- | -------- | --------- | --------- |
| LSTM/GRU                     | 10–30MB  | 50–150MB  | 200MB–1GB |
| Transformer                  | 50–100MB | 200–500MB | 1–5GB     |
| CNN (1D)                     | 5–20MB   | 30–100MB  | 150–500MB |
| Traditional (ARIMA, Prophet) | <1MB     | 1–10MB    | N/A       |

## Tabular Data

**What it is**: Models for structured data in rows and columns (like databases or spreadsheets).  

**Input formats**:
- CSV, Excel, databases  
- Mixed types: Numerical, categorical, dates  
- Size: From hundreds to millions of rows  

**Key difference**: Features have different meanings and scales, unlike images/text  

### Common Tasks

=== "Classification"

    **What**: Predict categorical outcomes.  

    **Input**: Row of features  
    **Output**: Class label + probability  

    !!! example "Loan approval"
        ```
        Input: {
          "age": 35,
          "income": 75000,
          "credit_score": 720,
          "loan_amount": 50000,
          "employment_years": 8
        }
        Output: {
          "approved": true,
          "confidence": 0.87,
          "risk_category": "low"
        }
        ```

    **Common models**:  
    - **XGBoost** — Gradient boosting, 1–100MB  
    - **LightGBM** — Fast gradient boosting, 1–50MB  
    - **CatBoost** — Handles categories well, 1–100MB  
    - **Random Forest** — Ensemble trees, 5–500MB  
    - **Neural Networks** — TabNet, FT-Transformer, 10–200MB  

    !!! info "Use cases"
        - Fraud detection  
        - Credit scoring  
        - Customer churn prediction  
        - Medical diagnosis  
        - Employee attrition  

=== "Regression"

    **What**: Predict continuous numerical values.  

    **Input**: Row of features  
    **Output**: Numerical prediction  

    !!! example "House price prediction"
        ```
        Input: {
          "bedrooms": 3,
          "bathrooms": 2,
          "sqft": 1800,
          "location": "downtown",
          "year_built": 1995
        }
        Output: {
          "predicted_price": 425000,
          "confidence_interval": [400000, 450000]
        }
        ```

    **Common models**:  
    - **XGBoost/LightGBM** — 1–100MB  
    - **Linear Regression** — <1MB  
    - **Neural Networks** — 10–200MB  
    - **Random Forest** — 5–500MB  

    !!! info "Use cases"
        - House price prediction  
        - Sales forecasting  
        - Risk assessment  
        - Demand estimation  
        - Salary prediction  

=== "Ranking"

    **What**: Order items by relevance or importance.  

    **Input**: Multiple rows to rank  
    **Output**: Ranked list with scores  

    !!! example "Product ranking"
        ```
        Input: [
          {"product_id": 1, "features": [...]},
          {"product_id": 2, "features": [...]},
          {"product_id": 3, "features": [...]}
        ]
        Output: [
          {"product_id": 2, "score": 0.94},
          {"product_id": 1, "score": 0.78},
          {"product_id": 3, "score": 0.45}
        ]
        ```

    **Common models**:  
    - **LambdaMART (LightGBM)** — 10–100MB  
    - **XGBoost Ranker** — 10–100MB  
    - **Neural ranking models** — 50–500MB  

    !!! info "Use cases"
        - Search results ranking  
        - Recommendation systems  
        - Ad targeting  
        - Product recommendations  
        - Job matching  

### Tabular Model Sizes

| Model Type       | Small Dataset (<10K rows) | Medium Dataset (10K-1M) | Large Dataset (>1M) |
| ---------------- | ------------------------- | ----------------------- | ------------------- |
| XGBoost/LightGBM | 1–10MB                    | 10–50MB                 | 50–500MB            |
| Random Forest    | 5–50MB                    | 50–200MB                | 200MB–2GB           |
| Neural Networks  | 10–50MB                   | 50–200MB                | 200MB–1GB           |
| Linear Models    | <1MB                      | 1–5MB                   | 5–20MB              |

## Generative Models

**What it is**: Models that create new content (images, text, audio, video) rather than just analyzing existing content.  

**Key characteristic**: Generate novel outputs, not just classify or predict  

### Common Tasks

=== "Text Generation"

    See Large Language Models (LLMs) section for details.

=== "Image Generation"

    **What**: Create new images from prompts or modify existing images  

    **Types**:

    - **Text-to-Image**: Generate images from text prompts  
      ```
      Input: "A serene lake surrounded by mountains at sunset"
      Output: [Generated photorealistic image]
      ```  
    - **Image-to-Image**: Modify existing images  
      ```
      Input: [Sketch] + "make it photorealistic"
      Output: [Photorealistic version]
      ```  
    - **Inpainting**: Fill missing areas in images  
      ```
      Input: [Image with masked area] + "fill with flowers"
      Output: [Image with flowers in masked area]
      ```  

    **Common models**:  
    - **Stable Diffusion (v1.5, v2.1, SDXL)** — 2–7GB  
    - **DALL-E 2/3** — OpenAI, API only  
    - **Midjourney** — API only  
    - **Imagen** — Google, limited access  

    !!! info "Use cases"
        - Marketing content creation  
        - Product visualization  
        - Concept art  
        - Stock image generation  
        - Image editing/enhancement  

=== "Audio Generation"

    **What**: Create music, sound effects, or speech  

    **Types**:

    - Music Generation  
      ```
      Input: "Jazz piano, upbeat, 30 seconds"
      Output: [Generated music clip]
      ```  
    - Sound Effect Generation  
      ```
      Input: "Thunderstorm with rain"
      Output: [Generated audio]
      ```  
    - Voice Cloning  
      ```
      Input: 5-second voice sample + text
      Output: Speech in that voice
      ```  

    **Common models**:  
    - **MusicGen** — Meta, 300MB–1.5GB  
    - **AudioLDM** — Text-to-audio, 1–2GB  
    - **Bark** — Text-to-speech with emotions, 1–8GB  
    - **Jukebox** — OpenAI, music, 5GB  

    !!! info "Use cases"
        - Content creation (YouTube, podcasts)  
        - Sound design (games, films)  
        - Music production  
        - Voiceovers  

=== "Video Generation"

    **What**: Create or manipulate videos  

    **Types**:

    - Text-to-Video  
      ```
      Input: "A dog running through a field"
      Output: [4-second video clip]
      ```  
    - Image-to-Video  
      ```
      Input: [Still image] + "animate"
      Output: [Video with motion]
      ```  
    - Video Editing  
      ```
      Input: [Video] + "replace person with cartoon"
      Output: [Modified video]
      ```  

    **Common models**:  
    - **Runway Gen-2** — API only  
    - **Pika** — API only  
    - **Stable Video Diffusion** — 3–5GB  
    - **AnimateDiff** — Animation from Stable Diffusion, 2–4GB  

    !!! info "Challenges"
        - Temporal consistency (avoid flickering)  
        - Computational cost (very high)  
        - Limited duration (typically 4–10 seconds)  
        - Quality not yet photorealistic  

    !!! info "Use cases"
        - Short-form content  
        - Animation  
        - Video effects  
        - Concept visualization  

=== "3D Generation"

    **What**: Create 3D models and scenes  

    **Types**:

    - Text-to-3D  
      ```
      Input: "A wooden chair"
      Output: [3D model file]
      ```  
    - Image-to-3D  
      ```
      Input: [Photo of object]
      Output: [3D reconstruction]
      ```  

    **Common models**:  
    - **DreamFusion** — Text-to-3D, research  
    - **Point-E** — OpenAI, 300MB–1GB  
    - **Shap-E** — OpenAI, 1–2GB  
    - **NeRF variants** — Scene reconstruction, varies  

    !!! info "Use cases"
        - Game asset creation  
        - Product visualization  
        - AR/VR content  
        - 3D printing  

### Generative Model Sizes

| Type  | Small     | Medium | Large   |
| ----- | --------- | ------ | ------- |
| Image | 2–3GB     | 4–7GB  | 10–15GB |
| Audio | 300MB–1GB | 1–3GB  | 5–10GB  |
| Video | 3–5GB     | 7–12GB | 20GB+   |
| 3D    | 300MB–1GB | 1–5GB  | 10GB+   |

## Domain Selection Guide

### Quick Decision Matrix

| Your Input         | Your Output        | Recommended Domain                 |
| ------------------ | ------------------ | ---------------------------------- |
| Images             | Class labels       | Computer Vision (Classification)   |
| Images             | Object locations   | Computer Vision (Object Detection) |
| Images             | Pixel-level masks  | Computer Vision (Segmentation)     |
| Text               | Class labels       | NLP (Classification)               |
| Text               | Generated text     | LLM or NLP (Generation)            |
| Text               | Embeddings/vectors | Embeddings (Text)                  |
| Audio              | Text transcript    | Speech (ASR)                       |
| Audio              | Class labels       | Speech (Audio Classification)      |
| Text + Audio       | Speech             | Speech (TTS)                       |
| Text + Images      | Answers/captions   | Multimodal                         |
| Time series data   | Future values      | Time Series (Forecasting)          |
| Time series data   | Anomalies          | Time Series (Anomaly Detection)    |
| Tabular / CSV data | Categories         | Tabular (Classification)           |
| Tabular / CSV data | Numbers            | Tabular (Regression)               |
| Text / Images      | New content        | Generative Models                  |

### By Industry

| Industry           | Common Use Cases                               | Recommended Domains                          |
| ------------------ | ---------------------------------------------- | -------------------------------------------- |
| E-commerce         | Product search, recommendations, visual search | Computer Vision, Embeddings, Tabular         |
| Finance            | Fraud detection, credit scoring, forecasting   | Tabular, Time Series                         |
| Healthcare         | Medical imaging, patient risk, diagnosis       | Computer Vision, Tabular, Time Series        |
| Manufacturing      | Defect detection, predictive maintenance       | Computer Vision, Time Series, Tabular        |
| Media              | Content moderation, recommendation, generation | Computer Vision, NLP, Multimodal, Generative |
| Telecommunications | Churn prediction, network optimization         | Tabular, Time Series                         |
| Retail             | Demand forecasting, customer analytics         | Time Series, Tabular, Computer Vision        |
| Automotive         | Autonomous driving, maintenance                | Computer Vision, Time Series, Multimodal     |