# Document-Based Question Generation System — Full Process Guide

## 1. High-Level Overview

When a user uploads a document (PDF, DOCX, TXT, or image), the system automatically creates questions from the content. The user can choose question types such as multiple choice (MCQ), short answer, or multiple response. The generated questions can then be reviewed, edited, and exported.

---

## 2. User Flow

1. User uploads a document.
2. System extracts text from the file.
3. Text is cleaned and preprocessed.
4. User selects question types (MCQ, short answer, etc.).
5. Questions are generated using NLP models or templates.
6. For MCQs, distractor options are also created.
7. Questions are displayed for user review.
8. User edits or exports final questions.

---

## 3. System Components

### Frontend (React)
- File upload form.
- Question type selection.
- Review and edit interface.
- Export functionality.

### Backend (FastAPI)
- Handles uploads, processing, and question generation.
- API endpoints for upload, processing status, and result retrieval.
- Communicates with NLP and text extraction modules.

### Text Extraction Service
- Extracts raw text from uploaded files.
- Tools:
  - `pdfminer.six` or `PyMuPDF` for PDFs.
  - `python-docx` for Word files.
  - `pytesseract` for OCR (images).

### NLP Preprocessing
- Cleans and tokenizes text.
- Identifies sentences, entities, and key phrases.
- Tools: `spaCy`, `NLTK`, or `transformers`.

### Question Generation
- **Template-based**: Uses rule-based sentence templates.
- **Model-based**: Uses pretrained models like T5 or BART.

### Distractor Generation
- Finds plausible incorrect options for MCQs.
- Methods: Similarity search, WordNet, or embeddings.

### Database (MongoDB)
- Stores files, text, generated questions, and user edits.

### Worker/Queue
- Processes uploaded documents asynchronously (Celery, RQ, or FastAPI background tasks).

---

## 4. Step-by-Step System Process

### Step A — Upload File
Frontend sends file to `/api/upload`. Backend saves it and creates a job record.

### Step B — Extract Text
Text is extracted depending on the file type:
- PDF → pdfminer or PyMuPDF.
- DOCX → python-docx.
- Image → pytesseract (OCR).

### Step C — Preprocess Text
Clean and segment text into sentences. Remove unnecessary content.

### Step D — Identify Candidate Sentences
Extract entities (NER) and key phrases (RAKE/TextRank).

### Step E — Generate Questions
- Replace entities with blanks and generate WH-questions (Who, What, When, Where).
- Or use a transformer model (T5/BART) for better results.

### Step F — Generate Distractors
Use similar entities, semantic embeddings, or lists (cities, people, etc.) to create wrong options.

### Step G — Store Results
Save questions in MongoDB with metadata (type, difficulty, etc.).

### Step H — Display and Edit
Frontend retrieves generated questions and allows edits or exports.

---

## 5. Example MongoDB Schema

### Documents
```json
{
  "_id": "doc1",
  "filename": "example.pdf",
  "status": "done",
  "raw_text": "Photosynthesis converts CO2 to oxygen..."
}
```

### Questions
```json
{
  "_id": "q1",
  "doc_id": "doc1",
  "type": "mcq",
  "question_text": "What process converts carbon dioxide into oxygen?",
  "options": ["Respiration", "Photosynthesis", "Evaporation", "Condensation"],
  "correct_index": 1
}
```

---

## 6. Example FastAPI Endpoints

- `POST /api/upload` — Upload file.
- `GET /api/job/{id}/status` — Check processing status.
- `GET /api/job/{id}/questions` — Retrieve generated questions.
- `POST /api/job/{id}/edit` — Save user edits.
- `GET /api/job/{id}/export` — Download final questions.

---

## 7. Question Generation Pseudocode

```python
for sentence in sentences:
    entities = extract_entities(sentence)
    keyphrases = extract_keyphrases(sentence)
    for cand in entities + keyphrases:
        if cand.type == "PERSON":
            question = f"Who {transform(sentence, cand)}?"
        elif cand.type == "DATE":
            question = f"When {transform(sentence, cand)}?"
        else:
            question = f"What {transform(sentence, cand)}?"
        save_question(question, cand.text)
```

---

## 8. Suggested Libraries

| Purpose | Library |
|----------|----------|
| Text extraction | pdfminer.six, PyMuPDF, python-docx |
| OCR | pytesseract |
| NLP | spaCy, NLTK, transformers |
| Question generation | T5/BART (Hugging Face) |
| Similarity search | sentence-transformers |
| Backend | FastAPI |
| Database | MongoDB |

---

## 9. Beginner Roadmap

1. Create FastAPI upload endpoint.
2. Extract text from TXT/DOCX/PDF.
3. Use spaCy to split sentences and extract entities.
4. Implement simple template-based questions.
5. Add MCQs and distractors.
6. Build React frontend to upload, review, and export.
7. Improve generation with transformer models.

---

## 10. MVP Plan

**Goal:** Build a working prototype that can extract text and create simple questions.

**Includes:**
- Upload & extraction.
- Short-answer and MCQ generation (template-based).
- Review & export.

**Later enhancements:**
- Transformer-based QG.
- Difficulty scoring.
- Advanced distractor generation.
- User feedback integration.

---

## 11. Example Data Flow

1. User uploads file → FastAPI saves & starts processing.
2. Worker extracts text → generates questions → stores in MongoDB.
3. User retrieves and edits generated questions.
4. User exports final version.

---

## 12. Key Challenges & Tips

- Scanned PDFs require OCR → slower and less accurate.
- Start small (short text first).
- Always let users review generated questions.
- Log accepted/rejected questions to improve model quality.

---

## 13. Recommended Learning Resources

- [spaCy Docs](https://spacy.io/usage)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/)
- [MongoDB Python Driver](https://pymongo.readthedocs.io/)
- [Text Extraction with PyMuPDF](https://pymupdf.readthedocs.io/en/latest/)

---

© 2025 Dasun — Question Generation Project Blueprint
