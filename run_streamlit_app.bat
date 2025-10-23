@echo off
echo 🚀 Starting RAG Streamlit App...
echo.
echo Make sure you have:
echo - Created .env file with your HuggingFace API token
echo - Added Chapter 01.pdf to this folder
echo - Installed requirements: pip install -r requirements.txt
echo.
echo Starting web app...
echo Open your browser to http://localhost:8501
echo.
streamlit run rag_streamlit_app.py