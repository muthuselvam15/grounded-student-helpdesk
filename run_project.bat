@echo off
echo ==========================================
echo Grounded Student Helpdesk - Setup
echo ==========================================
python -m pip install -r requirements.txt
echo.
echo Make sure Ollama is installed and running.
echo Pulling required models...
ollama pull gemma4
ollama pull embeddinggemma
echo.
echo Starting Streamlit...
python -m streamlit run streamlit_app.py
pause
