conda activate as
pip install -r requirements.txt
uvicorn backend.fastapi_server:app --host 0.0.0.0 --port 2000 --reload
cd frontend
npm start