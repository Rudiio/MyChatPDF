from fastapi import FastAPI
from src.rag import Rag
from fastapi.responses import HTMLResponse

app = FastAPI(title='MyChatPdf',
              description='Chat with you pdf files locally')
rag = Rag()

@app.get('/',response_class=HTMLResponse)
def index():
    return """<html>
    <h1>Hello, welcome to  MyChatPdf</h1>
    </html>"""

@app.post('/ingest')
def ingest_document(document_path:str):
    """Ingest a specified document"""

    try:
        rag.ingest(document_path)
        return {'status':'done'}
    except:
        return {'status':'failed'}


