
from pydantic import BaseModel, Field, InstanceOf

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

class DocumentStorer(BaseModel):

    loader : InstanceOf[PyPDFLoader] = Field(default=None,description="Pdf loader")
    splitter : InstanceOf[RecursiveCharacterTextSplitter] = Field(default=RecursiveCharacterTextSplitter(),
                                                                  description="Text splitter")
    db : InstanceOf[Chroma] = Field(default=Chroma(),description="Chroma vector database")
    document : InstanceOf[Document] = Field(default=None,description="Document placeholder")
    embeddings : InstanceOf[HuggingFaceEmbeddings] = Field(default=HuggingFaceEmbeddings(),description="Embeddings model")

    def load_file(self,pdf:str) -> None:
        self.loader = PyPDFLoader(pdf)
        self.document = self.loader.load()

        self.splitter = RecursiveCharacterTextSplitter(chunk_size=300,
                                                       chunk_overlap=100,)
        
        chunked_document = self.splitter.split_documents(self.document)

    def create_vector_store(self,) -> None:
        self.model = "sentence-transformers/all-MiniLM-l6-v2"
        model_kwargs = {'device':'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model,     # Provide the pre-trained model's path
            model_kwargs=model_kwargs, # Pass the model configuration options
            encode_kwargs=encode_kwargs # Pass the encoding options
            )

if __name__=="__main__":
    docstore = DocumentStorer()
    docstore.load_file('./article')
    docstore.create_vector_store()
    