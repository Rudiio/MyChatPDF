import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from pydantic import BaseModel, Field, InstanceOf

from langchain.prompts.chat import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import Runnable,RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms.ollama import Ollama

from dotenv import load_dotenv,find_dotenv

from rich.console import Console

console = Console()

class Rag(BaseModel):

    loader : InstanceOf[PyPDFLoader] = Field(default=None,description="Pdf loader")
    splitter : InstanceOf[RecursiveCharacterTextSplitter] = Field(default=None,
                                                                  description="Text splitter")
    vector_db : InstanceOf[Chroma] = Field(default=None,description="Chroma vector database")
    embeddings : InstanceOf[HuggingFaceEmbeddings] = Field(default=None,description="Embeddings model")
    retriever : InstanceOf[VectorStoreRetriever] = Field(default=None,description="Document retriever")
    chain : InstanceOf[Runnable] = Field(default=None,description="Chat chain")

    def ingest(self,pdf:str) -> None:
        """Ingest the pdf document"""
        # Load the document
        self.load_embeddings()

        with console.status("[bold green]Ingesting the document..."):
            self.loader = PyPDFLoader(pdf)
            document = self.loader.load()

            # Transforming the document
            self.splitter = RecursiveCharacterTextSplitter(chunk_size=300,
                                                        chunk_overlap=100,)
            
            chunked_document = self.splitter.split_documents(document)

            # Storing the document
            self.vector_db = Chroma.from_documents(documents=chunked_document,
                                                embedding=self.embeddings)
        self.retriever = self.vector_db.as_retriever()


    def load_embeddings(self,) -> None:
        """Load the embeddings model"""
        model = "sentence-transformers/all-MiniLM-l6-v2"
        model_kwargs = {'device':'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        self.embeddings = HuggingFaceEmbeddings(model_name=model,     # Provide the pre-trained model's path
                                                model_kwargs=model_kwargs, # Pass the model configuration options
                                                encode_kwargs=encode_kwargs # Pass the encoding options
                                                )
    
    def create_retrieval_qa(self,):
        """Create the retrieval qa chain"""
        load_dotenv(find_dotenv())

        with console.status('[bold green]Creating the retrieval chain'):
            prompt = ChatPromptTemplate.from_messages([
                ('system','You are a helpful assistant.'),
                ('human','Answer this question {question} using this context\n {context}.')
            ])
            llm = Ollama(model='gemma:2b-instruct-q4_0')
            
            def format_doc(documents):
                return '\n\n'.join(doc.page_content for doc in documents)
            
            self.chain = {'context':self.retriever | format_doc,'question':RunnablePassthrough()} | prompt | llm | StrOutputParser()

    def answer(self,question:str):
        """Answer teh asked question with rag"""
        with console.status('[bold green]Running the model'):
            return self.chain.invoke(question)


    @property
    def retriever(self):
        return self.retriever

if __name__=="__main__":
    docstore = Rag()
    docstore.ingest('./article.pdf')
    docstore.create_retrieval_qa()
    print(docstore.answer('What is Neo4j?'))
    
    