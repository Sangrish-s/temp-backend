from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import pymupdf
import fitz
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from mistralai import Mistral
from dotenv import load_dotenv
from src.sec_edgar import download_sec_html
from src.sec_api import request_recent_filings
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_core.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_text_splitters import TokenTextSplitter
from bs4 import BeautifulSoup
from langchain.schema.document import Document
import os
from enum import Enum

app = FastAPI()

# Load environment variables
load_dotenv(".env")
MISTRAL_API_KEY = os.environ.get("MISTRAL_API_KEY")
client = Mistral(api_key=MISTRAL_API_KEY)
model_name = "mistral-small-latest"

# CORS setup
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock data for demonstration purposes
recent_filings_data = {
    "AAPL": ["10-K", "10-Q", "8-K"],
    "GOOGL": ["10-K", "10-Q"],
    "AMZN": ["10-K", "10-Q", "S-1"],
}

class FilingType(str, Enum):
    ten_k = "10-K"
    ten_q = "10-Q"
    eight_k = "8-K"
    s_1 = "S-1"

class FilingRequest(BaseModel):
    company: str
    filing_type: str = "*"

# Functions

def get_text_chunks_langchain(text):
    text_splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = [Document(page_content=x) for x in text_splitter.split_text(text)]
    return docs

def get_response_from_gpt(message):
    completion = client.chat.complete(
        model=model_name,
        messages=[
            {"role": "user", "content": message},
        ],
    )
    return completion.choices[0].message.content

def get_response_from_url(url, prompt):
    html = download_sec_html(url)
    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text()
    docs = get_text_chunks_langchain(text)
    print(docs)
    llm = client.chat.complete(
        model=model_name,
        messages=[
            {"role": "user", "content": prompt.format(docs=docs)}
        ],
    )
    response = llm.choices[0].message.content

    return response

def get_summary_from_url(url):
    prompt = """The following is a set of summaries:
    {docs}
    Summarize this and avoid the boiler plate info.
    Helpful Answer:
    """
    response = get_response_from_url(url, prompt)
    return response

def extract_text_from_pdf(pdf_file):
    doc = fitz.open(stream=pdf_file, filetype="pdf")
    text = ""
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text += page.get_text()
    return text

# Function to summarize text using Mistral API
def summarize_text_with_mistral(text):
    prompt = (
        f"Summarize the following text:\n\n{text}\n\n"
        "The summary should be concise and cover the key points."
    )
    try:
        print(f"{prompt}")
        response = client.chat.complete(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        summary = response.choices[0].message.content
        return summary
    except Exception as e:
        print(f"Error during summarization: {e}")
        raise HTTPException(status_code=500, detail="Error occurred while summarizing the text.")

def get_key_points_from_url(url, key_point_type):
    response = get_response_from_url(url, "List out key points " + key_point_type + " of {docs}.")
    return response

def get_red_flags_from_url(url):
    response = get_response_from_url(url, "Find red flags in this: {docs}")
    return response

def get_who_is_involved_from_url(url):
    response = get_response_from_url(url, "Who is involved: {docs}")
    return response

def get_who_is_impacted_from_url(url):
    response = get_response_from_url(url, "Who is impacted: {docs}")
    return response

def get_what_does_it_mean_for_investors_from_url(url):
    response = get_response_from_url(url, "What does it mean for investors: {docs}")
    return response

# FastAPI Routes
@app.get("/")
async def home():
    return JSONResponse(content={"status": "running", "version": "v1.0.0"}, status_code=200)

@app.get("/recent_filings")
async def get_recent_filings(company: str, filing_type: FilingType = FilingType.ten_k):
    company = company
    recent_filings = request_recent_filings(company, filing_type.value, 10, 0)
    
    if recent_filings is None:
        raise HTTPException(status_code=404, detail="Company not found")
    
    return {"company": company, "filings": recent_filings}

@app.get("/get_filing_html")
async def get_filing_html(url: str):
    if not url.startswith("https://www.sec.gov/"):
        raise HTTPException(status_code=400, detail="Invalid SEC URL")

    try:
        html = download_sec_html(url)
        return html
    except Exception as e:
        print(f"There was an error downloading the SEC HTML for {url}: {e}")
        raise HTTPException(status_code=404, detail="Error downloading SEC HTML")

@app.get("/get_summary")
async def get_summary(url: str):
    if not url.startswith("https://www.sec.gov/"):
        raise HTTPException(status_code=400, detail="Invalid SEC URL")

    try:
        summary = get_summary_from_url(url)
        return summary
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail="Error downloading or summarizing")

@app.post("/summarize_pdf/")
async def summarize_pdf(file: UploadFile = File(...)):
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Invalid file type. Please upload a PDF file.")
        pdf_content = await file.read()
        text = extract_text_from_pdf(pdf_content)
        words = text.split()
        if len(words) > 10000:
            text = " ".join(words[:10000])
        summary = summarize_text_with_mistral(text)

        return JSONResponse(content={"summary": summary})
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="An error occurred during the processing of the PDF.")


@app.get("/get_key_points")
async def get_key_points(url: str, filing_type: FilingType):
    if not url.startswith("https://www.sec.gov/"):
        raise HTTPException(status_code=400, detail="Invalid SEC URL")

    try:
        summary = get_key_points_from_url(url, filing_type.value)
        return summary
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail="Error downloading or retrieving key points")

@app.get("/get_red_flags")
async def get_red_flags(url: str):
    if not url.startswith("https://www.sec.gov/"):
        raise HTTPException(status_code=400, detail="Invalid SEC URL")

    try:
        summary = get_red_flags_from_url(url)
        return summary
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail="Error downloading or retrieving red flags")

@app.get("/get_who_is_involved")
async def get_who_is_involved(url: str):
    if not url.startswith("https://www.sec.gov/"):
        raise HTTPException(status_code=400, detail="Invalid SEC URL")

    try:
        summary = get_who_is_involved_from_url(url)
        return summary
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail="Error downloading or retrieving involved parties")

@app.get("/get_who_is_impacted")
async def get_who_is_impacted(url: str):
    if not url.startswith("https://www.sec.gov/"):
        raise HTTPException(status_code=400, detail="Invalid SEC URL")

    try:
        summary = get_who_is_impacted_from_url(url)
        return summary
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail="Error downloading or retrieving impacted parties")

@app.get("/get_what_does_it_mean")
async def get_what_does_it_mean(url: str):
    try:
        summary = get_what_does_it_mean_for_investors_from_url(url)
        return summary
    except Exception as e:
        print(e)
        raise HTTPException(status_code=404, detail="Error downloading or retrieving implications for investors")

# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
