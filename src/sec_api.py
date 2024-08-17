from sec_api import QueryApi
from dotenv import load_dotenv
import yfinance as yf
import os

load_dotenv(".env")
SEC_API_KEY = os.environ.get("SEC_API_KEY")
queryApi = QueryApi(api_key=SEC_API_KEY)

company_to_ticker = {
    "Apple Inc": "AAPL",
    "Alphabet Inc": "GOOGL",
    "Amazon.com Inc": "AMZN",
    # Add more mappings as needed
}

def get_ticker_from_name(company_name):
    # Attempt to get the ticker from the lookup table
    ticker = company_to_ticker.get(company_name)
    if ticker:
        return ticker
    else:
        print(f"Ticker for company '{company_name}' not found")
        return None
def request_recent_filings(company_name, form_type=None, size=10, _from=0):
    # Convert company name to ticker
    ticker = get_ticker_from_name(company_name)
    if not ticker:
        print(f"Ticker for company '{company_name}' not found")
        return []
    
    query_parts = []
    
    if ticker:
        query_parts.append(f"ticker:{ticker}")
    
    if form_type:
        if isinstance(form_type, str):
            query_parts.append(f"formType:\"{form_type}\"")
        elif isinstance(form_type, list):
            form_type_query = " AND ".join([f"formType:\"{ft}\"" for ft in form_type])
            query_parts.append(form_type_query)
    
    query = " AND ".join(query_parts)
    
    print(f"Constructed query: {query}")
    try:
        response = queryApi.get_filings(
            {
                "query": query,
                "from": str(_from),
                "size": str(size),
                "sort": [{"filedAt": {"order": "desc"}}]
            }
        )
        return response.get("filings", [])
    except Exception as e:
        print(f"There was an error getting filings for the ticker {ticker}: {e}")
        return []

# Example usage
filings = request_recent_filings("Apple Inc", "10-K")
print(filings)
