from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from openpyxl.styles import Font
import pandas as pd
import numpy as np
import json
import time
import csv


def cosine_save_recommendations_to_excel(queries, all_recommendations, latencies, file_path):
    """
    Save queries and their recommendations to an Excel file.
    Each query will occupy a single row, with recommendations as a dictionary in another column.
    
    :param queries: List of query strings
    :param all_recommendations: List of lists containing recommendations for each query
    :param file_path: Path to save the Excel file
    """
    data = []
    for query, recommendations, latency in zip(queries, all_recommendations, latencies):
        rec_dict = {str(rec['id']): rec['product_name'] for rec in recommendations}
        rec_json = json.dumps(rec_dict)
        
        data.append({
            'Query': query,
            'Recommendations': rec_json,
            'Latency (seconds)': latency
        })
    
    df = pd.DataFrame(data)
    
    # Create a new Excel writer object
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        # Write the main dataframe to the first sheet
        df.to_excel(writer, index=False, sheet_name='Recommendations')
        
        # Adjust column widths
        worksheet = writer.sheets['Recommendations']
        worksheet.column_dimensions['A'].width = 50  # Adjust query column width
        worksheet.column_dimensions['B'].width = 100  # Adjust recommendations column width
        worksheet.column_dimensions['C'].width = 20
        worksheet.column_dimensions['D'].width = 25

        # Calculate average latency
        avg_latency = sum(latencies) / len(latencies)

        # Add average latency in the next column of the first row
        avg_latency_cell = worksheet.cell(row=2, column=4, value=f"Avg Latency: {avg_latency:.4f} s")
        avg_latency_cell.font = Font(bold=True)
        
    print(f"\n Recommendations saved to {file_path}")


def llm_save_recommendations_to_csv(queries, all_recommendations, latencies, file_path):
    data = []
    for i, (query, recommendation, latency) in enumerate(zip(queries, all_recommendations, latencies)):
        print(f"Debug - Processing item {i+1}")
        print(f"Debug - Recommendation type: {type(recommendation)}")
        
        if isinstance(recommendation, str):
            full_recommendation = recommendation
        elif isinstance(recommendation, list):
            full_recommendation = ' '.join(recommendation)
        else:
            full_recommendation = str(recommendation)
        
        print(f"Debug - Full recommendation length: {len(full_recommendation)}")
        
        data.append({
            'Query': query,
            'Recommendation': full_recommendation,
            'Latency': latency
        })

    with open(file_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=['Query', 'Recommendations', 'Latency'])
        writer.writeheader()
        for row in data:
            writer.writerow(row)

    print(f"Detailed recommendations saved to {file_path}")
 


def calculate_latency(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        latency = end_time - start_time
        return result, latency                     
    return wrapper



def creating_vectordb_faiss_llm():
    # Load the document, split it into chunks, embed each chunk and load it into the vector store.
    raw_documents = TextLoader('../../../state_of_the_union.txt').load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)
    db = FAISS.from_documents(documents, OpenAIEmbeddings())