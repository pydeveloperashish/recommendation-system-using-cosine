import streamlit as st
import pandas as pd
import numpy as np
from utils import calculate_latency
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
import faiss
import pickle
import os


class LargeScaleProductRecommender:
    def __init__(self, csv_path=None, chunk_size=10000):
        self.csv_path = csv_path
        self.chunk_size = chunk_size
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.index = None
        self.product_ids = []
        self.product_names = []
        self.product_descriptions = []

    def preprocess_data(self):
        if not self.csv_path:
            raise ValueError("CSV path is not set. Please provide a valid path.")

        print("Preprocessing data...")
        chunks = pd.read_csv(self.csv_path, chunksize=self.chunk_size)
        all_text = []

        for chunk in tqdm(chunks, desc="Processing chunks"):
            chunk['combined_text'] = chunk['name'] + ' ' + chunk['description']
            all_text.extend(chunk['combined_text'].tolist())
            self.product_ids.extend(chunk['id'].tolist())
            self.product_names.extend(chunk['name'].tolist())
            self.product_descriptions.extend(chunk['description'].tolist())

        print("Vectorizing data...")
        vectors = self.vectorizer.fit_transform(all_text).toarray()
        vectors = vectors.astype('float32')  # Faiss requires float32

        print("Building Faiss index...")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        faiss.normalize_L2(vectors)
        self.index.add(vectors)
        print("Preprocessing complete.")

    def save_system(self, file_path):
        print("Saving recommendation system...")
        with open(file_path, 'wb') as f:
            pickle.dump({
                'vectorizer': self.vectorizer,
                'product_ids': self.product_ids,
                'product_names': self.product_names,
                'product_descriptions': self.product_descriptions
            }, f)
        faiss.write_index(self.index, file_path + ".vectordb")
        print(f"System saved to {file_path} and {file_path}.vectordb")

    @classmethod
    def load_system(cls, file_path):
        print("Loading recommendation system...")
        recommender = cls()
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        recommender.vectorizer = data['vectorizer']
        recommender.product_names = data['product_names']
        recommender.product_descriptions = data['product_descriptions']
        
        # Handle case where product_ids are not in the saved system
        if 'product_ids' in data:
            recommender.product_ids = data['product_ids']
        else:
            print("Warning: Product IDs not found in the saved system. Generating placeholder IDs.")
            recommender.product_ids = list(range(len(recommender.product_names)))
        
        recommender.index = faiss.read_index(file_path + ".vectordb")
        print("System loaded successfully.")
        return recommender

    @calculate_latency
    def recommend(self, query, top_n=5):
        query_vector = self.vectorizer.transform([query]).toarray().astype('float32')
        faiss.normalize_L2(query_vector)
        distances, indices = self.index.search(query_vector, top_n)
        
        recommendations = []
        for i, idx in enumerate(indices[0]):
            recommendations.append({
                'id': self.product_ids[idx],
                'product_name': self.product_names[idx],
                'description': self.product_descriptions[idx],
                'similarity_score': float(distances[0][i])  # Convert np.float32 to Python float
            })
        
        return recommendations
    

# Streamlit app
def main():
    st.title("Cosine Product Recommendation System")

    # Load the recommendation system
    
    
    MODEL_PATH = os.path.join(os.getcwd(), 'models', 'cosine_recommendation_model')

    # Check if the system exists
    if os.path.exists(MODEL_PATH) and os.path.exists(MODEL_PATH + ".vectordb"):
        recommender = LargeScaleProductRecommender.load_system(MODEL_PATH)

    # User input
    user_query = st.text_area("Enter your product query:", height=100)

    if st.button("Get Recommendations"):
        if user_query:
            with st.spinner("Generating recommendations..."):
                recommendations, latency = recommender.recommend(user_query)

            st.success(f"Recommendations generated in {latency:.4f} seconds")

            # Display recommendations
            for i, rec in enumerate(recommendations, 1):
                st.markdown(f"**{i}. {rec['product_name']}**")
                st.markdown(f"Similarity: {rec['similarity_score']:.4f}")
                st.markdown(f"Description: {rec['description'][:200]}...")
                st.markdown("---")

        else:
            st.warning("Please enter a query to get recommendations.")

if __name__ == "__main__":
    main()