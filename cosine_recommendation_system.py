import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from utils import cosine_save_recommendations_to_excel, calculate_latency
import faiss
import pickle
from tqdm import tqdm
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


def main():
    # Example usage
    DATASET_PATH = os.path.join(os.getcwd(), 'dataset', 'final_processed_data.csv')  # Replace with your CSV file path
    MODEL_PATH = os.path.join(os.getcwd(), 'models', 'cosine_recommendation_model')
    EXCEL_FILE_OUTPUT_PATH = os.path.join(os.getcwd(), 'results', 'cosine_recommendations_reports.xlsx')


    # Check if the system exists
    if os.path.exists(MODEL_PATH) and os.path.exists(MODEL_PATH + ".vectordb"):
        # Load existing system
        try:
            recommender = LargeScaleProductRecommender.load_system(MODEL_PATH)
        except Exception as e:
            print(f"Error loading existing system: {e}")
            print("Creating a new system instead.")
            recommender = LargeScaleProductRecommender(DATASET_PATH)
            recommender.preprocess_data()
            recommender.save_system(MODEL_PATH)
    else:
        # Create and save new system
        recommender = LargeScaleProductRecommender(DATASET_PATH)
        recommender.preprocess_data()
        recommender.save_system(MODEL_PATH)

    # Example queries
    queries = [
        "I want to make a game like flappy bird where the bird flies and avoid obstacles",
        "I want a source code for a multiple choice test app, that is capable of limiting users to only a few questions until they pay for the full package to have full access to all questions, users can compete with each other and see their scores.",
        "Any programming language, No specific industry, I am looking to launch a service, UI and backend ,Project 2:Mobile applications and website development.Website design (front end)Mobile applications (front end)",
        "Any programming language , No specific industry , I am looking to launch a service , , mobile app , NEED A CHAT APPLICATION LIKE WHATSAPP WITH LOCATION TRACKER.  , TO BE USED FOR 200 COURIER DELIVERY BOYS AND TO TRACK THEIR MOVEMENT ON REALTIME BASIS. APP TO BE ANDROID BASED. WILL BE INSTALLED IN 200 MOBILES. TO BE TRACKED FROM ONE DESKTOP /ANDROID FROM CONTROLLING OFFICE.",
        "I need an app/software for real staff to communicate with customers without sharing their personal contacts. ,  As an example, when doing a graphic design, the designer can get a better idea if he can communicate directly with the client. But when they use their personal contact, they communicated through personal contact next time, that s the problem",
        "Any programming language, No specific industry, I am looking to launch a service, UI and backend, Web or mobile PWA(IOS / Android or both) TTS and STT and text to video app .1.Simple signup page with name, email, and phone number for verificationof OTP.2. Selection option to receive feedbackof voice or video Please let me know if you can do that, what would be the cost and run time .3.The user talks to the app and has voice / spoken words transcribed(STT) to send text queries to the ChatGPT API and when ChatGPT replies, they have the TTSuse a predefined voice to express the response to the user or have a predefined Avatar present the same as a video.",
        "Any programming language, No specific industry,I am looking to launch a service,UI and backend,currently using perfekto solutions paying for licenses.They are fast consuming business and they have consumer and business side.They have an online shop but also they have a business where they supply shops and they are their resellers.They need to have an access to accounts, customers, sales, stock / inventory, reports(breaks of each shop, each sales) 1500 productsThey currently have and need a portal for an online shopMore user friendly customer portal for consumers, warehouse invoice system that can be added as well.Office side, sales side,search needs to be integratedSome of the issues that they are facing: products needs to stay in the basket after a refresh, discounts to the shops, the discounts should stayTimeline: soon as possibleBudget: They need support for this productThey need training for the product",
        "Any programming language, No specific industry, I am looking to launch a service, UI and backend ,They need a product or algorithm that reads through charts, graphs, video and analysis what is written or potentially spoken. They currently have technology that goes through text but they want to improve it and add on that. So they said that pretty much anything that is build in that direction they might be potentially interested in.",
    ]


    all_recommendations = []
    latencies = []

    for query in queries:
        print(f"\nTop recommendations for query: '{query}'")
        recommendations, latency = recommender.recommend(query)
        all_recommendations.append(recommendations)
        latencies.append(latency)

        print(f"Latency: {latency:.4f} seconds")
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. [ID: {rec['id']}] {rec['product_name']} (Similarity: {rec['similarity_score']:.4f})")
            print(f"   Description: {rec['description'][:100]}...")  # Print first 100 chars of description

    # Calculate and print average latency
    avg_latency = sum(latencies) / len(latencies)
    print(f"\nAverage latency: {avg_latency:.4f} seconds")

    # Save recommendations to Excel
    cosine_save_recommendations_to_excel(queries, all_recommendations, latencies, EXCEL_FILE_OUTPUT_PATH)

if __name__ == "__main__":
    main()


   