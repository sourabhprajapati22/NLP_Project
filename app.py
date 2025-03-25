from flask import Flask, render_template, jsonify, request
import random
import pandas as pd
from textblob import TextBlob
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Dummy dataset
def generate_dummy_data():
    products = []
    for i in range(5):  # 5 products for demo
        reviews = [
            {
                "text": f"This is a sample review {j} for product {i}",
                "rating": random.choice([1, 2, 3, 4, 5])
            } for j in range(20)  # 20 reviews per product
        ]
        products.append({
            "product_id": i,
            "name": f"Product {i}",
            "reviews": reviews
        })
    return products

dataset = generate_dummy_data()

def sentiment_analysis(text):
    analysis = TextBlob(text)
    return "positive" if analysis.sentiment.polarity > 0 else "negative"

@app.route('/')
def home():
    return render_template('index.html', products=dataset)

@app.route('/sentiment_analysis', methods=['GET'])
def analyze_sentiments():
    results = []
    for product in dataset:
        sentiments = [sentiment_analysis(review["text"]) for review in product["reviews"]]
        positive_count = sentiments.count("positive")
        negative_count = sentiments.count("negative")
        total_reviews = len(sentiments)
        positive_percentage = (positive_count / total_reviews) * 100
        negative_percentage = (negative_count / total_reviews) * 100
        results.append({
            "product_id": product["product_id"],
            "name": product["name"],
            "positive_reviews": positive_count,
            "negative_reviews": negative_count,
            "positive_percentage": round(positive_percentage, 2),
            "negative_percentage": round(negative_percentage, 2)
        })
    return jsonify(results)

@app.route('/search_product', methods=['GET'])
def search_product():
    query = request.args.get('query', '').lower()
    filtered_products = [p for p in dataset if query in p['name'].lower()]
    return jsonify(filtered_products)

@app.route('/upload_product', methods=['POST'])
def upload_product():
    return jsonify({"message": "Product uploaded successfully (Dummy function)"})

if __name__ == '__main__':
    app.run(debug=True)
