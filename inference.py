import sqlite3
from myntra_sentiment_analysis.scraping.myntra_scraper import myntra_web_page,scrape_id_list,scrape_review
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import re
import json
import sys
import os
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import torch
import re
from transformers import pipeline
import emoji
from textblob import TextBlob
from transformers import AutoTokenizer, AutoModel
import joblib
from collections import Counter

# from sklearn.model_selection import train_test_split
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix,f1_score



file_current_path=Path(__file__).resolve().parent
print(file_current_path)
# sys.exit()


def write_jsonl_line(path, data):
    with open(path, 'a', encoding='utf-8') as f:
        json.dump(data, f)
        f.write('\n')

jsonl_path = os.path.join(file_current_path, "myntra_sentiment_analysis/data/raw/test_reviews.jsonl")

if os.path.exists(jsonl_path):
    os.remove(jsonl_path)
    print("File deleted successfully.")
else:
    print("File does not exist.")


emoji_sentiment = {
        '😍': ' positive_emoji ',
        '😊': ' positive_emoji ',
        '😃': ' positive_emoji ',
        '😁': ' positive_emoji ',
        '👍': ' positive_emoji ',
        '🔥': ' positive_emoji ',
        '💯': ' positive_emoji ',
        '🥰': ' positive_emoji ',
        '😢': ' negative_emoji ',
        '😡': ' negative_emoji ',
        '👎': ' negative_emoji ',
        '😠': ' negative_emoji ',
        '🤮': ' negative_emoji ',
        '😤': ' negative_emoji ',
        '😞': ' negative_emoji ',
        '❤️': ' positive_emoji ',
        '💖': ' positive_emoji ',
        '💘': ' positive_emoji ',
        '💗': ' positive_emoji ',
        '💓': ' positive_emoji ',
        '💞': ' positive_emoji ',
        '💟': ' positive_emoji ',
        '🧡': ' positive_emoji ',
        '💛': ' positive_emoji ',
        '💚': ' positive_emoji ',
        '💙': ' positive_emoji ',
        '💜': ' positive_emoji ',
        '🖤': ' positive_emoji ',
        '🤍': ' positive_emoji ',
        '🤎': ' positive_emoji ',
    }

def get_test_reviw(product_id):
    review_results=[]
    count=0
    print(product_id)
    html_content=scrape_review(product_id)

    if html_content==None:
        print('no content present.')
    soup = BeautifulSoup(html_content, "html.parser")

    count+=1

    data=soup.find_all("div", class_="user-review-userReviewWrapper")

    brand_name=soup.find("h1",class_="product-details-brand").getText()


    product_details=soup.find("h1",class_="product-details-name undefined").getText()
    # print(product_details)

    # current_price, _, actual_price, discount =soup.find("p",class_="product-details-discountContainer").getText(separator="||").split('||')

    # print(current_price, actual_price, discount, sep='\n')

    review_count=0
    for data_ in data:
        if review_count==500:
            break
        if len(data_.getText(separator="||").split("||"))!=6:
            continue
        rating, review, name, date, likes, dislikes= data_.getText(separator="||").split("||") #.find('div',class_="user-review-main user-review-showRating")

        # print(rating, review, name, date, likes, dislikes,sep='\n')

        record={
                'product_id':product_id,
                # 'rating':rating,
                'review':review,
                # 'name': name,
                # 'date': date,
                # 'likes': likes,
                # 'dislikes':dislikes,
                'brand_name':brand_name,
                'product_details': product_details,
                # 'current_price': current_price,
                # 'actual_price': actual_price,
                # 'discount':discount
                }
        
        write_jsonl_line(jsonl_path, record)

        review_count+=1


def get_sentiment_label(text):
    try:
        result = sentiment_pipeline(text[:512])[0]  # Truncate long reviews
        return int(result['label'][0])
    except:
        return "error"


aspects = [
        'fit', 'size', 'fabric', 'color', 'comfort', 'style', 'design', 'stitching',
        'softness', 'durability', 'washability', 'shrinkage', 'price', 'value',
        'brand', 'packaging', 'sleeve', 'collar', 'neckline', 'length',
        'waist', 'stretch', 'pockets', 'zippers', 'buttons', 'sole',
        'grip', 'warmth', 'hood', 'elasticity', 'tightness', 'looseness'
    ]

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
sentiment_pipeline = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")

def get_aspect_sentiment(text, aspects=aspects):
    sentiment_overall = TextBlob(text).sentiment.polarity
    result = classifier(text, aspects, multi_label=True)
    aspect_sentiments = {}

    for label, score in zip(result['labels'], result['scores']):
        if label in text.lower():  # ensure aspect is mentioned
            aspect_sentiments[label]= "positive" if sentiment_overall > 0 else "negative"
    return aspect_sentiments


def get_review(file_name):
    df=pd.read_json(file_name,lines=True)
    print("Number of Products: ",len(df.product_id.unique()))

    df['cleaned_review_without_emoji'] = df['review'].apply(clean_review)
    df['cleaned_review_with_emoji'] = df['review'].apply(clean_review_with_emojis)
    df = df[df['cleaned_review_with_emoji'].str.strip() != ""]
    
    

    df['sentiment_emoji']=df['cleaned_review_with_emoji'].apply(get_sentiment_label)
    df['sentiment']=df['cleaned_review_without_emoji'].apply(get_sentiment_label)

    df['sentiment_label_emoji'] = df['sentiment_emoji'].apply(lambda x: 'positive' if x > 3 else ('neutral' if x == 3 else 'negative'))
    df['sentiment_label'] = df['sentiment'].apply(lambda x: 'positive' if x > 3 else ('neutral' if x == 3 else 'negative'))

    
    new_df=pd.get_dummies(df[['sentiment_label_emoji','sentiment_label']]).set_index(df.product_id).reset_index()

    product_recommend=new_df.groupby('product_id').sum()

    product_recommend['recommend']=product_recommend.sentiment_label_positive.apply(lambda x:True if x>350 else False)


    print('====================pretrained public model=======================')

    print('Negative review count: ', product_recommend.sentiment_label_negative.values[0])
    print('neutral review count: ',product_recommend.sentiment_label_neutral.values[0])
    print('Positive reiview count: ',product_recommend.sentiment_label_positive.values[0])
    print('product recommend to customer: ',product_recommend.recommend.values[0])

    review = df['cleaned_review_without_emoji'][0]

    print(get_aspect_sentiment(review))

    print("===================custom trained model output=====================")

    # Model and tokenizer
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # Output path
    output_file = file_current_path / 'myntra_sentiment_analysis/data/processed/test_review_embeddings_flat.jsonl'
    print(output_file)

    with open(output_file, 'w') as f:
        for idx, row in tqdm(df.iterrows()):
            product_id = row['product_id']
            sentence = row['cleaned_review_without_emoji']
            sentiment = row['sentiment_label']  # <-- Add this line

            # Tokenize and move to device
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True).to(device)
            
            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().tolist()
            
            # Create flattened dict
            record = {'product_id': product_id}
            for i, val in enumerate(embedding):
                record[f'emb_{i}'] = val

            record['sentiment_label'] = sentiment  # <-- Add sentiment at the end

            # Save as json line
            f.write(json.dumps(record) + '\n')

    print(f" Saved flattened embeddings to: {output_file}")

    df_embedding=pd.read_json(output_file,lines=True)
    print(df_embedding.shape)

    X=df_embedding.drop(['product_id','sentiment_label'],axis=1)
    y=df_embedding.sentiment_label

    print('Load the models------------')

    rf_model=joblib.load(file_current_path / 'myntra_sentiment_analysis/models/baseline/random_forest_model.joblib')
    dt_model=joblib.load(file_current_path / 'myntra_sentiment_analysis/models/baseline/decision_tree_model.joblib')
    lr_model=joblib.load(file_current_path / 'myntra_sentiment_analysis/models/baseline/logistic_regression_model.joblib')


    print("RandomForest Result:- ")
    pred_output=rf_model.predict(X)
    final_output=Counter(pred_output)

    print('Negative review count: ', final_output['negative'])
    print('neutral review count: ',final_output['neutral'])
    print('Positive reiview count: ',final_output['positive'])
    
    if final_output['positive']>350:
        print('product recommend to customer: True')
    
    else:
        print('product recommend to customer: False')


    print("DecisionTree Result:- ")
    pred_output=dt_model.predict(X)
    final_output=Counter(pred_output)
    
    print('Negative review count: ', final_output['negative'])
    print('neutral review count: ',final_output['neutral'])
    print('Positive reiview count: ',final_output['positive'])
    
    if final_output['positive']>350:
        print('product recommend to customer: True')
    
    else:
        print('product recommend to customer: False')

    print("LogisticRegression Result:- ")
    pred_output=lr_model.predict(X)
    final_output=Counter(pred_output)
    
    print('Negative review count: ', final_output['negative'])
    print('neutral review count: ',final_output['neutral'])
    print('Positive reiview count: ',final_output['positive'])
    
    if final_output['positive']>350:
        print('product recommend to customer: True')
    
    else:
        print('product recommend to customer: False')






def replace_emojis(text):
    if not isinstance(text, str):
        return ""
    for emo, replacement in emoji_sentiment.items():
        text = text.replace(emo, replacement)
    return text

def clean_review_with_emojis(text):
    if not isinstance(text, str):
        return ""
    
    text = replace_emojis(text)  # keep useful emoji context
    
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"<.*?>", "", text)                   # remove HTML
    text = re.sub(r"[^a-zA-Z0-9\s.,!?_]", "", text)     # keep underscore in label
    text = re.sub(r"\s+", " ", text).strip()            # normalize whitespace
    return text

def clean_review(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)          # remove URLs
    text = re.sub(r"<.*?>", "", text)                   # remove HTML
    text = re.sub(r"[^a-zA-Z0-9\s.,!?]", "", text)      # remove special characters
    text = re.sub(r"\s+", " ", text).strip()            # remove extra whitespace
    return text



if __name__=="__main__":
    product_id=input("ID of the product: ")
    get_test_reviw(product_id)     # i have already sampled data
    print("Results saved to test_reviews.json")

    get_review(jsonl_path)

