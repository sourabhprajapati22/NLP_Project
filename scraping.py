from myntra_sentiment_analysis.scraping.myntra_scraper import myntra_web_page,scrape_id_list,scrape_review
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import re
import json
import sys
import os
from pathlib import Path


file_current_path=Path(__file__).resolve().parent
sys.exit()


def write_jsonl_line(path, data):
    with open(path, 'a', encoding='utf-8') as f:
        json.dump(data, f)
        f.write('\n')

# JSONL file paths
same_jsonl_path = os.path.join(file_current_path, "myntra_sentiment_analysis/data/raw/reviews_same.jsonl")
diff_jsonl_path = os.path.join(file_current_path, "myntra_sentiment_analysis/data/raw/reviews_diff.jsonl")


# step1: scrape the main page to get the products name ====================================
html_items_content=myntra_web_page()
print("step1 completed---------------------")



# Step2: different products name find =====================================================
soup = BeautifulSoup(html_items_content, "html.parser")

hrefs = [a.get("href") for a in soup.find_all("a") if a.get("href")]

pattern = re.compile(r'^[a-zA-Z-]+$')

filtered_hrefs = []
for href in hrefs:
    if 1 < len(href) < 20 and href.count('/') == 1:
        # If href starts with '/', consider only the part after the slash
        if href.startswith('/'):
            substring = href[1:]
        else:
            substring = href
        
        if pattern.fullmatch(substring):
            filtered_hrefs.append(substring)


results = {}
for p,i in enumerate(filtered_hrefs):
    if p==30:                # for final convert to 40
        break
    eight_digit_numbers=scrape_id_list(i)
    # with open("myntra_page.html", "r", encoding="utf-8") as file:
    #     html_content = file.read()
    # eight_digit_numbers = set(re.findall(r'\b\d{8}\b', html_content))
    if len(eight_digit_numbers)>50:
        print(p,i,len(eight_digit_numbers))
        results[i] = list(eight_digit_numbers)



with open(os.path.join(file_current_path,"myntra_sentiment_analysis/data/raw/results.json"), "w", encoding="utf-8") as json_file:
    json.dump(results, json_file, indent=2)

print("Results saved to results.json")
print("step2 completed---------------------")

# sys.exit()

items=list(results.keys())

# step3: for same product ==================================================================

review_results=[]
review_diff_results=[]
count=0
for product_id in results[items[0]]:
    if count==50: # after code complete convert to 50
        break
    print(product_id)
    html_content=scrape_review(product_id)

    if html_content==None:
        continue
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
        
        
        write_jsonl_line(same_jsonl_path, record)
        if count < 6:
            write_jsonl_line(diff_jsonl_path, record)

        review_count+=1

print("Results saved to reviews_same.jsonl")

# sys.exit()


# step4: for different product in multiple of 10 items============================================
id_review_complete=[]
for k in range(1,10):
    count=0
    for product_id in results[items[k]]:
        if count==5:
            break
        print(product_id)
        html_content=scrape_review(product_id)


        if html_content==None:
            continue
        soup = BeautifulSoup(html_content, "html.parser")

        count+=1
        

        

        data=soup.find_all("div", class_="user-review-userReviewWrapper")

        brand_name=soup.find("h1",class_="product-details-brand").getText()

        # print(brand_name)

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

            write_jsonl_line(diff_jsonl_path, record)
            review_count+=1

print("Results saved to reviews_diff.jsonl")