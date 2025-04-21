from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import time
import re

# scrape the main page to get the products name

def myntra_web_page():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=["--disable-http2"])
        context = browser.new_context(
            extra_http_headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/115.0 Safari/537.36"
            }
        )
        page = context.new_page()
        print("Chromium launched successfully!")
        url = "https://www.myntra.com"

        try:
            page.goto(url, timeout=9000, wait_until="domcontentloaded") #90000
            print("Page loaded successfully!")

            html_content = page.content()

            soup=BeautifulSoup(html_content,'html.parser')
            print(soup.find('li', class_="product-base"))


            # with open("myntra_page.html", "w", encoding="utf-8") as file:
            #     file.write(html_content)
            # print("HTML content saved as myntra_page.html")

        except Exception as e:
            print("Error occurred while navigating to URL:", e)
        browser.close()


    return html_content


# all products id list scrape

# def scrape_id_list(data_link):
#     with sync_playwright() as p:
#         browser = p.chromium.launch(headless=False, args=["--disable-http2"])
#         context = browser.new_context(
#             extra_http_headers={
#                 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
#                               "AppleWebKit/537.36 (KHTML, like Gecko) "
#                               "Chrome/115.0 Safari/537.36"
#             }
#         )
#         page = context.new_page()
#         url = f"https://www.myntra.com/{data_link}"

#         try:
#             page.goto(url, timeout=9000, wait_until="domcontentloaded") #90000
#             print("Page loaded successfully!")

#             html_content = page.content()

#             soup=BeautifulSoup(html_content,'html.parser')

#             with open("myntra_page.html", "w", encoding="utf-8") as file:
#                 file.write(html_content)
#             print("HTML content saved as myntra_page.html")

#         except Exception as e:
#             print("Error occurred while navigating to URL:", e)
#         browser.close()
#     # return html_content

def scrape_id_list(data_link):
    data=set()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            extra_http_headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/115.0 Safari/537.36"
            }
        )
        page = context.new_page()
        url = f"https://www.myntra.com/{data_link}"

        try:
            page.goto(url, timeout=90000, wait_until="domcontentloaded")
            print("Page loaded successfully!")

            page_num = 1
            while True:
                # Save current page
                html_content = page.content()
                # with open(f"myntra_page_{page_num}.html", "w", encoding="utf-8") as file:
                #     file.write(html_content)
                print(f"Saved page {page_num}")

                data.update(set(re.findall(r'\b\d{8}\b', html_content)))

                # Try clicking the Next button using JavaScript
                next_button = page.query_selector("li.pagination-next")
                if not next_button:
                    print("Next button not found. Exiting.")
                    break

                # Check if it's disabled (no event listener or class shows inactive)
                class_attr = next_button.get_attribute("class")
                if "disabled" in class_attr:
                    print("Next button is disabled. Exiting.")
                    break

                # Click using JS to trigger internal navigation handler
                page.evaluate("el => el.click()", next_button)
                page.wait_for_timeout(2000)  # Small wait for DOM updates
                page.wait_for_load_state("domcontentloaded")
                page_num += 1

                # Optional: stop after 5 pages
                if page_num > 10:
                    break

        except Exception as e:
            print("Error:", e)
        finally:
            browser.close()
    return data


# myntra scrape review



# def review_count(product_id):


def scrape_review(product_id):
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, args=["--disable-http2"])
        context = browser.new_context(
            extra_http_headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                              "AppleWebKit/537.36 (KHTML, like Gecko) "
                              "Chrome/115.0 Safari/537.36"
            }
        )
        page = context.new_page()
        url = f"https://www.myntra.com/reviews/{product_id}"

        try:
            page.goto(url, timeout=180000, wait_until="domcontentloaded")
            print("Page loaded successfully!")

            review_html=page.content()
            review_soup=BeautifulSoup(review_html,'html.parser')

            data=review_soup.find_all("div", class_="user-review-userReviewWrapper")
            # print(data)
            if not data:
                return None


            review_count=review_soup.find("div",class_="detailed-reviews-headline").getText(separator="||").split("||")[1]
            print('Review Count: ',review_count)
            if int(review_count)<540:  # for safe with image review only it's skip
                return None


            max_scroll_attempts = 140  # Limit to avoid infinite loops
            
            # previous_count=0
            for i in range(max_scroll_attempts):
                # Scroll a little instead of jumping to the bottom
                page.evaluate("window.scrollBy(0, 500)")  
                time.sleep(1)  # Wait for new reviews to load

                # Count current reviews
                reviews = page.query_selector_all("div[class*='review']")  # Adjust class based on Myntra's structure
                current_count = len(reviews)
                
                # print(f"Loaded Reviews: {current_count}")

                # if (i==4) and (previous_count==current_count):
                #     break
                # previous_count=current_count


            # Extract final HTML content
            html_content = page.content()

            # with open("myntra_page.html", "w", encoding="utf-8") as file:
            #     file.write(html_content)
            # print("HTML content saved as myntra_page.html")

        except Exception as e:
            print("Error occurred while navigating to URL:", e)
        browser.close()

        return html_content