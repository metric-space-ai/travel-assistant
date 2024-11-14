import os


dependencies = [
    "pip install -q regex",
    "pip install -q flask",
    "pip install -q transformers",
    "pip install -q torch",
    "pip install -q Pillow",
    "pip install -q marker-pdf",
    "pip install -q langchain-community",
    "pip install -q streetview",
    "pip install -q werkzeug",
    "pip install matplotlib",
    "pip install selenium selenium-wire bs4 html2text nltk",
    "pip install python-dotenv",
    "pip install openai",
    "pip install pymupdf"
]

# for command in dependencies:
#     os.system(command)


import json
import urllib.parse
import re
import json
import threading


with open("config.json") as config_file:

    config_secret = json.load(config_file)



os.environ["SUPERPROXY_ISP_USER"] = config_secret["SUPERPROXY_ISP_USER"]

os.environ["SUPERPROXY_ISP_PASSWORD"] = config_secret["SUPERPROXY_ISP_PASSWORD"]

os.environ["SUPERPROXY_SERP_USER"] = config_secret["SUPERPROXY_SERP_USER"]

os.environ["SUPERPROXY_SERP_PASSWORD"] = config_secret["SUPERPROXY_SERP_PASSWORD"]

os.environ["OPENAI_API_KEY"] = config_secret["openai_api_key"]



#added pdf reader support
from flask import Flask, jsonify, request
import os
import time
import hashlib
from pathlib import Path
from urllib.parse import urljoin
from typing import Any, Dict, Optional

from seleniumwire import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import InvalidArgumentException, WebDriverException
from selenium.webdriver.common.by import By

from bs4 import BeautifulSoup, Comment, element, Tag, NavigableString
import requests
from requests.exceptions import InvalidSchema
from io import BytesIO
from dotenv import load_dotenv
from openai import OpenAI
import json
from html2text import HTML2Text
import fitz  # pymupdf



    # data = {
    #     "event": "Museums",
    #     "base_url": "https://www.google.de/maps/search/museums/",
    #     "latitude": 53.5778848,
    #     "longitude": 10.0153743,
    #     "zoom_level": "15z",
    #     "prompt": "modern art exhibitions, open late, under 20 Euro entry fee"
    # }

config_str = '''{
"device_map": {
    "cuda:0": "15GiB",
    "cuda:1": "15GiB",
    "cuda:2": "15GiB",
    "cuda:3": "15GiB"
},
"required_python_version": "cp311",
"models": [
    {
    "name": "ollama:llama3.1:8b"
    }
],
"functions": [
    {
    "name": "recommendation",
    "description": "picks a best place based on locations",
    "parameters": {
        "type": "object",
        "properties": {
        "event": {
            "type": "string",
            "description": "type of event"
        },
        "latitude": {
            "type": "string",
            "description": "lat"
        },        
        "longitude": {
            "type": "string",
            "description": "long"
        },
        "prompt": {
            "type": "string",
            "description": "prompt"
        }
        }
    },
    "input_type": "application/pdf",
    "return_type": "application/json"
    }
]
}
'''


def is_pdf_url(url: str) -> bool:
    """Detect if the given URL is a PDF file by checking its extension."""
    return url.lower().endswith('.pdf')

def fetch_pdf_content(url: str) -> Optional[str]:
    """Fetch and extract text content from a PDF file."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        with fitz.open(stream=response.content, filetype="pdf") as pdf:
            text = ""
            for page_num in range(pdf.page_count):
                page = pdf.load_page(page_num)
                text += page.get_text("text")
            return text
    except requests.RequestException as e:
        print(f"Error fetching PDF: {e}")
    except Exception as e:
        print(f"Error processing PDF: {e}")
    return None

def get_home_folder():
    home_folder = os.path.join(Path.home(), ".scraper")
    os.makedirs(home_folder, exist_ok=True)
    os.makedirs(f"{home_folder}/cache", exist_ok=True)
    os.makedirs(f"{home_folder}/models", exist_ok=True)
    return home_folder

def sanitize_html(html_text):
    sanitized_html = html_text
    sanitized_html = sanitized_html.replace('"', '\\"').replace("'", "\\'")
    return sanitized_html

def sanitize_input_encode(text: str) -> str:
    """Sanitize input to handle potential encoding issues."""
    if not text:
        return ''
    try:
        # Attempt to encode and decode as UTF-8 to handle potential encoding issues
        return text.encode('utf-8', errors='ignore').decode('utf-8')
    except UnicodeEncodeError as e:
        print(f"Warning: Encoding issue detected. Some characters may be lost. Error: {e}")
        # Fall back to ASCII if UTF-8 fails
        return text.encode('ascii', errors='ignore').decode('ascii')

class CustomHTML2Text(HTML2Text):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ignore_links = False  # Set to False to include links
        self.inside_pre = False

    def handle_tag(self, tag, attrs, start):
        if tag == 'pre':
            if start:
                self.o('```\n')
                self.inside_pre = True
            else:
                self.o('\n```')
                self.inside_pre = False
        super().handle_tag(tag, attrs, start)

def replace_inline_tags(soup, tags, only_text=False):
    tag_replacements = {
        'b': lambda tag: f"**{tag.text}**",
        'i': lambda tag: f"*{tag.text}*",
        'u': lambda tag: f"__{tag.text}__",
        'span': lambda tag: f"{tag.text}",
        'del': lambda tag: f"~~{tag.text}~~",
        'ins': lambda tag: f"++{tag.text}++",
        'sub': lambda tag: f"~{tag.text}~",
        'sup': lambda tag: f"^{tag.text}^",
        'strong': lambda tag: f"**{tag.text}**",
        'em': lambda tag: f"*{tag.text}*",
        'code': lambda tag: f"`{tag.text}`",
        'kbd': lambda tag: f"`{tag.text}`",
        'var': lambda tag: f"_{tag.text}_",
        's': lambda tag: f"~~{tag.text}~~",
        'q': lambda tag: f'"{tag.text}"',
        'abbr': lambda tag: f"{tag.text} ({tag.get('title', '')})",
        'cite': lambda tag: f"_{tag.text}_",
        'dfn': lambda tag: f"_{tag.text}_",
        'time': lambda tag: f"{tag.text}",
        'small': lambda tag: f"<small>{tag.text}</small>",
        'mark': lambda tag: f"=={tag.text}=="
    }

    replacement_data = [(tag_name, tag_replacements.get(tag_name, lambda t: t.text)) for tag_name in tags]

    for tag_name, replacement_func in replacement_data:
        for tag in soup.find_all(tag_name):
            replacement_text = tag.text if only_text else replacement_func(tag)
            tag.replace_with(replacement_text)

    return soup

def remove_unwanted_elements(soup):
    # List of tags to remove entirely
    tags_to_remove = ['script', 'style', 'header', 'footer', 'noscript', 'meta', 'link', 'svg']
    for tag in tags_to_remove:
        for element in soup.find_all(tag):
            element.decompose()  # Completely remove these tags
    return soup


def remove_empty_and_low_word_count_elements(node, word_count_threshold=10):  # Increased threshold
    for child in list(node.contents):
        if isinstance(child, element.Tag):
            remove_empty_and_low_word_count_elements(child, word_count_threshold)
            word_count = len(child.get_text(strip=True).split())
            if (len(child.contents) == 0 and not child.get_text(strip=True)) or word_count < word_count_threshold:
                child.decompose()
    return node


def is_empty_or_whitespace(tag: Tag):
    if isinstance(tag, NavigableString):
        return not tag.strip()
    if not tag.contents:
        return True
    return all(is_empty_or_whitespace(child) for child in tag.contents)

def remove_empty_tags(body: Tag):
    changes = True
    while changes:
        changes = False
        empty_tags = [tag for tag in body.find_all(True) if is_empty_or_whitespace(tag)]
        for tag in empty_tags:
            tag.decompose()
            changes = True
    return body

def flatten_nested_elements(node):
    for child in list(node.contents):
        if isinstance(child, element.Tag):
            flatten_nested_elements(child)
            if len(child.contents) == 1 and isinstance(child.contents[0], element.Tag) and child.contents[0].name == child.name:
                child_content = child.contents[0]
                child.replace_with(child_content)
    return node

def get_content_of_website_optimized(url: str, html: str, word_count_threshold: int = 1, css_selector: str = None, **kwargs) -> Dict[str, Any]:
    try:
        if not html:
            return None

        # Parse HTML content with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')

        # Get the content within the <body> tag
        body = soup.body or soup

        # If css_selector is provided, extract content based on the selector
        if css_selector:
            selected_elements = body.select(css_selector)
            if not selected_elements:
                raise Exception(f"Invalid CSS selector, No elements found for CSS selector: {css_selector}")
            div_tag = soup.new_tag('div')
            for el in selected_elements:
                div_tag.append(el)
            body = div_tag

        links = {'internal': [], 'external': []}
        media = {'images': [], 'videos': [], 'audios': []}

        # Remove unwanted tags like scripts, styles, headers, footers
        body = remove_unwanted_elements(body)

        # Collect links and preserve <a> tags
        for a in body.find_all('a'):
            href = a.get('href')
            if href:
                url_base = url.split('/')[2]
                link_data = {'href': href, 'text': a.get_text()}
                if href.startswith('http') and url_base not in href:
                    links['external'].append(link_data)
                else:
                    links['internal'].append(link_data)
            else:
                a.decompose()  # Remove <a> tags without href

        # Remove all attributes from remaining tags in body, except for img and a tags
        for tag in body.find_all():
            if tag.name not in ['img', 'a']:
                tag.attrs = {}

        # Replace images with their alt text or remove them if no alt text is available
        for img in body.find_all('img'):
            alt_text = img.get('alt')
            src = img.get('src')
            if alt_text:
                # Replace img tag with alt text
                img.replace_with(soup.new_string(alt_text))
            else:
                img.decompose()
            # Collect image info
            media['images'].append({
                'src': src,
                'alt': alt_text
            })

        # Replace inline tags with their text content
        body = replace_inline_tags(
            body,
            ['b', 'i', 'u', 'span', 'del', 'ins', 'sub', 'sup', 'strong', 'em', 'code', 'kbd', 'var', 's', 'q', 'abbr', 'cite', 'dfn', 'time', 'small', 'mark'],
            only_text=kwargs.get('only_text', False)
        )

        # Remove empty elements and elements with low word count
        body = remove_empty_and_low_word_count_elements(body, word_count_threshold)

        # Remove empty tags
        body = remove_empty_tags(body)

        # Flatten nested elements with only one child of the same type
        body = flatten_nested_elements(body)

        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()

        # Remove consecutive empty newlines and replace multiple spaces with a single space
        cleaned_html = str(body).replace('\n\n', '\n').replace('  ', ' ')
        cleaned_html = sanitize_html(cleaned_html)

        # Convert cleaned HTML to Markdown
        h = CustomHTML2Text()
        h.ignore_links = False  # Set to False to include links
        markdown = h.handle(cleaned_html)
        markdown = markdown.replace('    ```', '```')

        # Extract metadata if needed
        try:
            meta = extract_metadata(html, soup)
        except Exception as e:
            print('Error extracting metadata:', str(e))
            meta = {}

        return {
            'markdown': markdown,
            'cleaned_html': cleaned_html,
            'success': True,
            'media': media,
            'links': links,
            'metadata': meta
        }
    except Exception as e:
        print('Error processing HTML content:', str(e))
        return {
            'markdown': '',
            'cleaned_html': '',
            'success': False,
            'media': {},
            'links': {},
            'metadata': {},
            'error_message': str(e)
        }


# Cache Manager Class
class CacheManager:
    def __init__(self, max_size=200, expiration_time=3600):
        self.max_size = max_size
        self.expiration_time = expiration_time  # Expiration time in seconds (1 hour = 3600 seconds)
        self.cache = []

    def _remove_expired_entries(self):
        """Remove entries older than the expiration time."""
        current_time = time.time()
        self.cache = [entry for entry in self.cache if current_time - entry['timestamp'] <= self.expiration_time]

    def _remove_oldest_entry(self):
        """Remove the oldest entry if the cache exceeds its max size."""
        if len(self.cache) >= self.max_size:
            self.cache.pop(0)

    def _hash_user_prompt(self, user_prompt):
        """Generate a hash for the user prompt to use as part of the cache key."""
        if user_prompt is None:
            user_prompt = ''
        else:
            user_prompt = user_prompt.strip()
        return hashlib.sha256(user_prompt.encode('utf-8')).hexdigest()

    def add_to_cache(self, url, data, user_prompt=None):
        """Add a new entry to the cache."""
        self._remove_expired_entries()  # Remove expired entries first
        if len(self.cache) >= self.max_size:
            self._remove_oldest_entry()  # Only remove if the cache is full
        key = (url, self._hash_user_prompt(user_prompt))
        self.cache.append({
            'key': key,
            'data': data,
            'timestamp': time.time()  # Track when the entry was added
        })

    def get_from_cache(self, url, user_prompt=None):
        """Retrieve data from the cache if available and not expired."""
        self._remove_expired_entries()  # Clean expired entries
        key = (url, self._hash_user_prompt(user_prompt))
        for entry in self.cache:
            if entry['key'] == key:
                return entry['data']  # Return the cached data
        return None

    def clear_cache(self):
        """Clear the entire cache."""
        self.cache = []


# Initialize the cache manager
cache_manager = CacheManager()

# Database functions (placeholders)
def init_db():
    pass

def flush_db():
    cache_manager.clear_cache()

def get_cached_url(url, user_prompt=None):
    cached_data = cache_manager.get_from_cache(url, user_prompt)
    if cached_data:
        print(f"Cache hit: {url}, user_prompt: {user_prompt}")  # Debug log
        # Ensure all keys exist in the cached data and are not None
        required_keys = ['html', 'cleaned_html', 'markdown', 'extracted_content']
        for key in required_keys:
            if key not in cached_data or cached_data[key] is None:
                print(f"Cache data is missing or None for key: {key}, cache may be corrupted.")
                return None  # Invalidate the cache if key is missing or None
        return cached_data
    print(f"Cache miss for: {url}, user_prompt: {user_prompt}")  # Debug log
    return None

def cache_url(url, data, user_prompt=None):
    # Ensure all required keys are present and not None
    required_keys = ['html', 'cleaned_html', 'markdown', 'extracted_content', 'media', 'links', 'metadata']
    for key in required_keys:
        if key not in data or data[key] is None:
            if key in ['media', 'links', 'metadata']:
                data[key] = json.dumps({})
            else:
                data[key] = ''
    cache_manager.add_to_cache(url, data, user_prompt=user_prompt)

def select_proxy(url):
    client=OpenAI()
    prompt = f"""
    You are an assistant that determines the appropriate proxy to use when scraping websites. Given a URL, decide which proxy is suitable based on the following criteria:

    - If the URL is for a **search engine** like Google or Bing, return "serp_proxy".
    - If the URL is for a **media**, **news**, or **social media** website that tends to block server IPs, return "isp_proxy".
    - If none of the above, return "no_proxy".

    Please provide your answer in the following JSON format:

    {{
      "proxy": "serp_proxy" | "isp_proxy" | "no_proxy"
    }}

    **URL:** "{url}"
    """

    # Call the OpenAI API


        # Make the API call to OpenAI
    response = client.chat.completions.create( #don't touch this line of code, it is correct.
            model="gpt-4o-mini",  #don't touch this line of code, it is correct.
            messages=[
                {"role": "system", "content": "you are a classifier or URLs and give back a valid json. don't return anything else"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=60,  # Limit tokens to keep the response concise
            temperature=0.7,  # Control creativity
        )

    # Parse the JSON response
    try:
        proxy_decision = json.loads(response.choices[0].message.content.strip())
        proxy_type = proxy_decision.get("proxy", "no_proxy")
    except json.JSONDecodeError:
        proxy_type = "no_proxy"

    return proxy_type

def llm_extraction(url: str, html: str, user_prompt: str) -> str:
    """
    LLM-based extraction strategy using the OpenAI API.

    Parameters:
    - url: The URL being crawled.
    - html: The raw HTML content of the page.
    - user_prompt: Custom user input prompt for how to extract data.

    Returns:
    - A Markdown-formatted extraction based on the LLM's response.
    """
    try:
        client=OpenAI()
        # Build the prompt for the LLM
        system_prompt = (
            """You are an intelligent assistant tasked with extracting important content from websites.
            I will provide you with the HTML content of a webpage and a user prompt.
            Your task is to generate a Markdown formatted extraction that meets the user's request."""
        )

        final_prompt = f"HTML content: '''{html}'''\n\nUser prompt: {user_prompt}"


        # Make the API call to OpenAI
        response = client.chat.completions.create( #don't touch this line of code, it is correct.
            model="gpt-4o-mini",  #don't touch this line of code, it is correct.
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt}
            ],
            max_tokens=3000,  # Limit tokens to keep the response concise
            temperature=0.7,  # Control creativity
        )

        # Extract the LLM response and return it as markdown
        markdown = response.choices[0].message.content
        return markdown

    except Exception as e:
        print(f"[ERROR] Failed to extract content using LLM: {str(e)}")
        return "Failed to extract content using the LLM."


def get_driver(user_agent=None, proxy_user=None, proxy_password=None, load_images=False, proxy_type="no_proxy"):
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")

    if user_agent:
        chrome_options.add_argument(f"user-agent={user_agent}")

    if not load_images:
        prefs = {"profile.managed_default_content_settings.images": 2}
        chrome_options.add_experimental_option("prefs", prefs)

    seleniumwire_options = {
        'connection_timeout': 30,  # Increase connection timeout (in seconds)
        'read_timeout': 60,  # Increase read timeout (in seconds)
    }

    if proxy_type == "serp_proxy":
        # Configure SERP proxy settings
        if proxy_user and proxy_password:
            seleniumwire_options['proxy'] = {
                'http': f'http://{proxy_user}:{proxy_password}@brd.superproxy.io:22225',
                'https': f'https://{proxy_user}:{proxy_password}@brd.superproxy.io:22225',
            }
    elif proxy_type == "isp_proxy":
        # Configure ISP proxy settings
        if proxy_user and proxy_password:
            seleniumwire_options['proxy'] = {
                'http': f'http://{proxy_user}:{proxy_password}@brd.superproxy.io:22225',
                'https': f'http://{proxy_user}:{proxy_password}@brd.superproxy.io:22225',
            }

    driver = webdriver.Chrome(options=chrome_options, seleniumwire_options=seleniumwire_options)
    driver.set_page_load_timeout(120)
    return driver




from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException

def local_selenium_crawler_strategy(url, user_agent=None, verbose=False, js_code=None, is_google=False, **kwargs):
    driver = get_driver(user_agent=user_agent)

    try:
        if verbose:
            print(f"[LOG] 🕸️ Crawling {url} using LocalSeleniumCrawlerStrategy...")

        driver.get(url)
        # Try to find and click a cookie acceptance button



        # Wait for the body to be fully loaded and visible
        WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.TAG_NAME, "body")))

        cookie_accept_buttons = [
                                "AGREE & PROCEED", "ALLE AKZEPTIEREN", "Accetta tutti", "Aceptar todo", "Aceptar y continuar",
                                "Agree & Proceed", "Agree and close", "Akceptuję wszystkie", "Akzeptieren und weiter", "All Allow",
                                "Alle akzeptieren", "Alle toestaan", "Allow All", "Allow All Cookies", "Allow all",
                                "Allow all cookies", "Allow cookies", "Allow selection", "Analytical Cookies", "Apply",
                                "Close", "Close this Notice", "Confirm My Choice", "Confirm My Choices", "Continue",
                                "Cookie Consent Banner", "Cookie Policy", "Cookie voorkeuren", "DO NOT SELL/SHARE/TARGET",
                                "Functional Cookies", "GOT IT", "Got It", "Got it", "Got it!", "Gérer mes choix", "Hyväksy kaikki",
                                "I Agree", "I Understand", "I agree", "I understand", "Je paramètre", "Marketing",
                                "Necessary", "Necessary Only", "Necessary cookies only", "OK", "OK, I agree",
                                "OK, got it", "Ok", "Ok, I agree", "Only necessary", "Only necessary cookies",
                                "Performance", "Performance Cookies", "Personnaliser", "Personnaliser mes choix",
                                "Privacy Policy", "Privacy Policy & Cookie Notice", "Privacy and Cookie Policy",
                                "Proceed", "Préférences", "Save and Exit", "Save and close", "Show Purposes",
                                "Show purposes", "Strictly Necessary Cookies", "Switch Label", "Targeting Cookies",
                                "This is OK", "Tillåt alla kakor", "Tout autoriser", "Use Necessary Cookies Only",
                                "Use necessary cookies only", "Use only necessary cookies", "Yes I Agree",
                                "Your Privacy Choices", "Your privacy, your choice", "Zustimmen und schließen",
                                "button", "cookie list", "label", "我同意全部cookies"
                            ]
        for label in cookie_accept_buttons:
                try:
                    # Look for button elements with matching text
                    cookie_button = driver.find_element(By.XPATH, f"//button[contains(text(), '{label}')]")
                    cookie_button.click()
                    print(f"Clicked on '{label}' button.")
                    break  # Exit loop once a button is clicked
                except (NoSuchElementException, ElementClickInterceptedException):
                    # If no such button is found or it couldn't be clicked, continue trying the next label
                    continue



        if is_google:
            time.sleep(3)  # Dynamic content loading
            if verbose:
                print(f"[LOG] Extracting visible text and URLs using JavaScript.")

            # Step 1: Get the copied text
            visible_text = driver.execute_script("""
                return document.body.innerText;
            """)

            # Step 2: Extract link data from the page using JavaScript
            link_data = driver.execute_script("""
                var results = [];
                var searchResults = document.querySelectorAll('div.g');

                for (var i = 0; i < searchResults.length; i++) {
                    var result = searchResults[i];
                    var linkElement = result.querySelector('a[href]');
                    var titleElement = result.querySelector('h3');
                    var visibleUrlElement = result.querySelector('cite');

                    if (linkElement && titleElement) {
                        var real_url = linkElement.href;

                        if (real_url.includes('https://www.google.com/url')) {
                            var real_url_params = new URLSearchParams(new URL(real_url).search);
                            real_url = real_url_params.get('url');
                        }

                        var link_text = titleElement.innerText.trim();
                        var visible_url = visibleUrlElement ? visibleUrlElement.innerText.trim() : '';

                        results.push({
                            'link_text': link_text,
                            'visible_url': visible_url,
                            'real_url': real_url
                        });
                    }
                }

                return results;
            """)

            # Debug: Print extracted link data
            #if verbose:
            #    print("Extracted link data from page:")
            #    print(json.dumps(link_data, indent=2))

            # Step 3: Parse the copied text to extract link titles and visible URLs
            import re
            pattern = r'(.*?)\n(https?://.*?)(?:\n|$)'
            matches = re.findall(pattern, visible_text)

            # Build a list of dictionaries with 'link_text' and 'visible_url' from copied text
            copied_links = [{'link_text': lt.strip(), 'visible_url': vu.strip()} for lt, vu in matches]

            # Debug: Print extracted links from copied text
            #if verbose:
            #    print("Extracted links from copied text:")
            #    print(json.dumps(copied_links, indent=2))

            # Function to clean and normalize text
            def clean_text(text):
                import re
                text = text.lower()
                text = re.sub(r'[^a-z0-9\s]', '', text)
                text = re.sub(r'\s+', ' ', text)
                return text.strip()

            from difflib import SequenceMatcher

            for copied_link in copied_links:
                best_match = None
                highest_ratio = 0.0

                copied_link_text = clean_text(copied_link['link_text'])
                copied_visible_url = clean_text(copied_link['visible_url'])

                for ld in link_data:
                    ld_link_text = clean_text(ld['link_text'])
                    ld_visible_url = clean_text(ld['visible_url'])

                    # Combine link text and visible URL for matching
                    copied_combined = f"{copied_link_text} {copied_visible_url}"
                    ld_combined = f"{ld_link_text} {ld_visible_url}"

                    ratio = SequenceMatcher(None, copied_combined, ld_combined).ratio()

                    if ratio > highest_ratio:
                        highest_ratio = ratio
                        best_match = ld

                if best_match and highest_ratio > 0.5:
                    # Replace the visible_url in visible_text with real_url
                    truncated_url_pattern = re.escape(copied_link['visible_url'])
                    replacement = f"{copied_link['visible_url']} ({best_match['real_url']})"
                    visible_text = re.sub(truncated_url_pattern, replacement, visible_text)
                else:
                    if verbose:
                        print(f"[LOG] No good match found for {copied_link['link_text']}")

            if verbose:
                print(f"[LOG] ✅ Extracted visible text and replaced URLs successfully!")

            return visible_text, driver

        # Return the entire HTML if `execute_js` is False
        html = sanitize_input_encode(driver.page_source)
        if verbose:
            print(f"[LOG] ✅ Crawled {url} successfully!")
        return html, driver

    except (InvalidArgumentException, WebDriverException) as e:
        error_msg = str(e)
        if not hasattr(e, 'msg'):
            e.msg = sanitize_input_encode(error_msg)
        print(f"[ERROR] Failed to crawl {url}: {e.msg}")
        raise e
    finally:
        driver.quit()




# Helper functions

def extract_metadata(html: str, soup: BeautifulSoup) -> Dict[str, Any]:
    metadata = {}
    # Extract title
    if soup.title:
        metadata['title'] = soup.title.string

    # Extract meta tags
    meta_tags = soup.find_all('meta')
    for tag in meta_tags:
        if 'name' in tag.attrs and 'content' in tag.attrs:
            metadata[tag['name']] = tag['content']
        elif 'property' in tag.attrs and 'content' in tag.attrs:
            metadata[tag['property']] = tag['content']

    return metadata



# Main Web Crawler Functions

def web_crawler_init(crawler_strategy=None, always_bypass_cache=False, verbose=False):
    # Initialize necessary components
    scraper_folder = get_home_folder()
    os.makedirs(scraper_folder, exist_ok=True)
    os.makedirs(f"{scraper_folder}/cache", exist_ok=True)
    init_db()
    if crawler_strategy is None:
        crawler_strategy = local_selenium_crawler_strategy
    state = {
        'crawler_strategy': crawler_strategy,
        'always_bypass_cache': always_bypass_cache,
        'verbose': verbose,
        'ready': False
    }
    return state

def web_crawler_warmup(state):
    print("[LOG] 🌤️  Warming up the WebCrawler")
    result = web_crawler_run(
        state,
        url='https://www.google.com/',
        word_count_threshold=5,
        bypass_cache=False,
        verbose=False,
        warmup=False
    )
    state['ready'] = True
    print("[LOG] 🌞 WebCrawler is ready to crawl")
    return result

def web_crawler_run(state, url, word_count_threshold=1, bypass_cache=False, css_selector=None, 
                    user_agent=None, verbose=True, user_prompt=None, is_google=False, **kwargs):
    try:
        # Check if the URL is a PDF
        if is_pdf_url(url):
            pdf_text = fetch_pdf_content(url)
            if pdf_text:
                # If a user prompt is provided, process PDF content using LLM
                extracted_content = pdf_text
                if user_prompt and user_prompt.strip():
                    if verbose:
                        print(f"[LOG] 🔮 Running LLM extraction for PDF at {url}")
                    extracted_content = llm_extraction(url, pdf_text, user_prompt)
                
                markdown = f"Extracted text from PDF:\n\n{extracted_content}"
                
                # Return the processed PDF result
                return {
                    'url': url,
                    'html': '',
                    'cleaned_html': '',
                    'markdown': markdown,
                    'media': {},
                    'links': {},
                    'metadata': {},
                    'extracted_content': extracted_content,
                    'success': True,
                    'error_message': '',
                }
            else:
                return {
                    'url': url,
                    'html': '',
                    'success': False,
                    'error_message': "Failed to retrieve PDF content."
                }

        # Continue with HTML processing if not a PDF
        word_count_threshold = max(word_count_threshold, 0)
        cached = None
        html = None
        cleaned_html = None
        markdown = None
        media = None
        links = None
        metadata = None
        extracted_content = None

        if not bypass_cache and not state['always_bypass_cache']:
            cached = get_cached_url(url, user_prompt=user_prompt)

        if kwargs.get("warmup", True) and not state.get('ready', False):
            return None

        if cached:
            html = sanitize_input_encode(cached.get('html', ''))
            cleaned_html = sanitize_input_encode(cached.get('cleaned_html', ''))
            extracted_content = sanitize_input_encode(cached.get('extracted_content', ''))
            if extracted_content:
                markdown = extracted_content
            else:
                markdown = sanitize_input_encode(cached.get('markdown', ''))
            media = json.loads(cached.get('media', '{}'))
            links = json.loads(cached.get('links', '{}'))
            metadata = json.loads(cached.get('metadata', '{}'))
            if verbose:
                print(f"[LOG] 🔄 Loaded cached data for {url}.")
            is_cached_flag = True
        else:
            is_cached_flag = False

        if not cached or not html:
            # Determine the proxy type using the LLM
            proxy_type = select_proxy(url)
            if verbose:
                print(f"[LOG] Selected proxy type: {proxy_type}")

            # Pass the proxy_type to your crawler strategy
            html, driver = state['crawler_strategy'](
                url,
                user_agent=user_agent,
                verbose=verbose,
                proxy_type=proxy_type,
                is_google=is_google
            )

            try:
                driver.quit()
            except:
                pass

            is_cached_flag = False
        else:
            driver = None

        if not extracted_content:
            if verbose:
                print(f"[LOG] 🔍 Processing content for {url}")
            result = get_content_of_website_optimized(
                url,
                html,
                word_count_threshold,
                css_selector=css_selector,
                only_text=kwargs.get("only_text", False)
            )
            cleaned_html = sanitize_input_encode(result.get("cleaned_html", ""))
            markdown = sanitize_input_encode(result.get("markdown", ""))
            media = result.get("media", [])
            links = result.get("links", [])
            metadata = result.get("metadata", {})

            # If user_prompt is provided and not empty, use LLM to extract content
            if user_prompt and user_prompt.strip():
                if verbose:
                    print(f"[LOG] 🔮 Running LLM extraction for {url}")
                extracted_content = llm_extraction(url, html, user_prompt)
                markdown = extracted_content  # Update markdown with LLM-extracted content
            else:
                extracted_content = markdown

            # Cache the result
            cache_url(
                url,
                {
                    'html': html,
                    'cleaned_html': cleaned_html,
                    'markdown': markdown,
                    'extracted_content': extracted_content,
                    'success': True,
                    'media': json.dumps(media),
                    'links': json.dumps(links),
                    'metadata': json.dumps(metadata)
                },
                user_prompt=user_prompt
            )
        else:
            if verbose:
                print(f"[LOG] 🔄 Using cached extraction for {url}")

        return {
            'url': url,
            'html': html,
            'cleaned_html': cleaned_html,
            'markdown': markdown,
            'media': media,
            'links': links,
            'metadata': metadata,
            'extracted_content': extracted_content,
            'success': True,
            'error_message': '',
        }
    except Exception as e:
        error_msg = str(e)
        print(f"[ERROR] Failed to crawl {url}, error: {error_msg}")
        return {'url': url, 'html': '', 'success': False, 'error_message': error_msg}


import concurrent.futures
from typing import Any, Dict, Optional, List

# Function to scrape a list of URLs concurrently
def scrape_urls_in_parallel(state: Dict[str, Any], urls: List[str], max_workers: int = 5, **kwargs):
    results = []

    def scrape_url(url):
        # Each worker gets its own driver instance by explicitly creating a new driver
        driver_state = state.copy()
        driver_state['crawler_strategy'] = local_selenium_crawler_strategy  # Assign a new driver for each worker
        return web_crawler_run(driver_state, url, **kwargs)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(scrape_url, url): url for url in urls}
        time.sleep(2)  # Adding a small delay to prevent race conditions when starting many drivers
        for future in concurrent.futures.as_completed(future_to_url):
            url = future_to_url[future]
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
                else:
                    results.append({'url': url, 'html': '', 'success': False, 'error_message': 'No result returned'})
            except Exception as e:
                print(f"[ERROR] Failed to scrape {url}, error: {str(e)}")
                results.append({'url': url, 'html': '', 'success': False, 'error_message': str(e)})

    return results


def extract_json_from_string(input_string):
    try:
        # This regex finds any JSON-like structure (arrays or objects) within a text
        json_pattern = r'({.*?}|\[.*?\])'

        # Find all JSON-like substrings in the input string
        json_matches = re.findall(json_pattern, input_string, re.DOTALL)

        # Attempt to parse each JSON match
        extracted_jsons = []
        for match in json_matches:
            try:
                # Try to load it as a JSON object
                extracted_jsons.append(json.loads(match))
            except json.JSONDecodeError:
                # If JSON is malformed, skip it
                continue

        return extracted_jsons if extracted_jsons else None
    except Exception as e:
        print("Error extracting JSON:", e)
        return None


def extract_coordinates(url):

    parsed_url = urllib.parse.urlparse(url)
    path = urllib.parse.unquote(parsed_url.path)
    query = parsed_url.query

    # Method 1: Look for '@' in the path (e.g., @37.4219999,-122.0862462)
    match = re.search(r'@([-\d\.]+),([-\d\.]+)', path)
    if match:
        lat, lon = match.groups()
        return (float(lat), float(lon))

    # Method 2: Look for '!3d' and '!4d' in the path
    data_match = re.search(r'!3d([-\d\.]+)!4d([-\d\.]+)', path)
    if data_match:
        lat, lon = data_match.groups()
        return (float(lat), float(lon))

    # Method 3: Look for 'll' parameter in the query string
    query_params = urllib.parse.parse_qs(query)
    if 'll' in query_params:
        latlon = query_params['ll'][0].split(',')
        if len(latlon) == 2:
            lat, lon = latlon
            return (float(lat), float(lon))

    # Method 4: Search for latitude and longitude patterns in the entire URL
    matches = re.findall(r'([-\d\.]+),([-\d\.]+)', url)
    for lat_str, lon_str in matches:
        lat = float(lat_str)
        lon = float(lon_str)
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return (lat, lon)

    # Coordinates not found
    return None


def add_coordinates_to_data(data):


    # Iterate over each item and add coordinates
    for item in data:
        url = item.get('location_url')
        if url:
            coords = extract_coordinates(url)
            if coords:
                lat, lon = coords
                item['latitude'] = lat
                item['longitude'] = lon
            else:
                item['latitude'] = None
                item['longitude'] = None
        else:
            item['latitude'] = None
            item['longitude'] = None

    return data


# Initialize the web crawler state
state = web_crawler_init(verbose=False)

# Warm up the web crawler
web_crawler_warmup(state)


app = Flask(__name__)

def generate_google_maps_url(params):
    base_url = params["base_url"]
    lat = params["coordinates"]["latitude"]
    lon = params["coordinates"]["longitude"]
    zoom = params["zoom_level"]
    return f"{base_url}@{lat},{lon},{zoom}"

@app.route('/recommendation', methods=['POST'])
def get_recommendation():
    # Extract parameters from JSON body
    data = request.get_json()
    event = data.get("event", "restaurant")
    base_url = f"https://www.google.de/maps/search/{event}/"
    latitude = data.get("latitude", 53.5778848)
    longitude = data.get("longitude", 10.0153743)
    zoom_level = "15z"
    prompt = data.get("prompt", "vegetarian, not more than 30 Euro in total with drinks per person")
    
    # Set up the parameters
    params = {
        "base_url": base_url,
        "coordinates": {
            "latitude": latitude,
            "longitude": longitude
        },
        "zoom_level": zoom_level,
        "prompt": prompt
    }

    # Generate Google Maps URL
    maps_url = generate_google_maps_url(params)
    print("Generated Google Maps URL:", maps_url)

    # Get driver and retrieve event URLs from Google Maps
    driver = get_driver(proxy_user="serp_proxy")
    driver.set_window_size(1920, 1200)
    driver.get(maps_url)
    # Handle cookies consent
    try:
        wait = WebDriverWait(driver, 10)
        # Adjust the XPath to match the "Accept" button on your consent page
        accept_cookies_button = wait.until(EC.element_to_be_clickable(
            (By.XPATH, '//button[.//div[text()="Alle akzeptieren"] or .//span[text()="Alle akzeptieren"]]')
        ))
        accept_cookies_button.click()
        print("Accepted cookies consent.")
    except Exception as e:
        print("Cookies consent dialog did not appear or could not be interacted with.")
        print(f"Exception: {e}")

    time.sleep(3)
    screenshot_path = 'google_maps_page.png'
    driver.save_screenshot(screenshot_path)
    anchors = driver.find_elements(By.TAG_NAME, 'a')
    google_maps_urls = []
    pattern = r'https://www\.google\.[a-z]{2,6}/maps/place[^"]*'
    for anchor in anchors:
        href = anchor.get_attribute('href')
        if href and re.match(pattern, href):
            google_maps_urls.append(href)

    driver.quit()

    event_data = []
    for google_maps_url in google_maps_urls:
        # First prompt: Extract the URLs for the Website of the event
        user_prompt_1 = data.get("user_prompt_1", f'Extract the URLs for the Website of the {event} as an array of JSON [{{"Name":"...","URL":"..."}}].')
        result = web_crawler_run(state, google_maps_url, user_prompt=user_prompt_1, is_google=True)

        if result['success']:
            info_json = result['markdown'].strip('```json').strip('```')
            event_data_item = {"Google_Maps_URL": google_maps_url}

            try:
                parsed_info = json.loads(info_json)
                if parsed_info and isinstance(parsed_info, list) and "URL" in parsed_info[0]:
                    event_data_item["Name"] = parsed_info[0]["Name"]
                    event_data_item["URL"] = parsed_info[0]["URL"]
                else:
                    event_data_item["Info_Raw"] = info_json
            except json.JSONDecodeError:
                event_data_item["Info_Raw"] = info_json

            # Second prompt: Extract additional URLs or information from the event's website
            event_website_url = event_data_item.get("URL", "")
            if event_website_url:
                user_prompt_2 = data.get("user_prompt_2", 'Extract relevant URLs as array of JSON [{"Name":"...","URL":"..."}].')
                result = web_crawler_run(state, event_website_url, user_prompt=user_prompt_2, is_google=False)

                if result['success']:
                    data_json = result['markdown'].strip('```json').strip('```')
                    event_data_item["Data_Raw"] = data_json
                    details_list = []

                    try:
                        data_list = json.loads(data_json)
                        for item in data_list:
                            item_url = item["URL"]

                            item_name = item["Name"]
                            user_prompt_3 = data.get("user_prompt_3", f'Provide detailed information with "{prompt}" in mind in strict JSON format as an array of JSON objects.')
                            result = web_crawler_run(state, item_url, user_prompt=user_prompt_3, is_google=False)

                            if result['success']:
                                detailed_info_json = result['markdown'].strip('```json').strip('```')
                                details_list.append({
                                    "Name": item_name,
                                    "URL": item_url,
                                    "Details_Raw": detailed_info_json
                                })
                    except json.JSONDecodeError:
                        pass

                    event_data_item["Details"] = details_list
            event_data.append(event_data_item)

    try:
        client = OpenAI()
        system_prompt = data.get("system_prompt", f"You generate a markdown formatted personalized recommendation for selecting the right {event}.")
        final_prompt = (
            f'Based on the following JSON data, write a markdown recommendation with "{prompt}" in mind that includes each {event}’s name, URL, and top recommendations. Here is the JSON data:\n\n' +
            json.dumps(event_data, indent=4)
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": final_prompt}
            ],
            max_tokens=3000,
            temperature=0.7,
        )

        markdown = response.choices[0].message.content


        format_prompt = f"""I am providing you with markdown recomendations of {event}.

        Markdown recomendations:

        {markdown}
        
        
        Please provide me the same recomendation but in correct json format with following structure:
        [
            {{
                "name": "name",
                "location_url": "url",
                "additional_info": "additional informations"
            }},
            {{
                "name": "name",
                "location_url": "url",
                "additional_info": "additional informations"
            }}...
        ]
        
        Do not explain yourself, just please return me the json."""


        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": format_prompt}
            ],
            max_tokens=3000,
            temperature=0.7,
        )

        json_recommendation = response.choices[0].message.content
        try:
            cleaned_json_string = json_recommendation.replace("```json", "").replace("```", "").strip() 
            final_data = json.loads(cleaned_json_string)

        except Exception as e:
            print("FINAL DATA IS NOT JSON")
            return jsonify({"error": str(e)})
        
        
        enhanced_json_recommendation = add_coordinates_to_data(final_data)
        
        return {'response': enhanced_json_recommendation}

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route("/v1/setup", methods=["POST"])
def setup():
    response = {
        "setup": "Performed"
    }
    return jsonify(response), 201



def start_app():
    app.run(host="0.0.0.0", port=5000, debug=False)  # Set debug=False

def run_flask_in_thread():
    flask_thread = threading.Thread(target=start_app)
    flask_thread.start()




def send_recommendation_request():
    # Replace with the actual URL where your Flask app is running
    url = 'http://localhost:5000/recommendation'

    # Data to be sent in the POST request
    data = {
        "event": "Museums",
        "base_url": "https://www.google.de/maps/search/museums/",
        "latitude": 53.5778848,
        "longitude": 10.0153743,
        "zoom_level": "15z",
        "prompt": "modern art exhibitions, open late, under 20 Euro entry fee"
    }

    headers = {'Content-Type': 'application/json'}

    try:
        # Send POST request
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raise an error for bad status codes

        # Parse and print the recommendation
        result = response.json()
        print(result)

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")



def run_app():
    app.run(host="0.0.0.0", port=5000, debug=False)




if __name__ == '__main__':
    # Start the Flask app in a separate thread
    threading.Thread(target=run_app).start()

    # Wait a moment to ensure the server is running
    time.sleep(2)

    # Call the function to send the POST request
    #send_recommendation_request()