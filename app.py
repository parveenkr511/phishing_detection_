from flask import Flask, request, render_template
import joblib
import pandas as pd
import re
import requests
from urllib.parse import urlparse
from difflib import SequenceMatcher
from bs4 import BeautifulSoup

app = Flask(__name__)
# Load your trained stacking model (ensure this matches your filename)
model = joblib.load("phishing_url_detection.pkl")

from urllib.parse import urlparse, urlunparse

def normalize_url(url):
    parsed = urlparse(url)
    # Remove trailing slash if path is empty or just '/'
    path = parsed.path if parsed.path != "/" else ""
    # Remove unnecessary parts like params, query, fragment if not needed
    normalized = urlunparse((parsed.scheme, parsed.netloc, path, '', '', ''))
    return normalized.lower()

def extract_features(URL):
    URL = normalize_url(URL)
    features = {
        'url_length': len(URL),
        'num_dots': URL.count('.'),
        'num_hyphens': URL.count('-'),
        'num_at': URL.count('@'),
        'has_https': int(URL.lower().startswith("https")),
        'has_ip': int(bool(re.search(r'\d+\.\d+\.\d+\.\d+', URL))),
        'num_subdirs': URL.count('/'),
        'num_parameters': URL.count('='),
        'num_percent': URL.count('%'),
        'num_www': URL.count('www'),
        'num_digits': sum(c.isdigit() for c in URL),
        'num_letters': sum(c.isalpha() for c in URL),
        # placeholders for content-based features
        'is_live': 0,
        'page_title_length': 0,
        'num_forms': 0,
        'num_links': 0,
        'num_input_fields': 0,
        'has_login_keyword': 0,
        'domain_similarity_score': 0.0
    }

    try:
        resp = requests.get(URL, timeout=5)
        features['is_live'] = int(resp.status_code < 400)

        soup = BeautifulSoup(resp.text, 'html.parser')
        title = soup.title.string.strip() if soup.title and soup.title.string else ""
        features['page_title_length'] = len(title)

        features['num_forms']        = len(soup.find_all('form'))
        features['num_links']        = len(soup.find_all('a'))
        features['num_input_fields'] = len(soup.find_all('input'))

        login_regex = re.compile(r'login|signin|verify|secure|auth', re.IGNORECASE)
        features['has_login_keyword'] = int(
            bool(login_regex.search(URL)) or bool(login_regex.search(resp.text))
        )

        domain = urlparse(URL).netloc.lower()
        norm_title = re.sub(r'[^a-z0-9]', '', title.lower())
        features['domain_similarity_score'] = SequenceMatcher(None, domain, norm_title).ratio()

    except Exception:
        # on any error, leave content-based features at default
        pass

    return features

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    url = ""
    # default to light gray if no submission
    bg_color = "#f4f6f9"

    if request.method == 'POST':
        url = request.form['url'].strip()
        feats = extract_features(url)

        if feats['is_live'] == 0:
            result = "Site Not Live ⚠️"
            bg_color = "#f1c40f"   # yellow
        else:
            df = pd.DataFrame([feats])
            pred = model.predict(df)[0]
            if pred == 1:
                result = "Phishing ❌"
                bg_color = "#e74c3c"  # red
            else:
                result = "Legitimate ✅"
                bg_color = "#2ecc71"  # green

    return render_template(
        'index.html',
        result=result,
        url=url,
        bg_color=bg_color
    )

import os

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
