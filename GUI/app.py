from flask import Flask, render_template, request
import FeatureExtraction
import pickle
import warnings
import xgboost as xgb
import pandas as pd
from urllib.parse import urlparse
import os
import requests
from requests.exceptions import RequestException, SSLError, ConnectionError, Timeout
import socket
import urllib3
import pyshorteners
import tldextract
from transformers import pipeline
import re

app = Flask(__name__, static_url_path='/static')

# Suppress warnings
warnings.filterwarnings('ignore')

# Create instances
feature_extractor = FeatureExtraction.FeatureExtraction()

# Suppress SSL verification warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Path to local BART model directory
BART_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../bart-large-mnli-local')

# Load zero-shot classifier (offline)
try:
    zero_shot_classifier = pipeline('zero-shot-classification', model=BART_MODEL_PATH)
except Exception as e:
    zero_shot_classifier = None
    print(f"Could not load BART model: {e}")

def get_model_path(model_name):
    """
    Get the correct path for model files regardless of how the application is run
    """
    # Get the directory where the current file (app.py) is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, model_name)

def check_url_connectivity(url):
    """
    Check if the URL is accessible and has valid HTTPS connection
    Returns: (bool, str) - (is_accessible, message)
    """
    try:
        # Add https:// if not present
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url

        # Set a reasonable timeout
        timeout = 15
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # First try to resolve the domain
        domain = urlparse(url).netloc.lower()
        try:
            socket.gethostbyname(domain)
        except socket.gaierror:
            return False, "Domain name could not be resolved. Please check if the URL is correct."

        # Try HTTPS connection first with SSL verification disabled
        try:
            response = requests.get(url, timeout=timeout, headers=headers, verify=False)
            if response.status_code == 200:
                return True, "Website is accessible"
            else:
                # Try with www. prefix
                if not domain.startswith('www.'):
                    www_url = url.replace('://', '://www.')
                    try:
                        response = requests.get(www_url, timeout=timeout, headers=headers, verify=False)
                        if response.status_code == 200:
                            return True, "Website is accessible with www prefix"
                    except:
                        pass
                
                return False, f"Website returned status code {response.status_code}. The website might be temporarily unavailable."
        except SSLError:
            # If SSL fails, try HTTP
            try:
                url = url.replace('https://', 'http://')
                response = requests.get(url, timeout=timeout, headers=headers)
                if response.status_code == 200:
                    return True, "Website is accessible via HTTP"
                else:
                    return False, f"Website returned status code {response.status_code}. The website might be temporarily unavailable."
            except RequestException:
                return False, "Website is not accessible. The domain might be inactive or the server is not responding."
    except (ConnectionError, Timeout):
        return False, "Could not connect to the website. The domain might be inactive or the server is not responding."
    except Exception as e:
        print(f"Connection error details: {str(e)}")  # Print detailed error for debugging
        return False, f"Error checking website: {str(e)}"

# Load XGBoost model
try:
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        tree_method='hist',
        n_jobs=1
    )
    xgb_model_path = get_model_path('XGBoostModel_12000.sav')
    xgb_model.load_model(xgb_model_path)
    print("XGBoost model loaded successfully")
except Exception as e:
    print(f"Error loading XGBoost model: {e}")
    xgb_model = None

# Load Random Forest model
# try:
#     rf_model_path = get_model_path('RFmodel_12000.sav')
#     with open(rf_model_path, 'rb') as f:
#         rf_model = pickle.load(f)
#     print("Random Forest model loaded successfully")
# except Exception as e:
#     print(f"Error loading Random Forest model: {e}")
#     rf_model = None

def preprocess_data(data):
    """Preprocess the data to match model's expected format"""
    # Ensure all required features are present
    required_features = [
        'long_url',
        'having_@_symbol',
        'redirection_//_symbol',
        'prefix_suffix_seperation',
        'sub_domains',
        'having_ip_address',
        'shortening_service',
        'https_token',
        'web_traffic',
        'domain_registration_length',
        'dns_record',
        'age_of_domain',
        'statistical_report'
    ]
    
    # Drop non-numeric columns
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    
    # Handle any missing values
    numeric_data = numeric_data.fillna(0)
    
    # Ensure all required features are present
    for feature in required_features:
        if feature not in numeric_data.columns:
            numeric_data[feature] = 0
    
    # Reorder columns to match model's expected order
    return numeric_data[required_features]

@app.route('/')
def index():
    return render_template("landing.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/detectURL')
def detectURL():
    return render_template("home.html")

def get_reasons(features):
    reasons = []
    feature_descriptions = {
        'long_url': 'URL length is suspiciously long',
        'having_@_symbol': 'URL contains @ symbol (high risk)',
        'redirection_//_symbol': 'URL contains suspicious redirection (high risk)',
        'prefix_suffix_seperation': 'Domain contains hyphens',
        'sub_domains': 'URL has multiple subdomains',
        'having_ip_address': 'URL contains IP address (high risk)',
        'shortening_service': 'URL uses URL shortening service (high risk)',
        'https_token': 'URL has suspicious HTTPS tokens',
        'web_traffic': 'Suspicious web traffic patterns',
        'domain_registration_length': 'Domain registration period is short',
        'dns_record': 'No DNS record found',
        'age_of_domain': 'Domain is very new',
        'statistical_report': 'Statistical analysis indicates suspicious patterns'
    }
    
    url = request.form['url']
    
    # Check for number-for-letter substitutions
    number_letter_map = {
        '0': 'o',
        '1': 'i',
        '3': 'e',
        '4': 'a',
        '5': 's',
        '7': 't',
        '8': 'b'
    }
    
    suspicious_chars = []
    for char in url:
        if char in number_letter_map:
            suspicious_chars.append(f"'{char}' (mimicking '{number_letter_map[char]}')")
    
    if suspicious_chars:
        reasons.append("⚠️ HIGH RISK: This URL uses numbers to mimic letters")
        reasons.append("Suspicious character substitutions detected:")
        for char in suspicious_chars:
            reasons.append(f"- {char}")
        reasons.append("This is a common phishing technique to make malicious URLs look legitimate")
    
    # Check typo-squatting
    typo_check = feature_extractor.check_typo_squatting(request.form['url'])
    if typo_check['is_typo_squatting']:
        reasons.append(f"⚠️ HIGH RISK: This appears to be a typo-squatting attempt")
        reasons.append(f"This domain is trying to impersonate {typo_check['company_name']}'s official website ({typo_check['original_domain']})")
        reasons.append("Common typo-squatting techniques detected:")
        reasons.append("- Using numbers instead of letters (e.g., '0' instead of 'o')")
        reasons.append("- Using similar-looking characters")
        reasons.append("- Slight misspellings of the original domain")
        reasons.append(f"Please visit the official website: {typo_check['original_domain']}")
        return reasons
    
    # Add warning for suspicious TLD or domain
    if feature_extractor.check_suspicious_tld(request.form['url']):
        reasons.append("WARNING: This website uses a suspicious top-level domain commonly associated with malicious sites")
    if feature_extractor.check_suspicious_domain(request.form['url']):
        reasons.append("WARNING: This domain contains suspicious patterns that may indicate phishing")
    
    # Show feature warnings based on ML model features
    for feature, value in features.items():
        if value == 1 and feature in feature_descriptions:
            reasons.append(feature_descriptions[feature])
    
    # Add additional context
    if '.gov' in request.form['url']:
        reasons.append("This appears to be a government website (.gov domain)")
    elif not reasons:
        reasons.append("No suspicious features detected")
    
    return reasons

def is_trusted_domain(url):
    """Check if the domain is from a trusted TLD"""
    trusted_tlds = ['.gov.np', '.edu.np']
    parsed = urlparse(url)
    domain = parsed.netloc.lower()
    return any(domain.endswith(tld) for tld in trusted_tlds)

@app.route('/getURL', methods=['GET', 'POST'])
def getURL():
    if request.method == 'POST':
        reasons = []  # Always initialize reasons
        raw_input = request.form.get('url', '').strip()
        # Extract the first URL from the input
        url_pattern = r'(https?://[^\s]+)'
        match = re.search(url_pattern, raw_input)
        url = match.group(0) if match else ''
        # Remove the URL from the text for NLP analysis
        text = re.sub(url, '', raw_input).strip() if url else raw_input
        print(f"\nProcessing URL: {url}")
        if text:
            print(f"Processing text: {text}")
        # Validate URL first
        is_valid, result = feature_extractor.validate_url(url)
        if not is_valid:
            return render_template("home.html", error="Invalid URL", reasons=[result])
        url = result
        print(f"Validated URL: {url}")
        original_url = url
        final_url = None
        is_shortened = False
        suspicious_redirect = False
        redirect_domain = None
        try:
            response = requests.head(url, allow_redirects=True, timeout=10)
            final_url = response.url
            # Extract registered domains for comparison
            original_domain = tldextract.extract(url).registered_domain
            final_domain = tldextract.extract(final_url).registered_domain
            is_shortened = final_domain != original_domain
            # If not shortened, keep final_url as url
            if not is_shortened:
                final_url = url
            if is_shortened:
                print(f"Shortened/Redirected URL detected. Original: {original_domain}")
                print(f"Final destination: {final_domain}")
                url = final_url
                redirect_domain = urlparse(final_url).netloc.lower()
        except Exception as e:
            print(f"Error resolving shortened URL: {str(e)}")
            try:
                shortener = pyshorteners.Shortener()
                expanded_url = None
                for service in shortener.available_shorteners:
                    try:
                        expanded_url = getattr(shortener, service).expand(url)
                        if expanded_url and expanded_url != url:
                            break
                    except Exception:
                        continue
                if expanded_url and expanded_url != url:
                    # Extract registered domains for comparison
                    original_domain = tldextract.extract(url).registered_domain
                    expanded_domain = tldextract.extract(expanded_url).registered_domain
                    is_shortened = expanded_domain != original_domain
                    # If not shortened, keep final_url as url
                    if not is_shortened:
                        final_url = url
                    else:
                        final_url = expanded_url
                    if is_shortened:
                        print(f"Shortened URL detected via pyshorteners. Original: {original_domain}")
                        print(f"Final destination: {expanded_domain}")
                        url = expanded_url
                        redirect_domain = urlparse(expanded_url).netloc.lower()
            except Exception as e:
                print(f"Error using pyshorteners: {str(e)}")
                if not redirect_domain:
                    is_shortened = False
                    final_url = url
        # At this point, url is set to the final destination if there was a redirect/shortener
        # Only run ML and feature extraction on the final destination
        data, phishing_reasons = feature_extractor.getAttributess(url)
        data = preprocess_data(data)
        xgb_proba = xgb_model.predict_proba(data)[0]
        # rf_proba = rf_model.predict_proba(data)[0]
        # weighted_proba = (0.6 * xgb_proba) + (0.4 * rf_proba)
        weighted_proba = xgb_proba
        # Print detailed probabilities in backend
        print("\nModel Probabilities:")
        print("-------------------")
        print(f"XGBoost Model:")
        print(f"  - Legitimate: {xgb_proba[0]*100:.1f}%")
        print(f"  - Phishing: {xgb_proba[1]*100:.1f}%")
        # print(f"\nRandom Forest Model:")
        # print(f"  - Legitimate: {rf_proba[0]*100:.1f}%")
        # print(f"  - Phishing: {rf_proba[1]*100:.1f}%")
        # print(f"\nCombined Weighted Probability (60% XGBoost, 40% RF):")
        # print(f"  - Legitimate: {weighted_proba[0]*100:.1f}%")
        # print(f"  - Phishing: {weighted_proba[1]*100:.1f}%")
        print("-------------------")
        # Change threshold: legitimate only if > 57%
        # is_legitimate = weighted_proba[0] > 0.57
        is_legitimate = xgb_proba[0] > 0.57
        is_phishing = not is_legitimate
        confidence = "HIGH" if abs(weighted_proba[1] - 0.5) > 0.3 else "MEDIUM"
        trust_level = "HIGH" if is_trusted_domain(url) else "NORMAL"
        value = None
        # NLP phishing intent detection
        nlp_result = None
        nlp_label = None
        nlp_score = 0
        text_message = None
        url_message = None
        if text and zero_shot_classifier:
            labels = ["phishing", "not phishing", "spam", "benign"]
            nlp_result = zero_shot_classifier(text, candidate_labels=labels)
            nlp_label = nlp_result['labels'][0]
            nlp_score = nlp_result['scores'][0]
            # User-friendly text analysis message
            print(f"NLP model prediction: {nlp_label} (score: {nlp_score:.2f})")
            if nlp_label in ['phishing', 'spam'] and nlp_score > 0.7:
                text_message = f"Text analysis: ⚠️ This message is likely phishing or spam!"
            elif nlp_label == 'benign' and nlp_score > 0.7:
                text_message = f"Text analysis: ✅ This message appears safe."
            elif nlp_label == 'not phishing' and nlp_score > 0.7:
                text_message = f"Text analysis: ✅ This message does not appear to be phishing."
            else:
                if nlp_label in ['phishing', 'spam']:
                    text_message = f"Text analysis: ⚠️ This message may be suspicious."
                else:
                    text_message = f"Text analysis: ⚠️ This message may be suspicious."
        # URL analysis message
        if url:
            if is_phishing:
                url_message = f"URL analysis: ⚠️ {confidence} RISK: This URL is Phishing"
            else:
                if trust_level == "HIGH":
                    url_message = f"URL analysis: ✅ This URL is Legitimate (Trusted domain: {urlparse(url).netloc})"
                else:
                    url_message = f"URL analysis: ✅ This URL is Legitimate"
        # Compose final reasons/messages
        reasons = []
        if text and url:
            if text_message:
                reasons.append(text_message)
            if url_message:
                reasons.append(url_message)
        elif text and not url:
            if text_message:
                reasons.append(text_message)
        elif url and not text:
            if url_message:
                reasons.append(url_message)
        # Add detailed reasons from phishing_reasons if URL was analyzed
        if url and phishing_reasons:
            reasons.extend(phishing_reasons)
        if is_phishing and url and not phishing_reasons:
            reasons.append("Multiple indicators suggest this is a phishing website")
        return render_template("home.html", error=None, reasons=reasons)

if __name__ == "__main__":
    app.run(debug=True)