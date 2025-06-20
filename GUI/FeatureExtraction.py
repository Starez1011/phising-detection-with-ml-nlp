# coding: utf-8

# packages
import pandas as pd
from urllib.parse import urlparse
import re
from bs4 import BeautifulSoup
import whois
import urllib.request
import time
import socket
from urllib.error import HTTPError
from datetime import  datetime
import ssl
import requests
from tld import get_tld
import tldextract
import numpy as np
from difflib import SequenceMatcher
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import os
import pyshorteners
import dns.resolver
import dns.exception
from datetime import datetime, timedelta


class FeatureExtraction:
    def __init__(self):
        self.suspicious_tlds = ['tk', 'ml', 'ga', 'cf', 'gq', 'xyz']
        self.suspicious_domains = ['login', 'signin', 'account', 'secure', 'webscr', 'banking']
        self.legitimate_tlds = ['com', 'org', 'net', 'edu', 'gov', 'co', 'io']
        self.legitimate_domains = ['google', 'microsoft', 'apple', 'amazon', 'facebook']
        self.web_traffic_cache = {}
        self.url_shortener_cache = {}
        self.shortener = pyshorteners.Shortener()
        self._load_web_traffic_data()

    def _load_web_traffic_data(self):
        """Load web traffic data from Tranco dataset"""
        try:
            # Get the absolute path to the dataset
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            dataset_path = os.path.join(project_root, 'web_traffic_dataset', 'tranco_top_1M.csv')
            
            # Load the entire dataset
            self.web_traffic_data = pd.read_csv(dataset_path, 
                                              names=['rank', 'domain'])
            # Convert domains to lowercase for case-insensitive matching
            self.web_traffic_data['domain'] = self.web_traffic_data['domain'].str.lower()
            print("Web traffic data loaded successfully")
        except Exception as e:
            print(f"Error loading web traffic data: {str(e)}")
            self.web_traffic_data = pd.DataFrame(columns=['rank', 'domain'])

    def _check_web_traffic(self, domain):
        """Check if domain has significant web traffic"""
        try:
            # Check cache first
            if domain in self.web_traffic_cache:
                return self.web_traffic_cache[domain]

            # Remove www. if present
            if domain.startswith('www.'):
                domain = domain[4:]

            # Check for suspicious patterns in domain
            suspicious_patterns = [
                r'\d+',  # Contains numbers
                r'[^a-zA-Z0-9.-]',  # Contains special characters
                r'[a-z]{2,}\d+[a-z]{2,}',  # Letters followed by numbers followed by letters
                r'\d+[a-z]{2,}\d+',  # Numbers followed by letters followed by numbers
            ]
            
            # If domain matches suspicious patterns, mark as suspicious
            if any(re.search(pattern, domain) for pattern in suspicious_patterns):
                self.web_traffic_cache[domain] = False
                return False

            # Check if domain is in the dataset
            domain_match = self.web_traffic_data[self.web_traffic_data['domain'] == domain.lower()]
            
            if not domain_match.empty:
                rank = domain_match.iloc[0]['rank']
                # Consider domains in top 1M as having significant traffic
                has_traffic = rank <= 1000000
                self.web_traffic_cache[domain] = has_traffic
                return has_traffic
            
            # For domains not in dataset, check if they look legitimate
            # Check for common TLDs and domain patterns
            tld = domain.split('.')[-1].lower()
            if tld in self.legitimate_tlds:
                # Check if domain name looks legitimate (no numbers, reasonable length)
                domain_name = domain.split('.')[0]
                if (len(domain_name) <= 20 and  # Reasonable length
                    not any(c.isdigit() for c in domain_name) and  # No numbers
                    not any(c in '_-' for c in domain_name)):  # No special characters
                    self.web_traffic_cache[domain] = True
                    return True
            
            # If we can't determine legitimacy, mark as suspicious
            self.web_traffic_cache[domain] = False
            return False
            
        except Exception as e:
            print(f"Error checking web traffic: {str(e)}")
            return False  # Return False (suspicious) on error to be safe

    def _check_shortened_url(self, url):
        """Check if a shortened URL is legitimate or suspicious"""
        try:
            # Check cache first
            if url in self.url_shortener_cache:
                return self.url_shortener_cache[url]

            # Get the domain
            domain = urlparse(url).netloc.lower()
            
            # Try to expand the URL using pyshorteners
            try:
                expanded_url = self.shortener.expand(url)
                if expanded_url and expanded_url != url:
                    final_url = expanded_url
                    final_domain = urlparse(final_url).netloc.lower()

                    # Check if final destination is suspicious
                    is_suspicious = False
                    reasons = []

                    # Check for HTTPS
                    if not final_url.startswith('https://'):
                        is_suspicious = True
                        reasons.append("Final destination does not use HTTPS")

                    # Check for suspicious patterns in final domain
                    if self._has_suspicious_patterns(final_domain):
                        is_suspicious = True
                        reasons.append("Final destination has suspicious domain patterns")

                    # Check if final domain is in our web traffic data
                    if not self._check_web_traffic(final_domain):
                        is_suspicious = True
                        reasons.append("Final destination has no significant web traffic")

                    # Check if the final domain is significantly different from the original
                    if self._is_significant_redirect(domain, final_domain):
                        is_suspicious = True
                        reasons.append("URL redirects to a significantly different domain")

                    # Check domain age of final destination
                    age = self._get_domain_age(final_domain)
                    if age and age < 180:  # Less than 6 months old
                        is_suspicious = True
                        reasons.append("Final destination domain is less than 6 months old")

                    # Check registration length of final destination
                    reg_length = self._get_domain_registration_length(final_domain)
                    if reg_length and reg_length < 365:  # Less than 1 year
                        is_suspicious = True
                        reasons.append("Final destination domain registration period is less than 1 year")

                    # Check DNS records of final destination
                    if not self._has_dns_record(final_domain):
                        is_suspicious = True
                        reasons.append("Final destination has no DNS records")

                    # Identify the shortener service
                    service = self._identify_shortener_service(domain)

                    result = {
                        'is_legitimate': not is_suspicious,
                        'service': service,
                        'final_url': final_url,
                        'reasons': reasons
                    }
                    self.url_shortener_cache[url] = result
                    return result

            except Exception as e:
                print(f"Error expanding URL: {str(e)}")
                # If pyshorteners fails, try direct HTTP request
                try:
                    response = requests.head(url, allow_redirects=True, timeout=5)
                    final_url = response.url
                    if final_url != url:
                        final_domain = urlparse(final_url).netloc.lower()
                        
                        is_suspicious = False
                        reasons = []

                        if not final_url.startswith('https://'):
                            is_suspicious = True
                            reasons.append("Final destination does not use HTTPS")

                        if self._has_suspicious_patterns(final_domain):
                            is_suspicious = True
                            reasons.append("Final destination has suspicious domain patterns")

                        if not self._check_web_traffic(final_domain):
                            is_suspicious = True
                            reasons.append("Final destination has no significant web traffic")

                        if self._is_significant_redirect(domain, final_domain):
                            is_suspicious = True
                            reasons.append("URL redirects to a significantly different domain")

                        # Check domain age of final destination
                        age = self._get_domain_age(final_domain)
                        if age and age < 180:
                            is_suspicious = True
                            reasons.append("Final destination domain is less than 6 months old")

                        # Check registration length of final destination
                        reg_length = self._get_domain_registration_length(final_domain)
                        if reg_length and reg_length < 365:
                            is_suspicious = True
                            reasons.append("Final destination domain registration period is less than 1 year")

                        # Check DNS records of final destination
                        if not self._has_dns_record(final_domain):
                            is_suspicious = True
                            reasons.append("Final destination has no DNS records")

                        service = self._identify_shortener_service(domain)

                        result = {
                            'is_legitimate': not is_suspicious,
                            'service': service,
                            'final_url': final_url,
                            'reasons': reasons
                        }
                        self.url_shortener_cache[url] = result
                        return result

                except Exception as e:
                    print(f"Error checking shortened URL: {str(e)}")
                    return {
                        'is_legitimate': False,
                        'service': 'Unknown',
                        'final_url': url,
                        'reasons': ['Error checking URL destination']
                    }

            # If not a URL shortener
            return {
                'is_legitimate': True,
                'service': 'Not a URL shortener',
                'final_url': url,
                'reasons': []
            }

        except Exception as e:
            print(f"Error in _check_shortened_url: {str(e)}")
            return {
                'is_legitimate': False,
                'service': 'Unknown',
                'final_url': url,
                'reasons': ['Error processing URL']
            }

    def _is_potential_shortener(self, domain):
        """Check if a domain is likely to be a URL shortener based on characteristics"""
        try:
            # Check domain characteristics
            domain_length = len(domain)
            parts = domain.split('.')
            
            # Short domains are more likely to be shorteners
            if domain_length <= 10:
                return True
                
            # Domains with very short parts are likely shorteners
            if any(len(part) <= 3 for part in parts):
                return True
                
            # Domains with many parts are less likely to be shorteners
            if len(parts) > 3:
                return False
                
            # Check for common shortener characteristics
            characteristics = [
                len(parts) == 2 and len(parts[0]) <= 3,  # e.g., t.co
                len(parts) == 2 and parts[0].endswith('ly'),  # e.g., bit.ly
                len(parts) == 3 and len(parts[0]) <= 3,  # e.g., goo.gl
                len(parts) == 2 and 'url' in parts[0],  # e.g., tinyurl
                len(parts) == 2 and 'link' in parts[0],  # e.g., shortlink
                len(parts) == 2 and parts[0].endswith('to'),  # e.g., amzn.to
            ]
            
            return any(characteristics)
        except:
            return False

    def _has_suspicious_patterns(self, domain):
        """Check for suspicious patterns in a domain"""
        try:
            # Check for patterns that are commonly used in phishing
            patterns = [
                r'\d+',  # Contains numbers
                r'[^a-zA-Z0-9.-]',  # Contains special characters
                r'[a-z]{2,}\d+[a-z]{2,}',  # Letters followed by numbers followed by letters
                r'\d+[a-z]{2,}\d+',  # Numbers followed by letters followed by numbers
            ]
            
            return any(re.search(pattern, domain) for pattern in patterns)
        except:
            return True  # If we can't check, assume suspicious

    def _identify_shortener_service(self, domain):
        """Identify the URL shortener service based on domain characteristics"""
        try:
            # Try to identify using pyshorteners
            for service in self.shortener.available_shorteners:
                if service in domain:
                    return f"{service.title()} URL Shortener"
            
            # If not found in pyshorteners, use domain characteristics
            parts = domain.split('.')
            tld = parts[-1]
            name = parts[0]
            
            # Generate service name based on characteristics
            if len(parts) == 2:
                if len(name) <= 3:
                    return f"{name.upper()} URL Shortener"
                elif name.endswith('ly'):
                    return f"{name[:-2].title()} URL Shortener"
                elif 'url' in name:
                    return f"{name.replace('url', '').title()} URL Shortener"
                elif 'link' in name:
                    return f"{name.replace('link', '').title()} URL Shortener"
                elif name.endswith('to'):
                    return f"{name[:-2].title()} URL Shortener"
            
            return 'URL Shortener'
        except:
            return 'Unknown Service'

    def _is_significant_redirect(self, original_domain, final_domain):
        """Check if the redirect is to a significantly different domain"""
        try:
            # Remove www. if present
            original_domain = original_domain.replace('www.', '')
            final_domain = final_domain.replace('www.', '')

            # If domains are exactly the same, it's not significant
            if original_domain == final_domain:
                return False

            # Check domain relationships
            original_parts = original_domain.split('.')
            final_parts = final_domain.split('.')
            
            # Check if one is a subdomain of the other
            if final_domain.endswith('.' + original_domain) or original_domain.endswith('.' + final_domain):
                return False
                
            # Check if they share a common parent domain
            if len(original_parts) >= 2 and len(final_parts) >= 2:
                if original_parts[-2:] == final_parts[-2:]:
                    return False
                    
            # Check domain similarity
            similarity = self._calculate_domain_similarity(original_domain, final_domain)
            return similarity < 0.5  # If domains are less than 50% similar, consider it significant
            
        except:
            return True  # If we can't determine, assume it's significant

    def _calculate_domain_similarity(self, domain1, domain2):
        """Calculate similarity between two domains"""
        try:
            # Remove TLDs for comparison
            domain1 = '.'.join(domain1.split('.')[:-1])
            domain2 = '.'.join(domain2.split('.')[:-1])
            
            # Calculate Levenshtein distance
            distance = self._levenshtein_distance(domain1, domain2)
            max_length = max(len(domain1), len(domain2))
            
            # Convert to similarity score (0 to 1)
            return 1 - (distance / max_length)
        except:
            return 0

    def _levenshtein_distance(self, s1, s2):
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
            
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
            
        return previous_row[-1]

    def _is_suspicious_subdomain(self, subdomain):
        """Dynamically check if a subdomain is suspicious using tldextract"""
        try:
            if not subdomain:  # No subdomain
                return False
                
            # Split into parts for multi-level subdomain analysis
            subdomain_parts = subdomain.split('.')
            
            # Analyze each subdomain level
            for part in subdomain_parts:
                # Skip empty parts
                if not part:
                    continue
                    
                # Check for suspicious characteristics
                suspicious_indicators = [
                    # Length checks
                    len(part) < 3,  # Very short subdomains
                    len(part) > 20,  # Unusually long subdomains
                    
                    # Pattern checks
                    bool(re.search(r'\d+', part)),  # Contains numbers
                    bool(re.search(r'[^a-zA-Z0-9-]', part)),  # Contains special characters
                    bool(re.search(r'[a-z]{2,}\d+[a-z]{2,}', part)),  # Letters-numbers-letters pattern
                    bool(re.search(r'\d+[a-z]{2,}\d+', part)),  # Numbers-letters-numbers pattern
                    
                    # Dynamic pattern analysis
                    bool(re.search(r'(?:login|signin|account|secure|webscr|verify|confirm|update|validate)', part)),
                    bool(re.search(r'(?:admin|administrator|support|help|service)', part)),
                    bool(re.search(r'(?:bank|pay|money|transfer|transaction)', part)),
                    
                    # Suspicious combinations
                    bool(re.search(r'(?:login|signin).*(?:secure|verify)', part)),
                    bool(re.search(r'(?:account|user).*(?:verify|confirm)', part)),
                    bool(re.search(r'(?:bank|pay).*(?:secure|verify)', part))
                ]
                
                if any(suspicious_indicators):
                    return True
                    
                # Check for unusual character distribution
                char_distribution = {}
                for char in part:
                    char_distribution[char] = char_distribution.get(char, 0) + 1
                
                # Check for repeated characters (potential typosquatting)
                if any(count > 3 for count in char_distribution.values()):
                    return True
                    
                # Check for mixed case in subdomain (unusual for legitimate sites)
                if part != part.lower() and part != part.upper():
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error checking subdomain: {str(e)}")
            return True  # If we can't check, assume suspicious

    def _validate_trusted_domain(self, domain):
        """Validate domains with trusted TLDs for suspicious subdomains using tldextract"""
        try:
            # Use tldextract to properly parse the domain
            extracted = tldextract.extract(domain)
            
            # Check if it's a trusted TLD
            trusted_tlds = ['gov', 'edu', 'org', 'mil']
            tld = extracted.suffix.lower()
            second_level = extracted.domain.lower()
            
            # Check for trusted TLD patterns dynamically
            is_trusted = (
                tld in trusted_tlds or
                (second_level and any(f"{tld}.{ext}" in f"{second_level}.{tld}" for ext in ['np', 'gov', 'edu']))
            )
            
            if is_trusted:
                # Check each subdomain level
                if extracted.subdomain:
                    subdomain_parts = extracted.subdomain.split('.')
                    for part in subdomain_parts:
                        if part and self._is_suspicious_subdomain(part):
                            print(f"⚠️ HIGH RISK: Suspicious subdomain '{part}' detected in trusted domain")
                            return False
                            
                # Try to resolve the domain to verify it exists
                try:
                    socket.gethostbyname(domain)
                except socket.gaierror:
                    # If domain doesn't resolve, it's suspicious
                    print(f"⚠️ HIGH RISK: Domain does not resolve: {domain}")
                    return False
                        
            return True
            
        except Exception as e:
            print(f"Error validating trusted domain: {str(e)}")
            return False

    def getAttributess(self, url):
        """Extract features from URL and return them along with any phishing reasons"""
        try:
            # Validate URL first
            is_valid, validated_url = self.validate_url(url)
            if not is_valid:
                return None, [validated_url]  # Return error message as reason
            
            url = validated_url
            print(f"Processing URL: {url}")
            
            # Extract features
            features = {}
            phishing_reasons = []
            
            # Always extract registered domain and subdomain for original and final URLs
            orig_ext = tldextract.extract(url)
            original_registered_domain = orig_ext.registered_domain
            original_subdomain = orig_ext.subdomain
            original_full_domain = f"{original_subdomain}.{original_registered_domain}" if original_subdomain else original_registered_domain
            
            final_url = self._resolve_shortened_url(url)
            final_ext = tldextract.extract(final_url)
            final_registered_domain = final_ext.registered_domain
            final_subdomain = final_ext.subdomain
            final_full_domain = f"{final_subdomain}.{final_registered_domain}" if final_subdomain else final_registered_domain
            
            # Detect redirect/shortening if registered domain changes or subdomain changes (excluding www)
            def normalize_subdomain(sub):
                return sub if sub and sub != 'www' else ''
            orig_norm_sub = normalize_subdomain(original_subdomain)
            final_norm_sub = normalize_subdomain(final_subdomain)
            is_shortened = (final_registered_domain != original_registered_domain) or (orig_norm_sub != final_norm_sub)
            
            if is_shortened:
                print(f"Shortened/Redirected URL detected. Original: {original_registered_domain}")
                print(f"Final destination: {final_registered_domain}")
                url = final_url
                domain = final_full_domain
                subdomain = final_subdomain
            else:
                domain = original_full_domain
                subdomain = original_subdomain
            
            # Always check for suspicious subdomains (not just for trusted TLDs)
            if subdomain and self._is_suspicious_subdomain(subdomain):
                phishing_reasons.append("⚠️ HIGH RISK: Suspicious subdomain detected")
                features = {k: 1 for k in [
                    'long_url', 'having_@_symbol', 'redirection_//_symbol',
                    'prefix_suffix_seperation', 'sub_domains', 'having_ip_address',
                    'https_token', 'web_traffic', 'domain_registration_length',
                    'dns_record', 'age_of_domain', 'statistical_report',
                    'shortening_service', 'suspicious_subdomain'
                ]}
                print("⚠️ HIGH RISK: URL classified as phishing due to suspicious subdomain")
                return pd.DataFrame([features]), phishing_reasons
            
            features['suspicious_subdomain'] = 0
            
            # Basic URL features
            features['long_url'] = 1 if len(url) > 54 else 0
            if features['long_url']:
                phishing_reasons.append("URL is unusually long")
            
            # Check for @ symbol
            features['having_@_symbol'] = 1 if '@' in url else 0
            if features['having_@_symbol']:
                phishing_reasons.append("URL contains @ symbol (high risk)")
            
            # Check for redirection - only if registered domains are different
            features['redirection_//_symbol'] = 1 if is_shortened else 0
            
            # Check for prefix-suffix separation (use full domain)
            features['prefix_suffix_seperation'] = 1 if '-' in domain and len(domain.split('-')) > 1 else 0
            if features['prefix_suffix_seperation']:
                phishing_reasons.append("Domain contains multiple hyphens")
            
            # Check subdomains (use full domain)
            subdomain_count = len(domain.split('.'))
            features['sub_domains'] = 1 if subdomain_count > 4 else 0
            if features['sub_domains']:
                phishing_reasons.append("URL has excessive subdomains")
            
            # Check for IP address (use full domain)
            features['having_ip_address'] = 1 if self._is_ip_address(domain) else 0
            if features['having_ip_address']:
                phishing_reasons.append("URL contains IP address (high risk)")
            
            # Check for HTTPS token - only if it's in the domain itself, not the protocol
            features['https_token'] = 1 if 'https' in domain else 0
            if features['https_token']:
                phishing_reasons.append("Domain contains 'https' (suspicious)")
            
            # Check web traffic (use registered domain)
            has_traffic = self._check_web_traffic(final_registered_domain if is_shortened else original_registered_domain)
            features['web_traffic'] = 0 if has_traffic else 1
            if not has_traffic:
                phishing_reasons.append("Domain has suspicious patterns or no significant web traffic")
            
            # Check domain registration length (use registered domain)
            reg_length = self._get_domain_registration_length(final_registered_domain if is_shortened else original_registered_domain)
            features['domain_registration_length'] = 1 if reg_length and reg_length < 365 else 0
            if features['domain_registration_length']:
                phishing_reasons.append("Domain registration period is less than 1 year")
            
            # Check DNS record (use registered domain)
            features['dns_record'] = 1 if not self._has_dns_record(final_registered_domain if is_shortened else original_registered_domain) else 0
            if features['dns_record']:
                phishing_reasons.append("No DNS record found")
            
            # Check domain age (use registered domain)
            age = self._get_domain_age(final_registered_domain if is_shortened else original_registered_domain)
            features['age_of_domain'] = 1 if age and age < 180 else 0
            if features['age_of_domain']:
                phishing_reasons.append("Domain is less than 6 months old")
            
            # Statistical report
            features['statistical_report'] = 1 if self._is_suspicious_pattern(url) else 0
            if features['statistical_report']:
                phishing_reasons.append("URL contains suspicious patterns")
            
            # Check for suspicious TLD
            if self.check_suspicious_tld(url):
                phishing_reasons.append("WARNING: This website uses a suspicious top-level domain commonly associated with malicious sites")
            
            # Check for suspicious domain patterns
            if self.check_suspicious_domain(url):
                phishing_reasons.append("WARNING: This domain contains suspicious patterns that may indicate phishing")
            
            # Mark if it was a shortened URL - only if registered domains are different
            features['shortening_service'] = 1 if is_shortened else 0
            if is_shortened:
                phishing_reasons.append(f"Final destination: {final_url}")
            
            return pd.DataFrame([features]), phishing_reasons
        except Exception as e:
            print(f"Error in getAttributess: {str(e)}")
            return None, [f"Error processing URL: {str(e)}"]

    def _is_ip_address(self, hostname):
        """Check if the hostname is an IP address"""
        try:
            parts = hostname.split('.')
            return len(parts) == 4 and all(part.isdigit() and 0 <= int(part) <= 255 for part in parts)
        except:
            return False

    def _is_shortening_service(self, url):
        """Check if URL is from a known URL shortening service"""
        shorteners = ['bit.ly', 't.co', 'goo.gl', 'tinyurl.com', 'is.gd', 'cli.gs', 'ow.ly', 'yfrog.com', 'migre.me', 'ff.im', 'tiny.cc', 'url4.eu', 'tr.im', 'twit.ac', 'su.pr', 'twurl.nl', 'snipurl.com', 'short.to', 'BudURL.com', 'ping.fm', 'post.ly', 'Just.as', 'bkite.com', 'snipr.com', 'fic.kr', 'loopt.us', 'doiop.com', 'twitthis.com', 'htxt.it', 'alturl.com', 'tiny.pl', 'urlzen.com', 'migre.me', 'xlinkz.info', 'metamark.net', 'sn.im', 'short.ie', 'kl.am', 'wp.me', 'rubyurl.com', 'om.ly', 'to.ly', 'bit.do', 't.co', 'lnkd.in', 'db.tt', 'qr.ae', 'adf.ly', 'goo.gl', 'bitly.com', 'cur.lv', 'tiny.pl', 'ow.ly', 'bit.ly', 'adcrun.ch', 'ity.im', 'q.gs', 'is.gd', 'po.st', 'bc.vc', 'twitthis.com', 'htxt.it', 'alturl.com', 'tiny.pl', 'urlzen.com', 'migre.me', 'xlinkz.info', 'metamark.net', 'sn.im', 'short.ie', 'kl.am', 'wp.me', 'rubyurl.com', 'om.ly', 'to.ly', 'bit.do', 't.co', 'lnkd.in', 'db.tt', 'qr.ae', 'adf.ly', 'goo.gl', 'bitly.com', 'cur.lv', 'tiny.pl', 'ow.ly', 'bit.ly', 'adcrun.ch', 'ity.im', 'q.gs', 'is.gd', 'po.st', 'bc.vc']
        return any(shortener in url.lower() for shortener in shorteners)

    def _has_dns_record(self, hostname):
        """Check if the hostname has DNS records using dnspython"""
        try:
            # Try to resolve A record
            try:
                dns.resolver.resolve(hostname, 'A')
                return True
            except dns.resolver.NXDOMAIN:
                return False
            except dns.resolver.NoAnswer:
                # If no A record, try MX record
                try:
                    dns.resolver.resolve(hostname, 'MX')
                    return True
                except (dns.resolver.NXDOMAIN, dns.resolver.NoAnswer):
                    return False
        except Exception as e:
            print(f"DNS resolution error: {str(e)}")
            return False

    def _is_suspicious_pattern(self, url):
        """Check for suspicious patterns in the URL"""
        try:
            # Get the domain
            domain = urlparse(url).netloc.lower()
            
            # Check if it's a potential shortener
            if self._is_potential_shortener(domain):
                # If it's a shortener, check its final destination
                final_url = self._resolve_shortened_url(url)
                if final_url != url:
                    # Check the final destination instead
                    return self._is_suspicious_pattern(final_url)
                return False
            
            # Check for patterns that are commonly used in phishing
            patterns = [
                r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}',  # IP address
                r'[!@#$%^&*()_+=<>?/\\|~`]',  # Only truly suspicious special characters
                r'(?:login|signin|account|secure|webscr)(?:[^a-zA-Z]|$)',  # Suspicious keywords at word boundaries
                r'\.(tk|ml|ga|cf|gq|xyz)$',  # Suspicious TLDs
                r'(?:\.|\/)(?:php|asp|jsp|exe|bat|cmd|sh|bash)(?:\.|\/|$)',  # Suspicious file extensions
                r'(?:\.|\/)(?:wp-|wp_|wp\.|wordpress)',  # WordPress-related paths
                r'(?:\.|\/)(?:admin|administrator|login|signin|signup|register|account|secure|webscr)(?:\.|\/|$)'  # Suspicious paths
            ]
            
            # Check if the domain is in our web traffic data
            if not self._check_web_traffic(domain):
                return True
                
            return any(re.search(pattern, url, re.IGNORECASE) for pattern in patterns)
        except:
            return True  # If we can't check, assume suspicious

    def check_suspicious_tld(self, url):
        """Check if the URL uses a suspicious TLD"""
        try:
            domain = urlparse(url).netloc.lower()
            tld = domain.split('.')[-1]
            return tld in self.suspicious_tlds
        except Exception as e:
            print(f"Error in check_suspicious_tld: {str(e)}")
            return False

    def check_suspicious_domain(self, url):
        """Check if the domain contains suspicious patterns"""
        try:
            domain = urlparse(url).netloc.lower()
            return any(pattern in domain for pattern in self.suspicious_domains)
        except Exception as e:
            print(f"Error in check_suspicious_domain: {str(e)}")
            return False

    def validate_url(self, url):
        """Validate and clean the URL"""
        try:
            # Add http:// if not present
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            
            # Parse the URL
            parsed = urlparse(url)
            
            # Check if the URL is valid
            if not all([parsed.scheme, parsed.netloc]):
                return False, "Invalid URL format. Please enter a valid URL."
            
            return True, url
        except Exception as e:
            print(f"Error in validate_url: {str(e)}")
            return False, "Invalid URL format. Please enter a valid URL."

    def _get_domain_age(self, domain):
        """Get domain age using python-whois"""
        try:
            w = whois.whois(domain)
            if w.creation_date:
                if isinstance(w.creation_date, list):
                    creation_date = w.creation_date[0]
                else:
                    creation_date = w.creation_date
                
                age = datetime.now() - creation_date
                return age.days
            return None
        except Exception as e:
            print(f"WHOIS lookup error: {str(e)}")
            return None

    def _get_domain_registration_length(self, domain):
        """Get domain registration length using python-whois"""
        try:
            w = whois.whois(domain)
            if w.expiration_date:
                if isinstance(w.expiration_date, list):
                    expiration_date = w.expiration_date[0]
                else:
                    expiration_date = w.expiration_date
                
                if w.creation_date:
                    if isinstance(w.creation_date, list):
                        creation_date = w.creation_date[0]
                    else:
                        creation_date = w.creation_date
                    
                    registration_length = expiration_date - creation_date
                    return registration_length.days
            return None
        except Exception as e:
            print(f"WHOIS lookup error: {str(e)}")
            return None

    def _resolve_shortened_url(self, url):
        """Resolve a shortened URL to its final destination"""
        try:
            # Always try to resolve the final destination, regardless of the shortener's subdomain
            try:
                response = requests.head(url, allow_redirects=True, timeout=5)
                final_url = response.url
                # Normalize URLs for comparison
                original_url = url.rstrip('/')
                final_url = final_url.rstrip('/')
                # If the only difference is http vs https, return original URL
                if original_url.replace('http://', 'https://') == final_url or \
                   final_url.replace('https://', 'http://') == original_url:
                    return url
                # If the only difference is a trailing slash, return original URL
                if original_url.rstrip('/') == final_url.rstrip('/'):
                    return url
                # If URLs are exactly the same after normalization, return original URL
                if original_url == final_url:
                    return url
                if final_url != url:
                    return final_url
            except requests.exceptions.RequestException as e:
                print(f"HTTP expansion failed: {str(e)}")
                # If HTTP request fails, it might be a suspicious domain
                if "getaddrinfo failed" in str(e):
                    return url  # Return original URL if domain doesn't exist
            # If HTTP request fails, try pyshorteners
            try:
                # Initialize shortener with all available services
                shortener = pyshorteners.Shortener()
                # Try each available service
                for service in shortener.available_shorteners:
                    try:
                        expanded_url = getattr(shortener, service).expand(url)
                        if expanded_url and expanded_url != url:
                            # Normalize URLs for comparison
                            original_url = url.rstrip('/')
                            expanded_url = expanded_url.rstrip('/')
                            # If the only difference is http vs https, return original URL
                            if original_url.replace('http://', 'https://') == expanded_url or \
                               expanded_url.replace('https://', 'http://') == original_url:
                                return url
                            # If the only difference is a trailing slash, return original URL
                            if original_url.rstrip('/') == expanded_url.rstrip('/'):
                                return url
                            # If URLs are exactly the same after normalization, return original URL
                            if original_url == expanded_url:
                                return url
                            return expanded_url
                    except:
                        continue
            except Exception as e:
                print(f"Pyshorteners expansion failed: {str(e)}")
            return url  # Return original URL if no expansion possible
        except Exception as e:
            print(f"Error resolving shortened URL: {str(e)}")
            return url

    