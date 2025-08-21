import requests
import json
import time
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET
import re
import sqlite3
import threading
from urllib.parse import urljoin
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class Filing:
    """Data class for SEC filing information"""
    cik: str
    company_name: str
    form_type: str
    filing_date: str
    accession_number: str
    primary_document: str
    filing_url: str

@dataclass
class FinancialMetrics:
    """Data class for financial metrics"""
    revenue: Optional[float] = None
    total_debt: Optional[float] = None
    shareholders_equity: Optional[float] = None
    cash_flow_operations: Optional[float] = None
    total_assets: Optional[float] = None
    current_assets: Optional[float] = None
    current_liabilities: Optional[float] = None
    net_income: Optional[float] = None
    filing_date: Optional[str] = None
    period_end_date: Optional[str] = None

@dataclass
class FinancialRatios:
    """Data class for calculated financial ratios"""
    debt_to_equity: Optional[float] = None
    current_ratio: Optional[float] = None
    roa: Optional[float] = None  # Return on Assets
    roe: Optional[float] = None  # Return on Equity
    debt_ratio: Optional[float] = None
    asset_turnover: Optional[float] = None

class RateLimiter:
    """Thread-safe rate limiter for SEC EDGAR API"""
    
    def __init__(self, max_requests: int = 10, time_window: int = 1):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        """Wait if rate limit would be exceeded"""
        with self.lock:
            now = time.time()
            # Remove requests older than time_window
            self.requests = [req_time for req_time in self.requests 
                           if now - req_time < self.time_window]
            
            if len(self.requests) >= self.max_requests:
                # Need to wait
                oldest_request = min(self.requests)
                wait_time = self.time_window - (now - oldest_request) + 0.1
                if wait_time > 0:
                    logger.info(f"Rate limiting: waiting {wait_time:.2f} seconds")
                    time.sleep(wait_time)
            
            self.requests.append(now)

class EDGARCollector:
    """
    Comprehensive SEC EDGAR data collector with rate limiting, caching, and financial analysis.
    
    This class provides methods to:
    - Fetch company filings from SEC EDGAR API
    - Extract and parse financial data from XBRL filings
    - Calculate key financial ratios
    - Cache data to minimize API calls
    - Handle errors and rate limiting gracefully
    """
    
    def __init__(self, user_agent: str = "CreditTech Intelligence Platform contact@example.com", 
                 cache_dir: str = "./cache"):
        """
        Initialize EDGAR collector with rate limiting and caching.
        
        Args:
            user_agent: User agent string required by SEC (must include contact info)
            cache_dir: Directory for caching responses
        """
        self.base_url = "https://data.sec.gov"
        self.user_agent = user_agent
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize rate limiter (SEC allows max 10 requests/second)
        self.rate_limiter = RateLimiter(max_requests=10, time_window=1)
        
        # Initialize session with retries and proper headers
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'Accept-Encoding': 'gzip, deflate',
            'Host': 'data.sec.gov'
        })
        
        # Initialize SQLite cache database
        self._init_cache_db()
        
        # Company CIK mapping cache
        self.cik_cache = {}
        
        logger.info(f"EDGARCollector initialized with user agent: {self.user_agent}")
    
    def _init_cache_db(self):
        """Initialize SQLite database for caching"""
        self.cache_db_path = self.cache_dir / "edgar_cache.db"
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        
        # Create tables for caching
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS filings_cache (
                cache_key TEXT PRIMARY KEY,
                data TEXT,
                timestamp REAL,
                expires_at REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS financial_data_cache (
                filing_url TEXT PRIMARY KEY,
                financial_data TEXT,
                timestamp REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS cik_mapping (
                ticker TEXT PRIMARY KEY,
                cik TEXT,
                company_name TEXT,
                timestamp REAL
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _get_cache_key(self, *args) -> str:
        """Generate cache key from arguments"""
        key_string = "|".join(str(arg) for arg in args)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _make_request(self, url: str, params: Optional[Dict] = None) -> Optional[Dict]:
        """
        Make rate-limited request to SEC EDGAR API
        
        Args:
            url: API endpoint URL
            params: Query parameters
            
        Returns:
            JSON response data or None if failed
        """
        response = None
        try:
            # Apply rate limiting
            self.rate_limiter.wait_if_needed()
            
            logger.debug(f"Making request to: {url}")
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            return response.json()
            
        except requests.exceptions.HTTPError as e:
            if response and response.status_code == 429:
                logger.warning("Rate limit exceeded, waiting longer...")
                time.sleep(5)
                return self._make_request(url, params)  # Retry once
            else:
                status_code = response.status_code if response else "Unknown"
                logger.error(f"HTTP error {status_code} for {url}: {e}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error for {url}: {e}")
        except Exception as e:
            logger.error(f"Unexpected error for {url}: {e}")
            
        return None
    
    def _get_cik_from_ticker(self, ticker: str) -> Optional[str]:
        """
        Get CIK (Central Index Key) from ticker symbol
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            CIK string or None if not found
        """
        ticker = ticker.upper().strip()
        
        # Check cache first
        if ticker in self.cik_cache:
            return self.cik_cache[ticker]
        
        # Check database cache
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT cik, company_name FROM cik_mapping WHERE ticker = ? AND timestamp > ?",
            (ticker, time.time() - 86400)  # Cache for 24 hours
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            cik, company_name = result
            self.cik_cache[ticker] = cik
            logger.debug(f"Found cached CIK for {ticker}: {cik}")
            return cik
        
        # Fetch from SEC company tickers JSON
        try:
            url = f"{self.base_url}/files/company_tickers.json"
            data = self._make_request(url)
            
            if data:
                # Search for ticker in the data
                for key, company_info in data.items():
                    if company_info.get('ticker', '').upper() == ticker:
                        cik = str(company_info['cik_str']).zfill(10)
                        company_name = company_info.get('title', '')
                        
                        # Cache the result
                        self.cik_cache[ticker] = cik
                        conn = sqlite3.connect(self.cache_db_path)
                        cursor = conn.cursor()
                        cursor.execute(
                            "INSERT OR REPLACE INTO cik_mapping (ticker, cik, company_name, timestamp) VALUES (?, ?, ?, ?)",
                            (ticker, cik, company_name, time.time())
                        )
                        conn.commit()
                        conn.close()
                        
                        logger.info(f"Found CIK for {ticker}: {cik} ({company_name})")
                        return cik
                
                logger.warning(f"Ticker {ticker} not found in SEC company tickers")
                
        except Exception as e:
            logger.error(f"Error fetching CIK for {ticker}: {e}")
        
        return None
    
    def get_company_filings(self, cik: str, form_type: str = "10-K", count: int = 10) -> List[Filing]:
        """
        Fetch company filings from SEC EDGAR API
        
        Args:
            cik: Central Index Key (10-digit string)
            form_type: Filing form type (10-K, 10-Q, 8-K, etc.)
            count: Maximum number of filings to retrieve
            
        Returns:
            List of Filing objects
        """
        cik = str(cik).zfill(10)  # Ensure 10-digit format
        
        # Check cache first
        cache_key = self._get_cache_key("filings", cik, form_type, count)
        cached_data = self._get_cached_data(cache_key, expires_hours=24)
        if cached_data:
            logger.debug(f"Using cached filings for CIK {cik}, form {form_type}")
            return [Filing(**filing) for filing in cached_data]
        
        filings = []
        
        try:
            # Get company submissions
            url = f"{self.base_url}/submissions/CIK{cik}.json"
            data = self._make_request(url)
            
            if not data:
                logger.warning(f"No data received for CIK {cik}")
                return filings
            
            company_name = data.get('name', 'Unknown')
            recent_filings = data.get('filings', {}).get('recent', {})
            
            if not recent_filings:
                logger.warning(f"No recent filings found for CIK {cik}")
                return filings
            
            # Extract filings of specified form type
            forms = recent_filings.get('form', [])
            filing_dates = recent_filings.get('filingDate', [])
            accession_numbers = recent_filings.get('accessionNumber', [])
            primary_documents = recent_filings.get('primaryDocument', [])
            
            found_count = 0
            for i, form in enumerate(forms):
                if found_count >= count:
                    break
                    
                if form == form_type and i < len(filing_dates) and i < len(accession_numbers):
                    accession_num = accession_numbers[i]
                    accession_num_clean = accession_num.replace('-', '')
                    
                    filing_url = f"{self.base_url}/Archives/edgar/data/{int(cik)}/{accession_num_clean}/{primary_documents[i]}"
                    
                    filing = Filing(
                        cik=cik,
                        company_name=company_name,
                        form_type=form,
                        filing_date=filing_dates[i],
                        accession_number=accession_num,
                        primary_document=primary_documents[i],
                        filing_url=filing_url
                    )
                    
                    filings.append(filing)
                    found_count += 1
            
            # Cache the results
            self._cache_data(cache_key, [filing.__dict__ for filing in filings], expires_hours=24)
            
            logger.info(f"Retrieved {len(filings)} {form_type} filings for CIK {cik}")
            
        except Exception as e:
            logger.error(f"Error retrieving filings for CIK {cik}: {e}")
        
        return filings
    
    def extract_financial_data(self, filing_url: str) -> FinancialMetrics:
        """
        Extract financial data from XBRL filing
        
        Args:
            filing_url: URL to the filing document
            
        Returns:
            FinancialMetrics object with extracted data
        """
        # Check cache first
        conn = sqlite3.connect(self.cache_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT financial_data FROM financial_data_cache WHERE filing_url = ? AND timestamp > ?",
            (filing_url, time.time() - 86400 * 7)  # Cache for 7 days
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            logger.debug(f"Using cached financial data for {filing_url}")
            return FinancialMetrics(**json.loads(result[0]))
        
        metrics = FinancialMetrics()
        
        try:
            # Get the XBRL data URL (replace .htm with _htm.xml)
            if filing_url.endswith('.htm'):
                xbrl_url = filing_url.replace('.htm', '_htm.xml')
            else:
                # Try to find XBRL data file
                base_url = '/'.join(filing_url.split('/')[:-1])
                # This is a simplified approach - in practice, you'd need to parse the filing index
                xbrl_url = f"{base_url}/xbrl_data.xml"
            
            # Make request for XBRL data
            self.rate_limiter.wait_if_needed()
            response = self.session.get(xbrl_url, timeout=30)
            
            if response.status_code == 200:
                # Parse XBRL XML
                root = ET.fromstring(response.content)
                
                # Define common XBRL tags for financial metrics
                # Note: Different companies may use different tags, so we try multiple variants
                tag_mappings = {
                    'revenue': ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 
                               'SalesRevenueNet', 'OperatingRevenues'],
                    'total_debt': ['DebtCurrent', 'DebtNoncurrent', 'LongTermDebt', 'ShortTermBorrowings'],
                    'shareholders_equity': ['StockholdersEquity', 'ShareholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
                    'cash_flow_operations': ['NetCashProvidedByUsedInOperatingActivities'],
                    'total_assets': ['Assets'],
                    'current_assets': ['AssetsCurrent'],
                    'current_liabilities': ['LiabilitiesCurrent'],
                    'net_income': ['NetIncomeLoss', 'ProfitLoss']
                }
                
                # Extract values from XBRL
                for metric, tags in tag_mappings.items():
                    value = self._extract_xbrl_value(root, tags)
                    if value is not None:
                        setattr(metrics, metric, value)
                
                # Extract dates
                metrics.filing_date = self._extract_filing_date(root)
                metrics.period_end_date = self._extract_period_end_date(root)
                
                logger.debug(f"Extracted financial data from XBRL: {xbrl_url}")
                
            else:
                # Fallback: try to extract from HTML filing (simplified approach)
                logger.warning(f"XBRL data not available, attempting HTML parsing for {filing_url}")
                metrics = self._extract_from_html_filing(filing_url)
                
        except ET.ParseError as e:
            logger.error(f"XML parsing error for {filing_url}: {e}")
        except Exception as e:
            logger.error(f"Error extracting financial data from {filing_url}: {e}")
        
        # Cache the results
        if metrics.revenue is not None or metrics.total_assets is not None:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO financial_data_cache (filing_url, financial_data, timestamp) VALUES (?, ?, ?)",
                (filing_url, json.dumps(metrics.__dict__), time.time())
            )
            conn.commit()
            conn.close()
        
        return metrics
    
    def _extract_xbrl_value(self, root: ET.Element, tags: List[str]) -> Optional[float]:
        """Extract numeric value from XBRL for given tags"""
        for tag in tags:
            # Try different namespace prefixes
            for prefix in ['us-gaap:', 'dei:', '']:
                full_tag = f"{prefix}{tag}"
                elements = root.findall(f".//{full_tag}")
                
                for element in elements:
                    try:
                        # Get the most recent period value
                        value_text = element.text
                        if value_text:
                            # Handle negative values in parentheses
                            value_text = value_text.replace('(', '-').replace(')', '')
                            value = float(value_text.replace(',', ''))
                            return value
                    except (ValueError, AttributeError):
                        continue
        
        return None
    
    def _extract_filing_date(self, root: ET.Element) -> Optional[str]:
        """Extract filing date from XBRL"""
        date_tags = ['DocumentPeriodEndDate', 'CurrentFiscalYearEndDate']
        for tag in date_tags:
            elements = root.findall(f".//{tag}")
            if elements and elements[0].text:
                return elements[0].text
        return None
    
    def _extract_period_end_date(self, root: ET.Element) -> Optional[str]:
        """Extract period end date from XBRL"""
        # This would need more sophisticated logic in practice
        return self._extract_filing_date(root)
    
    def _extract_from_html_filing(self, filing_url: str) -> FinancialMetrics:
        """
        Fallback method to extract financial data from HTML filing
        This is a simplified approach - in practice, you'd need more sophisticated parsing
        """
        metrics = FinancialMetrics()
        
        try:
            self.rate_limiter.wait_if_needed()
            response = self.session.get(filing_url, timeout=30)
            
            if response.status_code == 200:
                html_content = response.text
                
                # Use regex to find common financial statement patterns
                # This is very basic - production code would need more robust parsing
                patterns = {
                    'revenue': [r'Total\s+revenues?\s*[\$\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)', 
                               r'Net\s+sales\s*[\$\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)'],
                    'total_assets': [r'Total\s+assets\s*[\$\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)'],
                    'shareholders_equity': [r'Total\s+shareholders?\'\s+equity\s*[\$\s]*(\d{1,3}(?:,\d{3})*(?:\.\d+)?)']
                }
                
                for metric, regex_patterns in patterns.items():
                    for pattern in regex_patterns:
                        matches = re.findall(pattern, html_content, re.IGNORECASE)
                        if matches:
                            try:
                                value = float(matches[0].replace(',', '')) * 1000  # Usually in thousands
                                setattr(metrics, metric, value)
                                break
                            except ValueError:
                                continue
                
        except Exception as e:
            logger.error(f"Error extracting from HTML filing {filing_url}: {e}")
        
        return metrics
    
    def calculate_financial_ratios(self, financial_data: FinancialMetrics) -> FinancialRatios:
        """
        Calculate financial ratios from financial metrics
        
        Args:
            financial_data: FinancialMetrics object
            
        Returns:
            FinancialRatios object with calculated ratios
        """
        ratios = FinancialRatios()
        
        try:
            # Debt-to-Equity Ratio
            if financial_data.total_debt and financial_data.shareholders_equity:
                if financial_data.shareholders_equity != 0:
                    ratios.debt_to_equity = financial_data.total_debt / financial_data.shareholders_equity
            
            # Current Ratio
            if financial_data.current_assets and financial_data.current_liabilities:
                if financial_data.current_liabilities != 0:
                    ratios.current_ratio = financial_data.current_assets / financial_data.current_liabilities
            
            # Return on Assets (ROA)
            if financial_data.net_income and financial_data.total_assets:
                if financial_data.total_assets != 0:
                    ratios.roa = financial_data.net_income / financial_data.total_assets
            
            # Return on Equity (ROE)
            if financial_data.net_income and financial_data.shareholders_equity:
                if financial_data.shareholders_equity != 0:
                    ratios.roe = financial_data.net_income / financial_data.shareholders_equity
            
            # Debt Ratio
            if financial_data.total_debt and financial_data.total_assets:
                if financial_data.total_assets != 0:
                    ratios.debt_ratio = financial_data.total_debt / financial_data.total_assets
            
            # Asset Turnover
            if financial_data.revenue and financial_data.total_assets:
                if financial_data.total_assets != 0:
                    ratios.asset_turnover = financial_data.revenue / financial_data.total_assets
            
            logger.debug(f"Calculated financial ratios: D/E={ratios.debt_to_equity}, Current={ratios.current_ratio}, ROA={ratios.roa}, ROE={ratios.roe}")
            
        except Exception as e:
            logger.error(f"Error calculating financial ratios: {e}")
        
        return ratios
    
    def get_latest_metrics(self, ticker_symbol: str) -> Tuple[Optional[FinancialMetrics], Optional[FinancialRatios]]:
        """
        Get latest financial metrics and ratios for a company
        
        Args:
            ticker_symbol: Stock ticker symbol
            
        Returns:
            Tuple of (FinancialMetrics, FinancialRatios) or (None, None) if failed
        """
        try:
            # Get CIK from ticker
            cik = self._get_cik_from_ticker(ticker_symbol)
            if not cik:
                logger.warning(f"Could not find CIK for ticker {ticker_symbol}")
                return None, None
            
            # Get latest 10-K filing (annual report)
            filings = self.get_company_filings(cik, form_type="10-K", count=1)
            if not filings:
                # Try 10-Q if no 10-K available
                filings = self.get_company_filings(cik, form_type="10-Q", count=1)
            
            if not filings:
                logger.warning(f"No filings found for {ticker_symbol} (CIK: {cik})")
                return None, None
            
            # Extract financial data from latest filing
            latest_filing = filings[0]
            financial_data = self.extract_financial_data(latest_filing.filing_url)
            
            # Calculate ratios
            ratios = self.calculate_financial_ratios(financial_data)
            
            logger.info(f"Successfully retrieved latest metrics for {ticker_symbol}")
            return financial_data, ratios
            
        except Exception as e:
            logger.error(f"Error getting latest metrics for {ticker_symbol}: {e}")
            return None, None
    
    def cache_filing_data(self, cik: str, data: Dict[str, Any]):
        """
        Cache filing data to reduce API calls
        
        Args:
            cik: Central Index Key
            data: Data to cache
        """
        try:
            cache_key = self._get_cache_key("filing_data", cik)
            self._cache_data(cache_key, data, expires_hours=24)
            logger.debug(f"Cached filing data for CIK {cik}")
        except Exception as e:
            logger.error(f"Error caching filing data for CIK {cik}: {e}")
    
    def validate_financial_data(self, data: FinancialMetrics) -> bool:
        """
        Validate financial data for reasonableness
        
        Args:
            data: FinancialMetrics object to validate
            
        Returns:
            True if data appears valid, False otherwise
        """
        try:
            # Basic validation checks
            validations = []
            
            # Revenue should be positive if present
            if data.revenue is not None:
                validations.append(data.revenue >= 0)
            
            # Total assets should be positive if present
            if data.total_assets is not None:
                validations.append(data.total_assets > 0)
            
            # Current assets shouldn't exceed total assets
            if data.current_assets and data.total_assets:
                validations.append(data.current_assets <= data.total_assets * 1.1)  # Allow 10% buffer for timing differences
            
            # Shareholders equity can be negative, but if assets and liabilities exist, check balance sheet equation
            if all(x is not None for x in [data.total_assets, data.shareholders_equity, data.current_liabilities]):
                # Simplified check - in practice you'd need total liabilities
                if data.total_assets is not None and data.shareholders_equity is not None:
                    estimated_liabilities = data.total_assets - data.shareholders_equity
                    validations.append(estimated_liabilities >= 0)
            
            # At least some data should be present
            has_data = any(getattr(data, field) is not None for field in 
                          ['revenue', 'total_assets', 'shareholders_equity', 'net_income'])
            validations.append(has_data)
            
            is_valid = all(validations) and len(validations) > 0
            
            if not is_valid:
                logger.warning("Financial data failed validation checks")
            
            return is_valid
            
        except Exception as e:
            logger.error(f"Error validating financial data: {e}")
            return False
    
    def _cache_data(self, cache_key: str, data: Any, expires_hours: int = 24):
        """Cache data with expiration"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            expires_at = time.time() + (expires_hours * 3600)
            cursor.execute(
                "INSERT OR REPLACE INTO filings_cache (cache_key, data, timestamp, expires_at) VALUES (?, ?, ?, ?)",
                (cache_key, json.dumps(data), time.time(), expires_at)
            )
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Error caching data: {e}")
    
    def _get_cached_data(self, cache_key: str, expires_hours: int = 24) -> Optional[Any]:
        """Retrieve cached data if not expired"""
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT data FROM filings_cache WHERE cache_key = ? AND expires_at > ?",
                (cache_key, time.time())
            )
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return json.loads(result[0])
                
        except Exception as e:
            logger.error(f"Error retrieving cached data: {e}")
        
        return None
    
    def get_management_discussion(self, filing_url: str) -> Optional[str]:
        """
        Extract Management Discussion and Analysis section for sentiment analysis
        
        Args:
            filing_url: URL to the filing document
            
        Returns:
            MD&A text or None if not found
        """
        try:
            self.rate_limiter.wait_if_needed()
            response = self.session.get(filing_url, timeout=30)
            
            if response.status_code == 200:
                html_content = response.text
                
                # Look for MD&A section patterns
                mda_patterns = [
                    r'MANAGEMENT[\'S\s]+DISCUSSION\s+AND\s+ANALYSIS\s+OF\s+FINANCIAL\s+CONDITION\s+AND\s+RESULTS\s+OF\s+OPERATIONS(.*?)(?=QUANTITATIVE\s+AND\s+QUALITATIVE\s+DISCLOSURES|CONTROLS\s+AND\s+PROCEDURES|LEGAL\s+PROCEEDINGS)',
                    r'Item\s+2\.\s*Management[\'s\s]+Discussion\s+and\s+Analysis(.*?)(?=Item\s+3\.|Item\s+4\.)'
                ]
                
                for pattern in mda_patterns:
                    matches = re.findall(pattern, html_content, re.IGNORECASE | re.DOTALL)
                    if matches:
                        # Clean up the extracted text
                        mda_text = matches[0]
                        mda_text = re.sub(r'<[^>]+>', '', mda_text)  # Remove HTML tags
                        mda_text = re.sub(r'\s+', ' ', mda_text).strip()  # Normalize whitespace
                        
                        if len(mda_text) > 500:  # Ensure we have substantial content
                            logger.debug(f"Extracted MD&A section ({len(mda_text)} characters)")
                            return mda_text
                
                logger.warning(f"Could not find MD&A section in {filing_url}")
                
        except Exception as e:
            logger.error(f"Error extracting MD&A from {filing_url}: {e}")
        
        return None
    
    def get_collection_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about data collection performance
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            conn = sqlite3.connect(self.cache_db_path)
            cursor = conn.cursor()
            
            # Get cache statistics
            cursor.execute("SELECT COUNT(*) FROM filings_cache WHERE expires_at > ?", (time.time(),))
            active_cache_entries = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM financial_data_cache")
            financial_cache_entries = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM cik_mapping")
            cik_mappings = cursor.fetchone()[0]
            
            conn.close()
            
            stats = {
                'active_cache_entries': active_cache_entries,
                'financial_cache_entries': financial_cache_entries,
                'cik_mappings': cik_mappings,
                'cache_directory': str(self.cache_dir),
                'user_agent': self.user_agent
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting collection statistics: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    # Initialize collector
    collector = EDGARCollector(
        user_agent="CreditTech Intelligence Platform hackathon@example.com"
    )
    
    # Test with a well-known company
    ticker = "AAPL"
    print(f"\nTesting EDGAR collector with {ticker}...")
    
    # Get latest metrics
    financial_data, ratios = collector.get_latest_metrics(ticker)
    
    if financial_data:
        print(f"\nFinancial Metrics for {ticker}:")
        print(f"  Revenue: ${financial_data.revenue:,.0f}" if financial_data.revenue else "  Revenue: N/A")
        print(f"  Total Assets: ${financial_data.total_assets:,.0f}" if financial_data.total_assets else "  Total Assets: N/A")
        print(f"  Shareholders Equity: ${financial_data.shareholders_equity:,.0f}" if financial_data.shareholders_equity else "  Shareholders Equity: N/A")
        
        if ratios:
            print(f"\nFinancial Ratios for {ticker}:")
            print(f"  Debt-to-Equity: {ratios.debt_to_equity:.2f}" if ratios.debt_to_equity else "  Debt-to-Equity: N/A")
            print(f"  Current Ratio: {ratios.current_ratio:.2f}" if ratios.current_ratio else "  Current Ratio: N/A")
            print(f"  ROA: {ratios.roa:.2%}" if ratios.roa else "  ROA: N/A")
            print(f"  ROE: {ratios.roe:.2%}" if ratios.roe else "  ROE: N/A")
        
        # Validate data
        is_valid = collector.validate_financial_data(financial_data)
        print(f"\nData validation: {'PASSED' if is_valid else 'FAILED'}")
    
    else:
        print(f"Could not retrieve financial data for {ticker}")
    
    # Show collection statistics
    stats = collector.get_collection_statistics()
    print(f"\nCollection Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
