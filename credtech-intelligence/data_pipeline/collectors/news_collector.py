# Collector for news data
# File: /data_pipeline/collectors/news_collector.py

import requests
import time
import logging
import hashlib
import json
from typing import List, Dict, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional Redis import
try:
    import redis  # type: ignore
    REDIS_AVAILABLE = True
except ImportError:
    redis = None
    REDIS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InMemoryCache:
    """Simple in-memory cache fallback when Redis is not available"""
    
    def __init__(self):
        self._cache = {}
        self._expiry = {}
    
    def get(self, key: str) -> Optional[str]:
        if key in self._cache and key in self._expiry:
            if datetime.now().timestamp() < self._expiry[key]:
                return self._cache[key]
            else:
                # Expired, remove
                del self._cache[key]
                del self._expiry[key]
        return None
    
    def setex(self, key: str, ttl: int, value: str):
        self._cache[key] = value
        self._expiry[key] = datetime.now().timestamp() + ttl
    
    def ping(self):
        return True


@dataclass
class NewsConfig:
    """Configuration for News API settings"""
    api_key: str
    base_url: str = "https://newsapi.org/v2/everything"
    rate_limit_seconds: float = 1.0  # NewsAPI free tier: 1000 requests/day
    max_page_size: int = 100
    timeout_seconds: int = 10
    cache_ttl: int = 1800  # 30 minutes cache

class NewsCollector:
    """
    Comprehensive News API collector for financial news with caching and filtering
    """
    
    def __init__(self, config: NewsConfig, enable_cache: bool = True):
        """
        Initialize NewsCollector with API configuration
        
        Args:
            config: NewsConfig object with API settings
            enable_cache: Enable Redis caching for performance
        """
        self.config = config
        self.enable_cache = enable_cache
        self.cache: Optional[Any] = None
        
        # Initialize cache (Redis or in-memory fallback)
        if self.enable_cache:
            try:
                if REDIS_AVAILABLE and redis is not None:
                    self.cache = redis.Redis(
                        host='localhost',
                        port=6379,
                        db=1,
                        decode_responses=True,
                        socket_timeout=5
                    )
                    try:
                        if self.cache and hasattr(self.cache, 'ping'):
                            self.cache.ping()
                        logger.info("Redis cache initialized for news collector")
                    except Exception:
                        logger.warning("Redis connection failed, using in-memory cache")
                        self.cache = InMemoryCache()
                        logger.info("In-memory cache initialized for news collector")
                else:
                    logger.info("Redis not available, using in-memory cache")
                    self.cache = InMemoryCache()
                    logger.info("In-memory cache initialized for news collector")
            except Exception as e:
                logger.warning(f"Cache initialization failed: {e}. Running without cache.")
                self.enable_cache = False
                self.cache = None
        
        # Financial keywords for filtering
        self.financial_keywords = {
            'high_impact': [
                'earnings', 'merger', 'acquisition', 'ipo', 'bankruptcy', 
                'debt restructuring', 'chapter 11', 'liquidation', 'takeover'
            ],
            'medium_impact': [
                'ceo', 'cfo', 'management', 'leadership', 'regulatory', 
                'sec', 'fine', 'lawsuit', 'investigation', 'audit'
            ],
            'financial_general': [
                'revenue', 'profit', 'loss', 'dividend', 'stock', 'shares',
                'quarterly', 'annual', 'forecast', 'guidance', 'outlook'
            ]
        }
        
        # Company name variations for better matching
        self.company_aliases = {
            'apple': ['apple inc', 'aapl'],
            'microsoft': ['microsoft corp', 'msft'],
            'google': ['alphabet inc', 'googl', 'goog'],
            'amazon': ['amazon com inc', 'amzn'],
            'tesla': ['tesla inc', 'tsla'],
            'meta': ['meta platforms', 'facebook', 'fb'],
            'nvidia': ['nvidia corp', 'nvda']
        }
    
    def _get_cache_key(self, method_name: str, *args) -> str:
        """Generate cache key for method and arguments"""
        key_data = f"news:{method_name}:{':'.join(map(str, args))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[List[Dict]]:
        """Retrieve data from cache"""
        if not self.enable_cache or not self.cache:
            return None
        
        try:
            if hasattr(self.cache, 'get'):
                cached_data = self.cache.get(cache_key)
                if cached_data:
                    return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None
    
    def _set_cache(self, cache_key: str, data: List[Dict]):
        """Store data in cache"""
        if not self.enable_cache or not self.cache:
            return
        
        try:
            if hasattr(self.cache, 'setex'):
                self.cache.setex(cache_key, self.config.cache_ttl, json.dumps(data, default=str))
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def _build_company_query(self, company_name: str) -> str:
        """Build optimized search query for company"""
        base_name = company_name.lower().strip()
        
        # Check for known aliases
        for key, aliases in self.company_aliases.items():
            if base_name in aliases or key in base_name:
                return f'("{company_name}" OR "{key}")'
        
        return f'"{company_name}"'
    
    def fetch_news(self, query: str, from_date: Optional[str] = None, 
                   to_date: Optional[str] = None, max_articles: int = 500) -> List[Dict]:
        """
        Fetch news articles with pagination and rate limiting
        
        Args:
            query: Search query string
            from_date: ISO format date string (YYYY-MM-DD)
            to_date: ISO format date string (YYYY-MM-DD)
            max_articles: Maximum number of articles to fetch
            
        Returns:
            List of news article dictionaries
        """
        all_articles = []
        page = 1
        articles_fetched = 0
        
        headers = {"X-API-Key": self.config.api_key}
        
        while articles_fetched < max_articles:
            params = {
                "q": query,
                "from": from_date,
                "to": to_date,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": min(self.config.max_page_size, max_articles - articles_fetched),
                "page": page
            }
            
            try:
                response = requests.get(
                    self.config.base_url, 
                    headers=headers, 
                    params=params, 
                    timeout=self.config.timeout_seconds
                )
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('status') != 'ok':
                    logger.error(f"News API error: {data.get('message', 'Unknown error')}")
                    break
                
                articles = data.get('articles', [])
                if not articles:
                    break
                
                # Add metadata to each article
                for article in articles:
                    article['fetched_at'] = datetime.now().isoformat()
                    article['query_used'] = query
                    article['page'] = page
                
                all_articles.extend(articles)
                articles_fetched += len(articles)
                
                total_results = data.get('totalResults', 0)
                if articles_fetched >= total_results:
                    break
                
                page += 1
                
                # Rate limiting
                time.sleep(self.config.rate_limit_seconds)
                
            except requests.exceptions.RequestException as e:
                logger.error(f"HTTP error fetching news: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error fetching news: {e}")
                break
        
        logger.info(f"Fetched {len(all_articles)} news articles for query: {query}")
        return all_articles
    
    def filter_financial_news(self, articles: List[Dict]) -> List[Dict]:
        """
        Advanced filtering to keep only financial news with relevance scoring
        
        Args:
            articles: List of news articles
            
        Returns:
            Filtered articles with relevance scores
        """
        filtered_articles = []
        
        for article in articles:
            title = article.get('title', '').lower()
            description = article.get('description', '').lower() if article.get('description') else ''
            content = article.get('content', '').lower() if article.get('content') else ''
            
            combined_text = ' '.join([title, description, content])
            
            # Calculate relevance score
            relevance_score = 0
            matched_keywords = []
            
            # High impact keywords (weight: 3)
            for keyword in self.financial_keywords['high_impact']:
                if keyword in combined_text:
                    relevance_score += 3
                    matched_keywords.append(keyword)
            
            # Medium impact keywords (weight: 2)
            for keyword in self.financial_keywords['medium_impact']:
                if keyword in combined_text:
                    relevance_score += 2
                    matched_keywords.append(keyword)
            
            # General financial keywords (weight: 1)
            for keyword in self.financial_keywords['financial_general']:
                if keyword in combined_text:
                    relevance_score += 1
                    matched_keywords.append(keyword)
            
            # Only include articles with relevance score >= 2
            if relevance_score >= 2:
                article['relevance_score'] = relevance_score
                article['matched_keywords'] = list(set(matched_keywords))
                filtered_articles.append(article)
        
        # Sort by relevance score (descending)
        filtered_articles.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        logger.info(f"Filtered {len(filtered_articles)} relevant financial articles from {len(articles)} total")
        return filtered_articles
    
    def categorize_news_impact(self, article: Dict) -> str:
        """
        Categorize news by potential market impact level
        
        Args:
            article: News article dictionary
            
        Returns:
            Impact level: 'high', 'medium', or 'low'
        """
        title = article.get('title', '').lower()
        description = article.get('description', '').lower() if article.get('description') else ''
        combined_text = f"{title} {description}"
        
        # High impact indicators
        high_impact_terms = [
            'earnings', 'merger', 'acquisition', 'ipo', 'bankruptcy', 
            'debt restructuring', 'chapter 11', 'takeover', 'liquidation',
            'major lawsuit', 'sec investigation', 'criminal charges'
        ]
        
        # Medium impact indicators  
        medium_impact_terms = [
            'ceo', 'cfo', 'management change', 'leadership', 'regulatory',
            'fine', 'lawsuit', 'investigation', 'audit', 'compliance',
            'dividend cut', 'guidance', 'forecast'
        ]
        
        if any(term in combined_text for term in high_impact_terms):
            return 'high'
        elif any(term in combined_text for term in medium_impact_terms):
            return 'medium'
        else:
            return 'low'
    
    def extract_key_events(self, article: Dict) -> List[str]:
        """
        Extract key financial events from article using keyword matching
        
        Args:
            article: News article dictionary
            
        Returns:
            List of detected event types
        """
        title = article.get('title', '').lower()
        description = article.get('description', '').lower() if article.get('description') else ''
        combined_text = f"{title} {description}"
        
        events = []
        
        event_patterns = {
            'earnings_announcement': ['earnings', 'quarterly results', 'q1', 'q2', 'q3', 'q4'],
            'merger_acquisition': ['merger', 'acquisition', 'takeover', 'buyout'],
            'management_change': ['ceo', 'cfo', 'president', 'chairman', 'resign', 'appoint'],
            'regulatory_action': ['sec', 'fda', 'regulatory', 'investigation', 'fine'],
            'financial_restructuring': ['debt', 'restructuring', 'refinancing', 'bankruptcy'],
            'product_launch': ['launch', 'unveil', 'announce', 'release'],
            'legal_issue': ['lawsuit', 'litigation', 'settlement', 'court'],
            'dividend_action': ['dividend', 'payout', 'yield'],
            'guidance_update': ['guidance', 'forecast', 'outlook', 'expects']
        }
        
        for event_type, patterns in event_patterns.items():
            if any(pattern in combined_text for pattern in patterns):
                events.append(event_type)
        
        return events
    
    def get_company_news(self, ticker: str, days_back: int = 7) -> List[Dict]:
        """
        Get comprehensive company-specific news with analysis
        
        Args:
            ticker: Stock ticker symbol or company name
            days_back: Number of past days to search
            
        Returns:
            Enhanced list of news articles with metadata
        """
        # Check cache first
        cache_key = self._get_cache_key('company_news', ticker, days_back)
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result:
            logger.info(f"Retrieved company news from cache for {ticker}")
            return cached_result
        
        # Calculate date range
        to_date = datetime.now().date()
        from_date = to_date - timedelta(days=days_back)
        
        # Build search query
        query = self._build_company_query(ticker)
        
        # Fetch news
        logger.info(f"Fetching news for {ticker} from {from_date} to {to_date}")
        articles = self.fetch_news(
            query=query,
            from_date=str(from_date),
            to_date=str(to_date),
            max_articles=200
        )
        
        # Filter for financial relevance
        financial_news = self.filter_financial_news(articles)
        
        # Enhance each article with analysis
        for article in financial_news:
            article['ticker'] = ticker
            article['impact_level'] = self.categorize_news_impact(article)
            article['key_events'] = self.extract_key_events(article)
            article['analysis_timestamp'] = datetime.now().isoformat()
        
        # Cache the results
        self._set_cache(cache_key, financial_news)
        
        logger.info(f"Retrieved {len(financial_news)} relevant news articles for {ticker}")
        return financial_news
    
    def batch_get_company_news(self, tickers: List[str], days_back: int = 7, max_workers: int = 3) -> Dict[str, List[Dict]]:
        """
        Efficiently fetch news for multiple companies using parallel processing
        
        Args:
            tickers: List of ticker symbols or company names
            days_back: Number of past days to search
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping tickers to their news articles
        """
        results = {}
        
        def fetch_company_news(ticker: str) -> Tuple[str, List[Dict]]:
            try:
                news = self.get_company_news(ticker, days_back)
                return ticker, news
            except Exception as e:
                logger.error(f"Error fetching news for {ticker}: {e}")
                return ticker, []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_ticker = {executor.submit(fetch_company_news, ticker): ticker for ticker in tickers}
            
            for future in as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    ticker, news_articles = future.result(timeout=60)
                    results[ticker] = news_articles
                except Exception as e:
                    logger.error(f"Error processing news for {ticker}: {e}")
                    results[ticker] = []
        
        total_articles = sum(len(articles) for articles in results.values())
        logger.info(f"Batch processed news for {len(tickers)} companies: {total_articles} total articles")
        
        return results
    
    def get_market_news(self, days_back: int = 1, max_articles: int = 100) -> List[Dict]:
        """
        Get general market and economic news
        
        Args:
            days_back: Number of past days to search
            max_articles: Maximum number of articles to fetch
            
        Returns:
            List of market news articles
        """
        to_date = datetime.now().date()
        from_date = to_date - timedelta(days=days_back)
        
        market_query = "stock market OR financial markets OR economy OR federal reserve OR interest rates"
        
        articles = self.fetch_news(
            query=market_query,
            from_date=str(from_date),
            to_date=str(to_date),
            max_articles=max_articles
        )
        
        financial_news = self.filter_financial_news(articles)
        
        for article in financial_news:
            article['category'] = 'market_news'
            article['impact_level'] = self.categorize_news_impact(article)
        
        return financial_news
    
    def get_api_usage_stats(self) -> Dict:
        """
        Get API usage statistics and remaining quota
        
        Returns:
            Dictionary with usage statistics
        """
        # This would typically be tracked in a database or cache
        # For now, return basic info
        return {
            'api_provider': 'NewsAPI',
            'last_request': datetime.now().isoformat(),
            'daily_limit': 1000,  # Free tier limit
            'estimated_usage': 'Not tracked',
            'rate_limit_seconds': self.config.rate_limit_seconds
        }

# Example usage
if __name__ == "__main__":
    # Initialize collector
    config = NewsConfig(api_key="your_news_api_key_here")
    collector = NewsCollector(config)
    
    # Test company news
    try:
        apple_news = collector.get_company_news("Apple", days_back=3)
        print(f"Found {len(apple_news)} Apple news articles")
        
        if apple_news:
            print(f"Sample article: {apple_news[0]['title']}")
            print(f"Impact level: {apple_news[0]['impact_level']}")
            print(f"Key events: {apple_news[0]['key_events']}")
    
    except Exception as e:
        print(f"Error: {e}")
