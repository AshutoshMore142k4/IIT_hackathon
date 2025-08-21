# NLP processor for text data
# File: /data_pipeline/processors/nlp_processor.py

import openai
import time
import logging
import json
import hashlib
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import redis
from concurrent.futures import ThreadPoolExecutor, as_completed
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OpenAIConfig:
    """Configuration for OpenAI API settings"""
    api_key: str
    model: str = "gpt-3.5-turbo"
    max_tokens: int = 150
    temperature: float = 0.0
    max_requests_per_minute: int = 60
    cost_per_1k_tokens: float = 0.002  # GPT-3.5-turbo pricing
    daily_budget_limit: float = 10.0  # $10 daily limit

class NLPProcessor:
    """
    Advanced NLP processor using OpenAI API for financial news analysis
    with cost monitoring and fallback mechanisms
    """
    
    def __init__(self, config: OpenAIConfig, enable_cache: bool = True):
        """
        Initialize NLP processor with OpenAI configuration
        
        Args:
            config: OpenAIConfig object with API settings
            enable_cache: Enable Redis caching for expensive operations
        """
        self.config = config
        openai.api_key = config.api_key
        self.enable_cache = enable_cache
        
        # Cost tracking
        self.daily_cost = 0.0
        self.request_count = 0
        self.last_reset = datetime.now().date()
        
        # Initialize Redis cache
        if self.enable_cache:
            try:
                self.cache = redis.Redis(
                    host='localhost',
                    port=6379,
                    db=2,
                    decode_responses=True,
                    socket_timeout=5
                )
                self.cache.ping()
                logger.info("Redis cache initialized for NLP processor")
            except Exception as e:
                logger.warning(f"Cache initialization failed: {e}. Running without cache.")
                self.enable_cache = False
        
        # Rate limiting
        self.request_times = []
        
        # Fallback sentiment keywords for when API quota is exceeded
        self.sentiment_keywords = {
            'positive': [
                'growth', 'profit', 'success', 'strong', 'beat', 'exceed', 
                'bullish', 'optimistic', 'upgrade', 'outperform', 'buy'
            ],
            'negative': [
                'loss', 'decline', 'weak', 'miss', 'disappoint', 'bearish', 
                'pessimistic', 'downgrade', 'underperform', 'sell', 'crisis'
            ]
        }
    
    def _get_cache_key(self, method_name: str, text: str, *args) -> str:
        """Generate cache key for NLP operations"""
        text_hash = hashlib.md5(text.encode()).hexdigest()[:10]
        key_data = f"nlp:{method_name}:{text_hash}:{':'.join(map(str, args))}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _get_from_cache(self, cache_key: str) -> Optional[Dict]:
        """Retrieve NLP results from cache"""
        if not self.enable_cache:
            return None
        
        try:
            cached_data = self.cache.get(cache_key)
            if cached_data:
                return json.loads(cached_data)
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        return None
    
    def _set_cache(self, cache_key: str, data: Dict, ttl: int = 86400):
        """Store NLP results in cache (24 hours TTL)"""
        if not self.enable_cache:
            return
        
        try:
            self.cache.setex(cache_key, ttl, json.dumps(data, default=str))
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        now = time.time()
        
        # Remove requests older than 1 minute
        self.request_times = [t for t in self.request_times if now - t < 60]
        
        if len(self.request_times) >= self.config.max_requests_per_minute:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                logger.info(f"Rate limit reached. Sleeping for {sleep_time:.2f} seconds")
                time.sleep(sleep_time)
        
        self.request_times.append(now)
    
    def _check_budget(self, estimated_tokens: int) -> bool:
        """Check if request is within daily budget"""
        # Reset daily tracking if new day
        if datetime.now().date() > self.last_reset:
            self.daily_cost = 0.0
            self.request_count = 0
            self.last_reset = datetime.now().date()
        
        estimated_cost = (estimated_tokens / 1000) * self.config.cost_per_1k_tokens
        
        if self.daily_cost + estimated_cost > self.config.daily_budget_limit:
            logger.warning(f"Daily budget limit reached. Current cost: ${self.daily_cost:.2f}")
            return False
        
        return True
    
    def _make_openai_request(self, prompt: str, max_tokens: Optional[int] = None) -> Optional[str]:
        """Make OpenAI API request with error handling and cost tracking"""
        max_tokens = max_tokens or self.config.max_tokens
        
        # Check budget
        if not self._check_budget(max_tokens + len(prompt.split())):
            return None
        
        # Check rate limit
        self._check_rate_limit()
        
        try:
            response = openai.ChatCompletion.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=self.config.temperature
            )
            
            result = response.choices[0].message.content.strip()
            
            # Track cost
            tokens_used = response.usage.total_tokens
            cost = (tokens_used / 1000) * self.config.cost_per_1k_tokens
            self.daily_cost += cost
            self.request_count += 1
            
            logger.debug(f"OpenAI request completed. Tokens: {tokens_used}, Cost: ${cost:.4f}")
            return result
            
        except openai.error.RateLimitError:
            logger.warning("OpenAI rate limit exceeded. Sleeping for 60 seconds.")
            time.sleep(60)
            return None
        except openai.error.InvalidRequestError as e:
            logger.error(f"Invalid OpenAI request: {e}")
            return None
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None
    
    def _fallback_sentiment_analysis(self, text: str) -> str:
        """Fallback sentiment analysis using keyword matching"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in self.sentiment_keywords['positive'] if word in text_lower)
        negative_count = sum(1 for word in self.sentiment_keywords['negative'] if word in text_lower)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def analyze_news_sentiment(self, headlines: List[str]) -> List[Dict]:
        """
        Analyze sentiment of news headlines with caching and fallback
        
        Args:
            headlines: List of news headline texts
            
        Returns:
            List of dictionaries with headline, sentiment, and confidence
        """
        results = []
        
        for headline in headlines:
            if not headline or len(headline.strip()) == 0:
                results.append({
                    "headline": headline,
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "method": "default"
                })
                continue
            
            # Check cache first
            cache_key = self._get_cache_key('sentiment', headline)
            cached_result = self._get_from_cache(cache_key)
            
            if cached_result:
                results.append(cached_result)
                continue
            
            # Try OpenAI API
            prompt = f"""Analyze the sentiment of this financial news headline for credit risk assessment.
            
Headline: "{headline}"

Respond with only one word: positive, negative, or neutral."""

            ai_result = self._make_openai_request(prompt, max_tokens=10)
            
            if ai_result and ai_result.lower() in ['positive', 'negative', 'neutral']:
                sentiment = ai_result.lower()
                confidence = 0.85
                method = "openai"
            else:
                # Fallback to keyword-based analysis
                sentiment = self._fallback_sentiment_analysis(headline)
                confidence = 0.6
                method = "fallback"
                logger.info(f"Using fallback sentiment analysis for: {headline[:50]}...")
            
            result = {
                "headline": headline,
                "sentiment": sentiment,
                "confidence": confidence,
                "method": method,
                "timestamp": datetime.now().isoformat()
            }
            
            # Cache the result
            self._set_cache(cache_key, result)
            results.append(result)
        
        logger.info(f"Analyzed sentiment for {len(headlines)} headlines")
        return results
    
    def extract_financial_events(self, news_content: str) -> List[str]:
        """
        Extract key financial events from news content
        
        Args:
            news_content: Text content of news article
            
        Returns:
            List of detected financial events
        """
        if not news_content or len(news_content.strip()) == 0:
            return []
        
        cache_key = self._get_cache_key('events', news_content[:200])
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result and 'events' in cached_result:
            return cached_result['events']
        
        # Try OpenAI extraction
        prompt = f"""Extract key financial events from this news content. Focus on events that could impact credit risk.

Content: "{news_content[:1000]}"

List only the specific financial events mentioned (e.g., "earnings announcement", "merger", "debt refinancing"). 
Respond with events separated by commas, or "none" if no significant financial events found."""

        ai_result = self._make_openai_request(prompt, max_tokens=80)
        
        events = []
        if ai_result and ai_result.lower() != "none":
            # Parse events from AI response
            events = [event.strip().lower() for event in ai_result.split(',') if event.strip()]
            events = [event for event in events if len(event) > 2]  # Filter out short/invalid entries
        
        # Fallback: keyword-based extraction
        if not events:
            content_lower = news_content.lower()
            event_keywords = {
                'earnings announcement': ['earnings', 'quarterly results', 'financial results'],
                'merger': ['merger', 'acquisition', 'takeover'],
                'management change': ['ceo', 'chief executive', 'president', 'resigns', 'appointed'],
                'regulatory action': ['sec', 'investigation', 'regulatory', 'compliance'],
                'debt action': ['debt', 'refinancing', 'borrowing', 'credit facility'],
                'dividend action': ['dividend', 'payout', 'distribution'],
                'guidance update': ['guidance', 'forecast', 'outlook', 'expects']
            }
            
            for event_type, keywords in event_keywords.items():
                if any(keyword in content_lower for keyword in keywords):
                    events.append(event_type)
        
        result = {'events': events, 'timestamp': datetime.now().isoformat()}
        self._set_cache(cache_key, result)
        
        return events
    
    def classify_risk_impact(self, news_item: Dict) -> Dict:
        """
        Classify the impact of news on credit risk with detailed analysis
        
        Args:
            news_item: News article dictionary
            
        Returns:
            Dictionary with impact classification and reasoning
        """
        content = news_item.get('content') or news_item.get('description') or news_item.get('title') or ""
        
        if not content:
            return {
                'impact': 'neutral',
                'confidence': 0.0,
                'reasoning': 'No content available for analysis'
            }
        
        cache_key = self._get_cache_key('risk_impact', content[:300])
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result:
            return cached_result
        
        # Try OpenAI classification
        prompt = f"""Analyze how this news impacts a company's credit risk. Consider factors like financial health, debt capacity, and business stability.

News: "{content[:800]}"

Classify the impact as: positive, negative, or neutral
Provide a brief reason (1-2 sentences) for your classification.

Format: [IMPACT]: [REASON]"""

        ai_result = self._make_openai_request(prompt, max_tokens=100)
        
        impact = 'neutral'
        reasoning = 'Analysis unavailable'
        confidence = 0.5
        
        if ai_result:
            # Parse AI response
            try:
                parts = ai_result.split(':', 1)
                if len(parts) == 2:
                    impact_part = parts[0].strip().lower()
                    if impact_part in ['positive', 'negative', 'neutral']:
                        impact = impact_part
                        reasoning = parts[1].strip()
                        confidence = 0.8
            except Exception as e:
                logger.warning(f"Error parsing AI risk impact result: {e}")
        
        # Fallback: keyword-based classification
        if confidence < 0.8:
            content_lower = content.lower()
            
            positive_indicators = ['beat', 'exceed', 'strong', 'growth', 'profit', 'cash flow', 'upgrade']
            negative_indicators = ['miss', 'decline', 'loss', 'debt', 'lawsuit', 'investigation', 'downgrade']
            
            positive_score = sum(1 for indicator in positive_indicators if indicator in content_lower)
            negative_score = sum(1 for indicator in negative_indicators if indicator in content_lower)
            
            if positive_score > negative_score:
                impact = 'positive'
                reasoning = 'Positive financial indicators detected'
            elif negative_score > positive_score:
                impact = 'negative'
                reasoning = 'Negative financial indicators detected'
            else:
                impact = 'neutral'
                reasoning = 'Mixed or unclear impact indicators'
            
            confidence = 0.6
        
        result = {
            'impact': impact,
            'confidence': confidence,
            'reasoning': reasoning,
            'timestamp': datetime.now().isoformat()
        }
        
        self._set_cache(cache_key, result)
        return result
    
    def generate_news_summary(self, news_batch: List[Dict], max_summary_length: int = 200) -> str:
        """
        Generate concise summary of news batch highlighting credit risk factors
        
        Args:
            news_batch: List of news articles
            max_summary_length: Maximum length of summary
            
        Returns:
            Risk-focused summary text
        """
        if not news_batch:
            return "No news available for summary."
        
        # Extract headlines and key content
        headlines = [article.get('title', '') for article in news_batch[:5]]  # Limit to top 5
        combined_headlines = ' | '.join([h for h in headlines if h])
        
        cache_key = self._get_cache_key('summary', combined_headlines)
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result and 'summary' in cached_result:
            return cached_result['summary']
        
        # Try OpenAI summary
        prompt = f"""Summarize the key credit risk implications from these financial news headlines. Focus on factors that could affect the company's creditworthiness.

Headlines:
{combined_headlines}

Provide a concise summary (2-3 sentences) highlighting the main risk factors or positive developments."""

        ai_result = self._make_openai_request(prompt, max_tokens=max_summary_length//2)
        
        if ai_result and len(ai_result.strip()) > 10:
            summary = ai_result
        else:
            # Fallback: simple concatenation with risk keywords
            risk_keywords_found = []
            all_text = combined_headlines.lower()
            
            for keyword in ['earnings', 'debt', 'growth', 'loss', 'merger', 'regulatory']:
                if keyword in all_text:
                    risk_keywords_found.append(keyword)
            
            if risk_keywords_found:
                summary = f"Key developments include {', '.join(risk_keywords_found)} related news."
            else:
                summary = "General financial news without significant credit risk indicators."
        
        result = {'summary': summary, 'timestamp': datetime.now().isoformat()}
        self._set_cache(cache_key, result)
        
        return summary
    
    def calculate_sentiment_score(self, text: str, company_context: Optional[str] = None) -> float:
        """
        Calculate numerical sentiment score between -1 (negative) and 1 (positive)
        
        Args:
            text: Text to analyze
            company_context: Additional company context
            
        Returns:
            Sentiment score as float between -1 and 1
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        cache_key = self._get_cache_key('sentiment_score', text, company_context or '')
        cached_result = self._get_from_cache(cache_key)
        
        if cached_result and 'score' in cached_result:
            return cached_result['score']
        
        # Try OpenAI scoring
        prompt = f"""Rate the sentiment of this financial news for credit risk assessment on a scale from -1 (very negative for creditworthiness) to +1 (very positive for creditworthiness).

Text: "{text}"
"""
        
        if company_context:
            prompt += f"Company context: {company_context}"
        
        prompt += "\nRespond with only a decimal number between -1 and 1."
        
        ai_result = self._make_openai_request(prompt, max_tokens=10)
        
        score = 0.0
        if ai_result:
            try:
                score = float(ai_result.strip())
                score = max(-1.0, min(1.0, score))  # Clamp to valid range
            except ValueError:
                logger.warning(f"Invalid score from AI: {ai_result}")
        
        # Fallback: keyword-based scoring
        if score == 0.0 and ai_result is None:
            text_lower = text.lower()
            positive_count = sum(1 for word in self.sentiment_keywords['positive'] if word in text_lower)
            negative_count = sum(1 for word in self.sentiment_keywords['negative'] if word in text_lower)
            
            total_words = len(text.split())
            if total_words > 0:
                net_sentiment = (positive_count - negative_count) / max(total_words, 10)
                score = max(-1.0, min(1.0, net_sentiment * 5))  # Scale and clamp
        
        result = {'score': score, 'timestamp': datetime.now().isoformat()}
        self._set_cache(cache_key, result)
        
        return score
    
    def batch_analyze_news(self, news_articles: List[Dict], max_workers: int = 3) -> List[Dict]:
        """
        Analyze multiple news articles in parallel
        
        Args:
            news_articles: List of news article dictionaries
            max_workers: Maximum number of parallel workers
            
        Returns:
            Enhanced news articles with NLP analysis
        """
        def analyze_article(article: Dict) -> Dict:
            try:
                # Analyze sentiment
                headline = article.get('title', '')
                if headline:
                    sentiment_results = self.analyze_news_sentiment([headline])
                    if sentiment_results:
                        article['sentiment_analysis'] = sentiment_results[0]
                
                # Extract events
                content = article.get('content') or article.get('description') or ''
                if content:
                    article['financial_events'] = self.extract_financial_events(content)
                    article['risk_impact'] = self.classify_risk_impact(article)
                    article['sentiment_score'] = self.calculate_sentiment_score(content)
                
                article['nlp_processed'] = True
                article['nlp_timestamp'] = datetime.now().isoformat()
                
                return article
                
            except Exception as e:
                logger.error(f"Error analyzing article: {e}")
                article['nlp_error'] = str(e)
                return article
        
        enhanced_articles = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_article = {executor.submit(analyze_article, article): article for article in news_articles}
            
            for future in as_completed(future_to_article):
                try:
                    enhanced_article = future.result(timeout=30)
                    enhanced_articles.append(enhanced_article)
                except Exception as e:
                    original_article = future_to_article[future]
                    logger.error(f"Error processing article: {e}")
                    original_article['nlp_error'] = str(e)
                    enhanced_articles.append(original_article)
        
        logger.info(f"Completed NLP analysis for {len(enhanced_articles)} articles")
        return enhanced_articles
    
    def get_usage_stats(self) -> Dict:
        """
        Get API usage statistics and cost tracking
        
        Returns:
            Dictionary with usage and cost information
        """
        return {
            'daily_cost': round(self.daily_cost, 4),
            'daily_requests': self.request_count,
            'budget_limit': self.config.daily_budget_limit,
            'budget_remaining': round(self.config.daily_budget_limit - self.daily_cost, 4),
            'model_used': self.config.model,
            'cache_enabled': self.enable_cache,
            'last_reset': self.last_reset.isoformat(),
            'requests_per_minute_limit': self.config.max_requests_per_minute
        }

# Example usage
if __name__ == "__main__":
    # Initialize processor
    config = OpenAIConfig(api_key="your_openai_api_key_here")
    processor = NLPProcessor(config)
    
    # Test sentiment analysis
    test_headlines = [
        "Apple reports record quarterly earnings beating analyst expectations",
        "Tesla faces SEC investigation over safety concerns",
        "Microsoft announces dividend increase and stock buyback program"
    ]
    
    try:
        sentiment_results = processor.analyze_news_sentiment(test_headlines)
        for result in sentiment_results:
            print(f"Headline: {result['headline'][:50]}...")
            print(f"Sentiment: {result['sentiment']} (confidence: {result['confidence']})")
            print("---")
        
        # Check usage stats
        stats = processor.get_usage_stats()
        print(f"Usage stats: ${stats['daily_cost']} spent today")
        
    except Exception as e:
        print(f"Error: {e}")
