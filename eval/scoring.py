"""
Automated scoring and evaluation metrics for research agent performance.
Provides various scoring methods for answer quality, efficiency, and accuracy.
"""

import re
import string
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
from difflib import SequenceMatcher
import math

import structlog

logger = structlog.get_logger()


@dataclass 
class EvaluationScore:
    """Complete evaluation score for a research task."""
    answer_similarity: float  # 0.0 to 1.0
    source_relevance: float   # 0.0 to 1.0  
    efficiency_score: float   # 0.0 to 1.0
    completeness_score: float # 0.0 to 1.0
    overall_score: float      # 0.0 to 1.0
    details: Dict[str, Any]   # Additional scoring details


class ScoreCalculator:
    """Calculate various evaluation metrics for research agent performance."""
    
    def __init__(self):
        # Common stop words for text comparison
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can',
            'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 
            'they', 'me', 'him', 'her', 'us', 'them'
        }
    
    def calculate_answer_similarity(self, agent_answer: str, golden_answer: str) -> float:
        """
        Calculate semantic similarity between agent answer and golden answer.
        Uses multiple comparison methods and returns weighted average.
        """
        if not agent_answer or not golden_answer:
            return 0.0
        
        # Normalize texts
        agent_clean = self._normalize_text(agent_answer)
        golden_clean = self._normalize_text(golden_answer)
        
        # Calculate different similarity metrics
        scores = []
        
        # 1. Sequence matching (character-level similarity)
        seq_similarity = SequenceMatcher(None, agent_clean, golden_clean).ratio()
        scores.append(("sequence", seq_similarity, 0.2))
        
        # 2. Word-level similarity (Jaccard index)
        word_similarity = self._calculate_word_jaccard(agent_clean, golden_clean)
        scores.append(("word_jaccard", word_similarity, 0.3))
        
        # 3. Key phrase matching
        phrase_similarity = self._calculate_phrase_similarity(agent_clean, golden_clean)
        scores.append(("phrase", phrase_similarity, 0.25))
        
        # 4. Numeric value matching (if applicable)
        numeric_similarity = self._calculate_numeric_similarity(agent_answer, golden_answer)
        scores.append(("numeric", numeric_similarity, 0.15))
        
        # 5. Semantic keyword matching
        keyword_similarity = self._calculate_keyword_similarity(agent_clean, golden_clean)
        scores.append(("keyword", keyword_similarity, 0.1))
        
        # Calculate weighted average
        total_weight = sum(weight for _, _, weight in scores)
        weighted_sum = sum(score * weight for _, score, weight in scores)
        
        final_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        logger.debug("Answer similarity calculated", 
                    final_score=final_score,
                    component_scores={name: score for name, score, _ in scores})
        
        return min(1.0, max(0.0, final_score))
    
    def calculate_source_relevance(
        self, 
        sources_found: List[str], 
        expected_sources: List[str]
    ) -> float:
        """Calculate how well the agent found relevant sources."""
        if not expected_sources:
            return 1.0  # No specific sources required
        
        if not sources_found:
            return 0.0  # No sources found
        
        # Normalize source names for comparison
        found_normalized = [self._normalize_source_name(s) for s in sources_found]
        expected_normalized = [self._normalize_source_name(s) for s in expected_sources]
        
        # Calculate overlap
        matches = 0
        for expected in expected_normalized:
            for found in found_normalized:
                if self._sources_match(expected, found):
                    matches += 1
                    break
        
        # Score based on percentage of expected sources found
        relevance_score = matches / len(expected_sources)
        
        # Bonus for finding additional high-quality sources
        bonus = min(0.2, (len(sources_found) - len(expected_sources)) * 0.05) if len(sources_found) > len(expected_sources) else 0
        
        return min(1.0, relevance_score + bonus)
    
    def calculate_efficiency_score(
        self, 
        steps_taken: int, 
        execution_time_ms: int,
        max_steps: int,
        expected_time_ms: int = 120000  # 2 minutes default
    ) -> float:
        """Calculate efficiency based on steps and execution time."""
        # Step efficiency (fewer steps is better, up to a point)
        step_ratio = steps_taken / max_steps
        step_score = max(0.0, 1.0 - step_ratio) if step_ratio <= 1.0 else 0.0
        
        # Time efficiency (faster is better, but not too fast)
        time_ratio = execution_time_ms / expected_time_ms
        if time_ratio < 0.1:  # Too fast might indicate shallow research
            time_score = 0.5
        elif time_ratio <= 1.0:  # Within expected time
            time_score = 1.0 - (time_ratio * 0.3)  # Small penalty for longer time
        else:  # Overtime
            time_score = max(0.0, 1.0 - (time_ratio - 1.0) * 0.5)
        
        # Weighted combination
        efficiency_score = (step_score * 0.6) + (time_score * 0.4)
        
        return min(1.0, max(0.0, efficiency_score))
    
    def calculate_completeness_score(
        self, 
        agent_answer: str, 
        required_elements: List[str]
    ) -> float:
        """Calculate how completely the answer addresses required elements."""
        if not required_elements:
            return 1.0  # No specific requirements
        
        if not agent_answer:
            return 0.0  # No answer provided
        
        answer_lower = agent_answer.lower()
        elements_found = 0
        
        for element in required_elements:
            element_lower = element.lower()
            # Check for exact match or semantic similarity
            if element_lower in answer_lower or self._semantic_element_match(answer_lower, element_lower):
                elements_found += 1
        
        return elements_found / len(required_elements)
    
    def calculate_overall_score(
        self,
        answer_similarity: float,
        source_relevance: float, 
        efficiency_score: float,
        completeness_score: float,
        weights: Dict[str, float] = None
    ) -> float:
        """Calculate overall score with customizable weights."""
        if weights is None:
            weights = {
                'answer_similarity': 0.4,
                'source_relevance': 0.25,
                'efficiency': 0.15,
                'completeness': 0.2
            }
        
        overall = (
            answer_similarity * weights.get('answer_similarity', 0.4) +
            source_relevance * weights.get('source_relevance', 0.25) +
            efficiency_score * weights.get('efficiency', 0.15) +
            completeness_score * weights.get('completeness', 0.2)
        )
        
        return min(1.0, max(0.0, overall))
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _calculate_word_jaccard(self, text1: str, text2: str) -> float:
        """Calculate Jaccard similarity between word sets."""
        words1 = set(word for word in text1.split() if word not in self.stop_words and len(word) > 2)
        words2 = set(word for word in text2.split() if word not in self.stop_words and len(word) > 2)
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_phrase_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity based on common phrases (2-3 word combinations)."""
        def get_phrases(text: str, n: int = 2) -> set:
            words = text.split()
            if len(words) < n:
                return set()
            return {' '.join(words[i:i+n]) for i in range(len(words) - n + 1)}
        
        phrases1 = get_phrases(text1, 2).union(get_phrases(text1, 3))
        phrases2 = get_phrases(text2, 2).union(get_phrases(text2, 3))
        
        if not phrases1 and not phrases2:
            return 1.0
        if not phrases1 or not phrases2:
            return 0.0
        
        intersection = len(phrases1.intersection(phrases2))
        union = len(phrases1.union(phrases2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_numeric_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity based on numeric values found in texts."""
        # Extract numbers (including decimals, percentages, currencies)
        number_pattern = r'[\$€£¥]?[\d,]+\.?\d*[%]?'
        
        numbers1 = re.findall(number_pattern, text1)
        numbers2 = re.findall(number_pattern, text2)
        
        if not numbers1 and not numbers2:
            return 1.0  # No numbers to compare
        if not numbers1 or not numbers2:
            return 0.0  # One has numbers, other doesn't
        
        # Convert to standardized numeric values
        def parse_number(num_str: str) -> float:
            try:
                # Remove currency symbols and commas
                clean_num = re.sub(r'[\$€£¥,]', '', num_str)
                # Handle percentages
                if clean_num.endswith('%'):
                    return float(clean_num[:-1]) / 100
                return float(clean_num)
            except:
                return 0.0
        
        nums1 = [parse_number(n) for n in numbers1]
        nums2 = [parse_number(n) for n in numbers2]
        
        # Calculate similarity based on matching numbers
        matches = 0
        tolerance = 0.05  # 5% tolerance for numeric matches
        
        for n1 in nums1:
            for n2 in nums2:
                if n1 == 0 and n2 == 0:
                    matches += 1
                elif n1 != 0 and abs((n1 - n2) / n1) <= tolerance:
                    matches += 1
                elif n2 != 0 and abs((n1 - n2) / n2) <= tolerance:
                    matches += 1
        
        total_numbers = max(len(nums1), len(nums2))
        return matches / total_numbers if total_numbers > 0 else 0.0
    
    def _calculate_keyword_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity based on important keywords."""
        # Define domain-specific important keywords
        important_keywords = {
            'technology', 'research', 'development', 'innovation', 'company', 
            'market', 'industry', 'analysis', 'data', 'study', 'report',
            'billion', 'million', 'percent', 'increase', 'decrease', 'growth',
            'revenue', 'profit', 'investment', 'funding', 'acquisition',
            'artificial intelligence', 'machine learning', 'quantum', 'renewable',
            'climate', 'energy', 'sustainable', 'global', 'international'
        }
        
        def extract_keywords(text: str) -> set:
            keywords_found = set()
            text_lower = text.lower()
            for keyword in important_keywords:
                if keyword in text_lower:
                    keywords_found.add(keyword)
            return keywords_found
        
        keywords1 = extract_keywords(text1)
        keywords2 = extract_keywords(text2)
        
        if not keywords1 and not keywords2:
            return 1.0
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1.intersection(keywords2))
        union = len(keywords1.union(keywords2))
        
        return intersection / union if union > 0 else 0.0
    
    def _normalize_source_name(self, source: str) -> str:
        """Normalize source name for comparison."""
        source = source.lower()
        # Remove common prefixes/suffixes
        source = re.sub(r'^(www\.|https?://)', '', source)
        source = re.sub(r'\.(com|org|edu|gov|net).*$', '', source)
        return source.strip()
    
    def _sources_match(self, expected: str, found: str) -> bool:
        """Check if two sources match (with fuzzy matching)."""
        # Exact match
        if expected == found:
            return True
        
        # Partial match
        if expected in found or found in expected:
            return True
        
        # Domain matching for URLs
        if '.' in expected or '.' in found:
            exp_parts = expected.split('.')
            found_parts = found.split('.')
            if any(part in found_parts for part in exp_parts if len(part) > 2):
                return True
        
        return False
    
    def _semantic_element_match(self, text: str, element: str) -> bool:
        """Check if an element is semantically present in text."""
        # Simple semantic matching based on synonyms and related terms
        synonyms = {
            'population': ['inhabitants', 'residents', 'people', 'citizens'],
            'revenue': ['income', 'earnings', 'sales', 'turnover'],
            'market cap': ['market value', 'market capitalization', 'valuation'],
            'benefits': ['advantages', 'pros', 'positive', 'strengths'],
            'drawbacks': ['disadvantages', 'cons', 'negative', 'weaknesses'],
            'trends': ['patterns', 'developments', 'directions', 'movements'],
            'companies': ['firms', 'corporations', 'businesses', 'organizations']
        }
        
        element_words = element.split()
        for word in element_words:
            if word in text:
                return True
            if word in synonyms:
                for synonym in synonyms[word]:
                    if synonym in text:
                        return True
        
        return False


class PerformanceAnalyzer:
    """Analyze performance patterns across multiple evaluation runs."""
    
    def __init__(self):
        self.score_calculator = ScoreCalculator()
    
    def analyze_performance_trends(self, evaluation_history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze performance trends over multiple evaluation runs."""
        if not evaluation_history:
            return {}
        
        # Extract metrics over time
        timestamps = []
        success_rates = []
        execution_times = []
        quality_scores = []
        
        for eval_result in evaluation_history:
            if 'timestamp' in eval_result and 'summary' in eval_result:
                timestamps.append(eval_result['timestamp'])
                summary = eval_result['summary']
                success_rates.append(summary.get('overall_success_rate', 0))
                execution_times.append(summary.get('average_execution_time_ms', 0))
                quality_scores.append(summary.get('average_answer_quality', 0))
        
        if not timestamps:
            return {}
        
        # Calculate trends
        trends = {
            'success_rate_trend': self._calculate_trend(success_rates),
            'execution_time_trend': self._calculate_trend(execution_times),
            'quality_score_trend': self._calculate_trend(quality_scores),
            'total_evaluations': len(evaluation_history),
            'latest_success_rate': success_rates[-1] if success_rates else 0,
            'average_success_rate': sum(success_rates) / len(success_rates) if success_rates else 0,
            'best_success_rate': max(success_rates) if success_rates else 0,
            'worst_success_rate': min(success_rates) if success_rates else 0
        }
        
        # Performance categorization
        latest_success = success_rates[-1] if success_rates else 0
        if latest_success >= 0.9:
            performance_category = "excellent"
        elif latest_success >= 0.8:
            performance_category = "good"
        elif latest_success >= 0.7:
            performance_category = "fair"
        else:
            performance_category = "needs_improvement"
        
        trends['performance_category'] = performance_category
        
        return trends
    
    def identify_performance_bottlenecks(self, evaluation_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify common failure patterns and bottlenecks."""
        bottlenecks = {
            'common_failure_categories': {},
            'slow_task_types': [],
            'low_quality_patterns': [],
            'recommendations': []
        }
        
        for result in evaluation_results:
            if 'results' not in result:
                continue
            
            for task_result in result['results']:
                # Analyze failures
                if not task_result.get('success', False):
                    error = task_result.get('error_message', 'unknown')
                    if 'timeout' in error.lower():
                        category = 'timeout'
                    elif 'rate limit' in error.lower():
                        category = 'rate_limit'
                    elif 'network' in error.lower() or 'connection' in error.lower():
                        category = 'network'
                    else:
                        category = 'other'
                    
                    bottlenecks['common_failure_categories'][category] = \
                        bottlenecks['common_failure_categories'].get(category, 0) + 1
                
                # Analyze slow tasks
                if task_result.get('execution_time_ms', 0) > 300000:  # > 5 minutes
                    bottlenecks['slow_task_types'].append({
                        'benchmark_id': task_result.get('benchmark_id'),
                        'execution_time_ms': task_result.get('execution_time_ms')
                    })
                
                # Analyze low quality answers
                if task_result.get('answer_quality_score', 1.0) < 0.5:
                    bottlenecks['low_quality_patterns'].append({
                        'benchmark_id': task_result.get('benchmark_id'),
                        'quality_score': task_result.get('answer_quality_score'),
                        'steps_taken': task_result.get('steps_taken', 0)
                    })
        
        # Generate recommendations
        recommendations = []
        
        failure_categories = bottlenecks['common_failure_categories']
        if failure_categories.get('timeout', 0) > 2:
            recommendations.append("Consider increasing task timeout limits or optimizing tool execution speed")
        
        if failure_categories.get('rate_limit', 0) > 2:
            recommendations.append("Implement better rate limiting strategies or add delays between tool calls")
        
        if failure_categories.get('network', 0) > 2:
            recommendations.append("Add retry logic and better network error handling")
        
        if len(bottlenecks['slow_task_types']) > 3:
            recommendations.append("Optimize tool selection and execution for complex multi-step tasks")
        
        if len(bottlenecks['low_quality_patterns']) > 3:
            recommendations.append("Improve answer synthesis and quality validation mechanisms")
        
        bottlenecks['recommendations'] = recommendations
        
        return bottlenecks
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction from a series of values."""
        if len(values) < 2:
            return "insufficient_data"
        
        # Simple linear trend calculation
        n = len(values)
        x_mean = (n - 1) / 2
        y_mean = sum(values) / n
        
        numerator = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return "stable"
        
        slope = numerator / denominator
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"


# Utility functions for evaluation
def evaluate_single_task(
    agent_response: str,
    golden_answer: str,
    sources_found: List[str],
    expected_sources: List[str],
    execution_time_ms: int,
    steps_taken: int,
    max_steps: int
) -> EvaluationScore:
    """Evaluate a single task completion."""
    calculator = ScoreCalculator()
    
    # Calculate component scores
    answer_similarity = calculator.calculate_answer_similarity(agent_response, golden_answer)
    source_relevance = calculator.calculate_source_relevance(sources_found, expected_sources)
    efficiency = calculator.calculate_efficiency_score(steps_taken, execution_time_ms, max_steps)
    completeness = 1.0  # Default to full completeness, can be customized
    
    # Calculate overall score
    overall = calculator.calculate_overall_score(
        answer_similarity, source_relevance, efficiency, completeness
    )
    
    return EvaluationScore(
        answer_similarity=answer_similarity,
        source_relevance=source_relevance,
        efficiency_score=efficiency,
        completeness_score=completeness,
        overall_score=overall,
        details={
            'execution_time_ms': execution_time_ms,
            'steps_taken': steps_taken,
            'max_steps': max_steps,
            'sources_found_count': len(sources_found),
            'expected_sources_count': len(expected_sources)
        }
    )


if __name__ == "__main__":
    # Example usage
    calculator = ScoreCalculator()
    
    # Test answer similarity
    agent_answer = "Apple Inc. has a market capitalization of approximately $3 trillion as of 2024."
    golden_answer = "Apple's market cap is around $3 trillion in 2024."
    
    similarity = calculator.calculate_answer_similarity(agent_answer, golden_answer)
    print(f"Answer similarity: {similarity:.2f}")
    
    # Test source relevance
    sources_found = ["apple.com", "yahoo.finance", "bloomberg.com"]
    expected_sources = ["financial", "company_data", "news"]
    
    relevance = calculator.calculate_source_relevance(sources_found, expected_sources)
    print(f"Source relevance: {relevance:.2f}")
