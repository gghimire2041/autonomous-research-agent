"""
Specialized evaluation tasks for web research capabilities.
Tests the agent's ability to conduct comprehensive web-based research.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
import re

import structlog
from pydantic import BaseModel

from app.core.agent import ResearchAgent, TaskStatus
from eval.scoring import ScoreCalculator

logger = structlog.get_logger()


@dataclass
class WebResearchTask:
    """A web research evaluation task."""
    id: str
    title: str
    research_question: str
    expected_source_types: List[str]  # e.g., ["academic", "news", "government", "industry"]
    key_topics: List[str]  # Topics that should be covered
    evaluation_criteria: List[str]  # What to look for in a good answer
    complexity: str  # "simple", "moderate", "complex"
    estimated_duration_minutes: int
    requires_recent_data: bool = False


class WebResearchEvaluator:
    """Evaluator for web research capabilities."""
    
    def __init__(self):
        self.score_calculator = ScoreCalculator()
        self.research_tasks = self._create_web_research_tasks()
    
    def _create_web_research_tasks(self) -> List[WebResearchTask]:
        """Create comprehensive web research evaluation tasks."""
        return [
            # Technology Research
            WebResearchTask(
                id="tech_trends_2024",
                title="AI Technology Trends Analysis",
                research_question="What are the most significant trends in artificial intelligence technology for 2024, including breakthroughs, market developments, and future predictions?",
                expected_source_types=["tech_news", "research_papers", "industry_reports", "company_announcements"],
                key_topics=["machine_learning", "large_language_models", "AI_applications", "market_trends", "investment"],
                evaluation_criteria=[
                    "Identifies multiple recent AI developments",
                    "Covers both technical and business aspects", 
                    "Cites credible sources",
                    "Provides specific examples and data",
                    "Discusses future implications"
                ],
                complexity="moderate",
                estimated_duration_minutes=8,
                requires_recent_data=True
            ),
            
            WebResearchTask(
                id="renewable_energy_comparison",
                title="Renewable Energy Technologies Comparison",
                research_question="Compare solar, wind, and hydroelectric power in terms of efficiency, cost, environmental impact, and global adoption rates. Include recent developments and future outlook.",
                expected_source_types=["government", "environmental_agencies", "industry_data", "academic"],
                key_topics=["solar_energy", "wind_power", "hydroelectric", "efficiency", "costs", "environmental_impact"],
                evaluation_criteria=[
                    "Provides quantitative comparisons",
                    "Covers all three energy types thoroughly",
                    "Discusses environmental pros/cons",
                    "Includes recent cost and efficiency data",
                    "Mentions global adoption statistics"
                ],
                complexity="moderate",
                estimated_duration_minutes=10
            ),
            
            # Market Research
            WebResearchTask(
                id="ev_market_analysis",
                title="Electric Vehicle Market Analysis",
                research_question="Analyze the current state of the global electric vehicle market, including leading manufacturers, market share, growth trends, and barriers to adoption.",
                expected_source_types=["automotive_industry", "market_research", "financial_reports", "news"],
                key_topics=["Tesla", "traditional_automakers", "market_share", "sales_growth", "charging_infrastructure", "government_policies"],
                evaluation_criteria=[
                    "Identifies key market players and their positions",
                    "Provides market size and growth data",
                    "Discusses adoption challenges and solutions",
                    "Covers regional market differences",
                    "Includes recent sales/production figures"
                ],
                complexity="complex",
                estimated_duration_minutes=12,
                requires_recent_data=True
            ),
            
            # Scientific Research
            WebResearchTask(
                id="climate_change_impacts",
                title="Climate Change Regional Impacts",
                research_question="Research the specific impacts of climate change on agriculture, water resources, and coastal areas in Southeast Asia, including current effects and future projections.",
                expected_source_types=["scientific_journals", "UN_reports", "government_climate_data", "NGO_reports"],
                key_topics=["agriculture_impacts", "water_scarcity", "sea_level_rise", "extreme_weather", "adaptation_strategies"],
                evaluation_criteria=[
                    "Covers all three impact areas comprehensively",
                    "Focuses specifically on Southeast Asia",
                    "Includes both current and projected impacts",
                    "Cites authoritative climate sources",
                    "Discusses adaptation/mitigation measures"
                ],
                complexity="complex",
                estimated_duration_minutes=15
            ),
            
            # Comparative Analysis
            WebResearchTask(
                id="social_media_platforms_comparison",
                title="Social Media Platform Business Models",
                research_question="Compare the business models, user demographics, and revenue strategies of TikTok, Instagram, YouTube, and Twitter/X. How have these evolved recently?",
                expected_source_types=["business_news", "company_reports", "market_analysis", "user_statistics"],
                key_topics=["advertising_models", "user_demographics", "content_monetization", "platform_features", "revenue_growth"],
                evaluation_criteria=[
                    "Compares all four platforms systematically",
                    "Explains different monetization approaches",
                    "Provides user demographic data",
                    "Discusses recent changes and developments",
                    "Analyzes competitive positioning"
                ],
                complexity="moderate",
                estimated_duration_minutes=9,
                requires_recent_data=True
            ),
            
            # Policy Research
            WebResearchTask(
                id="ai_regulation_global",
                title="Global AI Regulation Landscape",
                research_question="Compare AI regulation approaches in the United States, European Union, China, and United Kingdom. What are the key differences and potential impacts on AI development?",
                expected_source_types=["government_documents", "legal_analysis", "policy_reports", "news"],
                key_topics=["EU_AI_Act", "US_executive_orders", "China_AI_regulations", "UK_approach", "regulatory_frameworks"],
                evaluation_criteria=[
                    "Covers regulatory approaches in all mentioned regions",
                    "Identifies key differences in approaches",
                    "Discusses specific regulations and policies",
                    "Analyzes potential industry impacts",
                    "Includes timeline of regulatory developments"
                ],
                complexity="complex",
                estimated_duration_minutes=18,
                requires_recent_data=True
            ),
            
            # Industry Analysis
            WebResearchTask(
                id="semiconductor_supply_chain",
                title="Global Semiconductor Supply Chain",
                research_question="Analyze the global semiconductor supply chain, including major manufacturers, geographic concentration, recent disruptions, and efforts to increase supply chain resilience.",
                expected_source_types=["industry_reports", "government_analysis", "company_data", "trade_statistics"],
                key_topics=["TSMC", "Samsung", "Intel", "supply_chain_bottlenecks", "geopolitical_factors", "manufacturing_locations"],
                evaluation_criteria=[
                    "Maps out key players in the supply chain",
                    "Discusses geographic concentration risks",
                    "Covers recent disruption events",
                    "Explains government/industry responses",
                    "Provides production and capacity data"
                ],
                complexity="complex",
                estimated_duration_minutes=20
            ),
            
            # Simple Research Tasks
            WebResearchTask(
                id="python_vs_javascript",
                title="Python vs JavaScript for Beginners",
                research_question="Compare Python and JavaScript as first programming languages for beginners, considering learning curve, job market, and typical applications.",
                expected_source_types=["educational_resources", "developer_surveys", "job_boards", "coding_bootcamps"],
                key_topics=["syntax_difficulty", "learning_resources", "job_opportunities", "use_cases", "community_support"],
                evaluation_criteria=[
                    "Fairly compares both languages",
                    "Considers beginner perspective",
                    "Discusses practical applications",
                    "Mentions job market demand",
                    "Provides learning resource recommendations"
                ],
                complexity="simple",
                estimated_duration_minutes=6
            ),
            
            WebResearchTask(
                id="remote_work_productivity",
                title="Remote Work Productivity Research",
                research_question="What does recent research show about remote work productivity compared to office work? Include studies, statistics, and expert opinions.",
                expected_source_types=["academic_studies", "business_research", "survey_data", "expert_opinions"],
                key_topics=["productivity_metrics", "employee_satisfaction", "collaboration_challenges", "work_life_balance", "company_policies"],
                evaluation_criteria=[
                    "Cites specific studies and research",
                    "Presents balanced view of pros/cons",
                    "Includes quantitative data where available",
                    "Covers different industry perspectives",
                    "Discusses implementation best practices"
                ],
                complexity="moderate",
                estimated_duration_minutes=8
            )
        ]
    
    async def evaluate_web_research_capability(
        self,
        agent: ResearchAgent,
        task_ids: Optional[List[str]] = None,
        complexity_filter: Optional[str] = None,
        max_concurrent: int = 2
    ) -> Dict[str, Any]:
        """
        Evaluate agent's web research capabilities across multiple tasks.
        
        Args:
            agent: Research agent to evaluate
            task_ids: Specific task IDs to run (None for all)
            complexity_filter: Filter by complexity level
            max_concurrent: Maximum concurrent evaluations
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting web research capability evaluation")
        
        # Filter tasks
        tasks_to_run = self.research_tasks
        if task_ids:
            tasks_to_run = [t for t in tasks_to_run if t.id in task_ids]
        if complexity_filter:
            tasks_to_run = [t for t in tasks_to_run if t.complexity == complexity_filter]
        
        # Run evaluations
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_single_evaluation(task: WebResearchTask) -> Dict[str, Any]:
            async with semaphore:
                return await self._evaluate_research_task(agent, task)
        
        results = await asyncio.gather(*[run_single_evaluation(task) for task in tasks_to_run])
        
        # Analyze results
        analysis = self._analyze_research_results(results, tasks_to_run)
        
        return {
            'evaluation_type': 'web_research',
            'timestamp': datetime.utcnow().isoformat(),
            'tasks_evaluated': len(tasks_to_run),
            'results': results,
            'analysis': analysis
        }
    
    async def _evaluate_research_task(
        self, 
        agent: ResearchAgent, 
        task: WebResearchTask
    ) -> Dict[str, Any]:
        """Evaluate a single research task."""
        start_time = datetime.utcnow()
        
        logger.info(f"Evaluating research task: {task.id}")
        
        try:
            # Start agent task
            agent_task_id = await agent.start_task(
                goal=task.research_question,
                max_steps=15  # Allow more steps for complex research
            )
            
            # Monitor execution with timeout
            timeout = timedelta(minutes=task.estimated_duration_minutes + 5)  # Add buffer
            
            while (datetime.utcnow() - start_time) < timeout:
                status = await agent.get_task_status(agent_task_id)
                
                if status.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    break
                
                await asyncio.sleep(3)
            else:
                # Timeout reached
                await agent.cancel_task(agent_task_id)
                return self._create_timeout_result(task, start_time)
            
            # Get final results
            final_status = await agent.get_task_status(agent_task_id)
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            if final_status.status == TaskStatus.COMPLETED and final_status.result:
                # Evaluate research quality
                evaluation_scores = await self._evaluate_research_quality(
                    final_status, task
                )
                
                return {
                'task_id': task.id,
                'title': task.title,
                'success': False,
                'execution_time_seconds': execution_time,
                'error': str(e),
                'overall_score': 0.0
            }
    
    def _create_timeout_result(self, task: WebResearchTask, start_time: datetime) -> Dict[str, Any]:
        """Create result object for timed-out task."""
        execution_time = (datetime.utcnow() - start_time).total_seconds()
        
        return {
            'task_id': task.id,
            'title': task.title,
            'success': False,
            'execution_time_seconds': execution_time,
            'error': f"Task timeout after {task.estimated_duration_minutes + 5} minutes",
            'overall_score': 0.0
        }
    
    async def _evaluate_research_quality(
        self, 
        task_result, 
        task: WebResearchTask
    ) -> Dict[str, Any]:
        """Evaluate the quality of research conducted."""
        response = task_result.result
        steps = task_result.steps
        
        scores = {}
        
        # 1. Topic Coverage Score
        scores['topic_coverage'] = self._evaluate_topic_coverage(response, task.key_topics)
        
        # 2. Source Quality Score  
        sources = self._extract_sources(steps)
        scores['source_quality'] = self._evaluate_source_quality(sources, task.expected_source_types)
        
        # 3. Information Depth Score
        scores['information_depth'] = self._evaluate_information_depth(response, task.evaluation_criteria)
        
        # 4. Recency Score (if required)
        if task.requires_recent_data:
            scores['recency'] = self._evaluate_recency(response)
        else:
            scores['recency'] = 1.0  # Not applicable
        
        # 5. Coherence and Structure Score
        scores['coherence'] = self._evaluate_coherence(response)
        
        # 6. Factual Accuracy Indicators
        scores['accuracy_indicators'] = self._evaluate_accuracy_indicators(response, steps)
        
        # Calculate overall score
        weights = {
            'topic_coverage': 0.25,
            'source_quality': 0.20,
            'information_depth': 0.25,
            'recency': 0.10,
            'coherence': 0.10,
            'accuracy_indicators': 0.10
        }
        
        overall_score = sum(scores[key] * weights[key] for key in weights)
        scores['overall_score'] = min(1.0, max(0.0, overall_score))
        
        return scores
    
    def _evaluate_topic_coverage(self, response: str, key_topics: List[str]) -> float:
        """Evaluate how well the response covers key topics."""
        if not key_topics:
            return 1.0
        
        response_lower = response.lower()
        topics_covered = 0
        
        for topic in key_topics:
            topic_variations = [
                topic.lower(),
                topic.lower().replace('_', ' '),
                topic.lower().replace('_', '-')
            ]
            
            if any(variation in response_lower for variation in topic_variations):
                topics_covered += 1
        
        return topics_covered / len(key_topics)
    
    def _evaluate_source_quality(self, sources: List[str], expected_types: List[str]) -> float:
        """Evaluate the quality and relevance of sources used."""
        if not expected_types:
            return 1.0
        
        if not sources:
            return 0.0
        
        # Source quality mapping
        quality_indicators = {
            'academic': ['edu', 'scholar', 'research', 'journal', 'university'],
            'government': ['gov', 'official', 'ministry', 'department', 'agency'],
            'news': ['news', 'times', 'post', 'reuters', 'bloomberg', 'cnn', 'bbc'],
            'industry': ['report', 'analysis', 'market', 'consulting', 'business'],
            'tech_news': ['techcrunch', 'verge', 'ars', 'wired', 'technology'],
            'financial': ['financial', 'bloomberg', 'reuters', 'wsj', 'finance']
        }
        
        # Count matches
        type_matches = 0
        for expected_type in expected_types:
            if expected_type in quality_indicators:
                indicators = quality_indicators[expected_type]
                for source in sources:
                    source_lower = source.lower()
                    if any(indicator in source_lower for indicator in indicators):
                        type_matches += 1
                        break
        
        base_score = type_matches / len(expected_types)
        
        # Bonus for diverse sources
        diversity_bonus = min(0.2, len(set(sources)) * 0.05)
        
        return min(1.0, base_score + diversity_bonus)
    
    def _evaluate_information_depth(self, response: str, evaluation_criteria: List[str]) -> float:
        """Evaluate depth and comprehensiveness of information."""
        if not evaluation_criteria:
            return 1.0
        
        response_lower = response.lower()
        criteria_met = 0
        
        # Define keywords for each type of criterion
        criterion_keywords = {
            'quantitative': ['percent', '%', 'million', 'billion', 'thousand', 'number', 'rate', 'statistics'],
            'comparison': ['compare', 'versus', 'vs', 'difference', 'similar', 'unlike', 'while', 'however'],
            'examples': ['example', 'such as', 'including', 'like', 'instance', 'case'],
            'recent': ['2024', '2023', 'recent', 'latest', 'current', 'new', 'updated'],
            'specific': ['specifically', 'particular', 'detailed', 'precisely', 'exactly'],
            'future': ['future', 'predict', 'forecast', 'expect', 'trend', 'outlook', 'projection']
        }
        
        for criterion in evaluation_criteria:
            criterion_lower = criterion.lower()
            
            # Direct keyword matching
            if any(word in response_lower for word in criterion_lower.split()):
                criteria_met += 1
                continue
            
            # Category-based matching
            for category, keywords in criterion_keywords.items():
                if category in criterion_lower:
                    if any(keyword in response_lower for keyword in keywords):
                        criteria_met += 1
                        break
        
        base_score = criteria_met / len(evaluation_criteria)
        
        # Length bonus for comprehensive responses
        word_count = len(response.split())
        if word_count > 500:
            length_bonus = min(0.1, (word_count - 500) / 5000)  # Up to 0.1 bonus
            return min(1.0, base_score + length_bonus)
        
        return base_score
    
    def _evaluate_recency(self, response: str) -> float:
        """Evaluate if the response includes recent information."""
        current_year = datetime.now().year
        response_lower = response.lower()
        
        # Look for recent years
        recent_years = [str(current_year), str(current_year - 1)]
        year_mentions = sum(1 for year in recent_years if year in response)
        
        # Look for recency indicators
        recency_indicators = ['recent', 'latest', 'current', 'new', 'updated', '2024', '2023']
        recency_mentions = sum(1 for indicator in recency_indicators if indicator in response_lower)
        
        # Score based on presence of recent information
        if year_mentions >= 2 and recency_mentions >= 3:
            return 1.0
        elif year_mentions >= 1 and recency_mentions >= 2:
            return 0.8
        elif year_mentions >= 1 or recency_mentions >= 2:
            return 0.6
        else:
            return 0.3
    
    def _evaluate_coherence(self, response: str) -> float:
        """Evaluate the coherence and structure of the response."""
        if not response:
            return 0.0
        
        sentences = response.split('.')
        if len(sentences) < 3:
            return 0.5  # Too short to evaluate properly
        
        # Check for structure indicators
        structure_score = 0.0
        
        # Look for transition words
        transitions = ['however', 'moreover', 'furthermore', 'additionally', 'in contrast', 
                      'on the other hand', 'therefore', 'consequently', 'meanwhile', 'subsequently']
        transition_count = sum(1 for trans in transitions if trans in response.lower())
        structure_score += min(0.3, transition_count * 0.1)
        
        # Look for organizational elements
        org_elements = ['first', 'second', 'third', 'finally', 'in conclusion', 'to summarize']
        org_count = sum(1 for elem in org_elements if elem in response.lower())
        structure_score += min(0.2, org_count * 0.1)
        
        # Check paragraph-like structure (double newlines or clear sections)
        if '\n\n' in response or response.count('\n') > 3:
            structure_score += 0.2
        
        # Sentence length variety (good writing has varied sentence lengths)
        sentence_lengths = [len(s.split()) for s in sentences if s.strip()]
        if sentence_lengths:
            avg_length = sum(sentence_lengths) / len(sentence_lengths)
            if 10 <= avg_length <= 25:  # Good average sentence length
                structure_score += 0.2
        
        # Repetition check (too much repetition indicates poor coherence)
        words = response.lower().split()
        word_freq = {}
        for word in words:
            if len(word) > 4:  # Only count meaningful words
                word_freq[word] = word_freq.get(word, 0) + 1
        
        if word_freq:
            max_freq = max(word_freq.values())
            if max_freq / len(words) > 0.1:  # Too much repetition
                structure_score -= 0.1
        
        return min(1.0, max(0.0, structure_score))
    
    def _evaluate_accuracy_indicators(self, response: str, steps: List) -> float:
        """Evaluate indicators of factual accuracy."""
        accuracy_score = 0.5  # Start with neutral score
        
        response_lower = response.lower()
        
        # Positive indicators
        positive_indicators = [
            'according to', 'reported by', 'data shows', 'study found',
            'research indicates', 'statistics show', 'survey reveals'
        ]
        positive_count = sum(1 for indicator in positive_indicators if indicator in response_lower)
        accuracy_score += min(0.3, positive_count * 0.1)
        
        # Source citations
        citation_patterns = [r'\(20\d{2}\)', r'\.com', r'\.org', r'\.edu', r'\.gov']
        citation_count = sum(1 for pattern in citation_patterns 
                           if len(re.findall(pattern, response)) > 0)
        accuracy_score += min(0.2, citation_count * 0.05)
        
        # Negative indicators (uncertainty without justification)
        uncertainty_without_context = ['might be', 'could be', 'possibly', 'maybe']
        uncertainty_count = sum(1 for indicator in uncertainty_without_context 
                              if indicator in response_lower)
        if uncertainty_count > 3:
            accuracy_score -= 0.1
        
        return min(1.0, max(0.0, accuracy_score))
    
    def _extract_tools_used(self, steps: List) -> List[str]:
        """Extract tools used during task execution."""
        tools = []
        for step in steps:
            if hasattr(step, 'tool_name') and step.tool_name:
                tools.append(step.tool_name)
        return list(set(tools))  # Remove duplicates
    
    def _extract_sources(self, steps: List) -> List[str]:
        """Extract sources accessed during research."""
        sources = []
        for step in steps:
            if hasattr(step, 'observation') and step.observation:
                # Extract URLs and domain names from observations
                urls = re.findall(r'https?://(?:www\.)?([^/\s]+)', step.observation)
                sources.extend(urls)
        return list(set(sources))  # Remove duplicates
    
    def _analyze_research_results(
        self, 
        results: List[Dict[str, Any]], 
        tasks: List[WebResearchTask]
    ) -> Dict[str, Any]:
        """Analyze overall research performance."""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.get('success', False)]
        
        # Overall metrics
        success_rate = len(successful_results) / len(results)
        avg_score = sum(r.get('overall_score', 0) for r in results) / len(results)
        avg_execution_time = sum(r.get('execution_time_seconds', 0) for r in successful_results) / len(successful_results) if successful_results else 0
        
        # Performance by complexity
        complexity_performance = {}
        for task in tasks:
            task_result = next((r for r in results if r.get('task_id') == task.id), None)
            if task_result:
                if task.complexity not in complexity_performance:
                    complexity_performance[task.complexity] = {
                        'total': 0, 'successful': 0, 'scores': [], 'avg_time': []
                    }
                
                perf = complexity_performance[task.complexity]
                perf['total'] += 1
                perf['scores'].append(task_result.get('overall_score', 0))
                
                if task_result.get('success', False):
                    perf['successful'] += 1
                    perf['avg_time'].append(task_result.get('execution_time_seconds', 0))
        
        # Calculate complexity metrics
        for complexity in complexity_performance:
            perf = complexity_performance[complexity]
            perf['success_rate'] = perf['successful'] / perf['total'] if perf['total'] > 0 else 0
            perf['average_score'] = sum(perf['scores']) / len(perf['scores']) if perf['scores'] else 0
            perf['average_time'] = sum(perf['avg_time']) / len(perf['avg_time']) if perf['avg_time'] else 0
        
        # Tool usage analysis
        all_tools_used = []
        for result in successful_results:
            if 'tools_used' in result:
                all_tools_used.extend(result['tools_used'])
        
        tool_usage_freq = {}
        for tool in all_tools_used:
            tool_usage_freq[tool] = tool_usage_freq.get(tool, 0) + 1
        
        # Source quality analysis
        all_sources = []
        for result in successful_results:
            if 'sources_accessed' in result:
                all_sources.extend(result['sources_accessed'])
        
        unique_sources = len(set(all_sources))
        avg_sources_per_task = len(all_sources) / len(successful_results) if successful_results else 0
        
        return {
            'overall_success_rate': success_rate,
            'average_score': avg_score,
            'average_execution_time_seconds': avg_execution_time,
            'tasks_evaluated': len(results),
            'successful_tasks': len(successful_results),
            'complexity_performance': complexity_performance,
            'tool_usage_frequency': tool_usage_freq,
            'source_analysis': {
                'unique_sources_accessed': unique_sources,
                'average_sources_per_task': avg_sources_per_task,
                'total_sources': len(all_sources)
            },
            'performance_insights': self._generate_performance_insights(
                complexity_performance, tool_usage_freq, success_rate, avg_score
            )
        }
    
    def _generate_performance_insights(
        self, 
        complexity_perf: Dict[str, Any],
        tool_usage: Dict[str, int],
        success_rate: float,
        avg_score: float
    ) -> List[str]:
        """Generate actionable insights from performance data."""
        insights = []
        
        # Overall performance insights
        if success_rate >= 0.9:
            insights.append("Excellent overall performance across research tasks")
        elif success_rate >= 0.7:
            insights.append("Good performance with room for improvement in complex tasks")
        else:
            insights.append("Performance needs improvement, focus on basic research skills")
        
        # Complexity-specific insights
        for complexity, perf in complexity_perf.items():
            if perf['success_rate'] < 0.6:
                insights.append(f"Struggles with {complexity} complexity tasks - consider additional training")
            elif perf['success_rate'] >= 0.9:
                insights.append(f"Excels at {complexity} complexity research tasks")
        
        # Tool usage insights
        if tool_usage:
            most_used_tool = max(tool_usage.items(), key=lambda x: x[1])
            if most_used_tool[1] > len(tool_usage) * 2:
                insights.append(f"Over-reliant on {most_used_tool[0]} tool - encourage diverse tool usage")
            
            if 'web_search' not in tool_usage:
                insights.append("Not utilizing web search effectively - may miss important information")
        
        # Score-based insights
        if avg_score < 0.6:
            insights.append("Low information quality scores - focus on source credibility and depth")
        elif avg_score >= 0.8:
            insights.append("High quality research outputs - maintain current standards")
        
        return insights


# Example usage and testing functions
async def run_web_research_evaluation():
    """Example function to run web research evaluation."""
    from app.core.agent import ResearchAgent
    
    agent = ResearchAgent()
    evaluator = WebResearchEvaluator()
    
    # Run evaluation on simple tasks
    results = await evaluator.evaluate_web_research_capability(
        agent=agent,
        complexity_filter="simple",
        max_concurrent=1
    )
    
    print("Web Research Evaluation Results:")
    print(f"Overall Success Rate: {results['analysis']['overall_success_rate']:.2%}")
    print(f"Average Score: {results['analysis']['average_score']:.2f}")
    print(f"Average Execution Time: {results['analysis']['average_execution_time_seconds']:.1f}s")
    
    if results['analysis']['performance_insights']:
        print("Performance Insights:")
        for insight in results['analysis']['performance_insights']:
            print(f"  • {insight}")


def create_custom_research_task(
    task_id: str,
    title: str,
    research_question: str,
    expected_source_types: List[str],
    key_topics: List[str],
    complexity: str = "moderate",
    estimated_minutes: int = 10
) -> WebResearchTask:
    """
    Create a custom web research task.
    
    Args:
        task_id: Unique identifier
        title: Human-readable title
        research_question: The research question
        expected_source_types: Types of sources expected
        key_topics: Key topics to cover
        complexity: Task complexity level
        estimated_minutes: Estimated completion time
        
    Returns:
        WebResearchTask instance
    """
    return WebResearchTask(
        id=task_id,
        title=title,
        research_question=research_question,
        expected_source_types=expected_source_types,
        key_topics=key_topics,
        evaluation_criteria=[
            "Provides comprehensive information",
            "Uses credible sources",
            "Covers key aspects thoroughly"
        ],
        complexity=complexity,
        estimated_duration_minutes=estimated_minutes
    )


if __name__ == "__main__":
    # Example of running the evaluation
    import asyncio
    
    # asyncio.run(run_web_research_evaluation())
    
    # Example of creating a custom research task
    custom_task = create_custom_research_task(
        task_id="custom_crypto_research",
        title="Cryptocurrency Market Analysis",
        research_question="What are the current trends in cryptocurrency adoption and regulation?",
        expected_source_types=["financial_news", "regulatory_documents", "market_analysis"],
        key_topics=["bitcoin", "ethereum", "regulation", "institutional_adoption"],
        complexity="moderate"
    )
    
    print(f"Created custom research task: {custom_task.id}")
    print(f"Research Question: {custom_task.research_question}")

                    'task_id': task.id,
                    'title': task.title,
                    'success': True,
                    'execution_time_seconds': execution_time,
                    'steps_taken': len(final_status.steps),
                    'agent_response': final_status.result,
                    'evaluation_scores': evaluation_scores,
                    'overall_score': evaluation_scores.get('overall_score', 0.0),
                    'tools_used': self._extract_tools_used(final_status.steps),
                    'sources_accessed': self._extract_sources(final_status.steps)
                }
            else:
                return {
                    'task_id': task.id,
                    'title': task.title,
                    'success': False,
                    'execution_time_seconds': execution_time,
                    'steps_taken': len(final_status.steps),
                    'error': final_status.error or "Task failed without specific error",
                    'overall_score': 0.0
                }
                
        except Exception as e:
            logger.error(f"Research task evaluation failed: {task.id}", error=str(e))
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            return {
