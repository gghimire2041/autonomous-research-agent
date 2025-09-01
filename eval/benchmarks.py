"""
Synthetic benchmark generation and evaluation framework.
Creates standardized tests to measure agent performance across different research tasks.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

import structlog
from pydantic import BaseModel

from app.core.agent import ResearchAgent, TaskStatus
from app.core.planner import PlannerType
from eval.scoring import EvaluationScore, ScoreCalculator

logger = structlog.get_logger()


class BenchmarkCategory(str, Enum):
    """Categories of benchmark tasks."""
    FACT_COLLECTION = "fact_collection"
    WEB_RESEARCH = "web_research"
    CALCULATION = "calculation"
    SYNTHESIS = "synthesis"
    MULTI_STEP = "multi_step"


@dataclass
class BenchmarkTask:
    """A single benchmark task definition."""
    id: str
    category: BenchmarkCategory
    goal: str
    expected_sources: List[str]
    golden_answer: str
    max_steps: int
    timeout_seconds: int
    difficulty: str  # "easy", "medium", "hard"
    tags: List[str]


@dataclass
class BenchmarkResult:
    """Result of running a benchmark task."""
    task_id: str
    benchmark_id: str
    success: bool
    execution_time_ms: int
    steps_taken: int
    sources_found: List[str]
    answer_quality_score: float
    error_message: Optional[str] = None
    agent_response: Optional[str] = None


class BenchmarkSuite:
    """Collection of benchmark tasks for systematic evaluation."""
    
    def __init__(self):
        self.tasks = self._generate_benchmark_tasks()
        self.score_calculator = ScoreCalculator()
    
    def _generate_benchmark_tasks(self) -> List[BenchmarkTask]:
        """Generate comprehensive benchmark task suite."""
        tasks = []
        
        # Fact Collection Tasks
        tasks.extend([
            BenchmarkTask(
                id="fact_collection_easy_01",
                category=BenchmarkCategory.FACT_COLLECTION,
                goal="Find the current population of Tokyo, Japan",
                expected_sources=["wikipedia", "government", "statistics"],
                golden_answer="Tokyo has approximately 14 million people in the city proper and 37-38 million in the greater metropolitan area.",
                max_steps=5,
                timeout_seconds=120,
                difficulty="easy",
                tags=["population", "geography", "facts"]
            ),
            BenchmarkTask(
                id="fact_collection_medium_01",
                category=BenchmarkCategory.FACT_COLLECTION,
                goal="Find the latest Nobel Prize winners in Physics (2023-2024)",
                expected_sources=["nobelprize.org", "news", "academic"],
                golden_answer="2023 Nobel Prize in Physics was awarded to Pierre Agostini, Ferenc Krausz, and Anne L'Huillier for experimental methods that generate attosecond pulses of light.",
                max_steps=7,
                timeout_seconds=180,
                difficulty="medium",
                tags=["awards", "science", "recent"]
            ),
            BenchmarkTask(
                id="fact_collection_hard_01",
                category=BenchmarkCategory.FACT_COLLECTION,
                goal="Find the top 5 venture capital firms by assets under management in 2024, with specific AUM figures",
                expected_sources=["financial", "industry_reports", "company_data"],
                golden_answer="Top 5 VC firms by AUM typically include Andreessen Horowitz, Sequoia Capital, Accel, Kleiner Perkins, and Benchmark, though exact figures vary by reporting period.",
                max_steps=10,
                timeout_seconds=300,
                difficulty="hard",
                tags=["finance", "venture_capital", "rankings"]
            )
        ])
        
        # Web Research Tasks
        tasks.extend([
            BenchmarkTask(
                id="web_research_easy_01",
                category=BenchmarkCategory.WEB_RESEARCH,
                goal="Research the benefits and drawbacks of solar energy",
                expected_sources=["energy.gov", "renewable_energy", "academic"],
                golden_answer="Solar energy benefits include renewable source, reduced electricity bills, and environmental friendliness. Drawbacks include weather dependence, high initial costs, and space requirements.",
                max_steps=8,
                timeout_seconds=240,
                difficulty="easy",
                tags=["energy", "environment", "pros_cons"]
            ),
            BenchmarkTask(
                id="web_research_medium_01",
                category=BenchmarkCategory.WEB_RESEARCH,
                goal="Compare the top 3 programming languages for machine learning in 2024",
                expected_sources=["developer_surveys", "github", "industry_analysis"],
                golden_answer="Python, R, and Julia are top ML languages. Python leads with extensive libraries (scikit-learn, TensorFlow, PyTorch), R excels in statistics, Julia offers high performance computing.",
                max_steps=10,
                timeout_seconds=300,
                difficulty="medium",
                tags=["programming", "machine_learning", "comparison"]
            ),
            BenchmarkTask(
                id="web_research_hard_01",
                category=BenchmarkCategory.WEB_RESEARCH,
                goal="Analyze the regulatory landscape for AI in the EU, US, and China, focusing on recent developments in 2024",
                expected_sources=["government", "legal", "policy_analysis"],
                golden_answer="EU leads with comprehensive AI Act, US focuses on executive orders and agency guidance, China emphasizes algorithmic regulation and data governance with specific AI provisions.",
                max_steps=15,
                timeout_seconds=480,
                difficulty="hard",
                tags=["regulation", "AI_policy", "comparative_analysis"]
            )
        ])
        
        # Calculation Tasks
        tasks.extend([
            BenchmarkTask(
                id="calculation_easy_01",
                category=BenchmarkCategory.CALCULATION,
                goal="Calculate the compound interest on $10,000 invested at 5% annually for 10 years",
                expected_sources=["calculator"],
                golden_answer="$16,288.95 (using compound interest formula: A = P(1 + r)^t)",
                max_steps=3,
                timeout_seconds=60,
                difficulty="easy",
                tags=["finance", "math", "compound_interest"]
            ),
            BenchmarkTask(
                id="calculation_medium_01",
                category=BenchmarkCategory.CALCULATION,
                goal="Find the current market cap of Apple Inc. and calculate what percentage it represents of the total S&P 500 market cap",
                expected_sources=["financial_data", "calculator"],
                golden_answer="Apple's market cap varies but typically represents 6-7% of S&P 500 total market cap (approximately $3 trillion out of $45 trillion total).",
                max_steps=6,
                timeout_seconds=180,
                difficulty="medium",
                tags=["finance", "market_analysis", "percentages"]
            )
        ])
        
        # Synthesis Tasks
        tasks.extend([
            BenchmarkTask(
                id="synthesis_medium_01",
                category=BenchmarkCategory.SYNTHESIS,
                goal="Research recent developments in quantum computing and synthesize the main trends and breakthroughs",
                expected_sources=["research_papers", "tech_news", "company_announcements"],
                golden_answer="Key trends include increasing qubit counts, improving error correction, cloud access expansion, and practical applications in optimization, cryptography, and drug discovery.",
                max_steps=12,
                timeout_seconds=360,
                difficulty="medium",
                tags=["quantum_computing", "technology_trends", "synthesis"]
            ),
            BenchmarkTask(
                id="synthesis_hard_01",
                category=BenchmarkCategory.SYNTHESIS,
                goal="Analyze the global semiconductor industry supply chain challenges and synthesize potential solutions from expert opinions",
                expected_sources=["industry_reports", "expert_analysis", "news"],
                golden_answer="Solutions include supply chain diversification, increased domestic production, strategic stockpiling, improved demand forecasting, and enhanced industry cooperation.",
                max_steps=18,
                timeout_seconds=600,
                difficulty="hard",
                tags=["semiconductors", "supply_chain", "industry_analysis"]
            )
        ])
        
        # Multi-step Research Tasks
        tasks.extend([
            BenchmarkTask(
                id="multi_step_hard_01",
                category=BenchmarkCategory.MULTI_STEP,
                goal="Research the top 3 renewable energy companies by revenue, find their recent financial performance, and calculate their combined market share in the renewable sector",
                expected_sources=["financial_reports", "industry_data", "company_info"],
                golden_answer="Top companies typically include NextEra Energy, Enel, and Iberdrola. Combined they represent approximately 15-20% of global renewable energy market share based on capacity and revenue.",
                max_steps=20,
                timeout_seconds=720,
                difficulty="hard",
                tags=["renewable_energy", "financial_analysis", "market_share"]
            )
        ])
        
        return tasks
    
    async def run_benchmark_suite(
        self, 
        agent: ResearchAgent,
        categories: Optional[List[BenchmarkCategory]] = None,
        difficulty_filter: Optional[str] = None,
        max_concurrent: int = 3
    ) -> Dict[str, Any]:
        """
        Run the complete benchmark suite or filtered subset.
        
        Args:
            agent: The research agent to evaluate
            categories: Filter by task categories
            difficulty_filter: Filter by difficulty level
            max_concurrent: Maximum concurrent task executions
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info("Starting benchmark evaluation suite")
        
        # Filter tasks based on criteria
        tasks_to_run = self._filter_tasks(categories, difficulty_filter)
        
        # Run tasks in batches
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_single_task(task: BenchmarkTask) -> BenchmarkResult:
            async with semaphore:
                return await self._run_benchmark_task(agent, task)
        
        # Execute tasks concurrently
        task_coroutines = [run_single_task(task) for task in tasks_to_run]
        results = await asyncio.gather(*task_coroutines, return_exceptions=True)
        
        # Handle exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Benchmark task failed", task_id=tasks_to_run[i].id, error=str(result))
                valid_results.append(BenchmarkResult(
                    task_id="unknown",
                    benchmark_id=tasks_to_run[i].id,
                    success=False,
                    execution_time_ms=0,
                    steps_taken=0,
                    sources_found=[],
                    answer_quality_score=0.0,
                    error_message=str(result)
                ))
            else:
                valid_results.append(result)
        
        # Calculate overall metrics
        evaluation_summary = self._calculate_suite_metrics(valid_results, tasks_to_run)
        
        logger.info("Benchmark evaluation completed", 
                   total_tasks=len(tasks_to_run),
                   successful_tasks=sum(1 for r in valid_results if r.success))
        
        return {
            "summary": evaluation_summary,
            "results": valid_results,
            "tasks_run": len(tasks_to_run),
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _filter_tasks(
        self, 
        categories: Optional[List[BenchmarkCategory]], 
        difficulty_filter: Optional[str]
    ) -> List[BenchmarkTask]:
        """Filter tasks based on criteria."""
        filtered = self.tasks
        
        if categories:
            filtered = [t for t in filtered if t.category in categories]
        
        if difficulty_filter:
            filtered = [t for t in filtered if t.difficulty == difficulty_filter]
        
        return filtered
    
    async def _run_benchmark_task(
        self, 
        agent: ResearchAgent, 
        task: BenchmarkTask
    ) -> BenchmarkResult:
        """Execute a single benchmark task."""
        start_time = time.time()
        
        try:
            logger.info("Running benchmark task", task_id=task.id, goal=task.goal)
            
            # Start the research task
            agent_task_id = await agent.start_task(
                goal=task.goal,
                planner_type=PlannerType.DELIBERATIVE,
                max_steps=task.max_steps
            )
            
            # Wait for completion with timeout
            timeout_time = start_time + task.timeout_seconds
            
            while time.time() < timeout_time:
                status = await agent.get_task_status(agent_task_id)
                
                if status.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                    break
                
                await asyncio.sleep(2)
            else:
                # Timeout reached
                await agent.cancel_task(agent_task_id)
                raise TimeoutError(f"Task timed out after {task.timeout_seconds} seconds")
            
            # Get final results
            final_status = await agent.get_task_status(agent_task_id)
            execution_time_ms = int((time.time() - start_time) * 1000)
            
            # Evaluate the results
            success = final_status.status == TaskStatus.COMPLETED and final_status.result
            
            # Calculate answer quality score
            answer_quality_score = 0.0
            sources_found = []
            
            if success and final_status.result:
                answer_quality_score = self.score_calculator.calculate_answer_similarity(
                    final_status.result, task.golden_answer
                )
                
                # Extract sources from execution steps
                for step in final_status.steps:
                    if step.tool_name == "web_search" or step.tool_name == "web_fetch":
                        # Extract domains/sources from step observations
                        sources_found.extend(self._extract_sources_from_observation(step.observation))
            
            return BenchmarkResult(
                task_id=agent_task_id,
                benchmark_id=task.id,
                success=success,
                execution_time_ms=execution_time_ms,
                steps_taken=len(final_status.steps),
                sources_found=list(set(sources_found)),  # Remove duplicates
                answer_quality_score=answer_quality_score,
                agent_response=final_status.result,
                error_message=final_status.error if final_status.status == TaskStatus.FAILED else None
            )
            
        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            logger.error("Benchmark task execution failed", task_id=task.id, error=str(e))
            
            return BenchmarkResult(
                task_id="failed",
                benchmark_id=task.id,
                success=False,
                execution_time_ms=execution_time_ms,
                steps_taken=0,
                sources_found=[],
                answer_quality_score=0.0,
                error_message=str(e)
            )
    
    def _extract_sources_from_observation(self, observation: Optional[str]) -> List[str]:
        """Extract source domains from tool observations."""
        if not observation:
            return []
        
        sources = []
        # Simple extraction of URLs/domains from observation text
        import re
        
        # Extract domains from URLs
        url_pattern = r'https?://(?:www\.)?([^/\s]+)'
        matches = re.findall(url_pattern, observation.lower())
        sources.extend(matches)
        
        # Look for known source indicators
        source_keywords = {
            'wikipedia': 'wikipedia',
            'github': 'github', 
            'stack overflow': 'stackoverflow',
            'government': 'gov',
            'academic': 'edu',
            'news': 'news'
        }
        
        for keyword, source in source_keywords.items():
            if keyword in observation.lower():
                sources.append(source)
        
        return sources
    
    def _calculate_suite_metrics(
        self, 
        results: List[BenchmarkResult], 
        tasks: List[BenchmarkTask]
    ) -> Dict[str, Any]:
        """Calculate overall benchmark suite metrics."""
        if not results:
            return {}
        
        total_tasks = len(results)
        successful_tasks = sum(1 for r in results if r.success)
        
        # Success rate by category and difficulty
        category_stats = {}
        difficulty_stats = {}
        
        for task in tasks:
            task_result = next((r for r in results if r.benchmark_id == task.id), None)
            if task_result:
                # Category stats
                if task.category.value not in category_stats:
                    category_stats[task.category.value] = {"total": 0, "successful": 0}
                category_stats[task.category.value]["total"] += 1
                if task_result.success:
                    category_stats[task.category.value]["successful"] += 1
                
                # Difficulty stats  
                if task.difficulty not in difficulty_stats:
                    difficulty_stats[task.difficulty] = {"total": 0, "successful": 0}
                difficulty_stats[task.difficulty]["total"] += 1
                if task_result.success:
                    difficulty_stats[task.difficulty]["successful"] += 1
        
        # Calculate rates
        for category in category_stats:
            stats = category_stats[category]
            stats["success_rate"] = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
        
        for difficulty in difficulty_stats:
            stats = difficulty_stats[difficulty]
            stats["success_rate"] = stats["successful"] / stats["total"] if stats["total"] > 0 else 0
        
        # Performance metrics
        successful_results = [r for r in results if r.success]
        
        avg_execution_time = sum(r.execution_time_ms for r in successful_results) / len(successful_results) if successful_results else 0
        avg_steps = sum(r.steps_taken for r in successful_results) / len(successful_results) if successful_results else 0
        avg_quality_score = sum(r.answer_quality_score for r in successful_results) / len(successful_results) if successful_results else 0
        
        return {
            "overall_success_rate": successful_tasks / total_tasks,
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "failed_tasks": total_tasks - successful_tasks,
            "average_execution_time_ms": avg_execution_time,
            "average_steps_taken": avg_steps,
            "average_answer_quality": avg_quality_score,
            "category_performance": category_stats,
            "difficulty_performance": difficulty_stats
        }
    
    def get_task_by_id(self, task_id: str) -> Optional[BenchmarkTask]:
        """Get a specific benchmark task by ID."""
        return next((t for t in self.tasks if t.id == task_id), None)
    
    def get_tasks_by_category(self, category: BenchmarkCategory) -> List[BenchmarkTask]:
        """Get all tasks in a specific category."""
        return [t for t in self.tasks if t.category == category]
    
    def get_tasks_by_difficulty(self, difficulty: str) -> List[BenchmarkTask]:
        """Get all tasks of a specific difficulty."""
        return [t for t in self.tasks if t.difficulty == difficulty]


# Convenience function for running evaluations
async def run_agent_evaluation(
    agent: ResearchAgent,
    categories: Optional[List[str]] = None,
    difficulty: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to evaluate an agent against benchmarks.
    
    Args:
        agent: Research agent to evaluate
        categories: List of category names to test
        difficulty: Difficulty level filter
        
    Returns:
        Evaluation results
    """
    suite = BenchmarkSuite()
    
    # Convert category strings to enums
    category_enums = None
    if categories:
        category_enums = [BenchmarkCategory(cat) for cat in categories if cat in BenchmarkCategory.__members__.values()]
    
    return await suite.run_benchmark_suite(
        agent=agent,
        categories=category_enums,
        difficulty_filter=difficulty
    )


if __name__ == "__main__":
    # Example usage
    async def main():
        from app.core.agent import ResearchAgent
        
        agent = ResearchAgent()
        suite = BenchmarkSuite()
        
        # Run evaluation on easy tasks only
        results = await suite.run_benchmark_suite(
            agent=agent,
            difficulty_filter="easy",
            max_concurrent=2
        )
        
        print("Evaluation Results:")
        print(f"Success Rate: {results['summary']['overall_success_rate']:.2%}")
        print(f"Average Execution Time: {results['summary']['average_execution_time_ms']:.0f}ms")
        print(f"Average Quality Score: {results['summary']['average_answer_quality']:.2f}")
    
    # asyncio.run(main())
