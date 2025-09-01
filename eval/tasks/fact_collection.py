"""
Specialized evaluation tasks for fact collection capabilities.
Tests the agent's ability to find and extract specific factual information.
"""

import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import structlog
from pydantic import BaseModel

from app.core.agent import ResearchAgent, TaskStatus
from eval.scoring import ScoreCalculator, EvaluationScore

logger = structlog.get_logger()


class FactCollectionTask(BaseModel):
    """A specific fact collection task definition."""
    id: str
    description: str
    query: str
    expected_facts: List[str]
    fact_types: List[str]  # e.g., ["number", "date", "name", "location"]
    verification_sources: List[str]
    difficulty: str
    max_execution_time: int = 300  # seconds


class FactCollectionEvaluator:
    """Specialized evaluator for fact collection tasks."""
    
    def __init__(self):
        self.score_calculator = ScoreCalculator()
        self.tasks = self._create_fact_collection_tasks()
    
    def _create_fact_collection_tasks(self) -> List[FactCollectionTask]:
        """Create comprehensive fact collection task suite."""
        return [
            # Basic factual queries
            FactCollectionTask(
                id="population_query_01",
                description="Find current population of major world cities",
                query="What is the current population of Tokyo, New York, and London?",
                expected_facts=[
                    "Tokyo: ~14 million (city), ~37-38 million (metro area)",
                    "New York: ~8.3 million (city), ~20 million (metro area)", 
                    "London: ~9 million (city), ~15 million (metro area)"
                ],
                fact_types=["number", "location"],
                verification_sources=["government_statistics", "UN_data", "city_official"],
                difficulty="easy"
            ),
            
            FactCollectionTask(
                id="company_founding_01", 
                description="Find founding dates and founders of tech companies",
                query="When were Google, Microsoft, and Apple founded, and who were their founders?",
                expected_facts=[
                    "Google: Founded 1998 by Larry Page and Sergey Brin",
                    "Microsoft: Founded 1975 by Bill Gates and Paul Allen",
                    "Apple: Founded 1976 by Steve Jobs, Steve Wozniak, and Ronald Wayne"
                ],
                fact_types=["date", "name", "company"],
                verification_sources=["company_official", "business_records", "biography"],
                difficulty="easy"
            ),
            
            # Scientific facts
            FactCollectionTask(
                id="scientific_constants_01",
                description="Find precise values of fundamental scientific constants",
                query="What are the exact values of the speed of light, Planck's constant, and Avogadro's number?",
                expected_facts=[
                    "Speed of light: 299,792,458 m/s",
                    "Planck's constant: 6.62607015×10^-34 J⋅Hz^-1",
                    "Avogadro's number: 6.02214076×10^23 mol^-1"
                ],
                fact_types=["number", "scientific_constant"],
                verification_sources=["NIST", "scientific_literature", "physics_reference"],
                difficulty="medium"
            ),
            
            # Economic/Financial facts
            FactCollectionTask(
                id="gdp_rankings_01",
                description="Find current GDP rankings of countries",
                query="What are the top 5 countries by nominal GDP in 2023-2024?",
                expected_facts=[
                    "1. United States (~$26.9 trillion)",
                    "2. China (~$17.7 trillion)",
                    "3. Japan (~$4.9 trillion)",
                    "4. Germany (~$4.3 trillion)",
                    "5. India (~$3.7 trillion)"
                ],
                fact_types=["number", "ranking", "country"],
                verification_sources=["World_Bank", "IMF", "government_statistics"],
                difficulty="medium"
            ),
            
            # Recent/Dynamic facts
            FactCollectionTask(
                id="recent_events_01",
                description="Find recent Nobel Prize winners",
                query="Who won the Nobel Prizes in Physics, Chemistry, and Medicine in 2023?",
                expected_facts=[
                    "Physics 2023: Pierre Agostini, Ferenc Krausz, Anne L'Huillier (attosecond pulses)",
                    "Chemistry 2023: Moungi Bawendi, Louis Brus, Alexei Ekimov (quantum dots)",
                    "Medicine 2023: Katalin Karikó, Drew Weissman (mRNA vaccines)"
                ],
                fact_types=["name", "award", "date", "field"],
                verification_sources=["nobelprize.org", "scientific_news", "official_announcement"],
                difficulty="medium"
            ),
            
            # Complex multi-part facts
            FactCollectionTask(
                id="climate_data_01",
                description="Find specific climate change statistics",
                query="What was the global average temperature increase since 1880, current CO2 levels, and the rate of sea level rise?",
                expected_facts=[
                    "Global temperature increase: ~1.1°C since 1880",
                    "Current CO2 levels: ~420+ ppm (as of 2024)",
                    "Sea level rise rate: ~3.3 mm/year"
                ],
                fact_types=["number", "measurement", "rate", "environmental"],
                verification_sources=["NOAA", "NASA", "IPCC", "climate_research"],
                difficulty="hard"
            ),
            
            # Technical specifications
            FactCollectionTask(
                id="tech_specs_01",
                description="Find technical specifications of latest hardware",
                query="What are the key specifications of the latest iPhone, Samsung Galaxy flagship, and Google Pixel?",
                expected_facts=[
                    "iPhone 15 Pro: A17 Pro chip, 48MP main camera, 6.1\" display",
                    "Samsung Galaxy S24 Ultra: Snapdragon 8 Gen 3, 200MP camera, 6.8\" display",
                    "Google Pixel 8 Pro: Tensor G3, 50MP main camera, 6.7\" display"
                ],
                fact_types=["specification", "technology", "product"],
                verification_sources=["manufacturer_official", "tech_reviews", "specification_database"],
                difficulty="hard",
                max_execution_time=420  # Longer for complex specs
            )
        ]
    
    async def evaluate_fact_collection_capability(
        self, 
        agent: ResearchAgent,
        task_ids: Optional[List[str]] = None,
        difficulty_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Evaluate agent's fact collection capabilities.
        
        Args:
            agent: Research agent to evaluate
            task_ids: Specific task IDs to run (None for all)
            difficulty_filter: Filter by difficulty level
            
        Returns:
            Detailed evaluation results
        """
        logger.info("Starting fact collection evaluation")
        
        # Filter tasks
        tasks_to_run = self.tasks
        if task_ids:
            tasks_to_run = [t for t in tasks_to_run if t.id in task_ids]
        if difficulty_filter:
            tasks_to_run = [t for t in tasks_to_run if t.difficulty == difficulty_filter]
        
        results = []
        
        for task in tasks_to_run:
            logger.info(f"Running fact collection task: {task.id}")
            
            try:
                result = await self._run_fact_collection_task(agent, task)
                results.append(result)
            except Exception as e:
                logger.error(f"Task {task.id} failed", error=str(e))
                results.append({
                    'task_id': task.id,
                    'success': False,
                    'error': str(e),
                    'score': 0.0
                })
        
        # Analyze results
        analysis = self._analyze_fact_collection_results(results, tasks_to_run)
        
        return {
            'evaluation_type': 'fact_collection',
            'timestamp': datetime.utcnow().isoformat(),
            'tasks_evaluated': len(tasks_to_run),
            'results': results,
            'analysis': analysis
        }
    
    async def _run_fact_collection_task(
        self, 
        agent: ResearchAgent, 
        task: FactCollectionTask
    ) -> Dict[str, Any]:
        """Run a single fact collection task."""
        start_time = datetime.utcnow()
        
        # Start agent task
        agent_task_id = await agent.start_task(
            goal=task.query,
            max_steps=10
        )
        
        # Wait for completion with timeout
        timeout = timedelta(seconds=task.max_execution_time)
        
        while (datetime.utcnow() - start_time) < timeout:
            status = await agent.get_task_status(agent_task_id)
            
            if status.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                break
                
            await asyncio.sleep(2)
        else:
            # Timeout
            await agent.cancel_task(agent_task_id)
            return {
                'task_id': task.id,
                'success': False,
                'error': 'Task timeout',
                'execution_time_ms': task.max_execution_time * 1000,
                'score': 0.0
            }
        
        # Get final results
        final_status = await agent.get_task_status(agent_task_id)
        execution_time = (datetime.utcnow() - start_time).total_seconds() * 1000
        
        # Evaluate fact extraction quality
        if final_status.status == TaskStatus.COMPLETED and final_status.result:
            fact_extraction_score = self._evaluate_fact_extraction(
                final_status.result, 
                task.expected_facts,
                task.fact_types
            )
            
            success = fact_extraction_score > 0.5
        else:
            fact_extraction_score = 0.0
            success = False
        
        return {
            'task_id': task.id,
            'agent_task_id': agent_task_id,
            'success': success,
            'execution_time_ms': execution_time,
            'steps_taken': len(final_status.steps),
            'agent_response': final_status.result,
            'expected_facts': task.expected_facts,
            'fact_extraction_score': fact_extraction_score,
            'score': fact_extraction_score,
            'error': final_status.error if final_status.status == TaskStatus.FAILED else None
        }
    
    def _evaluate_fact_extraction(
        self, 
        agent_response: str, 
        expected_facts: List[str],
        fact_types: List[str]
    ) -> float:
        """Evaluate how well facts were extracted from the response."""
        if not agent_response:
            return 0.0
        
        response_lower = agent_response.lower()
        facts_found = 0
        total_facts = len(expected_facts)
        
        for expected_fact in expected_facts:
            if self._is_fact_present(response_lower, expected_fact, fact_types):
                facts_found += 1
        
        # Base score from fact coverage
        coverage_score = facts_found / total_facts if total_facts > 0 else 0.0
        
        # Bonus for additional relevant facts
        additional_facts_bonus = min(0.1, self._count_additional_facts(agent_response) * 0.02)
        
        # Penalty for inaccurate facts
        accuracy_penalty = self._calculate_accuracy_penalty(agent_response, expected_facts)
        
        final_score = coverage_score + additional_facts_bonus - accuracy_penalty
        return max(0.0, min(1.0, final_score))
    
    def _is_fact_present(self, response: str, expected_fact: str, fact_types: List[str]) -> bool:
        """Check if a specific fact is present in the response."""
        import re
        
        expected_lower = expected_fact.lower()
        
        # Extract key components from expected fact
        if "number" in fact_types:
            # Extract numbers from expected fact
            expected_numbers = re.findall(r'[\d,]+\.?\d*', expected_fact)
            response_numbers = re.findall(r'[\d,]+\.?\d*', response)
            
            # Check if key numbers are present
            for num in expected_numbers:
                clean_num = num.replace(',', '')
                if clean_num in response.replace(',', ''):
                    return True
        
        if "date" in fact_types:
            # Extract years from expected fact
            expected_years = re.findall(r'\b(19|20)\d{2}\b', expected_fact)
            for year in expected_years:
                if year in response:
                    return True
        
        if "name" in fact_types:
            # Extract proper names (capitalized words)
            expected_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', expected_fact)
            for name in expected_names:
                if name.lower() in response:
                    return True
        
        # Fallback: check for partial string matching
        key_phrases = [phrase.strip() for phrase in expected_lower.split(',')]
        for phrase in key_phrases:
            if len(phrase) > 3 and phrase in response:
                return True
        
        return False
    
    def _count_additional_facts(self, response: str) -> int:
        """Count additional facts beyond the expected ones."""
        import re
        
        # Simple heuristic: count sentences with numbers, dates, or proper names
        sentences = response.split('.')
        additional_facts = 0
        
        for sentence in sentences:
            # Has numbers
            if re.search(r'\d+', sentence):
                additional_facts += 1
            # Has dates
            elif re.search(r'\b(19|20)\d{2}\b', sentence):
                additional_facts += 1
            # Has proper names (multiple capitalized words)
            elif len(re.findall(r'\b[A-Z][a-z]+\b', sentence)) >= 2:
                additional_facts += 1
        
        return max(0, additional_facts - 3)  # Subtract expected minimum
    
    def _calculate_accuracy_penalty(self, response: str, expected_facts: List[str]) -> float:
        """Calculate penalty for potentially inaccurate information."""
        # This is a simplified approach - in practice, you'd want more sophisticated fact checking
        penalty = 0.0
        
        # Check for contradictions or inconsistencies
        response_lower = response.lower()
        
        # Look for uncertainty markers
        uncertainty_markers = ['approximately', 'around', 'about', 'roughly', 'estimated']
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker in response_lower)
        
        # Small penalty for excessive uncertainty (might indicate lack of confidence)
        if uncertainty_count > 3:
            penalty += 0.05
        
        # Check for conflicting information (simplified)
        sentences = response.split('.')
        numbers_found = []
        for sentence in sentences:
            import re
            nums = re.findall(r'[\d,]+\.?\d*', sentence)
            numbers_found.extend(nums)
        
        # If multiple very different numbers for same type of fact, might be error
        if len(set(numbers_found)) > len(numbers_found) * 0.8:  # High diversity in numbers
            penalty += 0.1
        
        return penalty
    
    def _analyze_fact_collection_results(
        self, 
        results: List[Dict[str, Any]], 
        tasks: List[FactCollectionTask]
    ) -> Dict[str, Any]:
        """Analyze overall fact collection performance."""
        if not results:
            return {}
        
        successful_results = [r for r in results if r.get('success', False)]
        
        # Overall metrics
        success_rate = len(successful_results) / len(results)
        avg_score = sum(r.get('score', 0) for r in results) / len(results)
        avg_execution_time = sum(r.get('execution_time_ms', 0) for r in successful_results) / len(successful_results) if successful_results else 0
        
        # Performance by difficulty
        difficulty_performance = {}
        for task in tasks:
            task_result = next((r for r in results if r.get('task_id') == task.id), None)
            if task_result:
                if task.difficulty not in difficulty_performance:
                    difficulty_performance[task.difficulty] = {'total': 0, 'successful': 0, 'scores': []}
                
                difficulty_performance[task.difficulty]['total'] += 1
                if task_result.get('success', False):
                    difficulty_performance[task.difficulty]['successful'] += 1
                difficulty_performance[task.difficulty]['scores'].append(task_result.get('score', 0))
        
        # Calculate difficulty success rates
        for difficulty in difficulty_performance:
            perf = difficulty_performance[difficulty]
            perf['success_rate'] = perf['successful'] / perf['total'] if perf['total'] > 0 else 0
            perf['average_score'] = sum(perf['scores']) / len(perf['scores']) if perf['scores'] else 0
        
        # Fact type analysis
        fact_type_performance = {}
        for task in tasks:
            task_result = next((r for r in results if r.get('task_id') == task.id), None)
            if task_result:
                for fact_type in task.fact_types:
                    if fact_type not in fact_type_performance:
                        fact_type_performance[fact_type] = {'total': 0, 'successful': 0}
                    
                    fact_type_performance[fact_type]['total'] += 1
                    if task_result.get('success', False):
                        fact_type_performance[fact_type]['successful'] += 1
        
        # Calculate fact type success rates
        for fact_type in fact_type_performance:
            perf = fact_type_performance[fact_type]
            perf['success_rate'] = perf['successful'] / perf['total'] if perf['total'] > 0 else 0
        
        return {
            'overall_success_rate': success_rate,
            'average_score': avg_score,
            'average_execution_time_ms': avg_execution_time,
            'tasks_evaluated': len(results),
            'successful_tasks': len(successful_results),
            'difficulty_performance': difficulty_performance,
            'fact_type_performance': fact_type_performance,
            'strongest_areas': self._identify_strongest_areas(difficulty_performance, fact_type_performance),
            'improvement_areas': self._identify_improvement_areas(difficulty_performance, fact_type_performance)
        }
    
    def _identify_strongest_areas(
        self, 
        difficulty_perf: Dict[str, Any], 
        fact_type_perf: Dict[str, Any]
    ) -> List[str]:
        """Identify areas where the agent performs best."""
        strengths = []
        
        # Check difficulty levels
        for difficulty, perf in difficulty_perf.items():
            if perf['success_rate'] >= 0.8:
                strengths.append(f"{difficulty} difficulty tasks")
        
        # Check fact types
        for fact_type, perf in fact_type_perf.items():
            if perf['success_rate'] >= 0.8:
                strengths.append(f"{fact_type} extraction")
        
        return strengths
    
    def _identify_improvement_areas(
        self, 
        difficulty_perf: Dict[str, Any], 
        fact_type_perf: Dict[str, Any]
    ) -> List[str]:
        """Identify areas needing improvement."""
        improvements = []
        
        # Check difficulty levels
        for difficulty, perf in difficulty_perf.items():
            if perf['success_rate'] < 0.6:
                improvements.append(f"{difficulty} difficulty tasks")
        
        # Check fact types
        for fact_type, perf in fact_type_perf.items():
            if perf['success_rate'] < 0.6:
                improvements.append(f"{fact_type} extraction")
        
        return improvements


# Example usage and testing functions
async def run_fact_collection_evaluation():
    """Example function to run fact collection evaluation."""
    from app.core.agent import ResearchAgent
    
    agent = ResearchAgent()
    evaluator = FactCollectionEvaluator()
    
    # Run evaluation on easy tasks
    results = await evaluator.evaluate_fact_collection_capability(
        agent=agent,
        difficulty_filter="easy"
    )
    
    print("Fact Collection Evaluation Results:")
    print(f"Overall Success Rate: {results['analysis']['overall_success_rate']:.2%}")
    print(f"Average Score: {results['analysis']['average_score']:.2f}")
    print(f"Tasks Evaluated: {results['tasks_evaluated']}")
    
    if results['analysis']['strongest_areas']:
        print(f"Strongest Areas: {', '.join(results['analysis']['strongest_areas'])}")
    
    if results['analysis']['improvement_areas']:
        print(f"Needs Improvement: {', '.join(results['analysis']['improvement_areas'])}")


def create_custom_fact_task(
    task_id: str,
    description: str,
    query: str,
    expected_facts: List[str],
    fact_types: List[str],
    difficulty: str = "medium"
) -> FactCollectionTask:
    """
    Create a custom fact collection task.
    
    Args:
        task_id: Unique identifier for the task
        description: Human-readable description
        query: The research question to ask
        expected_facts: List of facts the agent should find
        fact_types: Types of facts (number, date, name, etc.)
        difficulty: Task difficulty level
        
    Returns:
        FactCollectionTask instance
    """
    return FactCollectionTask(
        id=task_id,
        description=description,
        query=query,
        expected_facts=expected_facts,
        fact_types=fact_types,
        verification_sources=["general"],
        difficulty=difficulty
    )


if __name__ == "__main__":
    # Example of running the evaluation
    import asyncio
    
    # asyncio.run(run_fact_collection_evaluation())
    
    # Example of creating a custom task
    custom_task = create_custom_fact_task(
        task_id="custom_population_01",
        description="Find population of specific cities",
        query="What is the population of Berlin, Germany?",
        expected_facts=["Berlin: approximately 3.7 million people"],
        fact_types=["number", "location"],
        difficulty="easy"
    )
    
    print(f"Created custom task: {custom_task.id}")
    print(f"Query: {custom_task.query}")
