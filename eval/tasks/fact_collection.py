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
