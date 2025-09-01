import asyncio
import json
import sys
from typing import Optional

import aiohttp
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
import structlog

logger = structlog.get_logger()
console = Console()
app = typer.Typer(help="Autonomous Research Agent CLI")


class AgentClient:
    """Client for interacting with the research agent API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        
    async def start_task(self, goal: str, planner: str = "reactive") -> str:
        """Start a new research task."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/task",
                json={"goal": goal, "planner_type": planner}
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return data["task_id"]
                else:
                    error = await response.text()
                    raise Exception(f"Failed to start task: {error}")
    
    async def get_status(self, task_id: str) -> dict:
        """Get task status."""
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{self.base_url}/api/v1/status/{task_id}"
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    error = await response.text()
                    raise Exception(f"Failed to get status: {error}")
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a running task."""
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/cancel/{task_id}"
            ) as response:
                return response.status == 200


@app.command()
def run(
    goal: str = typer.Argument(..., help="Research goal or question"),
    planner: str = typer.Option("reactive", help="Planner type (reactive/deliberative)"),
    server: str = typer.Option("http://localhost:8000", help="Agent server URL"),
    wait: bool = typer.Option(True, help="Wait for completion")
):
    """Run a research task."""
    asyncio.run(_run_task(goal, planner, server, wait))


async def _run_task(goal: str, planner: str, server: str, wait: bool):
    """Execute research task."""
    client = AgentClient(server)
    
    try:
        console.print(f"[bold blue]Starting research task:[/bold blue] {goal}")
        task_id = await client.start_task(goal, planner)
        console.print(f"[green]Task started with ID:[/green] {task_id}")
        
        if not wait:
            console.print(f"[yellow]Task running in background. Check status with:[/yellow] agent status {task_id}")
            return
        
        # Wait for completion with progress indicator
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Executing research task...", total=None)
            
            while True:
                status = await client.get_status(task_id)
                
                if status["status"] == "completed":
                    progress.update(task, description="Task completed!")
                    break
                elif status["status"] == "failed":
                    progress.update(task, description="Task failed!")
                    break
                elif status["status"] == "cancelled":
                    progress.update(task, description="Task cancelled!")
                    break
                
                progress.update(task, description=f"Running... ({len(status.get('steps', []))} steps)")
                await asyncio.sleep(2)
        
        # Display results
        final_status = await client.get_status(task_id)
        _display_results(final_status)
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@app.command()
def status(
    task_id: str = typer.Argument(..., help="Task ID"),
    server: str = typer.Option("http://localhost:8000", help="Agent server URL")
):
    """Check task status."""
    asyncio.run(_check_status(task_id, server))


async def _check_status(task_id: str, server: str):
    """Check task status."""
    client = AgentClient(server)
    
    try:
        status = await client.get_status(task_id)
        _display_results(status)
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


@app.command()
def cancel(
    task_id: str = typer.Argument(..., help="Task ID"),
    server: str = typer.Option("http://localhost:8000", help="Agent server URL")
):
    """Cancel a running task."""
    asyncio.run(_cancel_task(task_id, server))


async def _cancel_task(task_id: str, server: str):
    """Cancel task."""
    client = AgentClient(server)
    
    try:
        success = await client.cancel_task(task_id)
        if success:
            console.print(f"[green]Task {task_id} cancelled successfully[/green]")
        else:
            console.print(f"[red]Failed to cancel task {task_id}[/red]")
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        sys.exit(1)


def _display_results(status: dict):
    """Display task results in a formatted way."""
    console.print("\n[bold]Task Results[/bold]")
    console.print("="*50)
    
    # Basic info
    console.print(f"[bold]Task ID:[/bold] {status['task_id']}")
    console.print(f"[bold]Goal:[/bold] {status['goal']}")
    console.print(f"[bold]Status:[/bold] {status['status']}")
    console.print(f"[bold]Duration:[/bold] {status['total_duration_ms']}ms")
    
    # Results
    if status.get('result'):
        console.print(f"\n[bold green]Result:[/bold green]")
        console.print(status['result'])
    
    if status.get('error'):
        console.print(f"\n[bold red]Error:[/bold red]")
        console.print(status['error'])
    
    # Steps
    if status.get('steps'):
        console.print(f"\n[bold]Execution Steps ({len(status['steps'])}):[/bold]")
        
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Step", style="dim", width=6)
        table.add_column("Action", width=30)
        table.add_column("Tool", width=15)
        table.add_column("Duration", justify="right", width=10)
        
        for step in status['steps']:
            table.add_row(
                str(step['step_number']),
                step['action'][:50] + "..." if len(step['action']) > 50 else step['action'],
                step.get('tool_name', '-'),
                f"{step.get('duration_ms', 0)}ms"
            )
        
        console.print(table)


def main():
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()

