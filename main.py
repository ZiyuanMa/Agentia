import logging
import asyncio
import argparse
import json
from agentia.world import World
from agentia.agent import SimAgent
from agentia.utils import LLMClient
from agentia.logger_config import setup_logging, get_stats
from agentia.config import DEFAULT_SCENARIO_PATH

# Configure Logging
setup_logging()
logger = logging.getLogger("Agentia.Main")


def load_scenario(scenario_path: str):
    """Load world and agents from a scenario file."""
    logger.info(f"Loading scenario from {scenario_path}...")
    
    with open(scenario_path, 'r') as f:
        scenario = json.load(f)
    
    logger.info(f"Scenario: {scenario.get('name', 'Unnamed')}")
    logger.info(f"Description: {scenario.get('description', 'No description')}")
    
    return scenario.get("world", {}), scenario.get("agents", [])


def setup_scenario(scenario_path: str):
    """Initialize world and agents from a scenario file."""
    world_config, agents_data = load_scenario(scenario_path)
    
    llm_client = LLMClient()
    world = World(world_config, llm_client=llm_client)
    
    logger.info(f"Initializing {len(agents_data)} agents...")
    
    agents = []
    for agent_data in agents_data:
        agent = SimAgent(
            name=agent_data["name"],
            age=agent_data["age"],
            occupation=agent_data["occupation"],
            personality=agent_data["personality"],
            background=agent_data.get("background", "No background provided."),
            llm_client=llm_client,
            initial_goal=agent_data.get("initial_goal", "Explore the surroundings.")
        )
        agents.append(agent)
        world.place_agent(agent.name, agent_data["initial_location"])
            
    return world, agents


async def game_loop(world: World, agents: list[SimAgent], ticks: int = 5):
    logger.info("Starting Game Loop (async)...")
    stats = get_stats()
    
    for tick in range(1, ticks + 1):
        logger.info(f"--- TICK {tick} | {world.get_time_str()} ---")
        
        # 1. Build contexts and filter active agents
        active_agents = []
        contexts = []
        
        for agent in agents:
            lock_status = world.check_agent_lock(agent.name)
            if lock_status:
                if lock_status.get("expired"):
                    agent.update_state({"success": True, "message": lock_status["message"]})
                    logger.info(f"{agent.name} finished: {lock_status['message']}")
                    stats.record_event("lock_expired", f"{agent.name}: {lock_status['message']}")
                else:
                    logger.info(f"{agent.name} is busy: {lock_status['reason']}")
                    continue
            
            context_data = world.get_agent_context_data(agent.name, world.get_agent_location(agent.name))
            
            active_agents.append(agent)
            contexts.append(context_data)
        
        # 2. Concurrent decision making with asyncio.gather
        if active_agents:
            decisions = await asyncio.gather(*[
                agent.decide(ctx) 
                for agent, ctx in zip(active_agents, contexts)
            ])
            
            agent_decisions = dict(zip([a.name for a in active_agents], decisions))
        else:
            agent_decisions = {}

        # 3. Action Resolution (sequential - world state changes)
        for agent in agents:
            decision = agent_decisions.get(agent.name)
            if decision:
                stats.record_action(agent.name, decision.action_type)
                
                result = world.process_action(agent.name, decision)
                agent.update_state(result)
        
        # 4. Advance time and record tick
        world.advance_time()
        stats.record_tick()
    
    # Print summary at end
    summary = stats.get_summary()
    for line in summary.split('\n'):
        logger.info(line)
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentia - AI Agent Simulation")
    parser.add_argument("--scenario", "-s", type=str, default=DEFAULT_SCENARIO_PATH,
                        help=f"Path to scenario JSON file (default: {DEFAULT_SCENARIO_PATH})")
    parser.add_argument("--ticks", "-t", type=int, default=15,
                        help="Number of simulation ticks to run (default: 5)")
    parser.add_argument("--no-log-file", action="store_true",
                        help="Disable file logging")
    args = parser.parse_args()
    
    # Reconfigure logging if needed
    if args.no_log_file:
        setup_logging(enable_file=False)
    
    world_instance, agents_list = setup_scenario(args.scenario)
    asyncio.run(game_loop(world_instance, agents_list, ticks=args.ticks))
