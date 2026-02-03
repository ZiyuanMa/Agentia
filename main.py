import logging
import asyncio
import argparse
from world import World
from agent import SimAgent
from utils import LLMClient
import json
from config import TICK_DURATION_MINUTES
from logger_config import setup_logging, get_stats

# Configure Logging
setup_logging()
logger = logging.getLogger("Agentia.Main")

# Default paths
DEFAULT_WORLD_PATH = "data/world.json"
DEFAULT_AGENTS_PATH = "data/agents.json"

def setup_world(world_path: str, agents_path: str):
    logger.info(f"Initializing World from {world_path}...")
    llm_client = LLMClient()
    
    # Pass LLM client to World for EnvAgent
    world = World(world_path, llm_client=llm_client)
    
    logger.info(f"Initializing Agents from {agents_path}...")
    
    with open(agents_path, 'r') as f:
        agents_data = json.load(f)

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
        # Place agent in world (World is the single source of truth for location)
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
                    stats.record_event("lock_expired", f"{agent.name}: {lock_status['message'][:30]}")
                else:
                    logger.info(f"{agent.name} is busy: {lock_status['reason']}")
                    continue
            
            context_data = world.get_agent_context_data(agent.name, world.get_agent_location(agent.name))
            
            active_agents.append(agent)
            contexts.append(context_data)
        
        # 2. Concurrent decision making with asyncio.gather
        if active_agents:
            decisions = await asyncio.gather(*[
                agent.decide(tick, ctx) 
                for agent, ctx in zip(active_agents, contexts)
            ])
            
            agent_decisions = dict(zip([a.name for a in active_agents], decisions))
        else:
            agent_decisions = {}

        # 3. Action Resolution (sequential - world state changes)
        for agent in agents:
            decision = agent_decisions.get(agent.name)
            if decision:
                action_type = decision.get("action_type", "unknown")
                stats.record_action(agent.name, action_type)
                
                result = world.process_action(agent.name, decision, inventory=agent.inventory)
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
    parser.add_argument("--world", "-w", type=str, default=DEFAULT_WORLD_PATH,
                        help=f"Path to world JSON file (default: {DEFAULT_WORLD_PATH})")
    parser.add_argument("--agents", "-a", type=str, default=DEFAULT_AGENTS_PATH,
                        help=f"Path to agents JSON file (default: {DEFAULT_AGENTS_PATH})")
    parser.add_argument("--ticks", "-t", type=int, default=5,
                        help="Number of simulation ticks to run (default: 5)")
    parser.add_argument("--no-log-file", action="store_true",
                        help="Disable file logging")
    args = parser.parse_args()
    
    # Reconfigure logging if needed
    if args.no_log_file:
        setup_logging(enable_file=False)
    
    world_instance, agents_list = setup_world(args.world, args.agents)
    asyncio.run(game_loop(world_instance, agents_list, ticks=args.ticks))

