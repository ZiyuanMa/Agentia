"""
Enhanced logging configuration for Agentia.
Features: Colorized console, file output, statistics tracking.
"""
import logging
import sys
import os
import json
from datetime import datetime
from typing import Dict, Any
from collections import defaultdict


class ColorFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""
    GREY = "\x1b[38;20m"
    BLUE = "\x1b[34;20m"
    CYAN = "\x1b[36;20m"
    GREEN = "\x1b[32;20m"
    YELLOW = "\x1b[33;20m"
    RED = "\x1b[31;20m"
    MAGENTA = "\x1b[35;20m"
    BOLD_RED = "\x1b[31;1m"
    BOLD_BLUE = "\x1b[34;1m"
    BOLD_CYAN = "\x1b[36;1m"
    RESET = "\x1b[0m"
    
    FORMAT = "%(asctime)s - %(name)s - %(message)s"

    def format(self, record):
        # Default format
        log_fmt = self.GREY + self.FORMAT + self.RESET
        
        # Colorize based on logger name
        if "Agent." in record.name and "WorldEngine" not in record.name:
            # Individual agents get unique colors based on name hash
            agent_name = record.name.split(".")[-1]
            color_code = 91 + (hash(agent_name) % 6)  # Colors 91-96
            color = f"\x1b[{color_code}m"
            log_fmt = color + self.FORMAT + self.RESET
        elif "WorldEngine" in record.name:
            log_fmt = self.MAGENTA + self.FORMAT + self.RESET
        elif "Main" in record.name:
            log_fmt = self.BOLD_CYAN + self.FORMAT + self.RESET
        elif "World" in record.name:
            log_fmt = self.YELLOW + self.FORMAT + self.RESET
        elif record.levelno == logging.ERROR:
            log_fmt = self.RED + self.FORMAT + self.RESET
        elif record.levelno == logging.WARNING:
            log_fmt = self.YELLOW + self.FORMAT + self.RESET
             
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)


class FileFormatter(logging.Formatter):
    """Plain text formatter for file output."""
    def __init__(self):
        super().__init__(
            fmt="%(asctime)s | %(name)-25s | %(levelname)-8s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    def format(self, record):
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if hasattr(record, 'extra_data'):
            log_entry["data"] = record.extra_data
        return json.dumps(log_entry)


class SimulationStats:
    """Track simulation statistics."""
    
    def __init__(self):
        self.start_time = datetime.now()
        self.tick_count = 0
        self.action_counts: Dict[str, int] = defaultdict(int)
        self.agent_action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.world_engine_calls = 0
        self.api_calls = 0
        self.errors = 0
        self.events: list = []
    
    def record_action(self, agent_name: str, action_type: str):
        """Record an action taken by an agent."""
        self.action_counts[action_type] += 1
        self.agent_action_counts[agent_name][action_type] += 1
    
    def record_tick(self):
        """Record a completed tick."""
        self.tick_count += 1
    
    def record_world_engine_call(self):
        """Record a WorldEngine call."""
        self.world_engine_calls += 1
    
    def record_api_call(self):
        """Record an API call."""
        self.api_calls += 1
    
    def record_error(self):
        """Record an error."""
        self.errors += 1
    
    def record_event(self, event_type: str, details: str):
        """Record a notable event."""
        self.events.append({
            "tick": self.tick_count,
            "type": event_type,
            "details": details
        })
    
    def get_summary(self) -> str:
        """Generate a summary report."""
        duration = datetime.now() - self.start_time
        
        lines = [
            "",
            "=" * 60,
            "                  SIMULATION SUMMARY",
            "=" * 60,
            f"  Duration:        {duration.total_seconds():.1f} seconds",
            f"  Total Ticks:     {self.tick_count}",
            f"  API Calls:       {self.api_calls}",
            f"  WorldEngine Calls:  {self.world_engine_calls}",
            f"  Errors:          {self.errors}",
            "",
            "  --- Action Distribution ---",
        ]
        
        for action, count in sorted(self.action_counts.items()):
            lines.append(f"    {action:12s}: {count}")
        
        if self.agent_action_counts:
            lines.append("")
            lines.append("  --- Per-Agent Actions ---")
            for agent, actions in sorted(self.agent_action_counts.items()):
                action_strs = [f"{a}({c})" for a, c in sorted(actions.items())]
                lines.append(f"    {agent:12s}: {', '.join(action_strs)}")
        
        if self.events:
            lines.append("")
            lines.append("  --- Notable Events ---")
            for event in self.events[-10:]:  # Show last 10 events
                lines.append(f"    [Tick {event['tick']}] {event['type']}: {event['details'][:50]}")
        
        lines.append("=" * 60)
        lines.append("")
        
        return "\n".join(lines)
    
    def export_json(self, filepath: str):
        """Export stats to JSON file."""
        data = {
            "start_time": self.start_time.isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "tick_count": self.tick_count,
            "api_calls": self.api_calls,
            "world_engine_calls": self.world_engine_calls,
            "errors": self.errors,
            "action_counts": dict(self.action_counts),
            "agent_action_counts": {k: dict(v) for k, v in self.agent_action_counts.items()},
            "events": self.events
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# Global stats instance
_stats: SimulationStats = None


def get_stats() -> SimulationStats:
    """Get the global stats instance."""
    global _stats
    if _stats is None:
        _stats = SimulationStats()
    return _stats


def reset_stats():
    """Reset the global stats instance."""
    global _stats
    _stats = SimulationStats()


def setup_logging(log_dir: str = "logs", enable_file: bool = True, enable_json: bool = False):
    """
    Configure the logging system.
    
    Args:
        log_dir: Directory for log files
        enable_file: Whether to write logs to file
        enable_json: Whether to write structured JSON logs
    """
    # Suppress noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    handlers = []
    
    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter())
    handlers.append(console_handler)
    
    # File handlers
    if enable_file:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plain text log file
        file_handler = logging.FileHandler(
            os.path.join(log_dir, f"agentia_{timestamp}.log"),
            encoding='utf-8'
        )
        file_handler.setFormatter(FileFormatter())
        handlers.append(file_handler)
        
        # JSON log file (optional)
        if enable_json:
            json_handler = logging.FileHandler(
                os.path.join(log_dir, f"agentia_{timestamp}.jsonl"),
                encoding='utf-8'
            )
            json_handler.setFormatter(JSONFormatter())
            handlers.append(json_handler)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers
    )
    
    # Reset stats for new simulation
    reset_stats()
    
    logger = logging.getLogger("Agentia.Main")
    if enable_file:
        logger.info(f"Logs will be saved to: {log_dir}/")
