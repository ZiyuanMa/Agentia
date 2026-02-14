# Agentia: AI Micro-Society Simulator

Agentia is a lightweight, highly extensible virtual society simulator driven by Large Language Models (LLMs). Agents interact with the world through **Structured Action Output** (JSON Mode), allowing them to physically change their environment reliably.

## 🌟 Core Philosophy

Agentia solves the "rigid action" and "limited interaction" problems of traditional agent simulations (like Generative Agents) through a unique **Hybrid Environment Architecture**:
*   **Layer 1 (Fast Path)**: Deterministic physics handled by code (Moving, specific rigid interactions). Zero token cost.
*   **Layer 2 (Slow Path)**: Complex interactions handled by an **World Engine** (LLM). This "Game Master" decides the physical consequences of creative or undefined actions.

## ✨ Key Features

*   **Action as Function**: Agents output **Structured JSON decisions** (validated via Pydantic), ensuring reliability without the complexity of native function calling APIs.
*   **Graph-based Topology**: Locations are nodes in a `NetworkX` graph, supporting navigation and logical connections.
*   **Interactive Objects**: Everything from a coffee machine to a server rack has state and properties.
*   **WorldEngine System**: A dedicated LLM that acts as the physics engine for complex prompts like "fix the broken server" or "pour water on the computer".

## 🤖 Agent Capabilities

### SimAgent (The Residents)
These are the autonomous entities living in the world. Their reasoning focuses on "What should I do to achieve my goal?"
*   `Move`: Travel to a **connected** location (locations must be directly linked; you cannot teleport to unconnected areas).
*   `Talk`: Speak to others in the same location (broadcasts message to local agents).
*   `Interact`: Attempt to use/manipulate an object (e.g., "Use Coffee Machine", "Unlock Door").
*   `Wait`: Idling or observing.

### WorldEngine (The Environment)
This is the "Game Master" agent that resolves complex physics and causality when a SimAgent interacts with an object.
*   `UpdateObject`: Change an object's state (e.g., "Coffee Machine" -> "Broken").
*   `CreateObject`: Spawn new items (e.g., "Cup of Coffee").
*   `DestroyObject`: Remove items (e.g., "Cup of Coffee" consumed).
*   `TransferObject`: Move items between inventories or locations (e.g., Giving a key card).

Additionally, interactions can have **duration** (e.g., repairing takes 30 minutes), during which the agent is locked and cannot perform other actions.

## 🚀 Quick Start

### Prerequisites
*   Python 3.10+
*   OpenAI API Key (or compatible LLM endpoint)

### Installation

1.  **Clone the repository**
    ```bash
    git clone https://github.com/your-username/agentia.git
    cd agentia
    ```

2.  **Set up the environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

3.  **Configuration**
    Create a `.env` file in the root directory:
    ```env
    OPENAI_API_KEY=sk-your-api-key-here
    OPENAI_BASE_URL=https://api.openai.com/v1  # Optional: for other providers
    ```

### Running the Simulation

Run the main simulation loop:

```bash
python main.py
```

Options:
*   `--no-log-file`: Disable writing logs to disk.
*   `--scenario`, `-s`: Path to scenario JSON file (default: `data/scenario_office.json`)
*   `--ticks`, `-t`: Number of simulation ticks to run (default: 5)

Example:
```bash
python main.py --ticks 10 --scenario data/scenario_deep_space.json
```

## 📂 Project Structure

```
agentia/
├── main.py              # Entry point: Game Loop & concurrency management
├── world.py             # World Model: Graph topology, object state, time
├── agent.py             # Agent Model: Decision logic, memory, prompting
├── world_engine.py      # World Engine: "Game Master" logic
├── schemas.py           # Pydantic Models: Defines all interactions & data
├── config.py            # Configuration & Constants
├── data/                # JSON definitions for worlds and agents
└── tests/               # Unit tests
```

## 🛠️ Architecture Details

### The Decision Loop
1.  **Perception**: Agent receives a `Dynamic Context` (Time, Location, Active Objects, People).
2.  **Decision**: Agent outputs a JSON decision (`move`, `talk`, `interact`, `wait`).
3.  **Routing**:
    *   `move/talk` -> Handled immediately by `World` logic.
    *   `interact` -> Sent to `WorldEngine` if complex.
4.  **Execution**: State is updated, time advances.
*   **File Output**: Logs are saved to `logs/agentia_YYYYMMDD_HHMMSS.log`.

### Customizing Scenarios

Scenarios define a complete simulation environment including the world (locations, objects) and the agents that inhabit it.

Create a new scenario by defining a JSON file with this structure:

```json
{
    "name": "Coffee Shop Morning",
    "description": "A busy coffee shop during morning rush hour",
    "world": {
        "locations": [...],
        "objects": [...]
    },
    "agents": [...]
}
```

See `data/scenario_office.json` for a complete example.

## 📄 License
MIT License
