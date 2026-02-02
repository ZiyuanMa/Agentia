# Agentia: AI Micro-Society Simulator

Agentia is a lightweight, highly extensible virtual society simulator driven by Large Language Models (LLMs). It interacts with the world through **Structured Action Output** (JSON Mode), allowing agents to physically change their environment reliably.

## 🌟 Core Philosophy

Agentia solves the "rigid action" and "limited interaction" problems of traditional agent simulations (like Generative Agents) through a unique **Hybrid Environment Architecture**:
*   **Layer 1 (Fast Path)**: Deterministic physics handled by code (Moving, specific rigid interactions). Zero token cost.
*   **Layer 2 (Slow Path)**: Complex interactions handled by an **Environment Agent** (LLM). This "Game Master" decides the physical consequences of creative or undefined actions.

## ✨ Key Features

*   **Action as Function**: Agents output **Structured JSON decisions** (validated via Pydantic), ensuring reliability without the complexity of native function calling APIs.
*   **Graph-based Topology**: Locations are nodes in a `NetworkX` graph, supporting navigation and logical connections.
*   **Interactive Objects**: Everything from a coffee machine to a server rack has state and properties.
*   **EnvAgent System**: A dedicated LLM that acts as the physics engine for complex prompts like "fix the broken server" or "pour water on the computer".

## 🤖 Agent Capabilities

### SimAgent (The Residents)
These are the autonomous entities living in the world. Their reasoning focuses on "What should I do to achieve my goal?"
*   `MoveAction`: Travel to a connected location (e.g., from Hallway to Kitchen).
*   `TalkAction`: Speak to others in the same location (broadcasts message to local agents).
*   `InteractAction`: Attempt to use/manipulate an object (e.g., "Use Coffee Machine", "Unlock Door").
*   `WaitAction`: Idling or observing.

### EnvAgent (The Environment)
This is the "Game Master" agent that resolves complex physics and causality when a SimAgent interacts with an object.
*   `ModifyStateAction`: Change an object's state (e.g., "Coffee Machine" -> "Broken").
*   `CreateObjectAction`: Spawn new items (e.g., "Cup of Coffee").
*   `DestroyObjectAction`: Remove items (e.g., "Cup of Coffee"Consumed).
*   `TransferObjectAction`: Move items between inventories or locations (e.g., Giving a key card).
*   `LockAgentAction`: Freeze an agent for a duration to simulate time-consuming tasks (e.g., "Repairing Server - 30 mins").
*   `BroadcastAction`: Announce events to a location (e.g., "Loud explosion heard from Server Room").

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
*   `--world`: Path to world definition JSON (default: `data/world.json`)
*   `--agents`: Path to agent definition JSON (default: `data/agents.json`)
*   `--ticks`: Number of simulation ticks to run (default: 5)

Example:
```bash
python main.py --ticks 10 --world data/world_office.json
```

## 📂 Project Structure

```
agentia/
├── main.py              # Entry point: Game Loop & concurrency management
├── world.py             # World Model: Graph topology, object state, time
├── agent.py             # Agent Model: Decision logic, memory, prompting
├── env_agent.py         # Environment Agent: "Game Master" logic
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
    *   `interact` -> Sent to `EnvironmentAgent` if complex.
4.  **Execution**: State is updated, time advances.
*   **File Output**: Logs are saved to `logs/agentia_YYYYMMDD_HHMMSS.log`.

### Customizing the World
You can create new worlds by defining a JSON file in `data/`. See `data/world_office.json` for an example of defining Locations, Objects, and Connections.

## 📄 License
MIT License
