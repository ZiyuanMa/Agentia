import streamlit as st
import networkx as nx
import time
import asyncio
from datetime import timedelta
import graphviz

# Import simulation modules
import sys
import os
sys.path.append(os.getcwd())

from main import setup_scenario, game_loop
from agentia.world import World
from agentia.agent import SimAgent
from agentia.config import DEFAULT_SCENARIO_PATH

# ============================================
# 页面配置
# ============================================
st.set_page_config(
    page_title="Agentia Simulation",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================
# 自定义 CSS 样式
# ============================================
st.markdown("""
<style>
    /* 浅色主题基础 */
    .stApp {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 50%, #e2e8f0 100%);
    }
    
    /* 隐藏默认侧边栏 */
    [data-testid="stSidebar"] {
        display: none;
    }
    
    /* 主标题样式 */
    .main-title {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }
    
    /* 状态卡片 */
    .agent-card {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.3);
        border-radius: 12px;
        padding: 12px;
        margin: 8px 0;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    .agent-card:hover {
        background: rgba(255, 255, 255, 0.95);
        border-color: rgba(99, 102, 241, 0.5);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    /* 状态指示器 */
    .status-indicator {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    
    .status-idle { background: #10b981; box-shadow: 0 0 8px #10b981; }
    .status-busy { background: #f59e0b; box-shadow: 0 0 8px #f59e0b; }
    .status-talking { background: #3b82f6; box-shadow: 0 0 8px #3b82f6; }
    .status-moving { background: #8b5cf6; box-shadow: 0 0 8px #8b5cf6; }
    
    /* 日志条目 */
    .log-entry {
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 8px;
        font-family: 'JetBrains Mono', 'SF Mono', 'Consolas', monospace;
        font-size: 1rem;
        border-left: 3px solid transparent;
        animation: fadeIn 0.3s ease;
        color: #334155;
        line-height: 1.5;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateX(-10px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .log-move { background: rgba(59, 130, 246, 0.08); border-left-color: #3b82f6; }
    .log-talk { background: rgba(16, 185, 129, 0.08); border-left-color: #10b981; }
    .log-interact { background: rgba(245, 158, 11, 0.08); border-left-color: #f59e0b; }
    .log-system { background: rgba(139, 92, 246, 0.08); border-left-color: #8b5cf6; }
    .log-tick { background: rgba(241, 245, 249, 0.8); border-left-color: #64748b; font-weight: 600; color: #475569; }
    
    /* 地图容器 */
    .map-container {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 16px;
        padding: 20px;
        border: 1px solid rgba(148, 163, 184, 0.3);
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
    }
    
    /* 控制按钮样式 */
    .control-btn {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .control-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    .control-btn:disabled {
        opacity: 0.5;
        cursor: not-allowed;
        transform: none;
    }
    
    /* 信息面板 */
    .info-panel {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 12px;
        padding: 16px;
        border: 1px solid rgba(148, 163, 184, 0.3);
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
    }
    
    /* 时间显示 */
    .time-display {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.2rem;
        color: #d97706;
        font-weight: 600;
    }
    
    /* 隐藏 Streamlit 默认元素 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* 滚动条美化 */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(148, 163, 184, 0.2);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #6366f1 0%, #8b5cf6 100%);
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# Session State 初始化
# ============================================
def init_session_state():
    defaults = {
        'world': None,
        'agents': [],
        'tick': 0,
        'logs': [],
        'is_running': False,
        'scenario_loaded': False,
        'selected_scenario': DEFAULT_SCENARIO_PATH,
        'game_speed': 1.0
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ============================================
# 工具函数
# ============================================

def get_agent_status_icon(agent_name, world):
    """获取智能体状态图标"""
    lock_status = world.check_agent_lock(agent_name)
    if lock_status and not lock_status.get("expired"):
        return "🔒", "status-busy", f"busy: {lock_status.get('reason', '')}"
    return "🟢", "status-idle", "idle"

def format_log_entry(log: str) -> tuple:
    """格式化日志条目，返回 (样式类, 图标, 内容)"""
    if "TICK" in log:
        return "log-tick", "⏱️", log
    elif "move" in log.lower():
        return "log-move", "🚶", log
    elif "talk" in log.lower():
        return "log-talk", "💬", log
    elif "interact" in log.lower():
        return "log-interact", "👆", log
    elif "🔓" in log:
        return "log-system", "🔓", log
    else:
        return "log-system", "ℹ️", log

def render_map(world: World, agents: list[SimAgent]):
    """渲染现代化的地图"""
    if not world:
        st.info("🗺️ 请先加载场景")
        return
       
    graph = graphviz.Digraph()
    
    # 现代化样式
    graph.attr(
        rankdir='TB',
        bgcolor='transparent',
        nodesep='0.6',
        ranksep='1.2',
        fontname='Arial',
        pad='0.3'
    )
    
    # 颜色方案 - 浅色主题
    colors = {
        'empty': {'fill': '#ffffff', 'font': '#1e293b', 'border': '#94a3b8'},
        'occupied': {'fill': '#dbeafe', 'font': '#1e3a8a', 'border': '#3b82f6'}
    }
    
    # 添加节点
    for loc_id, loc in world.locations.items():
        agents_here = [a.name for a in agents if world.get_agent_location(a.name) == loc_id]
        is_occupied = len(agents_here) > 0
        
        color_scheme = colors['occupied'] if is_occupied else colors['empty']
        
        # 构建标签
        label = f"📍 {loc.name}"
        if agents_here:
            label += "\n" + "\n".join([f"👤 {name}" for name in agents_here])
        
        graph.node(
            loc_id,
            label=label,
            shape='box',
            style='rounded,filled',
            fillcolor=color_scheme['fill'],
            fontcolor=color_scheme['font'],
            color=color_scheme['border'],
            width='2.5',
            height='1.2',
            margin='0.3',
            fontname='Arial',
            fontsize='20'
        )
    
    # 边默认样式 - 无向边（双向连接）
    graph.attr('edge',
        color='#94a3b8',
        penwidth='1.5',
        arrowsize='0.7',
        dir='none'
    )
    
    # 添加边（避免重复）
    seen_edges = set()
    for loc_id, loc in world.locations.items():
        for target in loc.connected_to:
            edge = tuple(sorted((loc_id, target)))
            if edge not in seen_edges:
                # 检查是否有智能体在这条边上移动
                def is_moving(agent):
                    lock = world.check_agent_lock(agent.name)
                    if not lock:
                        return False
                    reason = lock.get('reason', '') or ''
                    return reason.startswith('moving')
                
                moving_here = any(is_moving(a) for a in agents)
                
                edge_color = '#7c3aed' if moving_here else '#94a3b8'
                graph.edge(loc_id, target, color=edge_color, penwidth='3' if moving_here else '2')
                seen_edges.add(edge)
    
    st.graphviz_chart(graph)

async def run_tick():
    """执行模拟步进"""
    if not st.session_state.world:
        return
    
    world = st.session_state.world
    agents = st.session_state.agents
    
    st.session_state.tick += 1
    current_time = world.get_time_str()
    
    # 添加 tick 分隔日志
    st.session_state.logs.append(f"--- TICK {st.session_state.tick} | {current_time} ---")
    
    # 1. 构建上下文
    active_agents = []
    contexts = []
    
    for agent in agents:
        lock_status = world.check_agent_lock(agent.name)
        if lock_status:
            if lock_status.get("expired"):
                agent.update_state({"success": True, "message": lock_status["message"]})
                st.session_state.logs.append(f"🔓 {agent.name}: {lock_status['message']}")
            else:
                continue
        
        context_data = world.get_agent_context_data(agent.name, world.get_agent_location(agent.name))
        active_agents.append(agent)
        contexts.append(context_data)
    
    # 2. 决策
    if active_agents:
        decisions = await asyncio.gather(*[
            agent.decide(ctx) 
            for agent, ctx in zip(active_agents, contexts)
        ])
        agent_decisions = dict(zip([a.name for a in active_agents], decisions))
    else:
        agent_decisions = {}
    
    # 3. 执行动作
    for agent in agents:
        decision = agent_decisions.get(agent.name)
        if decision:
            result = world.process_action(agent.name, decision)
            agent.update_state(result)
            
            # 格式化动作日志
            action_str = f"{agent.name}: {decision.action_type}"
            if decision.action_type == "talk":
                content = decision.get_validated_action().message
                action_str += f" '{content[:30]}...'" if len(content) > 30 else f" '{content}'"
            elif decision.action_type == "move":
                target = decision.get_validated_action().location_id
                action_str += f" → {target}"
            elif decision.action_type == "interact":
                target = decision.get_validated_action().object_id
                action_str += f" 👆 {target}"
            
            st.session_state.logs.append(f"▶️ {action_str} | {result['message']}")
    
    # 4. 推进时间
    world.advance_time()

# ============================================
# 顶部控制栏
# ============================================
with st.container():
    cols = st.columns([3, 2, 3])
    
    with cols[0]:
        st.markdown('<p class="main-title">🤖 Agentia Simulation</p>', unsafe_allow_html=True)
    
    with cols[1]:
        if st.session_state.world:
            tick = st.session_state.tick
            time_str = st.session_state.world.get_time_str()
            st.markdown(f'<p class="time-display">⏱️ Tick {tick} &nbsp;|&nbsp; 🕐 {time_str}</p>', unsafe_allow_html=True)
    
    with cols[2]:
        # 场景选择和控制
        scenario_cols = st.columns([2, 1])
        with scenario_cols[0]:
            scenario_path = st.text_input(
                "场景路径",
                st.session_state.selected_scenario,
                label_visibility="collapsed",
                placeholder="输入场景配置文件路径..."
            )
            st.session_state.selected_scenario = scenario_path
        
        with scenario_cols[1]:
            if st.button("🔄 加载场景", use_container_width=True, type="primary"):
                with st.spinner("加载中..."):
                    try:
                        world, agents = setup_scenario(scenario_path)
                        st.session_state.world = world
                        st.session_state.agents = agents
                        st.session_state.tick = 0
                        st.session_state.logs = ["✅ 场景加载成功"]
                        st.session_state.is_running = False
                        st.session_state.scenario_loaded = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"加载失败: {str(e)[:100]}")

st.divider()

# ============================================
# 主内容区
# ============================================

if not st.session_state.scenario_loaded:
    # 空状态引导
    st.markdown("""
    <div style="text-align: center; padding: 80px 20px; color: #475569;">
        <h2 style="color: #1e293b;">👋 欢迎来到 Agentia Simulation</h2>
        <p style="font-size: 1.1rem; margin-top: 20px;">
            这是一个多智能体模拟系统，智能体将在虚拟世界中自主决策、移动和交互。
        </p>
        <p style="margin-top: 30px;">
            点击右上角 <b>🔄 加载场景</b> 开始模拟
        </p>
    </div>
    """, unsafe_allow_html=True)
    
else:
    # 三列布局: 地图 | 日志 | 智能体状态
    col_map, col_logs, col_agents = st.columns([2.5, 1.5, 1])
    
    # ========== 地图区域 ==========
    with col_map:
        st.markdown('<div class="map-container">', unsafe_allow_html=True)
        st.markdown("### 🗺️ 世界地图")
        render_map(st.session_state.world, st.session_state.agents)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # ========== 日志区域 ==========
    with col_logs:
        st.markdown("### 📜 事件日志")
        
        # 控制按钮
        ctrl_cols = st.columns([1, 1, 1, 2])
        
        with ctrl_cols[0]:
            if st.button("▶️" if not st.session_state.is_running else "⏸️", use_container_width=True):
                st.session_state.is_running = not st.session_state.is_running
                st.rerun()
        
        with ctrl_cols[1]:
            if st.button("⏭️", use_container_width=True):
                st.session_state.is_running = False
                asyncio.run(run_tick())
                st.rerun()
        
        with ctrl_cols[2]:
            if st.button("⏹️", use_container_width=True):
                st.session_state.is_running = False
                st.session_state.tick = 0
                st.session_state.logs = []
                # 重新加载场景
                try:
                    world, agents = setup_scenario(st.session_state.selected_scenario)
                    st.session_state.world = world
                    st.session_state.agents = agents
                    st.rerun()
                except:
                    pass
        
        with ctrl_cols[3]:
            speed = st.slider("速度", 0.1, 2.0, st.session_state.game_speed, 0.1, label_visibility="collapsed")
            st.session_state.game_speed = speed
        
        # 日志显示区域
        log_container = st.container(height=450)
        with log_container:
            # 显示最近100条，按时间顺序（旧的在上面，新的在下面）
            for log in st.session_state.logs[-100:]:
                css_class, icon, content = format_log_entry(log)
                st.markdown(
                    f'<div class="log-entry {css_class}">{icon} {content}</div>',
                    unsafe_allow_html=True
                )
    
    # ========== 智能体状态区域 ==========
    with col_agents:
        st.markdown("### 👥 智能体")
        
        world = st.session_state.world
        for agent in st.session_state.agents:
            loc_id = world.get_agent_location(agent.name)
            location = world.locations.get(loc_id, None)
            loc_name = location.name if location else loc_id
            
            status_icon, status_class, status_text = get_agent_status_icon(agent.name, world)
            
            # 获取当前动作
            current_action = "待机"
            if st.session_state.logs:
                for log in reversed(st.session_state.logs):
                    if log.startswith(f"▶️ {agent.name}:"):
                        action = log.split(":")[1].split("|")[0].strip() if ":" in log else ""
                        current_action = action
                        break
            
            st.markdown(f"""
            <div class="agent-card">
                <div style="display: flex; align-items: center; margin-bottom: 8px;">
                    <span class="status-indicator {status_class}"></span>
                    <strong style="color: #1e293b; font-size: 1.1rem;">{agent.name}</strong>
                </div>
                <div style="color: #64748b; font-size: 0.95rem; margin-left: 16px;">
                    📍 {loc_name}<br>
                    🎯 {current_action[:20]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # 统计信息
        st.markdown("---")
        st.markdown("### 📊 统计")
        
        total_agents = len(st.session_state.agents)
        
        def is_active(agent):
            lock = world.check_agent_lock(agent.name)
            if not lock:
                return True  # 没有锁 = 活跃
            return lock.get("expired", False)
        
        active_count = sum(1 for a in st.session_state.agents if is_active(a))
        
        stats_cols = st.columns(2)
        with stats_cols[0]:
            st.metric("智能体", total_agents)
        with stats_cols[1]:
            st.metric("活跃", active_count)

    # ============================================
    # 自动运行循环
    # ============================================
    if st.session_state.is_running:
        time.sleep(st.session_state.game_speed)
        asyncio.run(run_tick())
        st.rerun()
