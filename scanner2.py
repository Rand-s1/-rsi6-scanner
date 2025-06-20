import requests
import ta
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import logging
import time
from typing import List, Dict, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置页面配置
st.set_page_config(
    page_title="鹅的RSI6 扫描器 Pro",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS样式
st.markdown("""
<style>
    /* 主要背景和主题 */
    .main {
        padding-top: 2rem;
    }
    
    /* 标题样式 */
    .big-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    
    /* 卡片样式 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #4ecdc4;
        margin: 1rem 0;
    }
    
    /* 按钮样式 */
    .stButton > button {
        width: 100%;
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 25px;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
    
    /* 数据表格样式 */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* 侧边栏样式 */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* 警告和信息框样式 */
    .stAlert {
        border-radius: 10px;
    }
    
    /* 进度条样式 */
    .stProgress > div > div {
        background: linear-gradient(90deg, #ff6b6b, #4ecdc4);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置常量
class Config:
    ENDPOINTS = ["https://api.bitget.com"]
    PRODUCT_TYPE = "usdt-futures"
    LIMIT = 100
    RSI_PERIOD = 6
    SLEEP_BETWEEN_REQUESTS = 0.5
    MAX_WORKERS = 10
    MIN_CANDLES_RELIABLE = 20
    
    # UI配置
    TIMEFRAMES = {
        "1小时": "1H",
        "4小时": "4H", 
        "1天": "1D"
    }
    
    # RSI区间配置
    RSI_RANGES = {
        "超卖区域": (0, 30),
        "中性区域": (30, 70),
        "超买区域": (70, 100)
    }

def create_header():
    """创建页面头部"""
    st.markdown('<h1 class="big-title">📈 鹅的RSI6 扫描器 Pro</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">🚀  Bitget USDT永续合约扫描</p>', unsafe_allow_html=True)
    
    # 添加分隔线
    st.markdown("---")

def create_sidebar():
    """创建侧边栏"""
    with st.sidebar:
        st.markdown("### ⚙️ 扫描设置")
        
        # 时间框架选择
        timeframe_display = st.selectbox(
            "📊 时间框架",
            options=list(Config.TIMEFRAMES.keys()),
            index=1,  # 默认4小时
            help="选择K线时间周期"
        )
        timeframe = Config.TIMEFRAMES[timeframe_display]
        
        st.markdown("### 🎯 RSI阈值设置")
        
        # RSI阈值设置
        col1, col2 = st.columns(2)
        with col1:
            rsi_low = st.number_input(
                "超卖线", 
                min_value=0.0, 
                max_value=50.0, 
                value=10.0, 
                step=1.0,
                help="RSI低于此值显示超卖信号"
            )
        with col2:
            rsi_high = st.number_input(
                "超买线", 
                min_value=50.0, 
                max_value=100.0, 
                value=90.0, 
                step=1.0,
                help="RSI高于此值显示超买信号"
            )
        
        # 高级设置
        with st.expander("🔧 高级设置"):
            show_charts = st.checkbox("显示图表分析", value=True)
            min_volume = st.number_input("最小成交量过滤", value=0.0, help="过滤低成交量币种")
            
        return timeframe, rsi_low, rsi_high, show_charts, min_volume

def ping_endpoint(endpoint: str) -> bool:
    """测试端点是否可用"""
    url = f"{endpoint}/api/v2/mix/market/candles"
    params = {
        "symbol": "BTCUSDT",
        "granularity": "4H",
        "limit": 1,
        "productType": Config.PRODUCT_TYPE,
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        return r.status_code == 200 and r.json().get("code") == "00000"
    except:
        return False

def get_working_endpoint() -> str:
    """获取可用端点"""
    for ep in Config.ENDPOINTS:
        for _ in range(3):
            if ping_endpoint(ep):
                return ep
            time.sleep(1)
    raise RuntimeError("无可用端点，请检查网络连接")

def get_usdt_symbols(base: str) -> List[str]:
    """获取USDT永续合约交易对"""
    url = f"{base}/api/v2/mix/market/contracts"
    params = {"productType": Config.PRODUCT_TYPE}
    
    try:
        r = requests.get(url, params=params, timeout=5)
        j = r.json()
        if j.get("code") != "00000":
            raise RuntimeError(f"获取交易对失败: {j}")
        symbols = [c["symbol"] for c in j["data"]]
        logger.info(f"找到 {len(symbols)} 个USDT永续合约")
        return symbols
    except Exception as e:
        logger.error(f"获取交易对错误: {e}")
        raise

def fetch_candles(base: str, symbol: str, granularity: str) -> pd.DataFrame:
    """获取K线数据"""
    url = f"{base}/api/v2/mix/market/candles"
    params = {
        "symbol": symbol,
        "granularity": granularity,
        "limit": Config.LIMIT,
        "productType": Config.PRODUCT_TYPE,
    }
    
    try:
        r = requests.get(url, params=params, timeout=10)
        j = r.json()
        if j.get("code") != "00000":
            return pd.DataFrame()
            
        cols = ["ts", "open", "high", "low", "close", "volume_base", "volume_quote"]
        df = pd.DataFrame(j["data"], columns=cols)
        df[["open", "high", "low", "close", "volume_base", "volume_quote"]] = df[
            ["open", "high", "low", "close", "volume_base", "volume_quote"]
        ].astype(float)
        df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
        return df.sort_values("ts").reset_index(drop=True)
    except Exception as e:
        logger.error(f"{symbol} K线获取失败: {e}")
        return pd.DataFrame()

def fetch_all_tickers(base: str) -> Dict[str, dict]:
    """批量获取ticker数据 - 修复版本"""
    url = f"{base}/api/v2/mix/market/tickers"
    params = {"productType": Config.PRODUCT_TYPE}
    
    try:
        r = requests.get(url, params=params, timeout=5)
        j = r.json()
        
        logger.info(f"Ticker API响应: code={j.get('code')}, msg={j.get('msg')}")
        
        if j.get("code") != "00000":
            logger.error(f"API返回错误: {j}")
            return {}
            
        if not isinstance(j.get("data"), list):
            logger.error(f"API数据格式错误: {type(j.get('data'))}")
            return {}
        
        tickers = {}
        for item in j["data"]:
            try:
                # 打印第一个item的结构，用于调试
                if len(tickers) == 0:
                    logger.info(f"Ticker数据结构示例: {list(item.keys())}")
                
                # 兼容不同的字段名
                symbol = item.get("symbol", "")
                if not symbol:
                    continue
                
                # 尝试不同的字段名
                change24h = 0.0
                if "change24h" in item:
                    change24h = float(item["change24h"]) * 100
                elif "chgUtc" in item:
                    change24h = float(item["chgUtc"]) * 100
                elif "changeUtc24h" in item:
                    change24h = float(item["changeUtc24h"]) * 100
                
                # 成交量字段
                volume = 0.0
                if "baseVolume" in item:
                    volume = float(item["baseVolume"])
                elif "baseVol" in item:
                    volume = float(item["baseVol"])
                elif "vol24h" in item:
                    volume = float(item["vol24h"])
                
                # 价格字段
                price = 0.0
                if "close" in item:
                    price = float(item["close"])
                elif "last" in item:
                    price = float(item["last"])
                elif "lastPr" in item:
                    price = float(item["lastPr"])
                
                tickers[symbol] = {
                    "change24h": change24h,
                    "volume": volume,
                    "price": price
                }
                
            except (ValueError, KeyError, TypeError) as e:
                logger.warning(f"处理ticker数据失败 {item.get('symbol', 'unknown')}: {e}")
                continue
        
        logger.info(f"成功获取 {len(tickers)} 个ticker数据")
        return tickers
        
    except requests.exceptions.RequestException as e:
        logger.error(f"网络请求失败: {e}")
        return {}
    except Exception as e:
        logger.error(f"获取ticker数据失败: {e}")
        return {}

def calculate_rsi_and_metrics(df: pd.DataFrame) -> Tuple[Optional[float], int, dict]:
    """计算RSI和其他技术指标"""
    try:
        close_series = pd.Series(df["close"].astype(float)).reset_index(drop=True)
        candle_count = len(close_series)
        
        if candle_count < Config.RSI_PERIOD + 1:
            return None, candle_count, {}
            
        # 计算RSI
        rsi_series = ta.momentum.RSIIndicator(close=close_series, window=Config.RSI_PERIOD).rsi()
        rsi = rsi_series.iloc[-1]
        
        # 计算其他指标
        metrics = {
            "sma_20": ta.trend.sma_indicator(close_series, window=20).iloc[-1] if candle_count >= 20 else None,
            "volatility": close_series.pct_change().std() * 100,
            "price_change": ((close_series.iloc[-1] - close_series.iloc[-2]) / close_series.iloc[-2]) * 100 if candle_count >= 2 else 0
        }
        
        return rsi, candle_count, metrics
        
    except Exception as e:
        logger.error(f"指标计算错误: {e}")
        return None, 0, {}

def fetch_candles_wrapper(args) -> tuple:
    """并行获取K线数据的包装函数"""
    base, symbol, granularity = args
    df = fetch_candles(base, symbol, granularity)
    if not df.empty:
        df["symbol"] = symbol
    return symbol, df

def create_statistics_cards(results: List[dict], total_symbols: int):
    """创建统计卡片"""
    col1, col2, col3, col4 = st.columns(4)
    
    oversold = len([r for r in results if r["rsi6"] < 30])
    overbought = len([r for r in results if r["rsi6"] > 70])
    gainers = len([r for r in results if r["change (%)"] > 0])
    
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <h3 style="color: #4ecdc4; margin: 0;">📊 总扫描数</h3>
            <h2 style="margin: 0.5rem 0;">{total_symbols}</h2>
            <p style="margin: 0; color: #666;">个交易对</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <h3 style="color: #ff6b6b; margin: 0;">🔥 超买信号</h3>
            <h2 style="margin: 0.5rem 0;">{overbought}</h2>
            <p style="margin: 0; color: #666;">RSI > 70</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown(f"""
        <div class="stat-card">
            <h3 style="color: #51cf66; margin: 0;">💎 超卖信号</h3>
            <h2 style="margin: 0.5rem 0;">{oversold}</h2>
            <p style="margin: 0; color: #666;">RSI < 30</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col4:
        st.markdown(f"""
        <div class="stat-card">
            <h3 style="color: #ffd43b; margin: 0;">📈 上涨币种</h3>
            <h2 style="margin: 0.5rem 0;">{gainers}</h2>
            <p style="margin: 0; color: #666;">24h涨幅 > 0</p>
        </div>
        """, unsafe_allow_html=True)

def create_rsi_distribution_chart(results: List[dict]):
    """创建RSI分布图表"""
    if not results:
        return None
        
    df = pd.DataFrame(results)
    
    # RSI分布直方图
    fig = px.histogram(
        df, 
        x="rsi6", 
        nbins=20,
        title="RSI6 分布图",
        labels={"rsi6": "RSI6 值", "count": "币种数量"},
        color_discrete_sequence=["#4ecdc4"]
    )
    
    # 添加超买超卖线
    fig.add_vline(x=30, line_dash="dash", line_color="green", annotation_text="超卖线 (30)")
    fig.add_vline(x=70, line_dash="dash", line_color="red", annotation_text="超买线 (70)")
    
    fig.update_layout(
        template="plotly_white",
        height=400,
        showlegend=False
    )
    
    return fig

def create_scatter_plot(results: List[dict]):
    """创建RSI vs 涨跌幅散点图"""
    if not results:
        return None
        
    df = pd.DataFrame(results)
    
    # 根据RSI区间着色
    def get_color(rsi):
        if rsi < 30:
            return "超卖"
        elif rsi > 70:
            return "超买" 
        else:
            return "中性"
    
    df["rsi_zone"] = df["rsi6"].apply(get_color)
    
    fig = px.scatter(
        df,
        x="rsi6",
        y="change (%)",
        color="rsi_zone",
        title="RSI6 vs 24小时涨跌幅",
        labels={"rsi6": "RSI6 值", "change (%)": "24h涨跌幅 (%)"},
        hover_data=["symbol"],
        color_discrete_map={
            "超卖": "#51cf66",
            "超买": "#ff6b6b", 
            "中性": "#868e96"
        }
    )
    
    # 添加分割线
    fig.add_hline(y=0, line_dash="dash", line_color="gray", annotation_text="涨跌分界线")
    fig.add_vline(x=30, line_dash="dash", line_color="green")
    fig.add_vline(x=70, line_dash="dash", line_color="red")
    
    fig.update_layout(
        template="plotly_white",
        height=400
    )
    
    return fig

def format_dataframe(df: pd.DataFrame, is_gainer: bool = True) -> pd.DataFrame:
    """格式化数据框显示"""
    if df.empty:
        return df
        
    # 添加趋势图标
    def add_trend_icon(row):
        change = row["change (%)"]
        rsi = row["rsi6"]
        
        if change > 5:
            trend = "🚀"
        elif change > 0:
            trend = "📈"
        elif change > -5:
            trend = "📉"
        else:
            trend = "💥"
            
        return f"{trend} {row['symbol']}"
    
    df_formatted = df.copy()
    df_formatted["交易对"] = df.apply(add_trend_icon, axis=1)
    df_formatted["24h涨跌"] = df_formatted["change (%)"].apply(lambda x: f"{x:+.2f}%")
    df_formatted["RSI6"] = df_formatted["rsi6"].apply(lambda x: f"{x:.1f}")
    df_formatted["K线数"] = df_formatted["k_lines"]
    df_formatted["备注"] = df_formatted["note"]
    
    return df_formatted[["交易对", "24h涨跌", "RSI6", "K线数", "备注"]]

def scan_symbols(base: str, symbols: List[str], granularity: str, rsi_low: float, rsi_high: float, min_volume: float = 0) -> Tuple[List[dict], dict]:
    """扫描交易对 - 修复版本"""
    start_time = time.time()
    results = []
    
    # 获取ticker数据
    with st.spinner("📊 正在获取市场数据..."):
        tickers = fetch_all_tickers(base)
        if not tickers:
            st.warning("⚠️ 无法获取完整的市场数据，将使用默认值")
            tickers = {}  # 继续执行，但使用空字典
    
    # 进度条容器
    progress_container = st.empty()
    status_container = st.empty()
    
    # 并行获取K线数据
    candle_data = {}
    total_symbols = len(symbols)
    processed = 0
    
    with ThreadPoolExecutor(max_workers=Config.MAX_WORKERS) as executor:
        futures = [executor.submit(fetch_candles_wrapper, (base, symbol, granularity)) for symbol in symbols]
        
        for future in as_completed(futures):
            symbol, df = future.result()
            processed += 1
            
            if not df.empty:
                candle_data[symbol] = df
                
            # 更新进度
            progress = processed / total_symbols
            progress_container.progress(progress, text=f"🔄 获取K线数据: {processed}/{total_symbols}")
            status_container.info(f"⏱️ 正在处理: {symbol}")
    
    # 清除进度显示
    progress_container.empty()
    status_container.empty()
    
    # 处理数据
    with st.spinner("🧮 正在计算技术指标..."):
        insufficient_data = []
        
        for symbol in symbols:
            try:
                if symbol not in candle_data:
                    continue
                    
                df = candle_data[symbol]
                rsi, candle_count, metrics = calculate_rsi_and_metrics(df)
                
                if rsi is None:
                    insufficient_data.append(symbol)
                    continue
                
                # 使用默认值如果ticker数据不可用
                ticker_data = tickers.get(symbol, {
                    "change24h": 0, 
                    "volume": 0, 
                    "price": 0
                })
                
                # 应用成交量过滤
                if ticker_data["volume"] < min_volume:
                    continue
                
                # 检查RSI条件
                if rsi < rsi_low or rsi > rsi_high:
                    note = ""
                    if candle_count < Config.MIN_CANDLES_RELIABLE:
                        note = f"数据较少({candle_count}根)"
                    
                    results.append({
                        "symbol": symbol,
                        "change (%)": round(ticker_data["change24h"], 2),
                        "rsi6": round(rsi, 2),
                        "k_lines": candle_count,
                        "note": note,
                        "volume": ticker_data["volume"],
                        "price": ticker_data["price"],
                        "volatility": metrics.get("volatility", 0)
                    })
                    
            except Exception as e:
                logger.warning(f"{symbol} 处理失败: {e}")
                continue
    
    # 确保scan_stats包含所有必需的字段
    scan_stats = {
        "scan_time": time.time() - start_time,
        "total_symbols": total_symbols,
        "processed_symbols": len(candle_data),
        "insufficient_data": len(insufficient_data),
        "results_count": len(results)
    }
    
    return results, scan_stats

def main():
    # 创建页面头部
    create_header()
    
    # 创建侧边栏并获取参数
    timeframe, rsi_low, rsi_high, show_charts, min_volume = create_sidebar()
    
    # 主要内容区域
    col1, col2 = st.columns([3, 1])
    
    with col2:
        # 扫描按钮
        if st.button("🚀 开始扫描", key="scan_button", help="点击开始扫描USDT永续合约"):
            scan_pressed = True
        else:
            scan_pressed = False
            
        # 显示当前设置
        with st.expander("📋 当前设置", expanded=True):
            st.write(f"⏰ **时间框架**: {timeframe}")
            st.write(f"📉 **超卖线**: {rsi_low}")
            st.write(f"📈 **超买线**: {rsi_high}")
            if min_volume > 0:
                st.write(f"📊 **最小成交量**: {min_volume:,.0f}")
    
    with col1:
        if not scan_pressed:
            # 显示使用说明
            st.markdown("""
            ### 🎯 使用指南
            
            **RSI6扫描器**是一个专业的技术分析工具，帮助您快速找到具有极端RSI值的交易机会：
            
            #### 📊 功能特点：
            - 🔄 **实时扫描**: 并行处理所有USDT永续合约
            - 📈 **多时间框架**: 支持1H、4H、1D级别分析  
            - 🎨 **可视化分析**: 直观的图表和统计信息
            - 📁 **数据导出**: 支持CSV格式下载
            - ⚡ **高性能**: 多线程处理，扫描速度快
            
            #### 🎯 交易信号：
            - 🟢 **超卖信号** (RSI < 30): 可能的买入机会
            - 🔴 **超买信号** (RSI > 70): 可能的卖出机会
            - ⚠️ **数据提醒**: 自动标注K线数据不足的币种
            
            #### 🚀 开始使用：
            1. 在左侧设置您的扫描参数
            2. 点击"开始扫描"按钮
            3. 等待扫描完成并查看结果
            4. 可选择下载数据进行进一步分析
            """)
            return
    
    if scan_pressed:
        try:
            # 获取API端点
            with st.spinner("🔗 连接到Bitget API..."):
                base = get_working_endpoint()
                st.success("✅ API连接成功")
            
            # 获取交易对
            with st.spinner("📋 获取交易对列表..."):
                symbols = get_usdt_symbols(base)
                st.success(f"✅ 找到 {len(symbols)} 个USDT永续合约")
            
            # 执行扫描
            results, scan_stats = scan_symbols(base, symbols, timeframe, rsi_low, rsi_high, min_volume)
            
            # 显示扫描统计
            st.success(f"✅ 扫描完成! 耗时 {scan_stats['scan_time']:.1f} 秒")
            
            if scan_stats['insufficient_data'] > 0:
                st.info(f"ℹ️ 有 {scan_stats['insufficient_data']} 个币种数据不足，已跳过")
            
            # 分类结果
            gainers = sorted([r for r in results if r["change (%)"] > 0], key=lambda x: x["rsi6"], reverse=True)
            losers = sorted([r for r in results if r["change (%)"] <= 0], key=lambda x: x["rsi6"])
            
            # 显示统计卡片
            create_statistics_cards(results, scan_stats['total_symbols'])
            
            # 🔥 第一部分：显示结果表格
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # 超买区域（涨幅榜）
            st.markdown(f"### 🔥 超买区域 (RSI6 {timeframe} > {rsi_high})")
            if gainers:
                gainers_df = pd.DataFrame(gainers)
                formatted_gainers = format_dataframe(gainers_df, True)
                st.dataframe(formatted_gainers, use_container_width=True, hide_index=True)
                
                # 下载按钮
                csv_data = gainers_df.to_csv(index=False)
                st.download_button(
                    label="📥 下载超买数据 CSV",
                    data=csv_data,
                    file_name=f"overbought_rsi6_{timeframe}_{current_time.replace(' ', '_').replace(':', '-')}.csv",
                    mime="text/csv",
                    key="download_gainers"
                )
            else:
                st.info("🤔 当前没有符合条件的超买信号")
            
            # 超卖区域（跌幅榜）  
            st.markdown(f"### 💎 超卖区域 (RSI6 {timeframe} < {rsi_low})")
            if losers:
                losers_df = pd.DataFrame(losers)
                formatted_losers = format_dataframe(losers_df, False)
                st.dataframe(formatted_losers, use_container_width=True, hide_index=True)
                
                # 下载按钮
                csv_data = losers_df.to_csv(index=False)
                st.download_button(
                    label="📥 下载超卖数据 CSV", 
                    data=csv_data,
                    file_name=f"oversold_rsi6_{timeframe}_{current_time.replace(' ', '_').replace(':', '-')}.csv",
                    mime="text/csv",
                    key="download_losers"
                )
            else:
                st.info("🤔 当前没有符合条件的超卖信号")
            
            # 📊 第二部分：显示图表分析（移到后面）
            if show_charts and results:
                st.markdown("---")  # 添加分隔线
                st.markdown("### 📊 数据分析")
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    rsi_chart = create_rsi_distribution_chart(results)
                    if rsi_chart:
                        st.plotly_chart(rsi_chart, use_container_width=True)
                
                with chart_col2:
                    scatter_chart = create_scatter_plot(results)
                    if scatter_chart:
                        st.plotly_chart(scatter_chart, use_container_width=True)
                
            # 扫描信息
            with st.expander("ℹ️ 扫描详情"):
                st.write(f"**扫描时间**: {current_time}")
                st.write(f"**处理时间**: {scan_stats['scan_time']:.2f} 秒")
                st.write(f"**总交易对数**: {scan_stats['total_symbols']}")
                st.write(f"**成功处理**: {scan_stats['processed_symbols']}")
                st.write(f"**符合条件**: {scan_stats['results_count']}")
                st.write(f"**数据不足**: {scan_stats['insufficient_data']}")
                
        except Exception as e:
            st.error(f"❌ 扫描过程中发生错误: {str(e)}")
            logger.error(f"扫描错误: {e}")

    # 页脚
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p>📈 RSI6 扫描器 Pro | 🚀 专业级量化交易工具</p>
        <p>⚠️ 投资有风险，交易需谨慎。本工具仅供参考，不构成投资建议。</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
