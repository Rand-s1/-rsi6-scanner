import requests
import ta  # 替换 talib
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
import logging
import time
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 配置
ENDPOINTS = ["https://api.bitget.com"]
PRODUCT_TYPE = "usdt-futures"
LIMIT = 100
RSI_PERIOD = 6
SLEEP_BETWEEN_REQUESTS = 0.5
MAX_WORKERS = 10
MIN_CANDLES_RELIABLE = 20  # 可靠RSI计算的最小K线数量

def ping_endpoint(endpoint: str) -> bool:
    """测试端点是否可用"""
    url = f"{endpoint}/api/v2/mix/market/candles"
    params = {
        "symbol": "BTCUSDT",
        "granularity": "4H",
        "limit": 1,
        "productType": PRODUCT_TYPE,
    }
    try:
        r = requests.get(url, params=params, timeout=5)
        logger.info(f"端点: {endpoint}, 状态码: {r.status_code}, 响应: {r.text[:200]}")
        return r.status_code == 200 and r.json().get("code") == "00000"
    except requests.exceptions.RequestException as e:
        logger.error(f"请求 {endpoint} 失败: {e}")
        return False
    except ValueError as e:
        logger.error(f"JSON 解析错误: {e}")
        return False

def get_working_endpoint() -> str:
    """获取可用端点"""
    logger.info("寻找可用 API 端点...")
    for ep in ENDPOINTS:
        logger.info(f"尝试端点: {ep}")
        for _ in range(3):
            if ping_endpoint(ep):
                logger.info(f"找到可用端点: {ep}")
                return ep
            logger.warning(f"{ep} 失败，稍后重试")
            time.sleep(1)
        logger.warning(f"{ep} 不可用")
    raise RuntimeError("无可用端点，请检查网络或 API")

def get_usdt_symbols(base: str) -> List[str]:
    """获取 USDT 永续合约交易对"""
    url = f"{base}/api/v2/mix/market/contracts"
    params = {"productType": PRODUCT_TYPE}
    try:
        r = requests.get(url, params=params, timeout=5)
        j = r.json()
        if j.get("code") != "00000":
            raise RuntimeError(f"获取交易对失败: {j}")
        symbols = [c["symbol"] for c in j["data"]]
        logger.info(f"找到 {len(symbols)} 个 USDT 永续合约: {symbols[:5]}...")
        return symbols
    except requests.exceptions.RequestException as e:
        logger.error(f"网络错误: {e}")
        raise
    except Exception as e:
        logger.error(f"错误: {e}")
        raise

def fetch_candles(base: str, symbol: str, granularity: str) -> pd.DataFrame:
    """获取 K线数据"""
    url = f"{base}/api/v2/mix/market/candles"
    params = {
        "symbol": symbol,
        "granularity": granularity,
        "limit": LIMIT,
        "productType": PRODUCT_TYPE,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        j = r.json()
        if j.get("code") != "00000":
            raise RuntimeError(f"{symbol} K线失败: {j}")
        cols = ["ts", "open", "high", "low", "close", "volume_base", "volume_quote"]
        df = pd.DataFrame(j["data"], columns=cols)
        df[["open", "high", "low", "close", "volume_base", "volume_quote"]] = df[
            ["open", "high", "low", "close", "volume_base", "volume_quote"]
        ].astype(float)
        df["ts"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
        return df.sort_values("ts").reset_index(drop=True)
    except requests.exceptions.RequestException as e:
        logger.error(f"{symbol} 网络错误: {e}")
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"{symbol} 错误: {e}")
        return pd.DataFrame()

def fetch_all_tickers(base: str) -> Dict[str, float]:
    """批量获取所有交易对的 ticker 数据"""
    url = f"{base}/api/v2/mix/market/tickers"
    params = {"productType": PRODUCT_TYPE}
    try:
        r = requests.get(url, params=params, timeout=5)
        j = r.json()
        logger.info(f"Ticker 批量响应: {j.get('code')} {j.get('msg')}")
        if j.get("code") != "00000" or not isinstance(j.get("data"), list):
            logger.error(f"获取 ticker 数据失败: {j}")
            return {}
        tickers = {item["symbol"]: float(item["change24h"]) * 100 for item in j["data"]}
        logger.info(f"获取 {len(tickers)} 个 ticker 数据")
        return tickers
    except requests.exceptions.RequestException as e:
        logger.error(f"网络错误: {e}")
        return {}
    except Exception as e:
        logger.error(f"错误: {e}")
        return {}

def calculate_rsi(df: pd.DataFrame) -> tuple:
    """计算 RSI6 - 使用 ta 库，返回 (rsi_value, candle_count)"""
    try:
        symbol = df['symbol'].iloc[0] if isinstance(df['symbol'], pd.Series) else df['symbol']
        logger.info(f"calculate_rsi: 交易对: {symbol}")
        
        # 确保数据是pandas Series格式
        close_series = pd.Series(df["close"].astype(float)).reset_index(drop=True)
        candle_count = len(close_series)
        logger.info(f"calculate_rsi: close 数据长度: {candle_count}")
        
        if candle_count < RSI_PERIOD + 1:
            logger.warning(f"{symbol} 数据不足: {candle_count} 根K线")
            return None, candle_count
            
        # 使用ta库计算RSI
        rsi_series = ta.momentum.RSIIndicator(close=close_series, window=RSI_PERIOD).rsi()
        rsi = rsi_series.iloc[-1]
        
        logger.info(f"calculate_rsi: RSI 计算结果: {rsi}")
        return rsi, candle_count
    except Exception as e:
        logger.error(f"RSI 计算错误: {e}")
        return None, 0

def fetch_candles_wrapper(args) -> tuple:
    """并行请求 K 线数据的包装函数"""
    base, symbol, granularity = args
    df = fetch_candles(base, symbol, granularity)
    if not df.empty:
        df["symbol"] = pd.Series([symbol] * len(df))
    return symbol, df

def scan_symbols(base: str, symbols: List[str], granularity: str, rsi_low: float, rsi_high: float) -> List[dict]:
    """扫描交易对，筛选 RSI6 满足条件的交易对"""
    start_time = time.time()
    results = []

    # 批量获取 ticker 数据
    logger.info("开始批量获取 ticker 数据...")
    tickers = fetch_all_tickers(base)
    if not tickers:
        st.error("无法获取 ticker 数据，请检查 API")
        return results

    # 创建进度条
    progress_container = st.empty()
    progress_bar = progress_container.progress(0.0, text="正在获取 K 线数据: 0%")

    # 并行获取 K 线数据
    logger.info(f"开始并行获取 {len(symbols)} 个交易对的 K 线数据...")
    candle_data = {}
    total_symbols = len(symbols)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [executor.submit(fetch_candles_wrapper, (base, symbol, granularity)) for symbol in symbols]
        for future in as_completed(futures):
            symbol, df = future.result()
            if not df.empty:
                candle_data[symbol] = df
            else:
                logger.warning(f"{symbol} K 线数据为空")
            # 更新进度条
            progress = len(candle_data) / total_symbols
            progress_bar.progress(progress, text=f"正在获取 K 线数据: {int(progress * 100)}%")

    # 移除进度条
    progress_container.empty()

    # 本地处理数据
    logger.info("开始本地处理数据...")
    for symbol in symbols:
        try:
            if symbol not in candle_data:
                continue
            df = candle_data[symbol]
            rsi, candle_count = calculate_rsi(df)
            if rsi is None:
                continue
            change = tickers.get(symbol, 0.0)
            if rsi < rsi_low or rsi > rsi_high:
                # 添加K线数量信息和备注
                note = ""
                if candle_count < MIN_CANDLES_RELIABLE:
                    note = f"数据较少({candle_count}根)"
                
                results.append({
                    "symbol": symbol,
                    "change (%)": round(change, 2),
                    "rsi6": round(rsi, 2),
                    "k_lines": candle_count,
                    "note": note
                })
        except Exception as e:
            logger.warning(f"{symbol} 处理失败: {e}")
            st.warning(f"{symbol} 处理失败: {e}")
            continue

    logger.info(f"扫描完成，耗时: {time.time() - start_time:.2f} 秒")
    return results

def main():
    st.set_page_config(page_title="鹅的 RSI6 扫描器 (BitgetUSDT 永续合约)", layout="wide")
    st.title("鹅的 RSI6 扫描器")
    st.markdown("扫描 Bitget USDT 永续合约，获取 4H 或 1D 时间框架下 RSI6 满足条件的交易对。")

    # 输入区域
    st.subheader("设置参数")
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        timeframe = st.selectbox("时间框架", ["4H", "1D"], index=0)
    with col2:
        rsi_low = st.number_input("RSI6 下限", min_value=0.0, max_value=100.0, value=10.0, step=1.0)
    with col3:
        rsi_high = st.number_input("RSI6 上限", min_value=0.0, max_value=100.0, value=90.0, step=1.0)

    # 显眼的扫描按钮
    if st.button("立即扫描", key="scan_button", help="点击开始扫描 USDT 永续合约"):
        with st.spinner("正在扫描 Bitget USDT 永续合约..."):
            try:
                # 获取端点
                base = get_working_endpoint()
                # 获取交易对
                symbols = get_usdt_symbols(base)
                if not symbols:
                    st.error("未找到 USDT 永续合约，请检查 API")
                    return

                # 扫描
                results = scan_symbols(base, symbols, timeframe, rsi_low, rsi_high)
                gainers = sorted([r for r in results if r["change (%)"] > 0], key=lambda x: x["rsi6"], reverse=True)
                losers = sorted([r for r in results if r["change (%)"] <= 0], key=lambda x: x["rsi6"])

                # 显示时间
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                st.write(f"**扫描时间**：{current_time}")
                
                # 显示数据说明
                st.info(f"💡 数据说明：K线数量少于{MIN_CANDLES_RELIABLE}根的币种会显示备注，这些RSI值可能不够准确。")

                # 涨幅榜
                st.subheader(f"涨幅榜 (RSI6 {timeframe} > {rsi_high})")
                gainers_df = pd.DataFrame(gainers)
                if not gainers_df.empty:
                    # 重新排序列，让备注更明显
                    gainers_df = gainers_df[["symbol", "change (%)", "rsi6", "k_lines", "note"]]
                    st.dataframe(gainers_df, use_container_width=True)
                    st.download_button(
                        label="下载涨幅榜 CSV",
                        data=gainers_df.to_csv(index=False),
                        file_name=f"gainers_rsi6_{timeframe}_{current_time.replace(' ', '_')}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("没有满足 RSI6 条件的上涨交易对。")

                # 跌幅榜
                st.subheader(f"跌幅榜 (RSI6 {timeframe} < {rsi_low})")
                losers_df = pd.DataFrame(losers)
                if not losers_df.empty:
                    # 重新排序列，让备注更明显
                    losers_df = losers_df[["symbol", "change (%)", "rsi6", "k_lines", "note"]]
                    st.dataframe(losers_df, use_container_width=True)
                    st.download_button(
                        label="下载跌幅榜 CSV",
                        data=losers_df.to_csv(index=False),
                        file_name=f"losers_rsi6_{timeframe}_{current_time.replace(' ', '_')}.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("没有满足 RSI6 条件的下跌交易对。")
            except Exception as e:
                st.error(f"扫描出错：{e}")
                logger.error(f"扫描错误：{e}")

if __name__ == "__main__":
    main()
