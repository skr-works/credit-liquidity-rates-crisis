import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import sys

# --- CONSTANTS (v7.1 Spec) ---
CONSTANTS = {
    'TICKERS': ['XLG', 'RSP', 'HYG', 'IEF', 'JPY=X', '^GSPC'],
    'PERIOD': '2y',
    'DISTORTION_THRESHOLD': 0.15,      # Condition: XLG/RSP Gap > 15%
    'CREDIT_LOOKBACK': 20,             # Trigger A: 20-day lookback
    'YEN_SHOCK_THRESHOLD': -0.03,      # Trigger B: 5-day change < -3% (Yen Surge)
    'RATE_SHOCK_THRESHOLD': -0.02,     # Trigger C: 10-day IEF change < -2%
    'SPX_FILTER_THRESHOLD': 0.0        # Trigger C: SPX must be negative
}

def fetch_and_process_data():
    print(f"[INFO] Fetching data for: {CONSTANTS['TICKERS']}")
    
    # 1. Download with Auto Adjust (Crucial for total return accuracy)
    try:
        raw_data = yf.download(
            CONSTANTS['TICKERS'], 
            period=CONSTANTS['PERIOD'], 
            auto_adjust=True, 
            progress=False,
            threads=True
        )['Close']
    except Exception as e:
        print(f"[ERROR] Failed to download data: {e}")
        sys.exit(1)

    # 2. Preprocessing: Strict DropNA
    # Forbidden: ffill (Do not fill holidays, it dulls the sensitivity)
    df = raw_data.dropna()
    
    if df.empty:
        print("[ERROR] DataFrame is empty after dropna. Check tickers or period.")
        sys.exit(1)
        
    print(f"[INFO] Data synced. Latest Date: {df.index[-1].strftime('%Y-%m-%d')}")
    return df

def calculate_indicators(df):
    results = {}
    
    # --- A. Structure Distortion (XLG / RSP) ---
    distortion_ratio = df['XLG'] / df['RSP']
    baseline_200 = distortion_ratio.rolling(200).mean()
    current_gap = (distortion_ratio / baseline_200) - 1
    
    results['distortion'] = {
        'val': distortion_ratio.iloc[-1],
        'baseline': baseline_200.iloc[-1],
        'gap': current_gap.iloc[-1]
    }

    # --- B. Credit Crunch (HYG / IEF) ---
    credit_ratio = df['HYG'] / df['IEF']
    credit_ma20 = credit_ratio.rolling(CONSTANTS['CREDIT_LOOKBACK']).mean()
    credit_min20 = credit_ratio.rolling(CONSTANTS['CREDIT_LOOKBACK']).min()
    
    results['credit'] = {
        'val': credit_ratio.iloc[-1],
        'ma20': credit_ma20.iloc[-1],
        'min20': credit_min20.iloc[-1]
    }

    # --- C. Market Context (S&P 500) ---
    spx_price = df['^GSPC']
    spx_ma50 = spx_price.rolling(50).mean()
    
    results['spx'] = {
        'price': spx_price.iloc[-1],
        'ma50': spx_ma50.iloc[-1],
        'change_10d': spx_price.pct_change(10).iloc[-1]
    }

    # --- D. Risk Parameters (Yen & Rate) ---
    # JPY=X: USD/JPY. A drop means Yen appreciation (Yen strength).
    results['yen_change_5d'] = df['JPY=X'].pct_change(5).iloc[-1]
    results['ief_change_10d'] = df['IEF'].pct_change(10).iloc[-1]

    return results

def evaluate_logic(indicators):
    # 1. Condition: Distortion
    is_distorted = indicators['distortion']['gap'] >= CONSTANTS['DISTORTION_THRESHOLD']
    
    # 2. Trigger A: Credit Crunch
    # Logic: Ratio < MA20 AND Ratio is at 20-day Low AND SPX is still strong (>MA50)
    # Using a small tolerance for float comparison on 'min'
    is_credit_low = indicators['credit']['val'] <= (indicators['credit']['min20'] * 1.0001)
    is_credit_downtrend = indicators['credit']['val'] < indicators['credit']['ma20']
    is_spx_high = indicators['spx']['price'] > indicators['spx']['ma50']
    
    trigger_a = is_credit_downtrend and is_credit_low and is_spx_high

    # 3. Trigger B: Unwind Shock (Yen Surge)
    trigger_b = indicators['yen_change_5d'] < CONSTANTS['YEN_SHOCK_THRESHOLD']

    # 4. Trigger C: Bad Rate Spike
    # Logic: Rates spiked (IEF crashed) AND Stocks fell. 
    # (Filters out "Good Rate Hikes" where stocks go up)
    is_rate_crash = indicators['ief_change_10d'] < CONSTANTS['RATE_SHOCK_THRESHOLD']
    is_stock_down = indicators['spx']['change_10d'] < CONSTANTS['SPX_FILTER_THRESHOLD']
    
    trigger_c = is_rate_crash and is_stock_down

    return {
        'condition': is_distorted,
        'trigger_a': trigger_a,
        'trigger_b': trigger_b,
        'trigger_c': trigger_c
    }

def print_report(inds, logic):
    print("\n" + "-"*50)
    print("[CALCULATION REPORT] v7.1")
    
    # 1. Distortion
    gap_pct = inds['distortion']['gap'] * 100
    cond_str = "[TRUE]" if logic['condition'] else "[FALSE]"
    print(f"\n1. Condition: Market Distortion (XLG/RSP)")
    print(f"   - Gap vs MA200: {gap_pct:+.2f}% (Threshold: +15%)")
    print(f"   >>> CONDITION: {cond_str}")

    # 2. Trigger A
    cred_val = inds['credit']['val']
    cred_ma = inds['credit']['ma20']
    cred_min = inds['credit']['min20']
    trig_a_str = "[TRUE]" if logic['trigger_a'] else "[FALSE]"
    print(f"\n2. Trigger A: Credit Crunch (HYG/IEF)")
    print(f"   - Current Ratio: {cred_val:.4f}")
    print(f"   - 20d Trend: {'Bearish' if cred_val < cred_ma else 'Bullish'}")
    print(f"   - At 20d Low?: {'YES' if cred_val <= cred_min * 1.0001 else 'NO'}")
    print(f"   - SPX Context: {'High (>MA50)' if inds['spx']['price'] > inds['spx']['ma50'] else 'Low'}")
    print(f"   >>> TRIGGER A: {trig_a_str}")

    # 3. Trigger B
    yen_chg = inds['yen_change_5d'] * 100
    trig_b_str = "[TRUE]" if logic['trigger_b'] else "[FALSE]"
    print(f"\n3. Trigger B: Liquidity Shock (USD/JPY)")
    print(f"   - 5-Day Change: {yen_chg:+.2f}% (Threshold: -3.0%)")
    print(f"   >>> TRIGGER B: {trig_b_str}")

    # 4. Trigger C
    ief_chg = inds['ief_change_10d'] * 100
    spx_chg = inds['spx']['change_10d'] * 100
    trig_c_str = "[TRUE]" if logic['trigger_c'] else "[FALSE]"
    print(f"\n4. Trigger C: Bad Rate Spike (Filter applied)")
    print(f"   - IEF 10d Change: {ief_chg:+.2f}% (Threshold: -2.0%)")
    print(f"   - SPX 10d Change: {spx_chg:+.2f}%")
    print(f"   >>> TRIGGER C: {trig_c_str}")

    print("-" * 50)
    
    # --- FINAL JUDGMENT ---
    if logic['trigger_a'] or logic['trigger_b']:
        level = "LEVEL 5: CRITICAL (崩壊)"
        msg = "【システムの逆回転】信用収縮(A) または 流動性枯渇(B) が発生。即時撤退推奨。"
    elif logic['trigger_c']:
        level = "LEVEL 4: WARNING (警戒)"
        msg = "【バリュエーション調整】悪い金利上昇(C) が発生。ポジション縮小推奨。"
    elif logic['condition']:
        level = "LEVEL 3: OVERHEATED (過熱)"
        msg = "【バブル温存】歪みは大だがトリガーなし。静観・準備。"
    else:
        level = "LEVEL 1: NORMAL (正常)"
        msg = "【順行】システムは正常稼働中。"

    print(f"\n{'#'*50}")
    print(f"   {level}")
    print(f"{'#'*50}")
    print(f"\n[MESSAGE]\n{msg}\n")
    print(f"{'#'*50}\n")

if __name__ == "__main__":
    df = fetch_and_process_data()
    indicators = calculate_indicators(df)
    logic = evaluate_logic(indicators)
    print_report(indicators, logic)
