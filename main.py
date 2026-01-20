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
    results['yen_change_5d'] = df['JPY=X'].pct_change(5).iloc[-1]
    results['ief_change_10d'] = df['IEF'].pct_change(10).iloc[-1]

    return results

def evaluate_logic(indicators):
    # 1. Condition: Distortion
    is_distorted = indicators['distortion']['gap'] >= CONSTANTS['DISTORTION_THRESHOLD']
    
    # 2. Trigger A: Credit Crunch
    is_credit_low = indicators['credit']['val'] <= (indicators['credit']['min20'] * 1.0001)
    is_credit_downtrend = indicators['credit']['val'] < indicators['credit']['ma20']
    is_spx_high = indicators['spx']['price'] > indicators['spx']['ma50']
    
    trigger_a = is_credit_downtrend and is_credit_low and is_spx_high

    # 3. Trigger B: Unwind Shock (Yen Surge)
    trigger_b = indicators['yen_change_5d'] < CONSTANTS['YEN_SHOCK_THRESHOLD']

    # 4. Trigger C: Bad Rate Spike
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
    # Helper to format percentages
    def fmt_pct(val):
        return f"{val*100:+.2f}%"

    print("\n" + "="*60)
    print("📊 市場構造・危機検知レポート (v7.1)")
    print("="*60)
    
    # --- 1. Market Distortion ---
    gap = inds['distortion']['gap']
    gap_str = fmt_pct(gap)
    threshold_str = fmt_pct(CONSTANTS['DISTORTION_THRESHOLD'])
    
    print(f"\n1. Condition: Market Distortion (市場の歪み)")
    print(f"   結果: {gap_str} (閾値: {threshold_str}) → [{'TRUE' if logic['condition'] else 'FALSE'}]")
    print("   [分析]:")
    
    if logic['condition']:
        print("   ⚠️ 危険水域です。トップ50社への資金集中が歴史的な水準(+15%超)に達しています。")
        print("   崩壊時のエネルギー（燃料）が満タンの状態です。着火に注意してください。")
    else:
        if gap > 0:
            print(f"   データ上は「正常範囲内」です。直近200日の平均的な歪み方と大きな差がありません。")
            print("   歪んでいる状態が常態化（Baseline化）しており、新たな乖離加速は見られません。")
        else:
            print("   歪みは解消されています。トップ50社とそれ以外が連動、あるいは循環物色されています。")

    # --- 2. Trigger A: Credit Crunch ---
    cred_val = inds['credit']['val']
    cred_ma = inds['credit']['ma20']
    trend_str = "Bearish(下落)" if cred_val < cred_ma else "Bullish(上昇)"
    
    print(f"\n2. Trigger A: Credit Crunch (信用の収縮)")
    print(f"   結果: Trend: {trend_str}, 最安値更新: {'YES' if logic['trigger_a'] else 'NO'} → [{'TRUE' if logic['trigger_a'] else 'FALSE'}]")
    print("   [分析]:")
    
    if logic['trigger_a']:
        print("   ⛔ 危険信号点灯！株価は高いのに、債券市場で「ジャンク債」が捨てられています。")
        print("   「質への逃避」が始まっています。典型的な暴落の先行指標です。")
    elif cred_val >= cred_ma:
        print("   ジャンク債が国債に対して強く、トレンドは上昇(Bullish)です。")
        print("   これは「倒産リスクなんて誰も気にしていない（イケイケドンドン）」という状態です。")
        print("   暴落の気配は微塵もありません。")
    else:
        print("   信用スプレッドはやや悪化していますが、決定的な安値更新には至っていません。")
        print("   まだ「調整」の範囲内です。")

    # --- 3. Trigger B: Liquidity Shock ---
    yen_chg = inds['yen_change_5d']
    yen_str = fmt_pct(yen_chg)
    thresh_yen = fmt_pct(CONSTANTS['YEN_SHOCK_THRESHOLD'])
    
    print(f"\n3. Trigger B: Liquidity Shock (円キャリー)")
    print(f"   結果: {yen_str} (閾値: {thresh_yen}) → [{'TRUE' if logic['trigger_b'] else 'FALSE'}]")
    print("   [分析]:")
    
    if logic['trigger_b']:
        print("   ⛔ 危険信号点灯！急激な「円高」が進行しています。")
        print("   円キャリー取引の巻き戻し（強制決済）による、世界的な換金売りリスクが高まっています。")
    elif yen_chg > 0:
        print("   プラス値は「ドル高・円安」を意味します。")
        print("   現在は真逆です。むしろ円安が進んでおり、キャリー取引による資金供給（燃料注入）が続いています。")
    else:
        print("   円高方向への動きですが、パニック的な水準（-3%超）ではありません。")
        print("   通常の変動範囲内です。")

    # --- 4. Trigger C: Bad Rate Spike ---
    ief_chg = inds['ief_change_10d']
    spx_chg = inds['spx']['change_10d']
    
    print(f"\n4. Trigger C: Bad Rate Spike (悪い金利上昇)")
    print(f"   結果: 債券 {fmt_pct(ief_chg)}, 株価 {fmt_pct(spx_chg)} → [{'TRUE' if logic['trigger_c'] else 'FALSE'}]")
    print("   [分析]:")
    
    if logic['trigger_c']:
        print("   ⚠️ 警告！「悪い金利上昇」です。")
        print("   金利急騰（債券急落）に対し、株価が耐えきれず下落しています。バリュエーション調整の合図です。")
    elif ief_chg < CONSTANTS['RATE_SHOCK_THRESHOLD']:
        print("   金利は急騰（債券急落）していますが、株価は上昇しています。")
        print("   これは典型的な『良い金利上昇（業績相場・トランプトレード）』です。")
        print("   フィルターが機能し、正常と判定しました。")
    else:
        print("   金利のパニック的な急騰は見られません。落ち着いています。")

    print("-" * 60)
    
    # --- FINAL JUDGMENT ---
    if logic['trigger_a'] or logic['trigger_b']:
        level = "LEVEL 5: CRITICAL (崩壊)"
        msg = "【システムの逆回転】信用収縮(A) または 流動性枯渇(B) が発生。\n即時撤退を推奨します。"
    elif logic['trigger_c']:
        level = "LEVEL 4: WARNING (警戒)"
        msg = "【バリュエーション調整】悪い金利上昇(C) が発生。\nポジション縮小を推奨します。"
    elif logic['condition']:
        level = "LEVEL 3: OVERHEATED (過熱)"
        msg = "【バブル温存】歪みは大ですがトリガーなし。\n静観・準備フェーズです。"
    else:
        level = "LEVEL 1: NORMAL (正常)"
        msg = "【順行】システムは正常稼働中。\n投資継続で問題ありません。"

    print(f"\n{'#'*60}")
    print(f"   {level}")
    print(f"{'#'*60}")
    print(f"\n[総合判定メッセージ]\n{msg}\n")
    print(f"{'#'*60}\n")

if __name__ == "__main__":
    df = fetch_and_process_data()
    indicators = calculate_indicators(df)
    logic = evaluate_logic(indicators)
    print_report(indicators, logic)
