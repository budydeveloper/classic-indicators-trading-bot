import ccxt
import os
import time
import pandas as pd
import numpy as np
import json  # Se importa el módulo json para formatear la salida

# ========================================
# 1. CONFIGURACIÓN DE BINGX (SPOT)
# ========================================

# Obtiene las claves API desde las variables de entorno (o introdúcelas directamente)
API_KEY = os.getenv("BINGX_API_KEY")
SECRET_KEY = os.getenv("BINGX_SECRET_KEY")

# Inicializa el exchange con ccxt y activa la limitación de tasa
exchange = ccxt.bingx({
    'apiKey': API_KEY,
    'secret': SECRET_KEY,
    'enableRateLimit': True,
})
# Activar el modo sandbox (prueba) si está disponible
exchange.set_sandbox_mode(True)

# Parámetros generales de trading
CAPITAL_TOTAL = 1000         # Capital total en USDT
TRADE_PERCENTAGE = 1.00      # Operar con el 100% del capital en cada trade

# Definir símbolo y timeframe (ajustado a 4 horas, de acuerdo al backtest)
SYMBOL = "BTC-USDT"
TIMEFRAME = "1m"
# Intervalo de espera entre iteraciones en segundos 
WAIT_SECONDS = 1 * 5

# ========================================
# 2. PARÁMETROS DE INDICADORES
# ========================================

# Bollinger Bands
bb_length     = 20
bb_multiplier = 2.0

# MACD
macd_fast   = 12
macd_slow   = 26
macd_signal = 9

# RSI
rsi_period   = 14
rsi_buy_thr  = 50
rsi_sell_thr = 50

# ADX
adx_period    = 14
adx_threshold = 25.0

# Estocástico (%K)
stoch_k_period  = 14
stoch_buy_thr   = 80
stoch_sell_thr  = 20

# SMA
sma_period = 200

# ========================================
# 3. FUNCIONES PARA CALCULAR INDICADORES
# ========================================

def calculate_bollinger_bands(df, length, multiplier):
    df['BB_Basis'] = df['close'].rolling(window=length).mean()
    df['BB_StdDev'] = df['close'].rolling(window=length).std()
    df['BB_Upper'] = df['BB_Basis'] + multiplier * df['BB_StdDev']
    df['BB_Lower'] = df['BB_Basis'] - multiplier * df['BB_StdDev']
    return df

def calculate_macd(df, fast, slow, signal):
    df['EMA_fast'] = df['close'].ewm(span=fast, adjust=False).mean()
    df['EMA_slow'] = df['close'].ewm(span=slow, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD_signal'] = df['MACD'].ewm(span=signal, adjust=False).mean()
    return df

def calculate_rsi(df, period):
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def calculate_adx(df, period):
    df['prev_close'] = df['close'].shift(1)
    df['tr1'] = df['high'] - df['low']
    df['tr2'] = (df['high'] - df['prev_close']).abs()
    df['tr3'] = (df['low'] - df['prev_close']).abs()
    df['TR'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['ATR'] = df['TR'].ewm(alpha=1/period, min_periods=period).mean()
    
    df['up_move'] = df['high'] - df['high'].shift(1)
    df['down_move'] = df['low'].shift(1) - df['low']
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    df['plus_di'] = 100 * df['plus_dm'].ewm(alpha=1/period, min_periods=period).mean() / df['ATR']
    df['minus_di'] = 100 * df['minus_dm'].ewm(alpha=1/period, min_periods=period).mean() / df['ATR']
    df['dx'] = 100 * np.abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['ADX'] = df['dx'].ewm(alpha=1/period, min_periods=period).mean()
    return df

def calculate_stoch(df, period):
    df['lowest_low'] = df['low'].rolling(window=period).min()
    df['highest_high'] = df['high'].rolling(window=period).max()
    df['stochK'] = 100 * (df['close'] - df['lowest_low']) / (df['highest_high'] - df['lowest_low'])
    return df

def calculate_sma(df, period):
    df['SMA'] = df['close'].rolling(window=period).mean()
    return df

# ========================================
# 4. LOOP DE TRADING (SPOT SIN LEVERAGE)
# ========================================

have_position = False
position_avg_price = None
position_quantity = None

print("🚀 Estrategia combinada (BB, MACD, RSI, ADX, Stoch, SMA) en SPOT sin leverage iniciada...")

while True:
    try:
        # 1) OBTENER DATOS DE MERCADO  
        limit = sma_period + 50
        ohlcv = exchange.fetch_ohlcv(SYMBOL, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
        
        # 2) CALCULAR LOS INDICADORES
        df = calculate_bollinger_bands(df, bb_length, bb_multiplier)
        df = calculate_macd(df, macd_fast, macd_slow, macd_signal)
        df = calculate_rsi(df, rsi_period)
        df = calculate_adx(df, adx_period)
        df = calculate_stoch(df, stoch_k_period)
        df = calculate_sma(df, sma_period)
        
        # Última barra (datos más recientes)
        latest = df.iloc[-1]
        price            = latest['close']
        bb_upper         = latest['BB_Upper']
        bb_lower         = latest['BB_Lower']
        macd_val         = latest['MACD']
        macd_signal_val  = latest['MACD_signal']
        rsi_val          = latest['RSI']
        adx_val          = latest['ADX']
        stochK_val       = latest['stochK']
        sma_val          = latest['SMA']
        
        # Verifica que todos los indicadores estén calculados
        if pd.isna(bb_upper) or pd.isna(macd_val) or pd.isna(rsi_val) or pd.isna(adx_val) or pd.isna(stochK_val) or pd.isna(sma_val):
            print("Esperando suficientes datos para calcular indicadores...")
            time.sleep(WAIT_SECONDS)
            continue
        
        # Muestra los valores de los indicadores para seguimiento
        print(f"\n[{pd.Timestamp.now()}] Precio actual: {price:.2f} USDT")
        print(f"   BB Upper: {bb_upper:.2f} | BB Lower: {bb_lower:.2f}")
        print(f"   MACD: {macd_val:.4f} | Signal: {macd_signal_val:.4f}")
        print(f"   RSI: {rsi_val:.2f} | ADX: {adx_val:.2f}")
        print(f"   StochK: {stochK_val:.2f} | SMA: {sma_val:.2f}")
        
        # 3) DEFINIR CONDICIONES DE ENTRADA Y SALIDA  
        buy_condition = (
            (price > bb_upper * 0.98) or 
            (macd_val > macd_signal_val) or 
            (rsi_val > (rsi_buy_thr - 10)) or 
            (adx_val > (adx_threshold - 5)) or 
            (stochK_val > (stoch_buy_thr - 10)) or 
            (price > sma_val)
        )
                            
        sell_condition = (
            (price < bb_lower) and 
            (macd_val < macd_signal_val) and 
            (rsi_val < rsi_sell_thr) and 
            (adx_val > adx_threshold) and 
            (stochK_val < stoch_sell_thr) and 
            (price < sma_val)
        )
        
        # 4) EJECUTAR ÓRDENES SEGÚN CONDICIONES
        if not have_position and buy_condition:
            trade_capital = CAPITAL_TOTAL * TRADE_PERCENTAGE
            quantity = trade_capital / price  # Calcula la cantidad a comprar
            print(f"🟢 Señal de COMPRA detectada: Ejecutando orden de compra de {quantity:.6f} {SYMBOL.split('-')[0]} a {price:.2f} USDT")
            order = exchange.create_market_order(
                SYMBOL,
                'buy',
                quantity,
                None,
                {"positionSide": "LONG"}
            )
            print("   Orden de compra ejecutada:")
            # Se imprime la respuesta de la orden en formato pretty
            print(json.dumps(order, indent=2, ensure_ascii=False))
            have_position = True
            position_avg_price = price
            position_quantity = quantity
        
        elif have_position and sell_condition:
            print(f"🔴 Señal de VENTA detectada: Ejecutando orden de venta de {position_quantity:.6f} {SYMBOL.split('-')[0]} a {price:.2f} USDT")
            order = exchange.create_market_order(
                SYMBOL,
                'sell',
                position_quantity,
                None,
                {"positionSide": "SHORT"}
            )
            print("   Orden de venta ejecutada:")
            print(json.dumps(order, indent=2, ensure_ascii=False))
            have_position = False
            position_avg_price = None
            position_quantity = None
        else:
            if not have_position:
                print("   No se detectaron señales de COMPRA.")
            else:
                print("   Posición abierta. Esperando señal de VENTA.")
    
    except Exception as e:
        print("❌ Error en la ejecución:", e)
    
    print(f"⏳ Esperando {WAIT_SECONDS:.0f} segundos para la próxima iteración...\n")
    time.sleep(WAIT_SECONDS)
