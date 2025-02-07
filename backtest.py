import pandas as pd
import numpy as np


def cargar_csv_btc(file_path, start_date=None, end_date=None):
    """
    Carga un archivo CSV con datos históricos de BTC y aplica filtros de fecha.

    Parámetros:
        file_path (str): Ruta del archivo CSV.
        start_date (str, opcional): Fecha de inicio en formato 'YYYY-MM-DD'.
        end_date (str, opcional): Fecha de fin en formato 'YYYY-MM-DD'.

    Retorna:
        pd.DataFrame: DataFrame con las columnas:
            - timestamp
            - open
            - high
            - low
            - close
            - volume
    """
    df = pd.read_csv(file_path)

    # Convertir la columna 'Date' a datetime y renombrar columnas
    df['Date'] = pd.to_datetime(df['Date'])
    df.rename(columns={
        'Date': 'timestamp',
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)

    # Seleccionar las columnas de interés
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Filtrar por fecha si se especifica
    if start_date is not None:
        df = df[df['timestamp'] >= pd.to_datetime(start_date)]
    if end_date is not None:
        df = df[df['timestamp'] <= pd.to_datetime(end_date)]

    return df


def calculate_bollinger_bands(df, length=20, multiplier=2.0):
    """
    Calcula las Bandas de Bollinger sobre el precio de cierre.

    Parámetros:
        df (pd.DataFrame): DataFrame con la columna 'close'.
        length (int): Número de periodos para la media móvil.
        multiplier (float): Multiplicador de la desviación estándar.

    Retorna:
        pd.DataFrame: DataFrame con las columnas:
            - BB_Basis: Media móvil simple.
            - BB_StdDev: Desviación estándar.
            - BB_Upper: Banda superior.
            - BB_Lower: Banda inferior.
    """
    df['BB_Basis'] = df['close'].rolling(window=length).mean()
    df['BB_StdDev'] = df['close'].rolling(window=length).std()
    df['BB_Upper'] = df['BB_Basis'] + (multiplier * df['BB_StdDev'])
    df['BB_Lower'] = df['BB_Basis'] - (multiplier * df['BB_StdDev'])
    return df


def calculate_macd(df, fastperiod=12, slowperiod=26, signalperiod=9):
    """
    Calcula el MACD (Moving Average Convergence Divergence).

    Parámetros:
        df (pd.DataFrame): DataFrame con la columna 'close'.
        fastperiod (int): Periodo para la EMA rápida.
        slowperiod (int): Periodo para la EMA lenta.
        signalperiod (int): Periodo para la línea de señal.

    Retorna:
        pd.DataFrame: DataFrame con las columnas:
            - EMA_fast, EMA_slow, MACD, MACD_signal, MACD_hist.
    """
    df['EMA_fast'] = df['close'].ewm(span=fastperiod, adjust=False).mean()
    df['EMA_slow'] = df['close'].ewm(span=slowperiod, adjust=False).mean()
    df['MACD'] = df['EMA_fast'] - df['EMA_slow']
    df['MACD_signal'] = df['MACD'].ewm(span=signalperiod, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
    return df


def calculate_rsi(df, period=14):
    """
    Calcula el RSI (Relative Strength Index).

    Parámetros:
        df (pd.DataFrame): DataFrame con la columna 'close'.
        period (int): Número de periodos para el cálculo del RSI.

    Retorna:
        pd.DataFrame: DataFrame con la columna 'RSI'.
    """
    df['change'] = df['close'].diff(1)
    df['gain'] = df['change'].mask(df['change'] < 0, 0)
    df['loss'] = df['change'].mask(df['change'] > 0, 0).abs()
    df['avg_gain'] = df['gain'].ewm(alpha=1/period, adjust=False).mean()
    df['avg_loss'] = df['loss'].ewm(alpha=1/period, adjust=False).mean()
    df['RS'] = df['avg_gain'] / df['avg_loss']
    df['RSI'] = 100 - (100 / (1 + df['RS']))
    return df


def calculate_adx(df, period=14):
    """
    Calcula el ADX (Average Directional Index) y sus componentes.

    Parámetros:
        df (pd.DataFrame): DataFrame con las columnas 'high', 'low', 'close'.
        period (int): Número de periodos para el cálculo.

    Retorna:
        pd.DataFrame: DataFrame con las columnas '+DM', '-DM', 'TR', '+DI', '-DI', 'DX' y 'ADX'.
    """
    df = df.copy()
    df['+DM'] = df['high'].diff()
    df['-DM'] = (-df['low'].diff())
    df['+DM'] = np.where((df['+DM'] > df['-DM']) & (df['+DM'] > 0), df['+DM'], 0)
    df['-DM'] = np.where((df['-DM'] > df['+DM']) & (df['-DM'] > 0), df['-DM'], 0)
    
    tr1 = df['high'] - df['low']
    tr2 = (df['high'] - df['close'].shift()).abs()
    tr3 = (df['low'] - df['close'].shift()).abs()
    df['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Suavizado al estilo Wilder usando EMA con alpha = 1/period
    df['TR_smoothed'] = df['TR'].ewm(alpha=1/period, adjust=False).mean()
    df['+DM_smoothed'] = df['+DM'].ewm(alpha=1/period, adjust=False).mean()
    df['-DM_smoothed'] = df['-DM'].ewm(alpha=1/period, adjust=False).mean()
    
    df['+DI'] = 100 * (df['+DM_smoothed'] / df['TR_smoothed'])
    df['-DI'] = 100 * (df['-DM_smoothed'] / df['TR_smoothed'])
    df['DX'] = 100 * ((df['+DI'] - df['-DI']).abs() / (df['+DI'] + df['-DI']))
    df['ADX'] = df['DX'].ewm(alpha=1/period, adjust=False).mean()
    return df


def calculate_stochastic(df, k_period=14, d_period=3):
    """
    Calcula el Oscilador Estocástico.

    Parámetros:
        df (pd.DataFrame): DataFrame con columnas 'high', 'low' y 'close'.
        k_period (int): Ventana para el cálculo de %K.
        d_period (int): Ventana para el promedio móvil de %K (%D).

    Retorna:
        pd.DataFrame: DataFrame con las columnas '%K' y '%D'.
    """
    df = df.copy()
    df['lowest_low'] = df['low'].rolling(window=k_period).min()
    df['highest_high'] = df['high'].rolling(window=k_period).max()
    # Se añade una pequeña constante (1e-9) para evitar división por cero.
    df['%K'] = 100 * ((df['close'] - df['lowest_low']) /
                      (df['highest_high'] - df['lowest_low'] + 1e-9))
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    return df


def calculate_sma(df, period=200):
    """
    Calcula la Media Móvil Simple (SMA) sobre el precio de cierre.

    Parámetros:
        df (pd.DataFrame): DataFrame con la columna 'close'.
        period (int): Número de periodos para la SMA.

    Retorna:
        pd.DataFrame: DataFrame con la columna 'SMA_{period}'.
    """
    df[f'SMA_{period}'] = df['close'].rolling(window=period).mean()
    return df


def backtest_all_indicators(
    df,
    initial_capital,
    fee,
    # Parámetros para Bollinger
    bb_length,
    bb_multiplier,
    # Parámetros para MACD
    macd_fast,
    macd_slow,
    macd_signal,
    # Parámetros para RSI
    rsi_period,
    rsi_buy_threshold,
    rsi_sell_threshold,
    # Parámetros para ADX
    adx_period,
    adx_threshold,
    # Parámetros para Estocástico
    stoch_k_period,
    stoch_d_period,
    stoch_buy_threshold,
    stoch_sell_threshold,
    # Parámetro para filtro SMA
    sma_period
):
    """
    Realiza el _backtesting_ de una estrategia que combina varios indicadores:
      - Bollinger, MACD, RSI, ADX, Estocástico y un filtro de tendencia (SMA).

    La estrategia establece:
      - COMPRA cuando se cumplen todas las siguientes condiciones:
            * close > BB_Upper
            * MACD > MACD_signal
            * RSI > rsi_buy_threshold
            * ADX > adx_threshold
            * %K (Estocástico) > stoch_buy_threshold
            * close > SMA
      - VENTA cuando se cumplen todas las siguientes condiciones:
            * close < BB_Lower
            * MACD < MACD_signal
            * RSI < rsi_sell_threshold
            * ADX > adx_threshold
            * %K (Estocástico) < stoch_sell_threshold
            * close < SMA

    Parámetros:
        df (pd.DataFrame): Datos históricos filtrados.
        initial_capital (float): Capital inicial en USDT.
        fee (float): Tasa de comisión por operación.
        Los demás parámetros configuran los indicadores.

    Retorna:
        tuple: (capital_final, trades_df, df (con indicadores), equity_curve)
    """
    df = df.copy()

    # Calcular indicadores
    df = calculate_bollinger_bands(df, length=bb_length, multiplier=bb_multiplier)
    df = calculate_macd(df, fastperiod=macd_fast, slowperiod=macd_slow, signalperiod=macd_signal)
    df = calculate_rsi(df, period=rsi_period)
    df_adx = calculate_adx(df, period=adx_period)
    df['ADX'] = df_adx['ADX']
    df_stoch = calculate_stochastic(df, k_period=stoch_k_period, d_period=stoch_d_period)
    df['STOCH_K'] = df_stoch['%K']
    df['STOCH_D'] = df_stoch['%D']
    df = calculate_sma(df, period=sma_period)
    sma_col = f'SMA_{sma_period}'

    capital = initial_capital
    position = 0  # 0: sin posición; 1: posición abierta
    btc_amount = 0.0

    trades_log = []
    equity_curve = [initial_capital]

    # Determinar el índice inicial para asegurar que haya datos suficientes en cada indicador
    start_idx = max(bb_length, macd_slow + macd_signal, rsi_period, adx_period, stoch_k_period, sma_period)

    for i in range(start_idx, len(df)):
        row = df.iloc[i]
        timestamp = row['timestamp']
        close_price = row['close']

        # Extraer valores de los indicadores
        bb_upper = row['BB_Upper']
        bb_lower = row['BB_Lower']
        macd_val = row['MACD']
        macd_sig = row['MACD_signal']
        rsi_val = row['RSI']
        adx_val = row['ADX']
        stoch_k = row['STOCH_K']
        sma_val = row[sma_col]

        # Omitir iteración si hay valores nulos
        if any(pd.isna([bb_upper, bb_lower, macd_val, macd_sig, rsi_val, adx_val, stoch_k, sma_val])):
            continue

        # --- Condición de COMPRA ---
        if position == 0:
            if (close_price > bb_upper and
                macd_val > macd_sig and
                rsi_val > rsi_buy_threshold and
                adx_val > adx_threshold and
                stoch_k > stoch_buy_threshold and
                close_price > sma_val):

                capital_before = capital
                # Se aplica la comisión al invertir
                buy_cost = capital * (1 - fee)
                btc_amount = buy_cost / close_price
                capital = 0
                position = 1

                trades_log.append({
                    'timestamp': timestamp,
                    'type': 'BUY 🚀',
                    'price': close_price,
                    'btc_amount': btc_amount,
                    'capital_before': capital_before,
                    'capital_after': capital,
                    'fee_paid': capital_before - buy_cost,
                })
                print(f"{timestamp} - BUY 🚀 a {close_price:.2f} USDT | Capital invertido: {buy_cost:.2f} USDT")

        # --- Condición de VENTA ---
        elif position == 1:
            if (close_price < bb_lower and
                macd_val < macd_sig and
                rsi_val < rsi_sell_threshold and
                adx_val > adx_threshold and
                stoch_k < stoch_sell_threshold and
                close_price < sma_val):

                btc_before = btc_amount
                gross_sale = btc_amount * close_price
                fee_paid = gross_sale * fee
                net_sale = gross_sale - fee_paid

                trades_log.append({
                    'timestamp': timestamp,
                    'type': 'SELL 🔻',
                    'price': close_price,
                    'btc_amount': btc_before,
                    'capital_before': 0,
                    'capital_after': net_sale,
                    'fee_paid': fee_paid,
                })
                capital = net_sale
                position = 0
                btc_amount = 0
                equity_curve.append(capital)
                print(f"{timestamp} - SELL 🔻 a {close_price:.2f} USDT | Capital recuperado: {net_sale:.2f} USDT")

    # Cerrar posición final si aún está abierta
    if position == 1 and btc_amount > 0:
        last_close = df.iloc[-1]['close']
        fee_paid = (btc_amount * last_close) * fee
        capital = (btc_amount * last_close) - fee_paid
        trades_log.append({
            'timestamp': df.iloc[-1]['timestamp'],
            'type': 'SELL (Cierre) 🔻',
            'price': last_close,
            'btc_amount': btc_amount,
            'capital_before': 0,
            'capital_after': capital,
            'fee_paid': fee_paid,
        })
        equity_curve.append(capital)
        position = 0
        btc_amount = 0

    trades_df = pd.DataFrame(trades_log)
    return capital, trades_df, df, equity_curve


def compute_trade_stats(trades_df, initial_capital, final_capital):
    """
    Calcula estadísticas de las operaciones realizadas.

    Parámetros:
        trades_df (pd.DataFrame): Registro de trades.
        initial_capital (float): Capital inicial.
        final_capital (float): Capital final después del backtesting.

    Retorna:
        dict: Estadísticas que incluyen:
            - pnl: Ganancia/pérdida total.
            - total_trades: Número total de operaciones.
            - winning_trades: Número de operaciones ganadoras.
            - losing_trades: Número de operaciones perdedoras.
            - win_rate: Tasa de aciertos.
            - avg_win: Ganancia promedio.
            - avg_loss: Pérdida promedio.
    """
    pnl = final_capital - initial_capital
    total_trades = len(trades_df)

    if total_trades == 0:
        return {
            'pnl': pnl,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0
        }

    trade_pnls = []
    # Se asume que cada operación se compone de un par BUY y SELL consecutivos.
    for i in range(len(trades_df) - 1):
        if trades_df.iloc[i]['type'].startswith('BUY') and trades_df.iloc[i + 1]['type'].startswith('SELL'):
            buy_capital = trades_df.iloc[i]['capital_before']
            sell_capital = trades_df.iloc[i + 1]['capital_after']
            trade_pnls.append(sell_capital - buy_capital)

    winning_trades = sum(1 for pnl in trade_pnls if pnl > 0)
    losing_trades = sum(1 for pnl in trade_pnls if pnl < 0)
    avg_win = sum(pnl for pnl in trade_pnls if pnl > 0) / winning_trades if winning_trades > 0 else 0.0
    avg_loss = sum(pnl for pnl in trade_pnls if pnl < 0) / losing_trades if losing_trades > 0 else 0.0
    total_closed = winning_trades + losing_trades
    win_rate = winning_trades / total_closed if total_closed > 0 else 0.0

    return {
        'pnl': pnl,
        'total_trades': total_trades,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss
    }


def compute_max_drawdown(equity_curve):
    """
    Calcula el máximo drawdown dado el recorrido del capital (_equity curve_).

    Parámetros:
        equity_curve (list): Lista de valores del capital a lo largo del tiempo.

    Retorna:
        float: Máximo drawdown en porcentaje.
    """
    peak = equity_curve[0]
    max_dd = 0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = (peak - value) / peak
        if dd > max_dd:
            max_dd = dd
    return max_dd


def main():
    csv_file = 'dataset/btc_1h_data_2018_to_2025.csv'
    INITIAL_CAPITAL = 1000.0
    FEE = 0.001

    # Bollinger Bands
    BB_LENGTH = 37
    BB_MULTIPLIER = 2.25

    # MACD
    MACD_FAST = 17
    MACD_SLOW = 36
    MACD_SIGNAL = 6

    # RSI
    RSI_PERIOD = 21
    RSI_BUY_THRESHOLD = 47
    RSI_SELL_THRESHOLD = 46

    # ADX
    ADX_PERIOD = 9
    ADX_THRESHOLD = 40

    # Estocástico
    STOCH_K_PERIOD = 8
    STOCH_D_PERIOD = 4
    STOCH_BUY_THRESHOLD = 90
    STOCH_SELL_THRESHOLD = 13

    # SMA (Filtro de tendencia)
    SMA_PERIOD = 196

    # Cargar datos completos
    df_original = cargar_csv_btc(csv_file)
    year_list = range(2018, 2026)

    result_by_year = []
    global_pnl = 0.0
    global_total_trades = 0
    global_winning_trades = 0
    global_losing_trades = 0
    global_sum_wins = 0.0
    global_sum_losses = 0.0
    valid_years = 0  # Contador de años procesados

    for year in year_list:
        start_date = f'{year}-01-01'
        end_date = f'{year}-12-31'
        df_year = df_original[(df_original['timestamp'] >= start_date) & (df_original['timestamp'] <= end_date)]

        # Verificar que existan suficientes datos para calcular los indicadores
        min_required = max(BB_LENGTH, MACD_SLOW + MACD_SIGNAL, RSI_PERIOD, ADX_PERIOD, STOCH_K_PERIOD, SMA_PERIOD)
        if len(df_year) < min_required:
            print(f"❌ Año {year}: Sin datos suficientes (filas={len(df_year)}).")
            continue

        final_capital, trades_df, df_indicators, equity_curve = backtest_all_indicators(
            df_year,
            initial_capital=INITIAL_CAPITAL,
            fee=FEE,
            bb_length=BB_LENGTH,
            bb_multiplier=BB_MULTIPLIER,
            macd_fast=MACD_FAST,
            macd_slow=MACD_SLOW,
            macd_signal=MACD_SIGNAL,
            rsi_period=RSI_PERIOD,
            rsi_buy_threshold=RSI_BUY_THRESHOLD,
            rsi_sell_threshold=RSI_SELL_THRESHOLD,
            adx_period=ADX_PERIOD,
            adx_threshold=ADX_THRESHOLD,
            stoch_k_period=STOCH_K_PERIOD,
            stoch_d_period=STOCH_D_PERIOD,
            stoch_buy_threshold=STOCH_BUY_THRESHOLD,
            stoch_sell_threshold=STOCH_SELL_THRESHOLD,
            sma_period=SMA_PERIOD
        )

        stats = compute_trade_stats(trades_df, INITIAL_CAPITAL, final_capital)
        max_dd = compute_max_drawdown(equity_curve)

        # Calcular returns % para el año
        year_returns_percent = ((final_capital / INITIAL_CAPITAL) - 1) * 100

        year_result = {
            'year': year,
            'start_date': df_year['timestamp'].min(),
            'end_date': df_year['timestamp'].max(),
            'rows': len(df_year),
            'final_capital': final_capital,
            'pnl': stats['pnl'],
            'total_trades': stats['total_trades'],
            'winning_trades': stats['winning_trades'],
            'losing_trades': stats['losing_trades'],
            'win_rate': stats['win_rate'],
            'avg_win': stats['avg_win'],
            'avg_loss': stats['avg_loss'],
            'max_drawdown': max_dd,
            'returns_percent': year_returns_percent
        }
        result_by_year.append(year_result)

        global_pnl += stats['pnl']
        global_total_trades += stats['total_trades']
        global_winning_trades += stats['winning_trades']
        global_losing_trades += stats['losing_trades']
        if stats['winning_trades'] > 0:
            global_sum_wins += stats['avg_win'] * stats['winning_trades']
        if stats['losing_trades'] > 0:
            global_sum_losses += stats['avg_loss'] * stats['losing_trades']
        valid_years += 1

    # Reporte anual
    print("\n======== RESULTADOS POR AÑO ========")
    for r in result_by_year:
        print(f"📅 Año {r['year']}:")
        print(f"   📆 Rango: {r['start_date'].date()} -> {r['end_date'].date()}")
        print(f"   📊 Filas: {r['rows']}")
        print(f"   💰 Capital final: {r['final_capital']:.2f} USDT")
        print(f"   📈 PnL: {r['pnl']:.2f} USDT")
        print(f"   💹 Returns: {r['returns_percent']:.2f}%")
        print(f"   🔄 Operaciones: {r['total_trades']}")
        print(f"   ✅ Ganadoras: {r['winning_trades']} | ❌ Perdedoras: {r['losing_trades']}")
        print(f"   🎯 Win Rate: {r['win_rate']*100:.2f}%")
        print(f"   📈 Avg Win: {r['avg_win']:.2f} USDT | 📉 Avg Loss: {r['avg_loss']:.2f} USDT")
        print(f"   ⚠️ Max Drawdown: {r['max_drawdown']*100:.2f}%\n")

    # Reporte global
    if global_total_trades > 0 and valid_years > 0:
        total_closed = global_winning_trades + global_losing_trades
        global_win_rate = global_winning_trades / total_closed if total_closed > 0 else 0.0
        global_avg_win = global_sum_wins / global_winning_trades if global_winning_trades > 0 else 0.0
        global_avg_loss = global_sum_losses / global_losing_trades if global_losing_trades > 0 else 0.0

        # Se calcula el capital inicial global como el capital inicial por cada año procesado.
        global_initial_capital = valid_years * INITIAL_CAPITAL
        global_returns_percent = ((global_pnl + global_initial_capital) / global_initial_capital - 1) * 100

        print("======== RESUMEN GLOBAL (TODOS LOS AÑOS) ========")
        print(f"💵 PnL Acumulado: {global_pnl:.2f} USDT")
        print(f"🔄 Total Operaciones: {global_total_trades}")
        print(f"   ✅ Ganadoras: {global_winning_trades} | ❌ Perdedoras: {global_losing_trades}")
        print(f"🎯 Win Rate Global: {global_win_rate*100:.2f}%")
        print(f"   📈 Avg Win: {global_avg_win:.2f} USDT | 📉 Avg Loss: {global_avg_loss:.2f} USDT")
        print(f"💹 Returns Global: {global_returns_percent:.2f}%")
    else:
        print("======== RESUMEN GLOBAL ========")
        print("No hubo operaciones en el período analizado. 😔")


if __name__ == '__main__':
    main()
