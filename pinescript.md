//@version=5
strategy("Estrategia Combinada (BB, MACD, RSI, ADX, Stoch y SMA) Spot", overlay=true, initial_capital=1000, default_qty_type=strategy.percent_of_equity, default_qty_value=100, commission_type=strategy.commission.percent, commission_value=0.1)

// ======== Parámetros de indicadores ========
bb_length       = input.int(20, "BB Length", minval=1)
bb_multiplier   = input.float(2.0, "BB Multiplier", step=0.1)
macd_fast       = input.int(12, "MACD Fast", minval=1)
macd_slow       = input.int(26, "MACD Slow", minval=1)
macd_signal     = input.int(9, "MACD Signal", minval=1)
rsi_period      = input.int(14, "RSI Period", minval=1)
rsi_buy_thr     = input.int(50, "RSI Buy Threshold", minval=1, maxval=100)
rsi_sell_thr    = input.int(50, "RSI Sell Threshold", minval=1, maxval=100)
adx_period      = input.int(14, "ADX Period", minval=1)
adx_threshold   = input.float(25.0, "ADX Threshold", step=0.1)
stoch_k_period  = input.int(14, "Stoch %K Period", minval=1)
stoch_d_period  = input.int(3,  "Stoch %D Period", minval=1)
stoch_buy_thr   = input.int(80, "Stoch Buy Threshold", minval=1, maxval=100)
stoch_sell_thr  = input.int(20, "Stoch Sell Threshold", minval=1, maxval=100)
sma_period      = input.int(200, "SMA Period", minval=1)

// =================== CÁLCULOS DE INDICADORES ===================
// Bollinger Bands
basis    = ta.sma(close, bb_length)
dev      = bb_multiplier * ta.stdev(close, bb_length)
bb_upper = basis + dev
bb_lower = basis - dev

// MACD
[macdLine, signalLine, _] = ta.macd(close, macd_fast, macd_slow, macd_signal)

// RSI
rsiValue = ta.rsi(close, rsi_period)

// ADX - CÁLCULO MANUAL
up       = high - high[1]
down     = low[1] - low
plusDM   = (up > down and up > 0) ? up : 0
minusDM  = (down > up and down > 0) ? down : 0
tr       = ta.tr(true)
atr      = ta.rma(tr, adx_period)
plusDI   = atr != 0 ? 100 * ta.rma(plusDM, adx_period) / atr : 0
minusDI  = atr != 0 ? 100 * ta.rma(minusDM, adx_period) / atr : 0
dx       = (plusDI + minusDI != 0) ? 100 * math.abs(plusDI - minusDI) / (plusDI + minusDI) : 0
adxValue = ta.rma(dx, adx_period)

// Estocástico (usamos %K)
stochK   = ta.stoch(close, high, low, stoch_k_period)

// SMA
smaValue = ta.sma(close, sma_period)

// =================== CONDICIONES DE ENTRADA Y SALIDA ===================
// Condición de BUY (entrada LONG)
buyCondition = close > bb_upper and macdLine > signalLine and rsiValue > rsi_buy_thr and adxValue > adx_threshold and stochK > stoch_buy_thr and close > smaValue

// Condición de SELL (salida normal)
sellCondition = close < bb_lower and macdLine < signalLine and rsiValue < rsi_sell_thr and adxValue > adx_threshold and stochK < stoch_sell_thr and close < smaValue

// =================== ORDENES Y GESTIÓN DE POSICIÓN ===================
// Entrada en posición LONG
if not strategy.position_size and buyCondition
    strategy.entry("Long", strategy.long)

// Salida normal cuando se cumplen las condiciones de SELL
if strategy.position_size > 0 and sellCondition
    strategy.close("Long", comment="Sell")

// =================== PLOTEOS ===================
plot(bb_upper, color=color.new(color.green, 0), title="BB Upper")
plot(basis,    color=color.new(color.gray, 0),  title="BB Basis")
plot(bb_lower, color=color.new(color.red, 0),   title="BB Lower")
plot(smaValue, color=color.new(color.blue, 0), title="SMA")

// Señales de compra y venta en el gráfico
plotshape(buyCondition and not strategy.position_size, title="Buy Signal", location=location.belowbar, color=color.green, style=shape.labelup, text="BUY", textcolor=color.white, size=size.normal)
plotshape(sellCondition and strategy.position_size > 0, title="Sell Signal", location=location.abovebar, color=color.red, style=shape.labeldown, text="SELL", textcolor=color.white, size=size.normal)

// =================== ALERTAS (Opcional) ===================
if buyCondition
    alert("Condición de COMPRA activada", alert.freq_once_per_bar)
if sellCondition
    alert("Condición de VENTA activada", alert.freq_once_per_bar)
