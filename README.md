# Trading Bot - MACD, RSI, ADX, Estocástico, Bollinger Bands y SMA. 🚀
## Estrategia con con Indicadores Técnicos Clásicos

Repositorio para backtesting y ejecución de un bot de trading en BTC, utilizando múltiples indicadores (MACD, RSI, ADX, Estocástico, Bollinger Bands y SMA).

---

## 📁 Contenido

- **dataset/**:  
  📊 Datos históricos de BTC en distintos timeframes (1D, 1H, 4H y 15min) (2018–2025).
  
- **backtest.py**:  
  🧪 Código para realizar el backtesting de la estrategia.
- **bot.py**:  
  🤖 Implementación del bot de trading.

---

## 📊 Información del Dataset

Cada archivo CSV contiene los siguientes campos:

- **Open time**: ⏰ Fecha y hora de inicio del período (vela).
- **Open**: 💲 Precio de apertura.
- **High**: 📈 Precio más alto alcanzado durante el período.
- **Low**: 📉 Precio más bajo alcanzado durante el período.
- **Close**: 🔒 Precio de cierre.
- **Volume**: 📊 Volumen de la criptomoneda operada.
- **Close time**: ⏲ Hora de cierre del período.
- **Quote asset volume**: 💰 Volumen total en la moneda cotizada.
- **Number of trades**: 🔢 Número de transacciones realizadas.
- **Taker buy base asset volume**: 📥 Volumen del activo base comprado por los 'takers'.
- **Taker buy quote asset volume**: 📤 Volumen en la moneda cotizada correspondiente a las compras de 'takers'.
- **Ignore**: 🚫 Columna reservada (sin uso relevante).

---

## ⚙️ Requerimientos

- **Python 3.7+**
- **Librerías de Python:**
  - [pandas](https://pandas.pydata.org/)
  - [numpy](https://numpy.org/)

Puedes instalarlas ejecutando:
```bash
pip install pandas numpy
