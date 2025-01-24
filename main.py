import requests
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import logging
import numpy as np
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor

# Настраиваем логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Класс для анализа данных криптовалют и генерации отчетов"""
    
    def __init__(self):
        self.api_url = "https://api.coingecko.com/api/v3"
        self.cached_data = {}
        self.prediction_days = 7  # Прогноз на неделю вперед

    def get_crypto_data(self, coin_id: str = "bitcoin") -> Optional[Dict]:
        """Получает данные о криптовалюте через API"""
        if coin_id in self.cached_data:
            return self.cached_data[coin_id]
            
        try:
            endpoint = f"{self.api_url}/coins/{coin_id}/market_chart"
            params = {
                "vs_currency": "usd",
                "days": "30",
                "interval": "daily"
            }
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            self.cached_data[coin_id] = response.json()
            return self.cached_data[coin_id]
        except requests.RequestException as e:
            logger.error(f"Ошибка при получении данных: {e}")
            return None

    def predict_future_prices(self, prices: List[float]) -> List[float]:
        """Предсказывает будущие цены используя линейную регрессию"""
        X = np.array(range(len(prices))).reshape(-1, 1)
        y = np.array(prices)
        
        model = LinearRegression()
        model.fit(X, y)
        
        future_dates = np.array(range(len(prices), len(prices) + self.prediction_days)).reshape(-1, 1)
        predictions = model.predict(future_dates)
        
        return predictions.tolist()

    def analyze_price_trends(self, data: Dict) -> Dict:
        """Анализирует тренды цен с расширенной аналитикой"""
        prices = [price[1] for price in data.get("prices", [])]
        if not prices:
            return {}
        
        # Рассчитываем дополнительные метрики
        price_changes = np.diff(prices)
        positive_days = sum(1 for change in price_changes if change > 0)
        
        return {
            "average_price": round(sum(prices) / len(prices), 2),
            "max_price": round(max(prices), 2),
            "min_price": round(min(prices), 2),
            "volatility": round((max(prices) - min(prices)) / min(prices) * 100, 2),
            "positive_days_percent": round(positive_days / len(price_changes) * 100, 2),
            "price_momentum": round(sum(price_changes[-5:]), 2)  # Momentum за последние 5 дней
        }

    def generate_report(self, coin_id: str = "bitcoin") -> None:
        """Генерирует расширенный визуальный отчет"""
        data = self.get_crypto_data(coin_id)
        if not data:
            logger.error("Не удалось получить данные для анализа")
            return

        # Анализ данных
        analysis = self.analyze_price_trends(data)
        
        # Создаем DataFrame
        df = pd.DataFrame(data["prices"], columns=["date", "price"])
        df["date"] = pd.to_datetime(df["date"], unit="ms")
        
        # Получаем предсказания
        predictions = self.predict_future_prices(df["price"].tolist())
        future_dates = pd.date_range(start=df["date"].iloc[-1], periods=len(predictions) + 1, closed="right")
        
        # Создаем график с двумя подграфиками
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
        
        # Верхний график - цены и предсказания
        ax1.plot(df["date"], df["price"], label=f"{coin_id.capitalize()} Price")
        ax1.plot(future_dates, predictions, '--', label="Прогноз", color='red')
        ax1.set_title(f"Анализ и прогноз цены {coin_id.capitalize()}")
        ax1.set_xlabel("Дата")
        ax1.set_ylabel("Цена (USD)")
        ax1.grid(True)
        ax1.legend()

        # Нижний график - объемы торгов
        volumes = [vol[1] for vol in data.get("total_volumes", [])]
        ax2.bar(df["date"], volumes, alpha=0.5, label="Объем торгов")
        ax2.set_title("Объем торгов")
        ax2.set_xlabel("Дата")
        ax2.set_ylabel("Объем (USD)")
        ax2.grid(True)
        ax2.legend()

        # Добавляем расширенную информацию
        info_text = (
            f"Средняя цена: ${analysis['average_price']}\n"
            f"Макс. цена: ${analysis['max_price']}\n"
            f"Мин. цена: ${analysis['min_price']}\n"
            f"Волатильность: {analysis['volatility']}%\n"
            f"% позитивных дней: {analysis['positive_days_percent']}%\n"
            f"Momentum (5d): ${analysis['price_momentum']}"
        )
        plt.figtext(0.02, 0.02, info_text, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(f"{coin_id}_analysis.png", dpi=300, bbox_inches='tight')
        logger.info(f"Расширенный анализ сохранен как {coin_id}_analysis.png")

def analyze_multiple_coins(coins: List[str]):
    """Параллельный анализ нескольких монет"""
    analyzer = DataAnalyzer()
    with ThreadPoolExecutor(max_workers=3) as executor:
        executor.map(analyzer.generate_report, coins)

def main():
    """Основная функция с расширенной демонстрацией"""
    # Анализируем топ-монеты параллельно
    coins = ["bitcoin", "ethereum", "dogecoin", "solana", "cardano"]
    analyze_multiple_coins(coins)
    
    logger.info("Анализ завершен! Проверьте сгенерированные отчеты.")

if __name__ == "__main__":
    main()
