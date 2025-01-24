import requests
import json
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
import logging

# Настраиваем логирование
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataAnalyzer:
    """Класс для анализа данных криптовалют и генерации отчетов"""
    
    def __init__(self):
        self.api_url = "https://api.coingecko.com/api/v3"
        self.cached_data = {}

    def get_crypto_data(self, coin_id: str = "bitcoin") -> Optional[Dict]:
        """Получает данные о криптовалюте через API"""
        try:
            endpoint = f"{self.api_url}/coins/{coin_id}/market_chart"
            params = {
                "vs_currency": "usd",
                "days": "30",
                "interval": "daily"
            }
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Ошибка при получении данных: {e}")
            return None

    def analyze_price_trends(self, data: Dict) -> Dict:
        """Анализирует тренды цен"""
        prices = [price[1] for price in data.get("prices", [])]
        if not prices:
            return {}
        
        return {
            "average_price": round(sum(prices) / len(prices), 2),
            "max_price": round(max(prices), 2),
            "min_price": round(min(prices), 2),
            "volatility": round((max(prices) - min(prices)) / min(prices) * 100, 2)
        }

    def generate_report(self, coin_id: str = "bitcoin") -> None:
        """Генерирует визуальный отчет"""
        data = self.get_crypto_data(coin_id)
        if not data:
            logger.error("Не удалось получить данные для анализа")
            return

        # Анализ данных
        analysis = self.analyze_price_trends(data)
        
        # Создаем DataFrame для визуализации
        df = pd.DataFrame(data["prices"], columns=["date", "price"])
        df["date"] = pd.to_datetime(df["date"], unit="ms")
        
        # Создаем график
        plt.figure(figsize=(12, 6))
        plt.plot(df["date"], df["price"], label=f"{coin_id.capitalize()} Price")
        plt.title(f"Анализ цены {coin_id.capitalize()} за последние 30 дней")
        plt.xlabel("Дата")
        plt.ylabel("Цена (USD)")
        plt.grid(True)
        plt.legend()
        
        # Добавляем текстовую информацию на график
        info_text = (
            f"Средняя цена: ${analysis['average_price']}\n"
            f"Макс. цена: ${analysis['max_price']}\n"
            f"Мин. цена: ${analysis['min_price']}\n"
            f"Волатильность: {analysis['volatility']}%"
        )
        plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                bbox=dict(facecolor='white', alpha=0.8),
                verticalalignment='top')
        
        # Сохраняем график
        plt.savefig(f"{coin_id}_analysis.png")
        logger.info(f"График сохранен как {coin_id}_analysis.png")

class CryptoPortfolio:
    """Класс для управления криптопортфелем"""
    
    def __init__(self):
        self.holdings: Dict[str, float] = {}
        self.transactions: List[Dict] = []

    def add_transaction(self, coin: str, amount: float, price: float) -> None:
        """Добавляет новую транзакцию"""
        transaction = {
            "coin": coin,
            "amount": amount,
            "price": price,
            "date": datetime.now().isoformat()
        }
        self.transactions.append(transaction)
        
        if coin in self.holdings:
            self.holdings[coin] += amount
        else:
            self.holdings[coin] = amount
        
        logger.info(f"Добавлена транзакция: {transaction}")

    def get_portfolio_summary(self) -> Dict:
        """Возвращает сводку по портфелю"""
        return {
            "holdings": self.holdings,
            "total_transactions": len(self.transactions)
        }

def main():
    """Основная функция для демонстрации возможностей"""
    analyzer = DataAnalyzer()
    portfolio = CryptoPortfolio()

    # Демонстрация анализа нескольких криптовалют
    coins = ["bitcoin", "ethereum", "dogecoin"]
    for coin in coins:
        analyzer.generate_report(coin)
        
        # Добавляем демо-транзакции
        portfolio.add_transaction(coin, 0.1, 1000.0)

    # Выводим сводку по портфелю
    summary = portfolio.get_portfolio_summary()
    logger.info(f"Сводка по портфелю: {json.dumps(summary, indent=2)}")

if __name__ == "__main__":
    main()
