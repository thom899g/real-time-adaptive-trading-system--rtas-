"""
RTAS Configuration Manager
Centralized configuration with validation and environment-aware settings
"""
import os
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional
from enum import Enum
from dotenv import load_dotenv

load_dotenv()

class TradingMode(Enum):
    """Trading system operational modes"""
    BACKTEST = "backtest"
    PAPER = "paper_trading"
    LIVE = "live_trading"


class RiskProfile(Enum):
    """Risk management profiles"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


@dataclass
class DatabaseConfig:
    """Firebase configuration with validation"""
    project_id: str
    private_key_id: str
    private_key: str
    client_email: str
    client_id: str
    database_url: str
    
    @classmethod
    def from_env(cls) -> 'DatabaseConfig':
        """Initialize from environment variables with validation"""
        required_vars = [
            'FIREBASE_PROJECT_ID',
            'FIREBASE_PRIVATE_KEY_ID',
            'FIREBASE_PRIVATE_KEY',
            'FIREBASE_CLIENT_EMAIL',
            'FIREBASE_CLIENT_ID',
            'FIREBASE_DATABASE_URL'
        ]
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing Firebase environment variables: {missing}")
        
        # Handle escaped newlines in private key
        private_key = os.getenv('FIREBASE_PRIVATE_KEY', '').replace('\\n', '\n')
        
        return cls(
            project_id=os.getenv('FIREBASE_PROJECT_ID', ''),
            private_key_id=os.getenv('FIREBASE_PRIVATE_KEY_ID', ''),
            private_key=private_key,
            client_email=os.getenv('FIREBASE_CLIENT_EMAIL', ''),
            client_id=os.getenv('FIREBASE_CLIENT_ID', ''),
            database_url=os.getenv('FIREBASE_DATABASE_URL', '')
        )


@dataclass
class ExchangeConfig:
    """Exchange API configuration"""
    name: str
    api_key: str
    secret: str
    sandbox: bool = True
    
    @classmethod
    def from_env(cls, exchange_name: str) -> 'ExchangeConfig':
        """Initialize exchange config from environment"""
        prefix = f"{exchange_name.upper()}_"
        return cls(
            name=exchange_name,
            api_key=os.getenv(f"{prefix}API_KEY", ""),
            secret=os.getenv(f"{prefix}SECRET", ""),
            sandbox=os.getenv(f"{prefix}SANDBOX", "true").lower() == "true"
        )


@dataclass
class ModelConfig:
    """Neural network and RL model configuration"""
    # Neural Network
    lstm_units: int = 128
    dense_units: int = 64
    dropout_rate: float = 0.2
    sequence_length: int = 60
    
    # Reinforcement Learning
    learning_rate: float = 0.001
    discount_factor: float = 0.95
    exploration_rate: float = 0.1
    replay_buffer_size: int = 10000
    batch_size: int = 32
    
    # Training
    train_interval: int = 100  # Number of trades between retraining
    validation_split: float = 0.2


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_position_size: float = 0.1  # Max 10% of portfolio per trade
    max_daily_loss: float = 0.02  # Max 2% daily loss
    stop_loss_pct: float = 0.02  # 2% stop loss
    take_profit_pct: float = 0.05  # 5% take profit
    max_open_positions: int = 5
    cooldown_period: int = 300  # 5 minutes between same asset trades


class RTASConfig:
    """Main configuration manager for RTAS"""
    
    def __init__(self, mode: TradingMode = TradingMode.PAPER):
        self.mode = mode
        self.database = DatabaseConfig.from_env()
        self.exchange = ExchangeConfig.from_env("binance")  # Default exchange
        self.model = ModelConfig()
        self.risk = RiskConfig()
        self.log_level = logging.INFO if mode == TradingMode.LIVE else logging.DEBUG
        
        # Trading parameters
        self.symbols = ["BTC/USDT", "ETH/USDT", "ADA/USDT"]
        self.update_interval = 5  # Seconds between data updates
        self.max_retries = 3
        self.retry_delay = 1  # Seconds
        
    def validate(self) -> bool:
        """Validate all configuration parameters"""
        try:
            # Validate database config
            assert self.database.project_id, "Firebase project ID required"
            assert self.database.private_key, "Firebase private key required"
            
            # Validate exchange config in live mode
            if self.mode == TradingMode.LIVE:
                assert self.exchange.api_key, "Exchange API key required for live trading"
                assert self.exchange.secret, "Exchange secret required for live trading"
            
            # Validate risk parameters
            assert 0 < self.risk.max_position_size <= 1, "Position size must be