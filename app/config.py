"""所有配置都从环境变量读，密钥不写进代码。"""

import os
from dataclasses import dataclass, field


def _env(name: str, default: str = ""):
    return field(default_factory=lambda: os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # 大模型：任何兼容 OpenAI 接口的服务都能用，默认 DeepSeek。换模型只改这三个变量。
    llm_api_key: str = _env("LLM_API_KEY", "")
    llm_base_url: str = _env("LLM_BASE_URL", "https://api.deepseek.com")
    llm_model: str = _env("LLM_MODEL", "deepseek-chat")
    # 评测时当裁判的模型，默认和上面同一个
    judge_model: str = _env("JUDGE_MODEL", "")
    tavily_api_key: str = _env("TAVILY_API_KEY", "")
    database_url: str = _env("DATABASE_URL", "sqlite:///data/invest_agent.db")
    timezone: str = _env("TIMEZONE", "Australia/Sydney")

    @property
    def demo_mode(self) -> bool:
        """没填模型密钥，或显式打开 DEMO_MODE 时，用假模型和假数据跑，方便别人不花钱就能看演示。"""
        return os.getenv("DEMO_MODE", "").lower() in ("1", "true", "yes") or not self.llm_api_key


def get_settings() -> Settings:
    return Settings()
