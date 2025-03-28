import os
import re
import json
import logging
import tempfile
import requests
from pydub import AudioSegment
from dotenv import load_dotenv
from telegram import Update, ReplyKeyboardMarkup, KeyboardButton, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters,
)
import torch
import whisper

# Настройка ffmpeg
os.environ["PATH"] += os.pathsep + r"C:\ffmpeg\bin"
AudioSegment.converter = r"C:\ffmpeg\bin\ffmpeg.exe"
AudioSegment.ffprobe = r"C:\ffmpeg\bin\ffprobe.exe"

# Загрузка переменных окружения
load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Константы
API_RETRY_ATTEMPTS = 10
MESSAGE_LIMIT = 4096

# Определяем устройство для whisper: GPU если доступно, иначе CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("medium", device=device)

def split_message(text, limit=MESSAGE_LIMIT):
    return [text[i:i + limit] for i in range(0, len(text), limit)]

def escape_markdown_v2(text: str) -> str:
    """Экранирует специальные символы для MarkdownV2."""
    escape_chars = r'_[]()~`>#+-=|{}.!'
    return re.sub(r'([%s])' % re.escape(escape_chars), r'\\\1', text)

async def transcribe_audio(file_path):
    """Транскрибирует аудиофайл с использованием модели Whisper."""
    try:
        result = model.transcribe(file_path, language="ru")
        text = result.get("text", "").strip()
    except Exception as e:
        text = f"Ошибка при транскрипции: {e}"
    return text

class APIClient:
    """
    Класс для работы с API нейросети.
    """
    API_URL = "https://openrouter.ai/api/v1/chat/completions"
    DEFAULT_MODEL = "deepseek/deepseek-chat-v3-0324:free"
    HEADERS = {
        "Authorization": OPENROUTER_API_KEY,
        "Content-Type": "application/json"
    }

    def summarize_text(self, transcript: str, system_prompt: str, model: str = None, history: list = None):
        if model is None:
            model = self.DEFAULT_MODEL

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": transcript})
        payload = {
            "model": model,
            "messages": messages
        }
        headers = self.HEADERS.copy()

        attempts = 0
        while attempts < API_RETRY_ATTEMPTS:
            attempts += 1
            logging.info("summarize_text(): обращение к API модели %s | попытка %d/%d", model, attempts, API_RETRY_ATTEMPTS)
            try:
                response = requests.post(
                    url=self.API_URL,
                    headers=headers,
                    data=json.dumps(payload)
                )
            except Exception as e:
                logging.exception("summarize_text(): Exception при обращении к API: %s", str(e))
                continue

            if response.status_code != 200:
                try:
                    data = response.json()
                    logging.error("summarize_text(): API вернул ошибку: %s", data.get("error"))
                except Exception:
                    logging.error("summarize_text(): API вернул статус код %s", response.status_code)
                continue

            data = response.json()
            if "error" in data:
                logging.error("summarize_text(): API вернул ошибку: %s", data["error"])
                continue

            message = data["choices"][0]["message"]
            content = message.get("content", "")
            if not content:
                logging.info("summarize_text(): пустой content, повтор запроса")
                continue

            reasoning = message.get("reasoning", "")
            return reasoning, content

        return "Server is busy right now", ""