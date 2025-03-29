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

class TelegramBot:
    """
    Telegram-бот для транскрипции аудио и выдачи структурированного содержания,
    с поддержкой уточняющих вопросов в режиме диалога.
    """
    def __init__(self, token: str):
        self.token = token
        self.api_client = APIClient()
        self.app = Application.builder().token(self.token).build()

        # pending_audio хранит путь к wav-файлу для каждого чата (для транскрипции после уточнения контекста)
        self.pending_audio = {}
        # awaiting_context: если бот ожидает ввод уточняющего контекста для аудио.
        # Если значение True – ждем текст от пользователя.
        self.awaiting_context = {}

        # Состояния для выбора модели и режима диалога
        self.selected_model = {}  # {chat_id: model_name}
        self.use_context = {}     # {chat_id: bool}
        self.chat_history = {}    # {chat_id: список сообщений}

        self.setup_handlers()

    def get_main_keyboard(self, chat_id):
        """
        Формирует клавиатуру для выбора модели и режима диалога.
        """
        context_button = "Отключить контекст" if self.use_context.get(chat_id, False) else "Режим диалога"
        keyboard = [
            [KeyboardButton("Выбрать модель")],
            [KeyboardButton(context_button)],
            [KeyboardButton("/start")]
        ]
        return keyboard

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        chat_id = update.effective_chat.id
        if chat_id not in self.selected_model:
            self.selected_model[chat_id] = APIClient.DEFAULT_MODEL
        if chat_id not in self.use_context:
            self.use_context[chat_id] = False
        if self.use_context.get(chat_id) and chat_id not in self.chat_history:
            self.chat_history[chat_id] = []
        reply_markup = ReplyKeyboardMarkup(self.get_main_keyboard(chat_id), resize_keyboard=True, one_time_keyboard=False)
        await update.message.reply_text(
            "Привет! Отправь голосовое сообщение или аудиофайл, и я переведу его в структурированный текст.\n"
            "После загрузки аудио можно ввести уточняющий контекст, чтобы достичь лучшего результата.",
            reply_markup=reply_markup
        )

    async def handle_voice_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка голосовых сообщений и аудиофайлов."""
        chat_id = update.effective_chat.id

        # Определяем тип файла
        file_id = None
        file_type = None
        if update.message.voice:
            file_id = update.message.voice.file_id
            file_type = "voice"
        elif update.message.audio:
            file_id = update.message.audio.file_id
            file_type = "audio"
        else:
            await update.message.reply_text("Неподдерживаемый формат файла.")
            return

        # Скачиваем файл во временное хранилище
        try:
            await context.bot.send_message(chat_id, "Скачиваю файл...")
            new_file = await context.bot.get_file(file_id)
        except Exception as e:
            if "File is too big" in str(e):
                await update.message.reply_text(
                    "Telegram накладывает ограничение на размер файла. Пожалуйста, отправьте файл до 20 МБ."
                )
                return
            else:
                await update.message.reply_text("Ошибка при получении файла.")
                return

        suffix = ".ogg" if file_type == "voice" else ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            file_path = tf.name
            await new_file.download_to_drive(file_path)

        # Конвертируем в wav
        try:
            if file_type == "voice":
                audio = AudioSegment.from_ogg(file_path)
            else:
                audio = AudioSegment.from_file(file_path)
        except Exception as e:
            await update.message.reply_text(f"Ошибка конвертации файла: {e}")
            os.remove(file_path)
            return

        wav_path = file_path + ".wav"
        audio.export(wav_path, format="wav")
        os.remove(file_path)

        # Сохраняем путь к аудиофайлу для последующей транскрипции
        self.pending_audio[chat_id] = wav_path

        # Спрашиваем, нужен ли уточняющий контекст
        buttons = [
            [InlineKeyboardButton("Уточнить", callback_data="ask_context_yes"),
             InlineKeyboardButton("Не нужно", callback_data="ask_context_no")]
        ]
        reply_markup = InlineKeyboardMarkup(buttons)
        await update.message.reply_text(
            "Нужно ли добавить уточняющий контекст для аудио?",
            reply_markup=reply_markup
        )
