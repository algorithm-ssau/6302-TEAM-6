import os
import re
import json
import logging
import tempfile
import requests
import assemblyai as aai
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

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")
aai.settings.api_key = ASSEMBLYAI_API_KEY

API_RETRY_ATTEMPTS = 15
MESSAGE_LIMIT = 4096

def split_message(text, limit=MESSAGE_LIMIT):
    return [text[i:i + limit] for i in range(0, len(text), limit)]

def escape_markdown_v2(text: str) -> str:
    """Экранирует специальные символы для MarkdownV2."""
    escape_chars = r'_[]()~>#+-=|{}.!'
    return re.sub(r'([%s])' % re.escape(escape_chars), r'\\\1', text)

async def transcribe_audio(file_path, context, chat_id):
    """Транскрибирует аудиофайл с использованием AssemblyAI."""
    try:
        config = aai.TranscriptionConfig(language_code="ru")
        transcript = aai.Transcriber(config=config).transcribe(file_path)
        if transcript.status == "error":
            await context.bot.send_message(
                chat_id,
                "Истёк ключ API для транскрипции аудио."
                "Попробуйте сделать запрос позже. Мы уже занимаемся этой проблемой."
            )
            return None
        text = transcript.text.strip()
    except Exception as e:
        await context.bot.send_message(chat_id, f"Неизвестная ошибка при транскрипции аудио:  {str(e)}."
                                                f"Попробуйте ещё раз.")
        return None
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
        messages.append({"role": "user", "content": "Распознанный текст из аудио пользователя: " + transcript})
        payload = {
            "model": model,
            "messages": messages
        }
        headers = self.HEADERS.copy()

        attempts = 0
        while attempts < API_RETRY_ATTEMPTS:
            attempts += 1
            logging.info("summarize_text(): обращение к API модели %s | попытка %d/%d",
                         model, attempts, API_RETRY_ATTEMPTS)
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
                except Exception as e:
                    logging.error("summarize_text(): API вернул статус код %s, исключение: %s",
                                  response.status_code, str(e))
                continue

            data = response.json()
            if "error" in data:
                logging.error("summarize_text(): API вернул ошибку: %s | Запрос был отправлен для модели: %s",
                              data["error"], model)
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

        self.pending_audio = {}
        self.awaiting_context = {}

        self.selected_model = {}
        self.use_context = {}
        self.chat_history = {}

        self.setup_handlers()

    def get_main_keyboard(self, chat_id):
        """
        Формирует клавиатуру для выбора модели и режима диалога.
        """
        context_button = "Отключить контекст" if self.use_context.get(chat_id, False) else "Режим диалога"
        keyboard = [
            [KeyboardButton("⚙️ Выбрать модель")],
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
        reply_markup = ReplyKeyboardMarkup(
            self.get_main_keyboard(chat_id),
            resize_keyboard=True, one_time_keyboard=False
        )
        await update.message.reply_text(
            "👋 Привет! Отправь голосовое сообщение или аудиофайл, и я переведу его в структурированный текст.\n"
            "🎯 После загрузки аудио можно ввести уточняющий контекст, чтобы достичь лучшего результата.",
            reply_markup=reply_markup
        )

    async def handle_voice_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка голосовых сообщений и аудиофайлов."""
        chat_id = update.effective_chat.id

        file_id = None
        file_type = None
        if update.message.voice:
            file_id = update.message.voice.file_id
            file_type = "voice"
        elif update.message.audio:
            file_id = update.message.audio.file_id
            file_type = "audio"
        else:
            await update.message.reply_text("Неподдерживаемый формат файла. Я принимаю голосовые сообщения, ogg и mp3.")
            return

        try:
            await context.bot.send_message(chat_id, "Скачиваю файл...")
            new_file = await context.bot.get_file(file_id)
        except Exception as e:
            if "File is too big" in str(e):
                await update.message.reply_text(
                    "Слишком большой размер файла. Пожалуйста, отправьте файл до 20 МБ."
                )
                return
            else:
                await update.message.reply_text("Ошибка получения файла. Попробуйте ещё раз.")
                return

        suffix = ".ogg" if file_type == "voice" else ".mp3"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
            file_path = tf.name
            await new_file.download_to_drive(file_path)

        self.pending_audio[chat_id] = file_path

        buttons = [
            [InlineKeyboardButton("Уточнить", callback_data="ask_context_yes"),
             InlineKeyboardButton("Не нужно", callback_data="ask_context_no")]
        ]
        reply_markup = InlineKeyboardMarkup(buttons)
        await update.message.reply_text(
            "Нужно ли добавить уточняющий контекст для аудио?",
            reply_markup=reply_markup
        )

    async def context_button_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка нажатия кнопок «Уточнить» / «Не нужно»."""
        query = update.callback_query
        chat_id = query.message.chat_id
        await query.answer()
        if query.data == "ask_context_yes":
            self.awaiting_context[chat_id] = True
            await query.edit_message_text("💬 Пожалуйста, введите уточняющий контекст."
                                          "\nНапример, "
                                          "\"Это требования заказчика к новому проекту о бронировании авиабилетов\".")
        elif query.data == "ask_context_no":
            self.awaiting_context[chat_id] = False
            await query.edit_message_text("🎧 Транскрибирую аудио...")
            await self.process_summarization(chat_id, additional_context=None, context=context)

    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка текстовых сообщений."""
        chat_id = update.effective_chat.id
        text = update.message.text.strip()

        if chat_id in self.awaiting_context and self.awaiting_context[chat_id] is True:
            additional_context = text
            self.awaiting_context[chat_id] = additional_context
            await update.message.reply_text("Транскрибирую аудио...")
            await self.process_summarization(chat_id, additional_context=additional_context, context=context)

        elif self.use_context.get(chat_id, False):
            await self.process_clarification(chat_id, question=text, context=context)
        else:
            await update.message.reply_text(
                "Я ожидаю голосовое сообщение или аудиофайл. Если хотите изменить настройки, используйте кнопки меню."
            )

    async def process_summarization(self, chat_id, additional_context, context: ContextTypes.DEFAULT_TYPE):
        """Проводит транскрипцию, формирует запрос и отправляет его в нейросеть."""
        file_path = self.pending_audio.pop(chat_id, None)
        if not file_path or not os.path.exists(file_path):
            await context.bot.send_message(chat_id, "Ошибка: аудиофайл не найден.")
            return

        try:
            transcript = await transcribe_audio(file_path, context, chat_id)
        except Exception as e:
            await context.bot.send_message(chat_id, "Ошибка обработки аудиофайла. Попробуйте позже.")
            return
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

        if transcript is None:
            return

        selected_model = self.selected_model.get(chat_id, APIClient.DEFAULT_MODEL)
        await context.bot.send_message(chat_id, "Отправляю запрос в языковую модель...")

        system_prompt = (
            "Ты — эксперт по анализу и структурированию информации. "
            "Твоя основная задача — внимательно проанализировать предоставленный текст, выделить его суть "
            "и важные детали, удаляя избыточные сведения. "
            "Помни, что текст получен из аудио, поэтому возможны ошибки в распознавании. "
            "Если по контексту можно однозначно восстановить или уточнить неточные фрагменты, сделай это "
            "для повышения качества анализа, но не указывай пользователю, какие именно слова были исправлены или "
            "восстановлены. Стремись точно передать содержимое текста, не добавляя информации, которой не было в "
            "исходном материале, и полагайся исключительно на предоставленный текст. "
            "Начни ответ с краткой суммаризации \"Кратко о тексте\", а затем переходи к более подробному "
            "пересказу, сохранив все важные детали. "
        )
        if additional_context:
            system_prompt += "\nДополнительный контекст: " + additional_context

        history = self.chat_history.get(chat_id) if self.use_context.get(chat_id, False) else None
        reasoning, content = self.api_client.summarize_text(transcript, system_prompt,
                                                            model=selected_model, history=history)
        if not content or content == "Server is busy right now":
            await context.bot.send_message(chat_id,
                                           "К сожалению, сервер языковой модели плохо себя ведёт. Попробуйте позже.")
        else:
            if self.use_context.get(chat_id, False):
                if chat_id not in self.chat_history:
                    self.chat_history[chat_id] = []
                self.chat_history[chat_id].append({"role": "user", "content": transcript})
                self.chat_history[chat_id].append({"role": "assistant", "content": content})
            for part in split_message(content):
                await context.bot.send_message(chat_id, escape_markdown_v2(part), parse_mode=ParseMode.MARKDOWN_V2)

    async def process_clarification(self, chat_id, question: str, context: ContextTypes.DEFAULT_TYPE):
        """Обрабатывает уточняющие вопросы в режиме диалога, используя историю переписки."""
        history = self.chat_history.get(chat_id, [])
        clarification_prompt = "Пользователь уточняет: " + question
        selected_model = self.selected_model.get(chat_id, APIClient.DEFAULT_MODEL)
        await context.bot.send_message(chat_id, "Отправляю уточняющий запрос в модель...")
        reasoning, content = self.api_client.summarize_text(
            clarification_prompt,
            system_prompt="Учти историю диалога и ответь на уточняющий вопрос.",
            model=selected_model,
            history=history
        )
        if not content or content == "Server is busy right now":
            await context.bot.send_message(chat_id,
                                           "К сожалению, сервер языковой модели плохо себя ведёт. Попробуйте позже.")
        else:
            self.chat_history[chat_id].append({"role": "user", "content": clarification_prompt})
            self.chat_history[chat_id].append({"role": "assistant", "content": content})
            for part in split_message(content):
                await context.bot.send_message(chat_id, escape_markdown_v2(part), parse_mode=ParseMode.MARKDOWN_V2)

    async def choose_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка запроса выбора модели."""
        keyboard = [
            [KeyboardButton("⚡ DeepSeek V3 685B"), KeyboardButton("DeepSeek R1")],
            [KeyboardButton("Gemini 2.5 Pro"), KeyboardButton("Qwen3 235B")],
            [KeyboardButton("Llama 4 Maverick"), KeyboardButton("Gemma 3 27B")],
        ]
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
        await update.message.reply_text("Выберите модель:", reply_markup=reply_markup)

    async def model_selection_handler(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Обработка выбора модели из меню."""
        chat_id = update.effective_chat.id
        user_text = update.message.text.strip()
        if user_text == "DeepSeek R1":
            self.selected_model[chat_id] = "deepseek/deepseek-r1:free"
        elif user_text == "Gemini 2.5 Pro":
            self.selected_model[chat_id] = "google/gemini-2.5-pro-exp-03-25:free"
        elif user_text == "Qwen3 235B":
            self.selected_model[chat_id] = "qwen/qwen3-235b-a22b:free"
        elif user_text == "⚡ DeepSeek V3 685B":
            self.selected_model[chat_id] = "deepseek/deepseek-chat-v3-0324:free"
        elif user_text == "Llama 4 Maverick":
            self.selected_model[chat_id] = "meta-llama/llama-4-maverick:free"
        elif user_text == "Gemma 3 27B":
            self.selected_model[chat_id] = "google/gemma-3-27b-it:free"
        reply_markup = ReplyKeyboardMarkup(self.get_main_keyboard(chat_id),
                                           resize_keyboard=True, one_time_keyboard=False)
        await update.message.reply_text(f"Выбрана модель {user_text}", reply_markup=reply_markup)

    async def set_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Переключение режима диалога."""
        chat_id = update.effective_chat.id
        current = self.use_context.get(chat_id, False)
        self.use_context[chat_id] = not current
        if not self.use_context[chat_id]:
            self.chat_history.pop(chat_id, None)
            response = "Режим диалога отключён. ❌ История очищена."
        else:
            self.chat_history[chat_id] = []
            response = "Режим диалога включён. ✅ Теперь вы можете задавать уточняющие вопросы."
        reply_markup = ReplyKeyboardMarkup(self.get_main_keyboard(chat_id),
                                           resize_keyboard=True, one_time_keyboard=False)
        await update.message.reply_text(response, reply_markup=reply_markup)

    def setup_handlers(self):
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, self.handle_voice_audio))
        self.app.add_handler(CallbackQueryHandler(self.context_button_handler))

        self.app.add_handler(MessageHandler(
            filters.Regex(r"^(DeepSeek R1|Gemini 2\.5 Pro|Qwen3 235B|⚡️ "
                          r"DeepSeek V3 685B|Llama 4 Maverick|Gemma 3 27B|Отмена)$"),
            self.model_selection_handler))
        self.app.add_handler(MessageHandler(filters.Regex(r"^(Режим диалога|Отключить контекст)$"), self.set_mode))
        self.app.add_handler(MessageHandler(filters.Regex(r"^Выбрать модель$"), self.choose_model))

        self.app.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.handle_text))

    def run(self):
        self.app.run_polling()

def main():
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO
    )
    bot = TelegramBot(TELEGRAM_TOKEN)
    bot.run()

if __name__ == '__main__':
    main()