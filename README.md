# 🤖 Transcriber & Summarizer Bot

### [Ссылка на бота в Telegram >](https://t.me/SsauAudioBot)

Телеграм-бот, который преобразует голосовые сообщения и аудиофайлы в текст, а затем с помощью языковой модели структурирует и выделяет основную информацию.

![Showcase](./images/showcase.gif)

## Особенности проекта

- 🎤 **Транскрипция аудио:**<br>Использование `Assembly AI` для преобразования голосовых сообщений и аудиофайлов в текст.
- 📑 **Структурирование информации:**<br>Применение API для анализа и выделения сути аудио с помощью языковых моделей.
- 💬 **Работа с контекстом:**<br>Поддержка уточняющих вопросов и дополнительного контекста для повышения точности результата.

## 📌 Как пользоваться

### Отправка аудио
Бот принимает голосовые сообщения (в том числе пересланные из другого диалога), а также аудиофайлы.<br>
Просто отправьте их в чат.

### Уточнение контекста
После получения аудио бот предложит добавить контекст.<br>
Это поможет языковой модели понять предметную область.<br>
- Нажимаем кнопку `Уточнить` и вводим контекст, например `Это требования заказчика к мобильному приложению о системе бронирования авиабилетов`.<br>
- Или `Не нужно`, если аудио не требует контекста для понимания.

### Смена модели
В боте доступно 6 языковых моделей:
- `DeepSeek V3 685B` **Рекомендуем.**<br>Оптимальная скорость и точность.
- `DeepSeek R1` **Поддерживает размышление.**<br>Дольше анализирует, но результат лучше.
- `Gemini 2.5 Pro`
- `Qwen3 235B`
- `Llama 4 Maverick`
- `Gemma 3 27B`

Чтобы выбрать модель, нажмите соответствующую кнопку в главном меню.

### Режим диалога
Если вам требуется загрузить несколько аудио или вы хотите задать вопросы по распознанному тексту, нажмите кнопку `Режим диалога`.<br>
В этом режиме бот будет сохранять транскрипции и ваши вопросы, позволяя вести полноценный диалог с языковой моделью. 

## Используемые технологии

- **Язык программирования:** Python
- **Библиотеки и инструменты:**
  - [python-dotenv](https://github.com/theskumar/python-dotenv) – для работы с переменными окружения
  - [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot) – для создания Telegram-бота
  - [AssemblyAI](https://www.assemblyai.com/) – для транскрипции аудио
  - [OpenRouter](https://openrouter.ai/) – для работы с языковыми моделями
  - [requests](https://docs.python-requests.org/) – для отправки HTTP-запросов к API


## ⚙️ Установка и запуск своего бота

### 1. Склонируйте репозиторий

```bash
git clone https://github.com/algorithm-ssau/6302-TEAM-6.git
```

### 2. Установите библиотеки

Для запуска бота потребуется установить несколько библиотек.<br>
Для этого запустите команду в терминале:
```bash
pip install -r requirements.txt
```

### 3. Получите токен бота

- Перейдите в [@BotFather](https://t.me/botfather) в Telegram
- Создайте нового бота: `/newbot` -> Введите `Публичное имя бота` -> Введите `Внутреннее имя бота` (будет использоваться в ссылке)
- Скопируйте токен:
```bash
Use this token to access the HTTP API:
123456789:abcdefghijk # здесь будет настоящий токен
Keep your token secure and store it safely, it can be used by anyone to control your bot.
```

### 4. Получите ключ OpenRouter
Для запросов к языковым моделям используется API OpenRouter.
- Перейдите на сайт [openrouter.ai](https://openrouter.ai/) и зарегистрируйтесь.
- В меню профиля выберите `Keys` -> `Create Key` -> Введите `Имя для ключа`, поле `Credit limit` оставьте пустым.
- Скопируйте ключ.

### 5. Получите ключ AssemblyAI
Для танскрипции аудио используется API AssemblyAI.
- Перейдите на сайт [assemblyai.com](https://www.assemblyai.com/) и зарегистрируйтесь.
- В меню выберите `API Keys`.
- Скопируйте ключ.

### 6. Настройте окружение
Создайте в корне проекта файл `.env` и заполните его:
```python
TELEGRAM_TOKEN=[токен, полученный на шаге №3]
OPENROUTER_API_KEY=[ключ, полученный на шаге №4]
ASSEMBLYAI_API_KEY=[ключ, полученный на шаге №5]
```

### 7. Запустите бота

## Разработчики
**Team 6 – 6302 – 2025** | Студенты группы 6302-020302D
- Я.В. Аграномов  / [Nu11Object](https://github.com/Nu11Object)
- Б.Г. Григорьева / [C1eem](https://github.com/C1eem)
- М.О. Долгова    / [MariaD04](https://github.com/MariaD04)
- Е.В. Мамонтов   / [JijaEdem666](https://github.com/JijaEdem666)
- Н.В. Пишков     / [LastHope777](https://github.com/LastHope777)
