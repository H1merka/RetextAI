"""Модуль для работы с языковой моделью."""
from transformers import T5ForConditionalGeneration, T5Tokenizer
"""Модуль для обновления состояния бота."""
from telegram import Update
"""Модуль для корректной работы в мессенджере."""
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes
)
"""Модуль для запуска синхронного кода в асинхронном."""
import nest_asyncio
"""Модуль для запуска асинхронного кода."""
import asyncio
"""Модуль для определения языка текста."""
from langdetect import detect, DetectorFactory

nest_asyncio.apply()

DetectorFactory.seed = 0
# Подключение языковой модели.
MODEL_NAME = 'cointegrated/rut5-base-paraphraser'
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

model.eval()


def paraphrase(text, sequences, beams=15, grams=4, do_sample=True):
    """Функция для перефразирования текста.

    :param text: Текст для перефразирования.
    :type text: str
    :param sequences: Количество вариантов перефразирования.
    :type sequences: int
    :param beams: Количество лучей для поиска лучшего результата.
    :type beams: int
    :param grams: Размер n-грамм для предотвращения повторений.
    :type grams: int
    :param do_sample: Флаг, включающий случайность в выбор следующего слова.
    :type do_sample: bool
    """
    x = tokenizer(text, return_tensors='pt', padding=True).to(model.device)
    max_size = int(x.input_ids.shape[1] * 1.5 + 10)
    out = model.generate(**x, encoder_no_repeat_ngram_size=grams,
                         num_beams=beams, max_length=max_size,
                         num_return_sequences=sequences,
                         do_sample=do_sample)
    return tokenizer.batch_decode(out, skip_special_tokens=True)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка начального сообщения от пользователя в боте."""
    context.user_data.clear()
    context.user_data['state'] = 'waiting_for_text'
    await update.message.reply_text(
        "Привет! Мы RetextAI - инструмент для\n"
        "перефразирования текстов. Здесь мы\n"
        "поможем Вам сделать тексты ярче\n"
        "и уникальнее, сохраняя их смысл, с\n"
        "помощью искусственного интеллекта.\n"
        "Чтобы начать работу, просто отправьте\n"
        "нам текст, который хотите обработать!\n\n"
        "Команды:\n"
        "/help - Инструкция по использованию"
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка команды help в боте."""
    await update.message.reply_text(
        "Инструкция:\n"
        "1. Отправьте текст, который вы хотите перефразировать.\n"
        "2. Укажите количество вариантов перефразирования (до 15).\n"
        "3. Я верну вам несколько вариантов перефразирования.\n"
        "4. Если результат вас не устраивает, я попробую снова.\n\n"
        "Вы можете отправить текст длиной до 500 символов."
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработка входящих сообщений от пользователя в боте."""
    state = context.user_data.get('state', 'waiting_for_text')
    user_input = update.message.text

    if state == 'waiting_for_text':
        # Проверка длины текста.
        if len(user_input) > 500:
            await update.message.reply_text(
             "Текст слишком длинный. "
             "Пожалуйста, отправьте текст до 500 символов."
            )
            return

        # Проверка языка.
        try:
            detected_language = detect(user_input)
            if detected_language != 'ru':
                await update.message.reply_text(
                 "Пожалуйста, отправьте текст на русском языке. "
                 "Мы принимаем только русский язык. "
                 "Пожалуйста, введите еще раз на русском языке."
                )
                return
        except Exception as e:
            await update.message.reply_text(f"Произошла ошибка при определении языка: {e}")
            return

        # Сохранение текста и переход к состоянию запроса количества вариантов.
        context.user_data['text_to_paraphrase'] = user_input
        context.user_data['state'] = 'waiting_for_options'
        await update.message.reply_text(
         "Сколько вариантов перефразирования вы хотите? (до 15)"
        )

    elif state == 'waiting_for_options':
        # Проверка, что ввод является целым числом.
        try:
            sequences = int(user_input)
            if sequences < 1 or sequences > 15:
                await update.message.reply_text(
                 "Пожалуйста, введите число от 1 до 15."
                )
                return

            context.user_data['sequences'] = sequences
            context.user_data['state'] = 'waiting_for_feedback'

            # Перефразирование текста.
            await update.message.reply_text("Обрабатываю текст...")
            text_to_paraphrase = context.user_data['text_to_paraphrase']
            paraphrased_texts = paraphrase(text_to_paraphrase, sequences=sequences)

            response = "\n\n".join(paraphrased_texts)
            await update.message.reply_text(f"Вот варианты перефразирования:\n{response}")

            await update.message.reply_text(
             "Вас устраивает результат? (Да/Нет)"
            )
            context.user_data['paraphrased_texts'] = paraphrased_texts
        except ValueError:
            await update.message.reply_text(
             "Пожалуйста, введите число от 1 до 15."
            )

    elif state == 'waiting_for_feedback':
        feedback = user_input.lower()
        if feedback == "да":
            await update.message.reply_text(
             "Отлично! Рад, что вам понравилось. Если хотите, отправьте новый текст."
            )
            context.user_data.clear()
            context.user_data['state'] = 'waiting_for_text'
        elif feedback == "нет":
            await update.message.reply_text("Попробую снова...")
            text_to_paraphrase = context.user_data['text_to_paraphrase']
            sequences = context.user_data['sequences']
            paraphrased_texts = paraphrase(text_to_paraphrase, sequences=sequences)

            response = "\n\n".join(paraphrased_texts)
            await update.message.reply_text(f"Вот новые варианты перефразирования:\n{response}")
            await update.message.reply_text(
             "Вас устраивает результат? (Да/Нет)"
            )
        else:
            await update.message.reply_text(
             "Пожалуйста, ответьте 'Да' или 'Нет'."
            )


async def main():
    """Основная функция."""
    TOKEN = "7299791761:AAFg8tlzs__2fIT5ZJs6SPcMtcUGbCcaoA0"

    application = Application.builder().token(TOKEN).build()

    # Добавление команд и обработчиков сообщений.
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запуск бота
    await application.run_polling()

if __name__ == '__main__':
    asyncio.run(main()) 
