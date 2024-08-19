import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, CallbackContext

# Загрузка модели и токенизатора
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(device)

async def generator(user_input, max_len=50, accuracy=1):
    input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)
    beams = [1, 5, 10, 15]
    out = model.generate(
        input_ids,
        max_length=max_len,
        num_beams=beams[accuracy],
        no_repeat_ngram_size=3,
        repetition_penalty=2.0,
        temperature=0.7,
        num_return_sequences=1,
        early_stopping=True,
        eos_token_id=tokenizer.eos_token_id
    )
    # Переносим сгенерированные данные на CPU для декодирования
    generated_text = tokenizer.decode(out[0].cpu(), skip_special_tokens=False)
    # Обрезаем полученный ответ для логичного завершения предложения
    stop = '<s>'
    if stop in generated_text:
        generated_text = generated_text.split(stop, 1)[0].strip()
    return generated_text

# Токен вашего Telegram-бота
TELEGRAM_TOKEN = '7342298864:AAGrtsYsapl6a6WMs3B27SgKKrgLItMPoog'

# Асинхронная функция для обработки сообщений Telegram
async def handle_message(update: Update, context: CallbackContext):
    user_input = update.message.text
    response = await generator(user_input, max_len=50, accuracy=1)
    await update.message.reply_text(response)

if __name__ == '__main__':
    # Инициализация Telegram-бота
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Обработчик сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Запуск Telegram-бота
    application.run_polling()
