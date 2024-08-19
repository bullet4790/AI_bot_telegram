import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from telegram import Update
from telegram.ext import Application, MessageHandler, filters, CallbackContext

# Определение устройства (GPU или CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Загрузка модели и токенизатора от Sberbank на основе GPT-2
model_name_or_path = "sberbank-ai/rugpt3large_based_on_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path)
model = GPT2LMHeadModel.from_pretrained(model_name_or_path).to(device)

# Асинхронная функция для генерации текста на основе пользовательского ввода
async def generator(user_input, max_len=50, accuracy=1):
    # Кодируем ввод пользователя в тензоры
    input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)
    
    # Настройка параметров для лучевой поисковой генерации (beam search)
    beams = [1, 5, 10, 15]
    
    # Генерация текста моделью с настройкой параметров для контроля качества
    out = model.generate(
        input_ids,
        max_length=max_len,            # Максимальная длина генерируемого текста
        num_beams=beams[accuracy],     # Количество лучей в beam search (настраиваемая точность)
        no_repeat_ngram_size=3,        # Запрет на повторение одинаковых фраз
        repetition_penalty=2.0,        # Штраф за повторение фраз
        temperature=0.7,               # Параметр для регулирования разнообразия
        num_return_sequences=1,        # Количество возвращаемых сгенерированных последовательностей
        early_stopping=True,           # Раннее завершение, когда достигнут конец текста
        eos_token_id=tokenizer.eos_token_id  # Идентификатор конца предложения
    )
    
    # Перенос сгенерированных данных на CPU для последующего декодирования
    generated_text = tokenizer.decode(out[0].cpu(), skip_special_tokens=False)
    
    # Обрезка текста до логичного завершения на основе специального символа
    stop = '<s>'
    if stop in generated_text:
        generated_text = generated_text.split(stop, 1)[0].strip()
    
    return generated_text

# Токен вашего Telegram-бота (его нужно заменить на реальный токен)
TELEGRAM_TOKEN = 'YOUR_BOT:TOKEN'

# Асинхронная функция для обработки входящих сообщений в Telegram
async def handle_message(update: Update, context: CallbackContext):
    user_input = update.message.text  # Получаем текст сообщения от пользователя
    response = await generator(user_input, max_len=50, accuracy=1)  # Генерируем ответ с помощью модели
    await update.message.reply_text(response)  # Отправляем сгенерированный ответ обратно пользователю

if __name__ == '__main__':
    # Инициализация Telegram-бота с использованием токена
    application = Application.builder().token(TELEGRAM_TOKEN).build()
    
    # Добавление обработчика для текстовых сообщений
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    
    # Запуск процесса поллинга для прослушивания новых сообщений
    application.run_polling()
