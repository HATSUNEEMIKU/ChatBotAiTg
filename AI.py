from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from pyrogram import Client, filters


blacklisted_ids = [1234]
# blacklisted_ids - Список айди пользователей, на которых бот не будет реагировать. 
#                   Указываем и свой айди, т.к если ты что-то будешь писать в избранном 
#                   бот так же будет реагировать на твои сообщения


owner = 6100723987 # вставляем айди юзербота



device = "cpu" # если что меняем на gpu (видеокарта), если ошибка, меняйте на cpu 

path_to_model = r"C:\Users\Miku\Desktop\hghj"
# path_to_model - тут указываем путь к твоей модели. У меня внутри папки был такой расклад:

#                   config.json
#                   generation_config.json
#                   pytorch_model.bin
#                   special_tokens_map.json
#                   spiece.model
#                   tokenizer_config.json

# Возможно, это кому - то будет важно


tokenizer = AutoTokenizer.from_pretrained(path_to_model, device=device)
model = AutoModelForSeq2SeqLM.from_pretrained(path_to_model).to(device)

def paraphrase(question, # из handle_private_message передаем сюда сообщение юзера
               
               num_beams=5,  # Количество лучей для широкого поиска

               num_beam_groups=5,  # Количество групп лучей для группового поиска

               num_return_sequences=1,  # Количество возвращаемых последовательностей

               repetition_penalty=10.0,  # Штраф за повторения

               diversity_penalty=0.1,  # Штраф за разнообразие

               no_repeat_ngram_size=1,  # Максимальный размер n-грамм, который не допускается повторять

               temperature=1,  # Параметр температуры для влияния на вероятности генерации

               max_length=512):  # Максимальная длина последовательности
    """
    Кто особо ничего не понял в параметрах, я объясню самое интересное:
    --------
    temperature = число - это то, насколько разнообразно будет модель генерировать ответы. 
    Чем больше - тем разнообразней он будет генерировать овтеты, но больше 1 не ставим.
    Если , когда ты решил поставить temperature = 3 выскочила ошибка, напиши вот так: temperature = 2.0
    -------
    max_length = число - это то, насколько большой текст может сгенерировать ответ ИИ. Но, учти,
    что это лишь максимальное количество символов, и бот может сгенерировать так и большой текст на 512 символов
    так и на 128 символов, зависит от модели скорее 

    """
    
    input_ids = tokenizer( # тут по сути ничего не трогаем
        f'Ответ: {question}',
        return_tensors="pt", padding="longest",
        max_length=max_length,
        truncation=True,
    ).input_ids.to(device)

    outputs = model.generate( # тут по сути ничего не трогаем
        input_ids, 
        temperature=temperature, 
        repetition_penalty=repetition_penalty,
        num_return_sequences=num_return_sequences, 
        no_repeat_ngram_size=no_repeat_ngram_size,
        num_beams=num_beams, 
        num_beam_groups=num_beam_groups,
        max_length=max_length, 
        diversity_penalty=diversity_penalty,
    )

    res = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return res

# Создаем клиента Pyrogram
api_id = '9860905'
api_hash = '94f5043cba54370fb8d4f501166188ae'
app = Client("my_account", api_id, api_hash)
with app:
    app.send_message('me', 'hello')

from datetime import datetime

# Флаг включения/выключения бота
bot_enabled = False

# Список пользователей, которым было отправлено уведомление
notification_sent_users = []

# Функция для обработки команды включения/выключения
@app.on_message(filters.command(["on", "off"]) & filters.user(owner))
def toggle_bot_status(client, message):
    global bot_enabled
    global notification_sent_users
    command = message.command[0]

    if command == "on":
        if not bot_enabled:
            notification_sent_users = []
            bot_enabled = True
            message.reply("Бот включен.")
    elif command == "off":
        bot_enabled = False
        message.reply("Бот выключен.")

# Функция-обработчик для личных сообщений
@app.on_message(filters.private & ~filters.user(blacklisted_ids))
def handle_private_message(client, message):
    if not bot_enabled:
        return

    user_id = message.from_user.id
    user_name = message.from_user.first_name
    question = message.text
    response = paraphrase(question)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"Время: {current_time}\nОт: {user_name} (user_id: {user_id})\nЗапрос: {question}\nОтвет: {response[0]}\n"

    if user_id not in notification_sent_users:
        client.send_message(user_id, "Владелец отошел, отвечает ИИ.")
        notification_sent_users.append(user_id)

    # Отправка сообщения в лог-канал
    app.send_message(-1001887474595, log_message)

    # Отправка ответа пользователю
    client.send_message(user_id, f"Привет, {user_name}! Вот что я могу сказать на ваш запрос: {response[0]}")


# Запуск клиента Pyrogram
app.run()



# модель была у меня T5 RUS , в простое , в бездействии, она не жрала оперативу и не нагружала ПК. 
#   Когда же кто-то писал мне в лс, модель работала , нагружала ПК, но не надолго. 