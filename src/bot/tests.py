import telebot

TOKEN = '6484128996:AAGO1ggOxUxOVG5gy5uZRnzf5pHEKfCGZ9k'

bot = telebot.TeleBot(TOKEN)







# триггерим по командам
@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Введите текст для классификации")


# триггерим по регулярному выражению
@bot.message_handler(regexp="By|Goodby|See you later")
def send_by(message):
    bot.reply_to(message, "By!")


is_hello = lambda x: x.lower() in ['привет', 'хай']


# триггерим по бинарной функции
@bot.message_handler(func=is_hello)
def send_welcome(message):
    bot.reply_to(message, "Hi!")


bot.polling()
