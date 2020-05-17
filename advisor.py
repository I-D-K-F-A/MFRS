#Importing extensions
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF #Factorization method NMF
import random
from datetime import date
from time import gmtime, strftime
import calendar
import re #Regular expression
import telebot
from telebot import types


#Telebot token
bot = telebot.TeleBot("")

def processing():

    global answer;
    global answer2;
    url = 'http://data-storage.kremppi.com/receipt_detailed.csv'


    #Importing dataset
    food = pd.read_csv('receipt_detailed.csv', index_col=0, sep=";")

    #Grouping objects for day of week and matrix transposing
    sorted_list = food.groupby(['Name']+['DOW'], as_index=True)
    work_frame = sorted_list.size().unstack(level=1)
    work_frame.fillna(0, inplace=True) #Replacing missed values to 0

    #Matrix factorization
    nmf = NMF(3)
    nmf.fit(work_frame)

    hmatrix = pd.DataFrame(np.round(nmf.components_,2), columns=work_frame.columns)
    wmatrix = pd.DataFrame(np.round(nmf.transform(work_frame),2), columns=hmatrix.index)
    wmatrix.index = work_frame.index
    blank_data = []

    factorized = pd.DataFrame(np.round(np.dot(wmatrix,hmatrix),2), columns=work_frame.columns)
    factorized.index = work_frame.index

    for i in range(0, factorized.shape[0]):
        r = factorized.loc[factorized.index[i]]
        rd = np.interp(r, (r.min(), r.max()), (0, +1))
        blank_data.append(rd)

    normalized = pd.DataFrame(np.round(blank_data,2), index=factorized.index, columns=factorized.columns)

    #Creating a median prices set
    median_prices_list = pd.DataFrame(food.groupby(['Name'], as_index=False)['Price'].median())
    median_prices_list.to_csv('median_prices.csv', header=True, index = None, encoding = 'utf-8', sep='&')
    pr = pd.read_csv('median_prices.csv', index_col=0, sep='&')
    normalized['Price'] = pr.index

    reg = re.compile('[^a-zA-Z0-9.? ]')
    count = 0
    answer = ''
    answer2 = ''

    for i in range (0, 10):
        name = random.choices(normalized.index, weights=normalized[calendar.day_name[date.today().weekday()]])
        pr = round(float(normalized.loc[(reg.sub('', str(name)))].Price), 2)
        answer += reg.sub('', str(name)) + ' ' + str(pr) + '€' + "\n"
        count += pr

    answer2 = 'Total: ' + str(round(count, 2)) + '€'


@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.send_message(message.chat.id, str(str(date.today()) + ' ' + str(calendar.day_name[date.today().weekday()]) + ' ' + str(strftime("%H:%M"))), reply_markup=keyboard())

@bot.message_handler(content_types=['text'])
def send_back(message):
    if message.text == 'Predict':
        bot.send_message(message.chat.id, 'Creating recommendation for ' + str(calendar.day_name[date.today().weekday()]))
        processing()
        bot.send_message(message.chat.id, answer)
        bot.send_message(message.chat.id, answer2)
    else:
        bot.send_message(message.chat.id, 'Select command on screen keyboard, or enter command "Predict".')
def keyboard():
	markup = types.ReplyKeyboardMarkup(one_time_keyboard=False, resize_keyboard=True)
	btn1 = types.KeyboardButton('Predict')
	markup.add(btn1)
	return markup

bot.infinity_polling(True)
