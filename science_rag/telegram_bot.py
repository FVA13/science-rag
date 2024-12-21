import logging
import threading
from telebot import (
    TeleBot,
    types,
)
from telebot.handler_backends import (
    StatesGroup,
    State,
)
from telebot.storage import StateMemoryStorage

from single_rag_agent import Agent
from config import BOT_TOKEN

# # for test
# class Agent:
#     def reset(self):
#         print('Reset')
#
#     def revoke(self, msg):
#         return "very useful information"


logging.basicConfig(level=logging.INFO)

state_storage = StateMemoryStorage()
bot = TeleBot(BOT_TOKEN, state_storage=state_storage)
llm_agent = Agent()
user_activity = {}


class ModelStates(StatesGroup):
    model_provider = State()
    model_name = State()


def reset_llm_agent(user_id):
    llm_agent.reset()
    logging.info(f"LLM agent reset for user {user_id}")


def schedule_reset(user_id, delay=300):
    if user_id in user_activity:
        user_activity[user_id].cancel()
    timer = threading.Timer(delay, reset_llm_agent, [user_id])
    user_activity[user_id] = timer
    timer.start()


@bot.message_handler(commands=['set_model_provider'])
def handle_model_provider(message: types.Message):
    bot.set_state(
        user_id=message.from_user.id,
        chat_id=message.chat.id,
        state=ModelStates.model_provider,
    )

    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    markup.add('Google', 'Anthropic', 'Ollama', 'OpenAI')

    bot.send_message(
        chat_id=message.chat.id,
        text="What's your preferable LLM provider?",
        reply_markup=markup,
    )

    schedule_reset(message.from_user.id)


@bot.message_handler(state=ModelStates.model_provider)
def handle_model_provider_selection(message: types.Message):
    selected_provider = message.text
    llm_agent.llm_provider = selected_provider
    bot.send_message(
        chat_id=message.chat.id,
        text=f"You have selected {selected_provider} as your LLM provider.",
        reply_markup=types.ReplyKeyboardRemove(),
    )
    bot.delete_state(user_id=message.from_user.id, chat_id=message.chat.id)

    schedule_reset(message.from_user.id)


@bot.message_handler(content_types=['text'])
def handle_user_request(message: types.Message):
    """Passes user's request to LLM"""
    if message.text.lower() == "end chat":
        reset_llm_agent(message.from_user.id)
        bot.send_message(chat_id=message.chat.id, text="Chat session ended. LLM agent has been reset.")
        return

    try:
        llm_response = llm_agent.revoke(message.text)
        bot.send_message(chat_id=message.chat.id, text=llm_response)
    except Exception as e:
        logging.error(f"Error handling user request: {e}")
        bot.send_message(chat_id=message.chat.id, text="Sorry, something went wrong. Please try again later.")

    schedule_reset(message.from_user.id)


def main():
    logging.info("Starting bot...")
    bot.enable_saving_states()
    bot.infinity_polling(skip_pending=True, allowed_updates=['message'])


if __name__ == "__main__":
    main()
