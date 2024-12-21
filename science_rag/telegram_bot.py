import logging
from telebot import (
    TeleBot,
    types,
)

from single_rag_agent import Agent
from config import BOT_TOKEN

logging.basicConfig(level=logging.INFO)

bot = TeleBot(BOT_TOKEN)
llm_agent = Agent()


@bot.message_handler(content_types=['text'])
def handle_user_request(message: types.Message):
    """Passes user's request to LLM"""
    try:
        llm_response = llm_agent.send_request(message.text)
        bot.send_message(chat_id=message.chat.id, text=llm_response)
    except Exception as e:
        logging.error(f"Error handling user request: {e}")
        bot.send_message(chat_id=message.chat.id, text="Sorry, something went wrong. Please try again later.")


def main():
    logging.info("Starting bot...")
    bot.infinity_polling(skip_pending=True, allowed_updates=['message'])


if __name__ == "__main__":
    main()
