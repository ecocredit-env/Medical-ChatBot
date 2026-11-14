"""Chat with the bot in the terminal:  python cli.py"""
from medical_chatbot import ChatBot, Session
from medical_chatbot.safety import EMERGENCY_NUMBER


def main():
    bot, session = ChatBot(), Session()
    print("Medical ChatBot. General health information only, not a doctor.")
    print(f"In an emergency call {EMERGENCY_NUMBER}. Type 'quit' to exit.\n")
    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.lower() in {"quit", "exit", "q"}:
            break
        reply = bot.reply(message, session)
        print(f"\nBot: {reply.text}\n")
    print("Take care.")


if __name__ == "__main__":
    main()
