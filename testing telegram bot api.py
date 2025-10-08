import requests

# Replace with your own Telegram bot token and chat ID
TELEGRAM_BOT_TOKEN = '7152256963:AAHV12D7CKjxBUntuqUNdeZsmq8lG8zqFzg'
TELEGRAM_CHAT_ID = '6139287790'

def send_telegram_message(message: str):
    """
    Send a Telegram message using the bot.
    """
    url = f"https://api.telegram.org/bot7152256963:AAHV12D7CKjxBUntuqUNdeZsmq8lG8zqFzg/sendMessage"
    payload = {
        'chat_id': TELEGRAM_CHAT_ID,
        'text': message,
        'parse_mode': 'HTML'
    }
    try:
        response = requests.post(url, data=payload)
        if response.status_code == 200:
            print("📲 Telegram message sent successfully.")
        else:
            print(f"❌ Failed to send Telegram message: {response.text}")
    except Exception as e:
        print(f"❌ Exception sending Telegram message: {e}")
send_telegram_message("Hello from the Telegram bot!")



