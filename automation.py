import imaplib
import email
from email.header import decode_header
import joblib
import json
import time

# Load configuration
def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

# Load trained model artifacts
model = joblib.load("model/classifier.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")

def get_email_body(msg):
    """Extracts plain text body from raw email message."""
    if msg.is_multipart():
        for part in msg.walk():
            content_type = part.get_content_type()
            content_disposition = str(part.get("Content-Disposition"))
            if content_type == "text/plain" and "attachment" not in content_disposition:
                return part.get_payload(decode=True).decode(errors="ignore")
    else:
        return msg.get_payload(decode=True).decode(errors="ignore")
    return ""

def process_inbox():
    config = load_config()
    
    # Connect to IMAP Server
    mail = imaplib.IMAP4_SSL(config["imap_server"])
    mail.login(config["email_user"], config["email_password"])
    mail.select("INBOX")

    # Search for unread emails
    status, response = mail.search(None, 'UNSEEN')
    email_ids = response[0].split()

    if not email_ids:
        print("No unread emails found.")
        mail.logout()
        return

    print(f"Found {len(email_ids)} unread email(s). Processing...")

    for e_id in email_ids:
        _, msg_data = mail.fetch(e_id, "(RFC822)")
        for response_part in msg_data:
            if isinstance(response_part, tuple):
                msg = email.message_from_bytes(response_part[1])
                
                # Extract Subject
                subject, encoding = decode_header(msg["Subject"])[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding or "utf-8", errors="ignore")
                
                # Extract Body
                body = get_email_body(msg)
                full_text = f"{subject} {body}"

                # Vectorize and Predict
                features = vectorizer.transform([full_text])
                prediction = model.predict(features)[0]
                confidence = max(model.predict_proba(features)[0]) * 100

                print(f"\n[Subject]: {subject[:50]}...")
                if prediction == 1:
                    print(f"--> Result: SPAM ({confidence:.1f}% confidence)")
                    # Move to Spam folder
                    mail.copy(e_id, config["spam_folder"])
                    mail.store(e_id, '+FLAGS', '\\Deleted')
                else:
                    print(f"--> Result: HAM ({confidence:.1f}% confidence)")

    mail.expunge()
    mail.logout()

if __name__ == "__main__":
    # Runs the check every 60 seconds
    print("Starting automated email spam monitoring service...")
    while True:
        try:
            process_inbox()
        except Exception as e:
            print(f"Error encountered: {e}")
        time.sleep(60)