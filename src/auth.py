import random
import smtplib
from email.message import EmailMessage
from datetime import datetime, timedelta

import streamlit as st

from src.database import get_or_create_user


def generate_otp():
    return str(random.randint(100000, 999999))


def send_email_otp(email):
    otp = generate_otp()

    st.session_state["otp"] = otp
    st.session_state["otp_email"] = email
    st.session_state["otp_expires"] = (
        datetime.now() + timedelta(minutes=5)
    )

    sender = st.secrets["EMAIL_ADDRESS"]
    password = st.secrets["EMAIL_PASSWORD"]

    message = EmailMessage()
    message["Subject"] = "Your Edge Energy Coordinator OTP"
    message["From"] = sender
    message["To"] = email

    message.set_content(
        f"""
Your Edge Energy Coordinator verification code is:

{otp}

This code expires in 5 minutes.
"""
    )

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender, password)
        smtp.send_message(message)


def verify_otp(email, entered_otp):

    if "otp" not in st.session_state:
        return False

    if email != st.session_state.get("otp_email"):
        return False

    if datetime.now() > st.session_state["otp_expires"]:
        return False

    if entered_otp != st.session_state["otp"]:
        return False

    user_id = get_or_create_user(email)

    st.session_state["authenticated"] = True
    st.session_state["user_id"] = user_id
    st.session_state["user_email"] = email

    # Remove OTP after successful login
    st.session_state.pop("otp", None)
    st.session_state.pop("otp_email", None)
    st.session_state.pop("otp_expires", None)

    return True


def logout():
    keys = [
        "authenticated",
        "user_id",
        "user_email",
        "otp",
        "otp_email",
        "otp_expires",
    ]

    for key in keys:
        st.session_state.pop(key, None)
