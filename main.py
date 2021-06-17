from spam_filter import SpamFilter
from db import db
import google_auth
import base64
from multiprocessing import Process
import time
from webserver import app


def run():
    spam_filter = SpamFilter()
    spam_filter.spam_filter()
    while True:
        users = db.get_all()
        for user in users:
            service = google_auth.create_service(user)
            try:
                mails_id = google_auth.get_mails(service)
                m_id, body = google_auth.get_user_messages_to_classify(service, mails_id)
                tmp = spam_filter.predict_text(body)
                label = ''
                if(tmp < 0.5):
                    label = 'SPAM'
                else:
                    label = 'HAM'
                google_auth.change_mail_label(service, m_id, label)
            except:
                print("error")
        time.sleep(60.0)



if __name__ == "__main__":
    p = Process(target=run())
    p.start()
    app.run(host='localhost', port=8080)
