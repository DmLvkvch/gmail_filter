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
                google_auth.change_mail_label(service, mails_id['messages'][1]['id'], 'SPAM')
                for id in mails_id['messages']:
                    tmp = google_auth.get_message(service, 'me', id['id'])
                    for q in tmp['payload']['parts']:
                        if q['mimeType'] == 'text/plain':
                            print(base64.b64decode(q['body']['data']))
            except:
                print(erro)
        time.sleep(60.0)



if __name__ == "__main__":
    p = Process(target=run())
    p.start()
    app.run(host='localhost', port=8080)
