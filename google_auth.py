import base64
import email
import pickle
import os.path
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
import base64
from google.oauth2.credentials import Credentials
from db import db

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']


def create_service(cred):
    creds = Credentials(token=cred['access_token'],
                        refresh_token=cred['refresh_token'],
                        token_uri='https://oauth2.googleapis.com/token',
                        client_id="43127102122-jj5ljv9ttfemsb938uvd7k3uh0im0i7v.apps.googleusercontent.com",
                        client_secret="MfbJ48kju5uMEfMQEwcCTSPR",
                        scopes=['https://www.googleapis.com/auth/gmail.modify'])
    service = build('gmail', 'v1', credentials=creds)
    return service


def auth():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    service = build('gmail', 'v1', credentials=creds)
    return service


def get_message(service, user_id, msg_id):
    try:
        return service.users().messages().get(userId=user_id, id=msg_id, format='full').execute()
    except Exception as error:
        print('An error occurred: %s' % error)


def parse_msg(msg):
    if msg.get("payload").get("body").get("data"):
        return base64.urlsafe_b64decode(msg.get("payload").get("body").get("data").encode("ASCII")).decode("utf-8")
    return msg.get("snippet")


def get_mime_message(service, user_id, msg_id):
    try:
        message = service.users().messages().get(userId=user_id, id=msg_id,
                                                 format='raw').execute()
        msg_str = base64.urlsafe_b64decode(message['raw'].encode("utf-8")).decode("utf-8")
        mime_msg = email.message_from_string(msg_str)
        return mime_msg
    except Exception as error:
        print('An error occurred: %s' % error)


def get_mails(service):
    results = service.users().messages().list(userId='me', includeSpamTrash=True).execute()
    return results


def change_mail_label(service, message_id, label):
    msg = get_message(service, 'me', message_id)
    labelsId = msg['labelIds']
    if label == 'SPAM':
        body = {
            "addLabelIds": ['SPAM', 'Label_1'],
            "removeLabelIds": []
        }
    else:
        body = {
            "addLabelIds": ['Label_1'],
            "removeLabelIds": ['SPAM']
        }
    response = service.users().messages().modify(userId='me', id=message_id, body=body).execute()


def get_filter(service):
    response = service.users().settings().filters().list(userId='me').execute()
    result = False
    return response


def check_filter(filter, message):
    criteria = filter['criteria']
    action = filter['action']
    headers = message['payload']['headers']
    sender = ''
    subject = ''
    for h in headers:
        if h['name'] == 'From':
            sender = h['value']
        if h['name'] == 'Subject':
            subject = h['value']
    return sender == criteria['from']



# response = service.users().labels().create(userId='me', body={
#   "labelListVisibility": "labelShow",
#   "messageListVisibility": "show",
#   "name": "filtered"
# }).execute()
#
# print(response)

def get_user_messages_to_classify(service, mails_id):
    msgs = []
    for id in mails_id['messages']:
        tmp = get_message(service, 'me', id['id'])
        filters = get_filter(service)
        result = False
        for f in filters['filter']:
            result = check_filter(f, tmp)
        if not result:
            body = tmp['payload']['parts'][0]['body']['data'].encode("ASCII")
            body = base64.urlsafe_b64decode(body).decode("utf-8")
            msgs.append((id, body))
    return msgs
