import requests
from flask import Flask, request, redirect
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

SCOPES = ['https://www.googleapis.com/auth/gmail.modify']
from google.oauth2.credentials import Credentials
import db
app = Flask(__name__)

flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)


@app.route('/auth')
def auth():
    print(flow.client_config)
    flow.redirect_uri = 'http://localhost:8080/code'
    url = flow.authorization_url()
    return redirect(url[0], code=200)


@app.route('/logout')
def logout():
    # https://support.google.com/accounts/answer/3466521?hl=ru
    return redirect('https://myaccount.google.com/permissions', code=200)


@app.route('/code')
def hello():
    auth_code = request.args.get('code')
    d = {"client_id": "43127102122-jj5ljv9ttfemsb938uvd7k3uh0im0i7v.apps.googleusercontent.com",
         "client_secret": "MfbJ48kju5uMEfMQEwcCTSPR",
         "code": auth_code,
         "grant_type": "authorization_code",
         "redirect_uri": 'http://localhost:8080/code'
         }
    access_token_response = requests.post('https://oauth2.googleapis.com/token', data=d)
    print(access_token_response)
    cred = access_token_response.json()
    creds = Credentials(token=cred['access_token'],
                          refresh_token=cred['refresh_token'],
                          token_uri='https://oauth2.googleapis.com/token',
                          client_id="43127102122-jj5ljv9ttfemsb938uvd7k3uh0im0i7v.apps.googleusercontent.com",
                          client_secret="MfbJ48kju5uMEfMQEwcCTSPR",
                          scopes=['https://www.googleapis.com/auth/gmail.modify'])
    service = build('gmail', 'v1', credentials=creds)
    user = service.users().getProfile(userId='me').execute()
    response = "You are successfully authorized in spam filter app as "+user['emailAddress']
    try:
        response = service.users().labels().create(userId='me', body={
          "labelListVisibility": "labelShow",
          "messageListVisibility": "show",
          "name": "filtered"
        }).execute()
        user = {
            'user_email': user['emailAddress'],
            'access_token': cred['access_token'],
            'refresh_token': cred['refresh_token'],
            'checked_label_id': response['id']
        }
        db.db.save_user(user)
    except:
        print(response)

    return response

