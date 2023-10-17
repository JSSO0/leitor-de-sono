from twilio.rest import Client

account_sid = 'AC175b265a77f02f5fcd20ef5e423e27dd'
auth_token = 'f3e4affe224be1f977beb8091412bb39'
client = Client(account_sid, auth_token)

message = client.messages.create(
  from_='+13345819367',
  body='teste',
  to='+5588981522648'
)

print(message.sid)