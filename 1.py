import requests
url = 'https://s3.flatfiles.coinapi.io/bucket/?prefix=trades/20230530/'
headers = {'Authorization' : '82dd5b47-b46b-413e-b00b-a7b37c3486e8'}
response = requests.get(url, headers=headers)
print(response)