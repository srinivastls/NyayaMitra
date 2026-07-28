import os
import requests

def download_pdf(url, folder):
    file_name = url.split('/')[-1]
    file_path = os.path.join(folder, file_name)
    try:
        response = requests.get(url)
        response.raise_for_status() 

        with open(file_path, 'wb') as f:
            f.write(response.content)
        print(f'Successfully downloaded: {file_name}')
    except requests.exceptions.RequestException as e:
        print(f'Failed to download: {file_name}. Error: {e}')

url = 'https://main.sci.gov.in/judgment/judis/'
folder = 'Supreme'

if not os.path.exists(folder):
    os.makedirs(folder)

for i in range(1,30000):
    ur=url+str(i)+".pdf"
    download_pdf(ur, folder)
