import urllib.parse
import requests
import json


def get_brave_search_api():
    headers = {
        'Accept': 'application/json',
        'Accept-Encoding': 'gzip',
        'X-Subscription-Token': '<YOUR BRAVE API KEY>'
    }

    def search(q: str):
        q = urllib.parse.quote(q, safe='')
        url = f'https://api.search.brave.com/res/v1/web/search?q={q}&summary=1&search_lang=zh-hans'
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return None

        json_ = json.loads(response.text)
        web_results = json_['web']['results'][:5]

        search_result = []
        for idx, web_result in enumerate(web_results):
            search_result.append(f'{idx + 1}. {web_result['description']}\n')

        return ''.join(search_result)

    return search


def get_bochaai_search_api():
    headers = {
        'Authorization': '<YOUR BOCHAAI API KEY>',
        'Content-Type': 'application/json'
    }

    def search(q: str):
        q = q.replace('/think', '').replace('/no think', '')
        payload = json.dumps({
            "query": f"{q}",
            "summary": True,
            "freshness": "noLimit",
            "count": 1
        })

        response = requests.post('https://api.bochaai.com/v1/web-search', headers=headers, data=payload)
        if response.status_code != 200:
            return None

        json_ = json.loads(response.text)
        if json_['code'] != 200:
            return None

        summary = json_['data']['webPages']['value'][0]['summary']
        return f'{summary}\n'

    return search
