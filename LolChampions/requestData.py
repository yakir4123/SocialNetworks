import os
import re
import json
import time
import asyncio
import requests

from pantheon import pantheon
from alive_progress import alive_bar


def requestsLog(url, status, headers):
    print("url:")
    print(url)
    print("status:")
    print(status)
    print("headers:")
    print(headers)


async def getSummonerId(name):
    try:
        data = await panth.getSummonerByName(name)
        return data['id'], data['accountId']
    except Exception as e:
        print(e)


async def getRecentMatchlist(accountId):
    try:
        data = await panth.getMatchlist(accountId, params={"endIndex": match_num})
        return data
    except Exception as e:
        print(e)


async def getRecentMatches(accountId):
    try:
        matchlist = await getRecentMatchlist(accountId)
        tasks = [panth.getMatch(match['gameId']) for match in matchlist['matches']]
        return await asyncio.gather(*tasks)
    except Exception as e:
        print(e)


def get_names(page):
    test_str = requests.get(f'https://eune.op.gg/ranking/ladder/page={page}').text
    test_sub = "userName"
    res = [i.start() for i in re.finditer(test_sub, test_str)][1:]
    all_names = [test_str[index + 9: index + 9 + test_str[index + 9:index + 100].find('"')] for index in res]
    return [name.replace("+", " ") for name in all_names]


if __name__ == '__main__':

    api_key = "RGAPI-9203fec0-368f-4ff7-9ef1-e2c745d5aa08"
    server = 'eun1'
    ranking_page = 104

    names = get_names(ranking_page)
    match_num = 100

    panth = pantheon.Pantheon(server, api_key, errorHandling=True, requestsLoggingFunction=requestsLog, debug=True)
    loop = asyncio.get_event_loop()

    with alive_bar(len(names), force_tty=True) as bar:
        for name in names:
            (summonerId, accountId) = loop.run_until_complete(getSummonerId(name))
            print("summonerId: " + summonerId)
            print("accountId: " + accountId)
            matches = loop.run_until_complete(getRecentMatches(accountId))
            with open(f'matches{os.sep}{name}_{match_num}.json', 'w', newline='\n', encoding='utf-8') as json_output:
                print(matches)
                print('Dumping the data.')
                json.dump(matches, json_output, indent=1)
            time.sleep(120)
            bar()
    print('done.')
