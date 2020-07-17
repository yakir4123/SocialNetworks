import os
import json
import asyncio
import time

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


if __name__ == '__main__':

    api_key = "RGAPI-27d296f7-2d1a-49cc-bdb8-3c55158c3145"
    server = 'eun1'

    # 2901 to 2925 top players
    names = ['gXJason', 'Teh', 'CuteEGrillCheese', 'Aira', 'LuzíkArbuzìk', 'LayxZyx', 'Waszk', 'Zëref', 'CofiKiller23', 'Fittamunden', 'Everything Ends', 'RaekH', 'Retso', 'Indecisiveness', 'Sufleks', 'The True Adrian', 'Crimson Behelit']
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
