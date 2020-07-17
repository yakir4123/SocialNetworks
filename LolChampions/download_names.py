import requests
import re

PAGE = "30"

test_str = requests.get('https://eune.op.gg/ranking/ladder/page=' + PAGE).text
test_sub = "userName"
res = [i.start() for i in re.finditer(test_sub, test_str)][1:]
all_names = [test_str[index+9: index+ 9 + test_str[index+9:index+100].find('"')] for index in res]
all_names = [name.replace("+"," ") for name in all_names]
print(all_names)
