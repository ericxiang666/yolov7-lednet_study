import random

def random_team(team):
    random.shuffle(team)

team = ['海王', '东京冷', '细节', '葡萄', '死神', '风', '张', '五五开', 'ray', '阿谢', '帝光', '山南', '溯', '肥羊']

first = {0,1,2} 
last = {11,12,13}

while any(team.index(member) in first for member in ['海王', '东京冷', '细节'])  and any(team.index(member) in last for member in [ '山南', '溯', '肥羊']):
    random_team(team)

print(team)