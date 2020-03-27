
tb_card = input()
cards = list(map(str,input().split()))
count = 0

for i in cards:
    if(i[0]==tb_card[0] or i[1]==tb_card[1]):
        count = 1
        break
    else:
        count = 0
        
if(count==1):
    print('YES')
else:
    print('NO')
