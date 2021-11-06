def solution(n, lost, reserve):
    # 자기자신에서 다 제거
    for x in range(1,n+1):
        if (x in lost) and (x in reserve): #잃어버린 자기 자신에게 여분의 체육복이 있는 경우
            lost.remove(x)
            reserve.remove(x)
            
    if n in lost and n-1 in reserve:
        lost.remove(n)
        reserve.remove(n-1) 
  
    #타인 한쪽 제거        
    for x in range(1,n+1):
        if x == 1: # 첫번째 사람이라 자신의 앞에 사람이 없는 경우
            if x in lost and x+1 in reserve:
                    lost.remove(x)
                    reserve.remove(x+1)

        else: # 중간 순번들 - 한쪽만 있는 경우
            # if (x in lost) and (((x-1 in reserve) or(x+1 in reserve)) and not ((x-1 in reserve) and (x+1 in reserve ))):
            if (x in lost) and ((x-1 in reserve) or(x+1 in reserve)):
                if x-1 in reserve:
                    lost.remove(x)
                    reserve.remove(x-1)
                elif x+1 in reserve:
                    lost.remove(x)   
                    reserve.remove(x+1)
                else:
                    print("error ocurred")                     

    return n-len(lost)

solution(10, [1, 3, 5, 6, 7, 10],[ 1, 2, 4,7,8,9]) #lost, reserve, return