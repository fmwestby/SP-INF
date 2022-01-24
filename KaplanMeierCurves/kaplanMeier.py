d_time = [1, 1, 2, 2, 2, 3, 5, 6, 8, 10]
d_status = [1, 1, 1, 0, 0, 1, 1, 0, 1, 0]

def test(d_time, d_status):
    S_hat = [1]
    days = [0]
    total = 10
    
    observed = 0
    censored = 0
    
    # first case
    if d_status[0] == 1:
        observed += 1
    else: 
        censored += 1
    
    
    
    for i in range(1, 10):
        
        if (d_time[i] not in days) and (d_status[i] != 0):
            days.append(d_time[i])
            
        
        while d_time[i] == d_time[i-1]:
            if d_status[i] == 1:
                observed += 1
            else:
                censored += 1
            
            i += 1
            
        tmp = S_hat[-1] * (1 - (observed / total))
        S_hat.append(S_hat[-1] * (1 - (observed / total)))
        
        total -= (observed + censored)
            
        observed = 0
        censored = 0
            
    return S_hat, days

def test2(t, status):
    days = [0]
    nAlive = [len(t)]
    nDied = [0]
    pDying = [0]
    pSurviving = [1]

    total = len(t)
    for i in range(1, len(t)):
        if (t[i] not in days) and (status[i] != 0):
            days.append(t[i])
            nDied.append(1)
            nAlive.append(total)
            #println(t[i])
            if (i-1 == 0):
                nAlive.append(total)
                nDied[-1] += 1
                nAlive[-1] -= 1
                total -= 1
                
            
            nAlive[-1] -= 1
            
        elif (t[i] in days and status[i] != 0):
            nDied[-1] += 1
            nAlive[-1] -= 1
            
        else:
            nAlive[-1] -= 1
            
        #nAlive[-1] -= 1
        total -= 1

    for i in range(1, len(nAlive)-1):
          
        pSurviving.append(pSurviving[-1] * ((nAlive[i] - nDied[i]) / nAlive[i]))

    print("days: ", days)
    print("nDied: ", nDied)
    print("nAlive: ", nAlive)
    print("pSurviving: ", pSurviving)
            
test2(d_time, d_status)

"""
days:  [0, 1, 2, 3, 5, 8]
nDied:  [0, 2, 1, 1, 1, 1]
nAlive:  [10, 10, 8, 5, 4, 2, 0]
pSurviving:  [1, 0.8, 0.7000000000000001, 0.56, 0.42000000000000004, 0.21000000000000002]
""" 