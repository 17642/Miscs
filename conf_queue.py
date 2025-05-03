import numpy as np
from collections import deque
import random
import math


ddh = 0.5/0.85

class bodyPoint:
    def __init__(self, LS, RS, LH, RH): # LS, RS, LH, RH is pos (x,y)
        self.LS = LS #왼쪽어깨
        self.RS = RS #오른쪽어깨
        self.LH = LH #왼쪽골반
        self.RH = RH # 오른쪽골반
        self.CS = ((LS[0]+RS[0])/2, (LS[1]+RS[1])/2) #어깨중점
        self.CH = ((LH[0]+RH[0])/2, (LH[1]+RH[1])/2) #골반중점
        self.HSH = self.CS[1] - self.CH[1] #어깨-골반 높이차

        
    def getSqrLength(self,a,b): #언젠가 최적화 예정
        return (a[0]-b[0])**2+(a[1]-b[1])**2
    
    def getMovement(self, lineA, lineB): #lineA, lineB는 튜플(점1,점2) -> 기울기 차만 계산한다.
        
        x1,y1= lineA[1][0] - lineA[0][0], lineA[1][1] - lineA[0][1]  
        x2,y2= lineB[1][0] - lineB[0][0], lineB[1][1] - lineB[0][1]    # 각 선의 변동치 

        return abs(y1/x1 - y2/x2)
    
    def calculate(self, target):
        if target == None:
            return None
        return CalculatedResult(self.getMovement((self.LS,self.RS),(target.LS, target.RS)),self.CH[1] - target.CH[1],(self.HSH+target.HSH)*ddh/2, (self.HSH+target.HSH)/2)
    
MAX_LIST = ["MAX_SHOULDER_ANGLE", "MAX_HIP_DOWN", "DANGEROUS_DOWN_HEIGHT"]
MIN_LIST = ["MIN_SHOULDER_HIP_HEIGHT"]

class CalculatedResult:
    def __init__(self, MSA, MHD, DDH, MSHH):
        self.MAX_SHOULDER_ANGLE = MSA
        self.MAX_HIP_DOWN = MHD
        self.DANGEROUS_DOWN_HEIGHT = DDH
        self.MIN_SHOULDER_HIP_HEIGHT = MSHH
    

class CalcStorage:
    def __init__(self, dataSize=5):
        self.base = deque(maxlen = dataSize - 1)
        self.datSize = dataSize

        self.MSA = 0
        self.MHD = 0
        self.DDH = 0
        self.MSHH = math.inf

        #최대값 최소값 초기화(참조로 관리)
        
        # 어차피 None을 제외한 모든 값에서 max,min을 추적해야 하는데, 
        # max, min은 그렇게 비싼건 아니니까 그냥 내외적 거리연산만 하고
        # 비교 연산은 매 프레임마다 해도 될듯 함.

        #아니라면 move 함수에 삭제되는 인덱스 파악하고, maxList랑 minList 업데이트 해야지
        for i in range(dataSize):
            self.base.append(deque([None]*(dataSize - 1 - i))) # None으로 값을 채운다.
    
    def move(self, list ,newData): # list는 아래의 ImgProcessDeque를 사용하면 됨
        self.base.append(deque([]))
        for i in range(self.datSize - 1):
            if list[i] == None:
                self.base[i].append(None) #옆에 쓴 노트에 그림 설명 있으니까 알아서 보셈
            self.base[i].append(list[i].calculate(newData))

    def __getitem__(self,find): #find는 튜플임.
        if(find[0]==find[1]): #이거 시전하면 [-1]을 찾을텐데 나는 좋지 않다고 봄.
            return None
        return self.base[max(find[0],find[1])][min(find[0],find[1])-1-find[0]]#언젠가 말로 설명해야지
    
    def checkAllDatas(self):
        for b in self.base:
            for z in b:
                if z == None:
                    continue
                self.MSA = max(self.MSA,z.MAX_SHOULDER_ANGLE)
                self.MHD = max(self.MHD, z.MAX_HIP_DOWN)
                self.DDH = max(self.DDH, z.DANGEROUS_DOWN_HEIGHT)
                self.MSHH = min(self.MSHH, z.MIN_SHOULDER_HIP_HEIGHT)


    def __str__(self):
        s=""
        for i in range(self.datSize):
            if i < len(self.base):
                s += f"{i}: {list(self.base[i])}\n"
        return s
        
    

        



class ImgProcessDeque:

    def __init__(self,queSize, CancelCount, FRCount):
        self.deq = deque([None]*queSize, maxlen = queSize)

        self.queSize = queSize
        self.CancelNoneCount = CancelCount
        self.FRCancelCount = FRCount

        self.FNon = queSize
        self.RNon = queSize
        self.NonCount = queSize

    def insertDeq(self, data): 
        self.deq.append(data)
        if self.NonCount == self.queSize and data != None: # 첫 삽입
            self.RNon = 0
            self.FNon -=1
            self.NonCount -=1
        elif self.NonCount == 0 and data != None: # 데이터가 가득 차 있을경우 넘김
            pass
        elif self.NonCount == self.queSize and data == None: # None이 가득 차있는데, 추가로 None이 들어가도 넘김
            pass
        else:
            if data == None:
                self.RNon +=1
                if self.FNon == 0: # None이 아닌 값이 제거되는 경우 
                    self.NonCount += 1 # NoneCount를 증가시킨다.
            else:
                self.RNon = 0
                if self.FNon != 0:
                    self.NonCount -=1
            #데이터가 밀리는 건 동일하므로 LNON 연산을 공유한다.
            if self.deq[0] == None: # 첫 원소 보는 함수 이름 기억안남
                if self.FNon != 0: # 이미 빈칸이 연속적일 경우
                    self.FNon -= 1 # 한칸 뺀다
                else: # 빈칸이 이번에 생긴 경우
                    for d in self.deq: # 순차적으로 탐색해 FNon을 증가시킨다
                        if d == None:
                            self.FNon +=1
                        else:
                            break # 빈칸이 끝나면 종료
            else:
                self.FNon = 0
            
    def __str__(self):
        return f"Deque: {list(self.deq)},\n maxQueSize = {self.queSize}\n\
Fnon, Rnon = ({self.FNon, self.RNon}), NoneCount = {self.NonCount}\n\
CancelNoneCount = {self.CancelNoneCount}\nFRNoneCount = {self.FRCancelCount}"
    
    def test(self, count, NoneProb=0.5):
        test_inputs = [None if random.random() < NoneProb else random.randint(1, 100) for _ in range(count)]
        for i, data in enumerate(test_inputs, 1):
            self.insertDeq(data)
            print(f"삽입 {i}회 (입력: {data}):")
            print(f"Deque: {list(self.deq)},\nFnon, Rnon = ({self.FNon, self.RNon}), NoneCount = {self.NonCount}")
            print("-" * 50)
        self.deq.clear()

    def cutNon(self):#에...
        return self.NonCount < self.CancelNoneCount and self.FNon + self.RNon < self.FRCancelCount
    
    def clear(self):
        self.deq.clear()
        self.deq.append([None]*self.queSize)
    
iq = ImgProcessDeque(20, 5, 3)
iq.test(100)
id = CalcStorage(5)
print(id)

