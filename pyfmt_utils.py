
from wcwidth import wcswidth



def wcfmt(x, w, align='<'): # align의 기본값은 '< : left', <, ><, >
    """ 동아시아문자 폭을 고려하여, 문자열 포매팅을 해 주는 함수.
    w 는 해당 문자열과 스페이스문자가 차지하는 너비.
    align 은 문자열의 수평방향 정렬 좌/우/중간.
    """
    x = str(x) # 해당 문자열
    l = wcswidth(x) # 문자열이 몇자리를 차지하는지를 계산.
    s = w-l # 남은 너비 = 사용자가 지정한 전체 너비 - 문자열이 차지하는 너비
    if s <= 0:
        return x
    if align == '<':
        return x + ' '*s
    if align == '><':
        sl = s//2 # 변수 좌측
        sr = s - sl # 변수 우측
        return ' '*sl + x + ' '*sr
    # '>': right align
    return ' '*s + x
