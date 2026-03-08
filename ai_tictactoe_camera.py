import cv2
import mediapipe as mp
import numpy as np
import math

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

drawing = []
board = [" "] * 9

player = None
computer = None
game_over = False


# ---------- FINGER DETECTION ---------- #

def index_finger_up(hand_landmarks):

    tip = hand_landmarks.landmark[8]
    pip = hand_landmarks.landmark[6]

    return tip.y < pip.y


# ---------- SHAPE DETECTION ---------- #

def detect_shape(points):

    if len(points) < 25:
        return None

    start = points[0]
    end = points[-1]

    dist = math.hypot(start[0]-end[0], start[1]-end[1])

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    width = max(xs) - min(xs)
    height = max(ys) - min(ys)

    ratio = width / (height+1)

    # O shape
    if dist < 50 and 0.7 < ratio < 1.3:
        return "O"

    # X shape
    if dist > 50:
        return "X"

    return None


# ---------- GAME LOGIC ---------- #

def check_winner(b, p):

    wins = [(0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)]

    for a,b1,c in wins:
        if b[a]==b[b1]==b[c]==p:
            return True
    return False


def minimax(board, is_max):

    if check_winner(board, computer):
        return 1
    if check_winner(board, player):
        return -1
    if " " not in board:
        return 0

    if is_max:
        best = -100
        for i in range(9):
            if board[i]==" ":
                board[i]=computer
                score=minimax(board,False)
                board[i]=" "
                best=max(best,score)
        return best
    else:
        best = 100
        for i in range(9):
            if board[i]==" ":
                board[i]=player
                score=minimax(board,True)
                board[i]=" "
                best=min(best,score)
        return best


def computer_move():

    best_score = -100
    move = None

    for i in range(9):

        if board[i]==" ":

            board[i]=computer
            score=minimax(board,False)
            board[i]=" "

            if score>best_score:
                best_score=score
                move=i

    if move is not None:
        board[move]=computer


# ---------- DRAW BOARD ---------- #

def draw_board(frame):

    h,w,_ = frame.shape
    step_x = w//3
    step_y = h//3

    for i in range(1,3):
        cv2.line(frame,(i*step_x,0),(i*step_x,h),(255,255,255),2)
        cv2.line(frame,(0,i*step_y),(w,i*step_y),(255,255,255),2)

    for i in range(9):

        x = (i%3)*step_x + step_x//2
        y = (i//3)*step_y + step_y//2

        if board[i]!=" ":
            cv2.putText(frame,board[i],(x-20,y+20),
                        cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,0),3)


# ---------- MAIN LOOP ---------- #

while True:

    ret, frame = cap.read()
    frame = cv2.flip(frame,1)

    rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    drawing_mode = False

    if results.multi_hand_landmarks:

        hand = results.multi_hand_landmarks[0]

        mp_draw.draw_landmarks(frame,hand,mp_hands.HAND_CONNECTIONS)

        if index_finger_up(hand):
            drawing_mode = True

            h,w,_ = frame.shape

            x = int(hand.landmark[8].x*w)
            y = int(hand.landmark[8].y*h)

            drawing.append((x,y))

            cv2.circle(frame,(x,y),6,(0,0,255),-1)

    # draw line
    for i in range(1,len(drawing)):
        cv2.line(frame,drawing[i-1],drawing[i],(255,0,0),2)

    # detect shape when finger lowered
    if not drawing_mode and len(drawing)>30 and not game_over:

        shape = detect_shape(drawing)

        if shape:

            if player is None:
                player = shape
                computer = "O" if player=="X" else "X"

            h,w,_ = frame.shape

            col = drawing[-1][0] // (w//3)
            row = drawing[-1][1] // (h//3)

            index = int(row*3 + col)

            if index<9 and board[index]==" ":

                board[index] = player

                if not check_winner(board,player):
                    computer_move()

        drawing = []

    draw_board(frame)

    # ----- RESULTS ----- #

    if player and check_winner(board,player):
        cv2.putText(frame,"YOU WIN",(50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),3)
        game_over = True

    elif computer and check_winner(board,computer):
        cv2.putText(frame,"COMPUTER WINS",(50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
        game_over = True

    elif " " not in board:
        cv2.putText(frame,"TIE MATCH",(50,50),
                    cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),3)
        game_over = True

    cv2.imshow("AI Tic Tac Toe",frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


