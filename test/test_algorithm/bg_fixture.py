from ...arc.algorithm.find_background import *
BG = [
    Background(X_train=[0, 0], y_train=[0, 0], X_test=[0], y_test=[0]),  # 0
    Background(X_train=[0, 0, 0, 0, 0], y_train=[
               0, 0, 0, 0, 0], X_test=[0], y_test=[0]),  # 1
    Background(X_train=[0, 0, 0, 0, 0], y_train=[
               0, 0, 0, 0, 0], X_test=[0], y_test=[0]),  # 2
    Background(X_train=[0, 0, 0, 0, 0], y_train=[
               0, 0, 0, 0, 0], X_test=[0], y_test=[0]),  # 3
    Background(X_train=[0, 0, 0, 0], y_train=[0, 0, 0, 0], X_test=[0], y_test=[0]),  # 4
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 5
    Background(X_train=[0, 0], y_train=[0, 0], X_test=[0], y_test=[0]),  # 6
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 7
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 8
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 9
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 10
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 11
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 12
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 13
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 14
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 15
    Background(X_train=[0, 0], y_train=[0, 0], X_test=[0], y_test=[0]),  # 16
    Background(X_train=[0, 0], y_train=[0, 0], X_test=[0], y_test=[0]),  # 17
    Background(X_train=[0, 0, 0, 0], y_train=[
               0, 0, 0, 0], X_test=[0], y_test=[0]),  # 18
    Background(X_train=[0, 0], y_train=[0, 0], X_test=[0], y_test=[0]),  # 19
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 20
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 21
    Background(X_train=[0, 0, 0, 0], y_train=[
               0, 0, 0, 0], X_test=[0], y_test=[0]),  # 22
    Background(X_train=[0, 0, 0, 0], y_train=[
               0, 0, 0, 0], X_test=[0], y_test=[0]),  # 23
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 24
    Background(X_train=[0, 0], y_train=[0, 0], X_test=[0], y_test=[0]),  # 25
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 26
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 27
    Background(X_train=[5, 2, 5], y_train=[5, 2, 5], X_test=[8], y_test=[8]),  # 28
    Background(X_train=[0, 0, 0, 0], y_train=[
               0, 0, 0, 0], X_test=[0], y_test=[0]),  # 29
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 30
    Background(X_train=[0, 0, 0, 0], y_train=[
               0, 0, 0, 0], X_test=[0], y_test=[0]),  # 31
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 32
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 33
    Background(X_train=[0, 0, 0, 0], y_train=[
               0, 0, 0, 0], X_test=[0], y_test=[0]),  # 34
    Background(X_train=[0, 0, 0, 0], y_train=[
               0, 0, 0, 0], X_test=[0], y_test=[0]),  # 35
    Background(X_train=[0, 0], y_train=[0, 0], X_test=[0], y_test=[0]),  # 36
    Background(X_train=[0, 0, 0, 0], y_train=[
               0, 0, 0, 0], X_test=[0], y_test=[0]),  # 37
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 38
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 39
    Background(X_train=[1, 3, 3], y_train=[1, 3, 3], X_test=[1], y_test=[1]),  # 40
    Background(X_train=[7, 7, 7], y_train=[7, 7, 7], X_test=[7], y_test=[7]),  # 41
    Background(X_train=[0, 0], y_train=[0, 0], X_test=[0], y_test=[0]),  # 42
    Background(X_train=[0, 0, 0, 0, 0], y_train=[
               0, 0, 0, 0, 0], X_test=[0], y_test=[0]),  # 43
    Background(X_train=[0, 0, 0, 0], y_train=[0, 0, 0, 0],
               X_test=[0, 0], y_test=[0, 0]),  # 44
    Background(X_train=[0, 0, 0, 0], y_train=[
               0, 0, 0, 0], X_test=[0], y_test=[0]),  # 45
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 46
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 47
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 48
    Background(X_train=[2, 1, 3], y_train=[2, 1, 3], X_test=[4], y_test=[4]),  # 49
    Background(X_train=[9, 8, 7], y_train=[9, 8, 7], X_test=[8], y_test=[8]),  # 50
    Background(X_train=[0, 0, 0, 0], y_train=[
               0, 0, 0, 0], X_test=[0], y_test=[0]),  # 51
    Background(X_train=[7, 7, 7], y_train=[7, 7, 7], X_test=[7], y_test=[7]),  # 52
    Background(X_train=[7, 7, 7], y_train=[7, 7, 7], X_test=[7], y_test=[7]),  # 53
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 54
    Background(X_train=[4, 4, 4], y_train=[4, 4, 4], X_test=[4], y_test=[4]),  # 55
    Background(X_train=[0, 0], y_train=[0, 0], X_test=[0], y_test=[0]),  # 56
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 57
    Background(X_train=[0, 0, 0, 0], y_train=[
               0, 0, 0, 0], X_test=[0], y_test=[0]),  # 58
    Background(X_train=[7, 7], y_train=[7, 7], X_test=[7], y_test=[7]),  # 59
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 60
    Background(X_train=[0, 0, 0, 0], y_train=[
               0, 0, 0, 0], X_test=[0], y_test=[0]),  # 61
    Background(X_train=[8, 8], y_train=[8, 8], X_test=[8, 8], y_test=[8, 8]),  # 62
    Background(X_train=[0, 0, 0, 0], y_train=[
               0, 0, 0, 0], X_test=[0], y_test=[0]),  # 63
    Background(X_train=[7, 7, 7], y_train=[7, 7, 7], X_test=[7], y_test=[7]),  # 64
    Background(X_train=[7, 7, 7], y_train=[7, 7, 7], X_test=[7], y_test=[7]),  # 65
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 66
    Background(X_train=[7, 7, 7], y_train=[7, 7, 7], X_test=[7], y_test=[7]),  # 67
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 68
    Background(X_train=[0, 0, 0, 0], y_train=[
               0, 0, 0, 0], X_test=[0], y_test=[0]),  # 69
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 70
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 71
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 72
    Background(X_train=[8, 8, 8], y_train=[8, 8, 8], X_test=[8], y_test=[8]),  # 73
    Background(X_train=[0, 0, 0, 0, 0], y_train=[
               0, 0, 0, 0, 0], X_test=[0], y_test=[0]),  # 74
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 75
    Background(X_train=[0, 0, 0, 0], y_train=[
               0, 0, 0, 0], X_test=[0], y_test=[0]),  # 76
    Background(X_train=[9, 9, 9, 9, 9], y_train=[
               0, 0, 0, 0, 0], X_test=[9], y_test=[0]),  # 77
    Background(X_train=[4, 4], y_train=[4, 4], X_test=[4], y_test=[4]),  # 78
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 79
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 80
    Background(X_train=[0, 0], y_train=[0, 0], X_test=[0, 0], y_test=[0, 0]),  # 81
    Background(X_train=[0, 0], y_train=[0, 0], X_test=[0], y_test=[0]),  # 82
    Background(X_train=[5, 5, 5], y_train=[5, 5, 5], X_test=[5], y_test=[5]),  # 83
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 84
    Background(X_train=[0, 0, 0, 0], y_train=[
               0, 0, 0, 0], X_test=[0], y_test=[0]),  # 85
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 86
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 87
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 88
    Background(X_train=[1, 1, 1], y_train=[1, 1, 1], X_test=[1], y_test=[1]),  # 89
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0],
               X_test=[0, 0], y_test=[0, 0]),  # 90
    Background(X_train=[7, 7], y_train=[7, 7], X_test=[7], y_test=[7]),  # 91
    Background(X_train=[0, 0], y_train=[0, 0], X_test=[0], y_test=[0]),  # 92
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 93
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 94
    Background(X_train=[7, 7], y_train=[7, 7], X_test=[7], y_test=[7]),  # 95
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 96
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 97
    Background(X_train=[0, 0, 0, 0], y_train=[
               0, 0, 0, 0], X_test=[0], y_test=[0]),  # 98
    Background(X_train=[0, 0, 0], y_train=[0, 0, 0], X_test=[0], y_test=[0]),  # 99
]
