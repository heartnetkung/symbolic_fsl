from ...arc.algorithm.find_background import *
BG = [
    {'X_train':[0, 0], 'y_train':[0, 0], 'X_test':[0], 'y_test':[0]},  # 0
    {'X_train':[0, 0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 1
    {'X_train':[0, 0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 2
    {'X_train':[0, 0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 3
    {'X_train':[0, 0, 0, 0], 'y_train':[0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 4
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 5
    {'X_train':[0, 0], 'y_train':[0, 0], 'X_test':[0], 'y_test':[0]},  # 6
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 7
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 8
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 9
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 10
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 11
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 12
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 13
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 14
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 15
    {'X_train':[0, 0], 'y_train':[0, 0], 'X_test':[0], 'y_test':[0]},  # 16
    {'X_train':[0, 0], 'y_train':[0, 0], 'X_test':[0], 'y_test':[0]},  # 17
    {'X_train':[0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 18
    {'X_train':[0, 0], 'y_train':[0, 0], 'X_test':[0], 'y_test':[0]},  # 19
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 20
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 21
    {'X_train':[0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 22
    {'X_train':[0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 23
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 24
    {'X_train':[0, 0], 'y_train':[0, 0], 'X_test':[0], 'y_test':[0]},  # 25
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 26
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 27
    {'X_train':[5, 2, 5], 'y_train':[5, 2, 5], 'X_test':[8], 'y_test':[8]},  # 28
    {'X_train':[0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 29
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 30
    {'X_train':[0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 31
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 32
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 33
    {'X_train':[0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 34
    {'X_train':[0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 35
    {'X_train':[0, 0], 'y_train':[0, 0], 'X_test':[0], 'y_test':[0]},  # 36
    {'X_train':[0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 37
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 38
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 39
    {'X_train':[1, 3, 3], 'y_train':[1, 3, 3], 'X_test':[1], 'y_test':[1]},  # 40
    {'X_train':[7, 7, 7], 'y_train':[7, 7, 7], 'X_test':[7], 'y_test':[7]},  # 41
    {'X_train':[0, 0], 'y_train':[0, 0], 'X_test':[0], 'y_test':[0]},  # 42
    {'X_train':[0, 0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 43
    {'X_train':[0, 0, 0, 0], 'y_train':[0, 0, 0, 0],
               'X_test':[0, 0], 'y_test':[0, 0]},  # 44
    {'X_train':[0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 45
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 46
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 47
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 48
    {'X_train':[2, 1, 3], 'y_train':[2, 1, 3], 'X_test':[4], 'y_test':[4]},  # 49
    {'X_train':[9, 8, 7], 'y_train':[9, 8, 7], 'X_test':[8], 'y_test':[8]},  # 50
    {'X_train':[0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 51
    {'X_train':[7, 7, 7], 'y_train':[7, 7, 7], 'X_test':[7], 'y_test':[7]},  # 52
    {'X_train':[7, 7, 7], 'y_train':[7, 7, 7], 'X_test':[7], 'y_test':[7]},  # 53
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 54
    {'X_train':[4, 4, 4], 'y_train':[4, 4, 4], 'X_test':[4], 'y_test':[4]},  # 55
    {'X_train':[0, 0], 'y_train':[0, 0], 'X_test':[0], 'y_test':[0]},  # 56
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 57
    {'X_train':[0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 58
    {'X_train':[7, 7], 'y_train':[7, 7], 'X_test':[7], 'y_test':[7]},  # 59
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 60
    {'X_train':[0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 61
    {'X_train':[8, 8], 'y_train':[8, 8], 'X_test':[8, 8], 'y_test':[8, 8]},  # 62
    {'X_train':[0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 63
    {'X_train':[7, 7, 7], 'y_train':[7, 7, 7], 'X_test':[7], 'y_test':[7]},  # 64
    {'X_train':[7, 7, 7], 'y_train':[7, 7, 7], 'X_test':[7], 'y_test':[7]},  # 65
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 66
    {'X_train':[7, 7, 7], 'y_train':[7, 7, 7], 'X_test':[7], 'y_test':[7]},  # 67
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 68
    {'X_train':[0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 69
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 70
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 71
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 72
    {'X_train':[8, 8, 8], 'y_train':[8, 8, 8], 'X_test':[8], 'y_test':[8]},  # 73
    {'X_train':[0, 0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 74
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 75
    {'X_train':[0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 76
    {'X_train':[9, 9, 9, 9, 9], 'y_train':[
               0, 0, 0, 0, 0], 'X_test':[9], 'y_test':[0]},  # 77
    {'X_train':[4, 4], 'y_train':[4, 4], 'X_test':[4], 'y_test':[4]},  # 78
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 79
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 80
    {'X_train':[0, 0], 'y_train':[0, 0], 'X_test':[0, 0], 'y_test':[0, 0]},  # 81
    {'X_train':[0, 0], 'y_train':[0, 0], 'X_test':[0], 'y_test':[0]},  # 82
    {'X_train':[5, 5, 5], 'y_train':[5, 5, 5], 'X_test':[5], 'y_test':[5]},  # 83
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 84
    {'X_train':[0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 85
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 86
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 87
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 88
    {'X_train':[1, 1, 1], 'y_train':[1, 1, 1], 'X_test':[1], 'y_test':[1]},  # 89
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0],
               'X_test':[0, 0], 'y_test':[0, 0]},  # 90
    {'X_train':[7, 7], 'y_train':[7, 7], 'X_test':[7], 'y_test':[7]},  # 91
    {'X_train':[0, 0], 'y_train':[0, 0], 'X_test':[0], 'y_test':[0]},  # 92
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 93
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 94
    {'X_train':[7, 7], 'y_train':[7, 7], 'X_test':[7], 'y_test':[7]},  # 95
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 96
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 97
    {'X_train':[0, 0, 0, 0], 'y_train':[
               0, 0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 98
    {'X_train':[0, 0, 0], 'y_train':[0, 0, 0], 'X_test':[0], 'y_test':[0]},  # 99
]
