

def play_sound():
    import winsound
    duration = 100 #ms
    frequency = 440
    for _ in range(5):
        winsound.Beep(frequency, duration)