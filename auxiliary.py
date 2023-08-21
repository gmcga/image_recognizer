

def play_sound():
    import winsound
    duration = 350 #ms
    frequency = 440
    for _ in range(4):
        winsound.Beep(frequency, duration)


def main():
    play_sound()


if __name__ == "__main__":
    main()