from gym_idsgame.envs.snort_reader import snort_alert_reader

def test():
    conf = snort_alert_reader.Config("reader.ini")
    reader = snort_alert_reader.LogReader(conf)
    reader.start_tail()

if __name__ == "__main__":
    #main()
    test()

