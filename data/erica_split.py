def get_jackrabbot_split():
    train = ['bytes-cafe-2019-02-07_0',
             'gates-ai-lab-2019-02-08_0',
             'gates-basement-elevators-2019-01-17_1',
             'hewlett-packard-intersection-2019-01-24_0',
             'huang-lane-2019-02-12_0',
             'jordan-hall-2019-04-22_0',
             'packard-poster-session-2019-03-20_1',
             'packard-poster-session-2019-03-20_2']


    val = ['stlc-111-2019-04-19_0',
           'svl-meeting-gates-2-2019-04-08_0',
           'svl-meeting-gates-2-2019-04-08_1',  # impercitible slight rotation
           'tressider-2019-03-16_0',
           'tressider-2019-03-16_1']

    test = ['stlc-111-2019-04-19_0',
            'svl-meeting-gates-2-2019-04-08_0',
            'svl-meeting-gates-2-2019-04-08_1',  # impercitible slight rotation
            'tressider-2019-03-16_0',
            'tressider-2019-03-16_1']

    return train, val, test
