class windows:
    _windows = []

    def __init__(self):
        self._windows=[]


    def set_windows(self, windows):
        self._windows.extend(windows)


    def get_windows(self):
        return self._windows[-8:]

    def del_windows(self):
        self._windows=[]









