



class Hash_Wrapper:
    
    def __init__(self, func: function, resize_height: int=-1, resize_width: int=-1, available_devices: list[str]=["cpu"]):
        self.func = func
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.available_devices = available_devices


    def get_info(self):
        return self.func, self.resize_height, self.resize_width, self.available_devices