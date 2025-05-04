from .PDQ import PDQHasher


class Hash_Wrapper:
    
    def __init__(self, name: str, func, colormode, resize_height: int=-1, resize_width: int=-1, available_devices: set[str]={"cpu"}):
        self.name = name
        self.func = func
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.available_devices = available_devices
        self.colormode = colormode


    def get_name(self):
        return self.name

    def get_info(self):
        return self.func, self.resize_height, self.resize_width, self.available_devices, self.colormode
    


class PDQ_Wrapper(Hash_Wrapper):
    
    def __init__(self, name: str, func, colormode, resize_height: int=-1, resize_width: int=-1, available_devices: set[str]={"cpu"}):
        self.PDQ = PDQHasher()
        self.func = self.generate_pdq
        self.name = name
        self.colormode = colormode,
        self.resize_height = resize_height
        self.resize_width = resize_width
        self.available_devices = available_devices



    def generate_pdq(self, tensor): #[1, H, W] -> [64]
        return 