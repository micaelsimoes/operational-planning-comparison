# ============================================================================================
#   Class Load
# ============================================================================================
class Load:

    def __init__(self):
        self.load_id = -1                       # Load ID
        self.bus = -1                           # bus number
        self.pd = 0.00                          # Active power consumption
        self.qd = 0.00                          # Reactive power consumption
        self.status = 1                         # status:
                                                #   1 - machine in service
                                                #   0 - machine out of service
        self.fl_reg = False                     # Indicates if the load is flexible
        self.flexibility = LoadFlexibility()    # Flexibility structure


# ======================================================================================================================
#   Class LOAD FLEXIBILITY
# ======================================================================================================================
class LoadFlexibility:

    def __init__(self):
        self.upward = list()                # Note: FL - increase consumption
        self.downward = list()
