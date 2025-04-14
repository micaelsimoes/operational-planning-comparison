from definitions import *


# ============================================================================================
#   Class Generator
# ============================================================================================
class Generator:

    def __init__(self):
        self.gen_id = -1                    # Generator ID
        self.bus = -1                       # bus number
        self.pmax = 0.0                     # Pmax, maximum real power output (MW)
        self.pmin = 0.0                     # Pmin, minimum real power output (MW)
        self.qmax = 0.0                     # Qmax, maximum reactive power output (MVAr)
        self.qmin = 0.0                     # Qmin, minimum reactive power output (MVAr)
        self.vg = 0.0                       # Vg, voltage magnitude setpoint (p.u.)
        self.status = 0                     # status:
                                            #   1 - machine in service
                                            #   0 - machine out of service
        self.pre_processed = False
        self.gen_type = GEN_CONV
        self.power_factor_control = False   # Power factor control
        self.max_pf = 0.80                  # - Maximum power factor
        self.min_pf = -0.80                 # - Minimum power factor

    def is_controllable(self):
        if self.gen_type in GEN_CONTROLLABLE_TYPES:
            return True
        return False

    def is_curtaillable(self):
        if self.gen_type in GEN_CURTAILLABLE_TYPES:
            return True
        return False

    def is_renewable(self):
        if self.gen_type in GEN_RENEWABLE_TYPES:
            return True
        return False
