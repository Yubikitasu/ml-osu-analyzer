class ExternalCalculator:
    NM = 0
    EZ = 2
    HR = 16
    DT = 64
    HT = 256
    NC = 512

    def __init__(self, base_cs: float, base_od: float, mods: int = 0):
        self.base_cs = base_cs
        self.base_od = base_od
        self.mods = mods

    def get_cs(self) -> float:
        cs = self.base_cs

        if self.mods & self.HR:
            cs = min(cs * 1.3, 10.0)
        elif self.mods & self.EZ:
            cs *= 0.5

        return cs

    def get_od(self) -> float:
        od = self.base_od

        # Apply HR / EZ
        if self.mods & self.HR:
            od = min(od * 1.4, 10.0)
        elif self.mods & self.EZ:
            od *= 0.5

        # Convert OD to 300 hit window
        hit_window = 80.0 - (6.0 * od)

        # Apply DT / NC / HT
        if self.mods & (self.DT | self.NC):
            speed_multiplier = 1.5
        elif self.mods & self.HT:
            speed_multiplier = 0.75
        else:
            speed_multiplier = 1.0

        final_window = hit_window / speed_multiplier

        # Convert back to effective OD
        return (80.0 - final_window) / 6.0

    def calculate(self) -> dict:
        return {
            "CS": round(self.get_cs(), 2),
            "OD": round(self.get_od(), 2)
        }