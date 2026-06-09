from pulse2percept.implants import ElectrodeArray, DiskElectrode
import collections as coll
import numpy as np

class iBionicsElectrodeArray(ElectrodeArray):

    def __init__(self, electrodesPositions):
        """Electrodes from iBionics project

        Electrodes will be named 'A0', 'A1', ...

        Parameters
        ----------

        """
        # The job of the constructor is to create the electrodes. We start
        # with an empty collection:
        self._electrodes = coll.OrderedDict()
        self.electrodes_positions = electrodesPositions[:-1,:]
        self.electrodes_positions = sorted(self.electrodes_positions, key=lambda e: (e[1], e[0]))
        self.shape = np.array([16,18])
        #print(self.electrodes_positions)

        # We then generate a number `n_electrodes` of electrodes:
        for n,c in enumerate(self.electrodes_positions, 0):
            # Create the disk electrode:
            electrode = DiskElectrode(c[0]*20.0 ,c[1]*20.0 , 0, 25)
            # Add the electrode to the collection:
            self.add_electrode('A' + str(n), electrode)
