import seisbench.data as sbd

iquique = sbd.Iquique()

waveform = iquique.get_waveforms(int(1))

print(waveform)