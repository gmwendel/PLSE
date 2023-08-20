import numpy as np
import rat
import ROOT as ROOT

###USED TO CONVERT RAT ROOT FILES TO NPZ FILES FOR TRAINING OR EVALUATION###

def main():
    args = get_args()
    total_waveforms, total_nphotons, total_hittimes = [], [], []
    for file in args.input_files:
        try:
            waveforms, hittimes, nphotons = process_file(file)
            print("Processed file: " + file)
            total_waveforms += [waveforms]
            total_nphotons += [nphotons]
            total_hittimes += [hittimes]
        except:
            print("Failed to process file: " + file)
            continue

    all_waveforms = np.vstack(total_waveforms)
    all_nphotons = np.hstack(total_nphotons)
    all_hittimes = np.vstack(total_hittimes)

    np.savez_compressed(args.output_file, waveforms=all_waveforms, nphotons=all_nphotons, hittimes=all_hittimes)
    print("Saved ", len(all_waveforms), "to: " + args.output_file)

def get_args():
    import argparse
    # Get command line inputs
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files',
                        help='Type = String; Location of rat files to be converted, e.g. $PWD/waveforms{1..10}.root',
                        nargs="+",
                        required=True
                        )
    parser.add_argument('-o', '--output_file',
                        help='Type = String;  Specify npz output file with waveform information; e.g. $PWD/waveforms.npz',
                        nargs=None,
                        required=True
                        )

    return parser.parse_args()


def fast_dsreader(filename):
    r = ROOT.RAT.DSReader(filename)
    ds = r.NextEvent()
    while ds:
        yield ROOT.RAT.DS.Root(ds)
        ds = r.NextEvent()


def process_file(input_filename):
    ## Input file
    dsreader = fast_dsreader(input_filename)

    allHitTimes, allFrontTimes, allNoiseFlags = [], [], []

    all_waveforms = []

    ##  Loop over all triggered events
    for i, event in enumerate(dsreader):

        ## Read in the event and get the digitizer information
        ev = event.GetEV(0)

        digitizer = ev.GetDigitizer()

        ## Get the digitizer dynamic range and the number of bits
        dynamic_range = digitizer.GetDynamicRange()  ## in mV
        nbits = digitizer.GetNBits()
        voltage_res = dynamic_range / (1 << nbits);

        ## Digitizer sampling rate and number of samples
        sampling_rate = digitizer.GetSamplingRate()
        time_step = 1.0 / sampling_rate
        nsamples = digitizer.GetNSamples()

        # print("Time Step: ",time_step)

        ## Loop over the hit PMTs
        for iPMT in range(ev.GetPMTCount()):

            ## Get the waveform for each PMT
            pmt = ev.GetPMT(iPMT)
            pmtID = pmt.GetID()
            waveform = digitizer.GetWaveform(pmtID);
            hwaveform = []

            for sample in range(waveform.size()):
                time = sample * time_step
                ## The waveform contains the ADC counts for each sample
                adc = int(waveform[sample])
                ## Convert to voltage
                voltage = adc * voltage_res
                hwaveform.append(voltage)

            ## Save the waveform to list
            all_waveforms.append(hwaveform)

    ## Now get truth info
    f = ROOT.TFile(input_filename, 'read')

    T = f.Get('T')
    nevents = T.GetEntries()

    ds = rat.RAT.DS.Root()
    T.SetBranchAddress('ds', ROOT.AddressOf(ds))
    mc = ds.GetMC()

    ## Get true hit times for all events
    allHitTimes = []
    allNPhotons = []
    for i_event in range(0, nevents):
        T.GetEvent(i_event)
        ev = ds.GetEV(0)
        mc = ds.GetMC()
        nhitpmts = ev.GetPMTCount()
        for i_pmt in range(nhitpmts):
            photon_times_on_this_pmt = []
            nphotons = mc.GetMCPMT(i_pmt).GetMCPhotonCount()
            allNPhotons.append(nphotons)
            for i_MCPhoton in range(nphotons):
                MCPhoton = mc.GetMCPMT(i_pmt).GetMCPhoton(i_MCPhoton)
                HitTime = MCPhoton.GetHitTime()
                photon_times_on_this_pmt.append(HitTime)
            # Make them all the same length so we can save as np array
            while i_MCPhoton < 250:
                photon_times_on_this_pmt.append(0)
                i_MCPhoton += 1
            allHitTimes.append(photon_times_on_this_pmt)
        if i_event % 500 == 0:
            print('processed event: ', i_event)

    return np.array(all_waveforms, dtype=np.float32), np.array(allHitTimes, dtype=np.float32), np.array(allNPhotons,
                                                                                                        dtype=np.int16)


if __name__ == "__main__":
    main()
