import numpy as np
import rat
import ROOT as ROOT


###USED TO CONVERT RAT ROOT FILES TO NPZ FILES FOR TRAINING OR EVALUATION###

def main():
    args = get_args()
    total_waveforms, total_nphotons, total_hittimes, total_event_id = [], [], [], []
    for file in args.input_files:

        waveforms, hittimes, nphotons, event_id = process_file(file)
        print("Processed file: " + file)
        total_waveforms += [waveforms]
        total_nphotons += [nphotons]
        total_hittimes += [hittimes]
        total_event_id += [event_id]

    all_waveforms = np.vstack(total_waveforms)
    all_nphotons = np.hstack(total_nphotons)
    all_hittimes = np.vstack(total_hittimes)
    all_event_id = np.vstack(total_event_id)

    np.savez_compressed(args.output_file, waveforms=all_waveforms, nphotons=all_nphotons, hittimes=all_hittimes, eventid=all_event_id)
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
    good_events = []
    ##  Loop over all triggered events
    for i, event in enumerate(dsreader):
        try:
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
                    voltage = adc * voltage_res - 1800
                    hwaveform.append(voltage)

                ## Save the waveform to list
                all_waveforms.append(hwaveform)
            good_events.append(i)

        except:
            print("Error on event: ", i)
            continue

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
    allevtinfo = []
    for i_event in good_events:
        T.GetEvent(i_event)
        ev = ds.GetEV(0)
        mc = ds.GetMC()
        nhitpmts = ev.GetPMTCount()
        for i_pmt in range(nhitpmts):
            # Pre-allocate the array with zeros
            photon_times_on_this_pmt = np.zeros(250, dtype=np.float32)

            nphotons = mc.GetMCPMT(i_pmt).GetMCPhotonCount()
            allNPhotons.append(nphotons)
            allevtinfo.append([i_event, i_pmt])

            # Fill in the actual hit times
            for i_MCPhoton in range(min(nphotons, 250)):
                MCPhoton = mc.GetMCPMT(i_pmt).GetMCPhoton(i_MCPhoton)
                HitTime = MCPhoton.GetFrontEndTime()
                photon_times_on_this_pmt[i_MCPhoton] = HitTime

            allHitTimes.append(photon_times_on_this_pmt)

        if i_event % 500 == 0:
            print('processed event: ', i_event)

    all_waveforms = np.array(all_waveforms, dtype=np.float32)
    #print(allHitTimes)
    allHitTimes = np.array(allHitTimes, dtype=np.float32)
    allNPhotons = np.array(allNPhotons, dtype=np.int16)
    allevtinfo = np.array(allevtinfo, dtype=np.int32)

    return all_waveforms, allHitTimes, allNPhotons, allevtinfo


if __name__ == "__main__":
    main()
