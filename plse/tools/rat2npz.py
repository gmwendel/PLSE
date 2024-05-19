import numpy as np
import ROOT
import rat
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(input_files, output_file):
    """Main function to process input files and save the data to a compressed NPZ file."""
    total_waveforms, total_nphotons, total_hittimes, total_event_id = [], [], [], []

    for file in input_files:
        waveforms, hittimes, nphotons, event_id = process_file(file)
        logging.info(f"Processed file: {file}")
        total_waveforms.append(waveforms)
        total_nphotons.append(nphotons)
        total_hittimes.append(hittimes)
        total_event_id.append(event_id)

    all_waveforms = np.vstack(total_waveforms)
    all_nphotons = np.hstack(total_nphotons)
    all_hittimes = np.vstack(total_hittimes)
    all_event_id = np.vstack(total_event_id)

    np.savez_compressed(output_file, waveforms=all_waveforms, nphotons=all_nphotons, hittimes=all_hittimes, eventid=all_event_id)
    logging.info(f"Saved {len(all_waveforms)} waveforms to: {output_file}")

def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert RAT ROOT files to NPZ files for training or evaluation.")
    parser.add_argument('-i', '--input_files', help='Input RAT files to be converted', nargs='+', required=True)
    parser.add_argument('-o', '--output_file', help='Output NPZ file with waveform information', required=True)
    return parser.parse_args()

def fast_dsreader(filename):
    """Generator to read RAT DS events from a file."""
    r = ROOT.RAT.DSReader(filename)
    ds = r.NextEvent()
    while ds:
        yield ROOT.RAT.DS.Root(ds)
        ds = r.NextEvent()

def process_file(input_filename):
    """Process a single RAT ROOT file and return waveform, hit times, photon counts, and event IDs."""
    dsreader = fast_dsreader(input_filename)
    all_waveforms = []
    good_events = []

    for i, event in enumerate(dsreader):
        try:
            ev, digitizer, sampling_rate, time_step = get_digitizer_info(event)
            process_event_waveforms(ev, digitizer, sampling_rate, time_step, all_waveforms)
            good_events.append(i)
        except Exception as e:
            logging.error(f"Error on event {i}: {e}")
            continue

    all_hit_times, all_nphotons, all_evt_info = get_truth_info(input_filename, good_events)

    return (
        np.array(all_waveforms, dtype=np.float32),
        np.array(all_hit_times, dtype=np.float32),
        np.array(all_nphotons, dtype=np.int16),
        np.array(all_evt_info, dtype=np.int32)
    )

def get_digitizer_info(event):
    """Extract digitizer information from an event."""
    ev = event.GetEV(0)
    digitizer = ev.GetDigitizer()
    dynamic_range = digitizer.GetDynamicRange()
    nbits = digitizer.GetNBits()
    sampling_rate = digitizer.GetSamplingRate()
    time_step = 1.0 / sampling_rate
    return ev, digitizer, sampling_rate, time_step

def process_event_waveforms(ev, digitizer, sampling_rate, time_step, all_waveforms):
    """Process waveforms for a single event."""
    for iPMT in range(ev.GetPMTCount()):
        pmt = ev.GetPMT(iPMT)
        pmtID = pmt.GetID()
        waveform = digitizer.GetWaveform(pmtID)

        # Vectorize the waveform calculation
        waveform_array = np.array(waveform, dtype=np.int32)
        dynamic_range = digitizer.GetDynamicRange()
        nbits = digitizer.GetNBits()
        voltage_res = dynamic_range / (1 << nbits)
        hwaveform = waveform_array * voltage_res - 1800

        all_waveforms.append(hwaveform)

def get_truth_info(input_filename, good_events):
    """Extract truth information for events."""
    f = ROOT.TFile(input_filename, 'read')
    T = f.Get('T')
    ds = rat.RAT.DS.Root()
    T.SetBranchAddress('ds', ROOT.AddressOf(ds))

    all_hit_times, all_nphotons, allevtinfo = [], [], []
    for i_event in good_events:
        T.GetEvent(i_event)
        ev = ds.GetEV(0)
        mc = ds.GetMC()
        nhitpmts = ev.GetPMTCount()

        for i_pmt in range(nhitpmts):
            photon_times_on_this_pmt = np.full(100, -999, dtype=np.float32)
            nphotons = mc.GetMCPMT(i_pmt).GetMCPhotonCount()
            all_nphotons.append(nphotons)
            allevtinfo.append([i_event, i_pmt])

            for i_MCPhoton in range(min(nphotons, 100)):
                MCPhoton = mc.GetMCPMT(i_pmt).GetMCPhoton(i_MCPhoton)
                photon_times_on_this_pmt[i_MCPhoton] = MCPhoton.GetFrontEndTime()

            all_hit_times.append(photon_times_on_this_pmt)

        if i_event % 500 == 0:
            logging.info(f'Processed event: {i_event}')

    return all_hit_times, all_nphotons, allevtinfo

if __name__ == "__main__":
    args = get_args()
    main(args.input_files, args.output_file)
