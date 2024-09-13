import numpy as np
import ROOT
import rat
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def main(input_files, output_file):
    """Main function to process input files and save the data to a compressed NPZ file."""
    total_waveforms, total_nphotons, total_hittimes, total_event_id = [], [], [], []
    total_event_info_regular = []  # For regular PMT ID mapping

    for file in input_files:
        waveforms, hittimes, nphotons, event_id, event_info_regular = process_file(file)
        logging.info(f"Processed file: {file}")
        total_waveforms.append(waveforms)
        total_nphotons.append(nphotons)
        total_hittimes.append(hittimes)
        total_event_id.append(event_id)
        total_event_info_regular.append(event_info_regular)

    # Stack the data
    all_waveforms = np.vstack(total_waveforms)
    all_nphotons = np.hstack(total_nphotons)
    all_hittimes = np.vstack(total_hittimes)
    all_event_id = np.vstack(total_event_id)
    all_event_info_regular = np.vstack(total_event_info_regular)

    np.savez_compressed(output_file,
                        waveforms=all_waveforms,
                        nphotons=all_nphotons,
                        hittimes=all_hittimes,
                        eventid=all_event_id,
                        event_info_regular=all_event_info_regular)

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

    all_hit_times, all_nphotons, all_evt_info, allevtinfo_regular = get_truth_info(input_filename, good_events)

    return (
        np.array(all_waveforms, dtype=np.float32),
        np.array(all_hit_times, dtype=np.float32),
        np.array(all_nphotons, dtype=np.int16),
        np.array(all_evt_info, dtype=np.int32),
        np.array(allevtinfo_regular, dtype=np.int32)  # Event info for regular PMTs
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
    """Process waveforms for a single event and sort by PMT ID."""

    # Loop over all PMT IDs in the event, sorted in ascending order
    for iPMT in sorted(ev.GetAllPMTIDs()):
        pmt = ev.GetOrCreatePMT(iPMT)
        pmtID = pmt.GetID()
        waveform = digitizer.GetWaveform(pmtID)

        # Vectorize the waveform calculation
        waveform_array = np.array(waveform, dtype=np.int32)
        dynamic_range = digitizer.GetDynamicRange()
        nbits = digitizer.GetNBits()
        voltage_res = dynamic_range / (1 << nbits)
        hwaveform = waveform_array * voltage_res - 1800  # TODO: fix 1800 so it's not set manually

        # Append the processed waveform to the all_waveforms list
        all_waveforms.append(hwaveform)


def get_truth_info(input_filename, good_events):
    """Extract truth information for events, including regular PMTs."""
    f = ROOT.TFile(input_filename, 'read')
    T = f.Get('T')
    ds = rat.RAT.DS.Root()
    T.SetBranchAddress('ds', ROOT.AddressOf(ds))

    all_hit_times, all_nphotons, allevtinfo, allevtinfo_regular = [], [], [], []

    for i_event in good_events:
        T.GetEvent(i_event)
        ev = ds.GetEV(0)
        mc = ds.GetMC()
        nhitpmts = ev.GetPMTCount()
        trigger_time = ev.GetCalibratedTriggerTime()

        # Store PMT information and photon times together for MC PMTs
        pmt_info_list = []  # To store (PMT ID, photon times, event info) for sorting later

        for i_pmt in range(nhitpmts):
            nphotons = mc.GetMCPMT(i_pmt).GetMCPhotonCount()
            pmt_id = mc.GetMCPMT(i_pmt).GetID()
            allevtinfo_entry = [i_event, pmt_id]  # Store event and PMT ID

            # Collect valid photon times for this PMT
            photon_times = [
                mc.GetMCPMT(i_pmt).GetMCPhoton(i_MCPhoton).GetFrontEndTime() - trigger_time + 60
                for i_MCPhoton in range(nphotons)
                if True
                # 0 < (mc.GetMCPMT(i_pmt).GetMCPhoton(i_MCPhoton).GetFrontEndTime() - trigger_time + 60) < 200
            ]

            # Only keep the first 100 sorted photon times
            photon_times = np.sort(photon_times)[:100]

            # Pad the array to length 100 with -999
            photon_times_padded = np.pad(photon_times, (0, max(0, 100 - len(photon_times))), constant_values=-999)
            all_nphotons.append(nphotons)
            # Append the PMT ID, photon times, and allevtinfo entry for sorting later
            pmt_info_list.append((pmt_id, photon_times_padded, allevtinfo_entry))

        # Sort the list by PMT ID (first element in the tuple)
        pmt_info_list_sorted = sorted(pmt_info_list, key=lambda x: x[0])

        # Append the sorted photon times and allevtinfo entries
        all_hit_times.extend([photon_times_padded for _, photon_times_padded, _ in pmt_info_list_sorted])
        allevtinfo_sorted = [allevtinfo_entry for _, _, allevtinfo_entry in pmt_info_list_sorted]

        # Overwrite allevtinfo with the sorted version
        allevtinfo.extend(allevtinfo_sorted)

        # Generate ID mapping for the regular PMTs (non-MC) and sort by PMT ID
        regular_pmt_info_list = []
        for i_pmt in sorted(ev.GetAllPMTIDs()):
            allevtinfo_regular_entry = [i_event, i_pmt]  # Store event and PMT ID for regular PMT
            regular_pmt_info_list.append(allevtinfo_regular_entry)

        # Sort and append for regular PMTs
        allevtinfo_regular_sorted = sorted(regular_pmt_info_list, key=lambda x: x[1])  # Sort by PMT ID
        allevtinfo_regular.extend(allevtinfo_regular_sorted)

        # Log progress
        if i_event % 500 == 0:
            logging.info(f'Processed event: {i_event}')

    return all_hit_times, all_nphotons, allevtinfo, allevtinfo_regular


if __name__ == "__main__":
    args = get_args()
    main(args.input_files, args.output_file)
