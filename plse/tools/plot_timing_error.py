import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import norm
from scipy.optimize import curve_fit

from plse.data import DataLoader


def gaussian_func(x, a, x0, sigma): 
    return a*np.exp(-(x-x0)**2/(2*sigma**2)) 


def plot_2d_true_vs_reco(true_times,output_times,output_dir):
    # True vs. reco
    finite_mask = np.isfinite(true_times[:,0])
    fig,ax = plt.subplots(figsize=(5,5))
    ax.hist2d(true_times[:,0][finite_mask],output_times[:,0][finite_mask],bins=np.linspace(0,30,101))
    ax.plot([0,0],[100,100],color='k',alpha=0.5)
    ax.set_aspect(1)
    ax.set_xlabel('True Times [ns]')
    ax.set_ylabel('Predicted Times [ns]')
    plt.savefig(output_dir+"timing_2dhist_true_vs_reco_1st.png")
    plt.close()


def plot_mpe_timing_error(true_times,output_times,output_dir):

    # Calculate the errors
    timing_error = true_times-output_times

    # Plot timing errors of all PE sorted by num of true PE
    fig,ax = plt.subplots(2,5,figsize=(12,5))
    ax = ax.flatten()
    for n in range(1,11):
        mask = np.sum(true_times!=-999,axis=1)==n
        finite_mask = np.isfinite(timing_error[mask])
        yy,xx,_ = ax[n-1].hist(timing_error[mask][finite_mask],bins=np.linspace(-20,20,101))

        # Add gaussian fit
        xx_centers = (xx[:-1]+xx[1:])/2.
        popt, pcov = curve_fit(gaussian_func, xx_centers, yy)
        ym = gaussian_func(xx, popt[0], popt[1], np.abs(popt[2]))
        ax[n-1].plot(xx, ym, c='k')

        ax[n-1].set_title('True %d PE waveforms \n'%n+r'$\mu$=%.2f, $\sigma$=%.2f'%(popt[1],np.abs(popt[2])),fontsize=10)
        ax[n-1].set_xlabel('Timing error [ns]')

    plt.suptitle('Timing resolution for all photons in waveform')
    plt.tight_layout()
    plt.savefig(output_dir+"timing_error_all_by_npe.png")
    plt.close()

    # Plot timing errors of 1st PE ("Leading edge") sorted by num of true PE
    fig,ax = plt.subplots(2,5,figsize=(12,5))
    ax = ax.flatten()
    for n in range(1,11):
        mask = np.sum(true_times!=-999,axis=1)==n
        finite_mask = np.isfinite(timing_error[:,0][mask])
        yy,xx,_ = ax[n-1].hist(timing_error[:,0][mask][finite_mask],bins=np.linspace(-20,20,101))

        # Add gaussian fit
        xx_centers = (xx[:-1]+xx[1:])/2.
        popt, pcov = curve_fit(gaussian_func, xx_centers, yy)
        ym = gaussian_func(xx, popt[0], popt[1], np.abs(popt[2]))
        ax[n-1].plot(xx, ym, c='k')

        ax[n-1].set_title('True %d PE waveforms \n'%n+r'$\mu$=%.2f, $\sigma$=%.2f'%(popt[1],np.abs(popt[2])),fontsize=10)
        ax[n-1].set_xlabel('Timing error [ns]')

    plt.suptitle('Timing resolution for 1st photon')
    plt.tight_layout()
    plt.savefig(output_dir+"timing_error_1st_by_npe.png")
    plt.close()


def get_xy_to_plot(xs, ys):
    # This turns a list of x and y values into a flat "step" for each waveform sample bin
    xs_to_plot = np.array(list(zip(xs[:-1], xs[1:]))).flatten()
    ys_to_plot = np.array(list(zip(ys, ys))).flatten()
    return xs_to_plot, ys_to_plot


def convert_times_to_waveform_bin(t):
    # 2 nanoseconds per bin
    # 30 is the offset to define where t=0 is
    return 30+t/2


def plot_visualization(waveforms=None, true_hit_times=None, reco_hit_times=None, 
                    n_waveforms_to_plot=20, n_start=0, vertical_spacing=100, output_dir=''):
    # The vertical_spacing param defines the gap between waveforms. If there's a really large waveform, you may need to make this larger so it doesn't obscure the one below it.
    # The "if i==10 else None" in labels is so that the legend is only populated once instead of for every waveform. Picking 10 was arbitrary.
    # When we save the waveforms, we pad timing array with -999, so we have to remove those before plotting. For example, a 2PE waveforms could have times [0.1,2.3,-999,-999,-999,etc.]
    plt.figure(figsize=(6, 8))
    for i in range(n_start,n_start+n_waveforms_to_plot):
        if waveforms is not None:
            xx = np.array(range(len(waveforms[i])+1))
            yy = waveforms[i]
            x, y = get_xy_to_plot(xx, yy+i*vertical_spacing)
            plt.plot(x,y,label='Waveform' if i==10 else None)
            plt.xlabel("waveform sample bin")
        if true_hit_times is not None:
            mask = true_hit_times[i]!=-999
            n_hits = np.sum(mask)
            plt.scatter(convert_times_to_waveform_bin(true_hit_times[i][mask]),i*np.ones(n_hits)*vertical_spacing,color='k',marker='x',label='Truth' if i==10 else None)
            # List the number of true PE on the left side of the plot for each waveform
            plt.text(0,i*vertical_spacing+8,'%d PE'%n_hits,fontsize='small')
        if reco_hit_times is not None:
            mask = true_hit_times[i]!=-999
            n_hits = np.sum(mask)
            plt.scatter(convert_times_to_waveform_bin(reco_hit_times[i][mask]),i*np.ones(n_hits)*vertical_spacing,alpha=0.5,label='Reco' if i==10 else None)
    plt.legend()
    # The y-scale doesn't matter for visualizing so just remove it.
    plt.yticks([])
    plt.savefig(output_dir+"timing_error_visualization.png")
    plt.close()


def plot_timing_error(input_files, prediction_file, output_dir):

    print("Loading truth and predictions...")

    # Load truth data
    dataloader = DataLoader(input_files)
    waveforms, encoded_npes, true_times = dataloader.load_good_data()

    # Load predictions
    output_times = np.load(prediction_file,'r')*100.

    # Make plots
    plot_2d_true_vs_reco(true_times,output_times,output_dir)
    plot_mpe_timing_error(true_times,output_times,output_dir)
    plot_visualization(waveforms=waveforms, true_hit_times=true_times, reco_hit_times=output_times, n_waveforms_to_plot=16, vertical_spacing=50, output_dir=output_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_files',
                        help='Type = String; Input locations of training set files, e.g. $PWD/{1..16}.root', nargs="+",
                        required=True)
    parser.add_argument('-p', '--prediction_file',
                        help='Type = String; Input locations of predicted set files, e.g. evaluated_output.npy',
                        required=True)
    parser.add_argument('-o', '--output_dir',
                        help='Type = String; Output location for plots, e.g. myplots',
                        required=True)
    args = parser.parse_args()

    plot_timing_error(args.input_files, args.prediction_file, args.output_dir)
