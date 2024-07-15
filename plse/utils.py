import os

def verify_output(output_dir, filename, overwrite=False):
    '''Verify that the output directory exists and that the output file won't be overwritten'''
    # Convert relative/symbolic path to an absolute path
    output_dir = os.path.realpath(output_dir)

    # If the output directory doesn't exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Construct the full output file path
    output_file = os.path.join(output_dir, filename)

    # Check if the file already exists
    if os.path.exists(output_file) and not overwrite:
        assert False, "Output file (%s) already exists. Use 'overwrite=True' to overwrite it."%output_file
    else:
        # File is able to be saved
        return output_dir, output_file
