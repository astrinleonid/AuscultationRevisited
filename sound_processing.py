import wave

def combine_wav_files(output_file, input_files):
    # Ensure there is at least one input file
    if not input_files:
        print("No input files provided.")
        return

    # Open the first file to set parameters
    try:
        with wave.open(input_files[0], 'rb') as infile:
            params = infile.getparams()
    except wave.Error as e:
        print(f"Failed to read parameters from {input_files[0]}: {e}")
        return

    print(params)

    # Open the output file with the parameters from the first file
    with wave.open(output_file, 'wb') as outfile:
        outfile.setparams(params)

        # Process each file
        for file in input_files:
            try:
                with wave.open(file, 'rb') as infile:
                    while True:
                        data = infile.readframes(1024)
                        if not data:
                            break
                        outfile.writeframes(data)
            except wave.Error as e:
                print(f"Error reading {file}: {e}")


