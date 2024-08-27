import numpy as np

def read_ppm(filename):
    ## Reads file `filename`, expecting the relative path to a correct
    ## PPM file.
    lines = str()
    with open(filename, 'r') as f:
        lines = f.readlines()

    no_comments_filter = lambda s: not ('#' in s)
    lines = " ".join(filter(no_comments_filter, lines))

    no_empty_filter = lambda s: len(s)

    f_split = lines.split()

    return f_split


def ppm_to_array(ppm_file):
    ## Reads file `filename`, expecting the relative path to a correct
    ## PPM file. Returns a `numpy.array`
    entries = read_ppm(ppm_file)
    assert len(entries) > 3, "PPM file invalid/incomplete."
    assert entries[0] == 'P3', "'Raw' PPM files not supported, only 'plain'."

    W = int(entries[1])
    H = int(entries[2])

    # Note: Always use `zeros`, not `empty`, as `empty` does not clear
    # the memory, and it needs to be overriden or it will lead to
    # bugs.
    arr = np.zeros((H,W), dtype=np.int32) 

    for i in range(H*W*3):
        arr_i = i // 3 # Actual index (without header offset)
        shift_v = 16 - (i % 3) * 8 # Bit-Shift to join the whole pixel in one number
        val = int(entries[i+4]) << shift_v # `i+4` to offset the header

        y_i = arr_i // W
        x_i = arr_i % W
        arr[y_i, x_i] += val


    return arr


    
