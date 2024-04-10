import numpy as np

class ParseResult:
    def __init__(self):
        self.epsilons = []
        self.progname = ""
        self.maximum_frame_errors = 30
        self.maximum_decoded_words = 45000000
        self.print_help = False

def print_help(progname):
    print(
        "Neural Belief Propagation Using Overcomplete Check Matrices\n"
        "\n"
        "Usage:\n"
        f"{progname} [options] [Epsilon] [Epsilon]...\n"
        "\n"
        "Positional arguments:\n"
        "  Epsilon                     optionally specify a list of epsilons at which to verify.\n"
        "                              If no epsilons are given, use default list of epsilons \n"
        "\n"
        "Options:\n"
        "  -r|--range START STEP STOP  Specify a decreasing range of epsilons to use, e.g.,\n"
        "                              \"-r 0.1 0.025 0.9\" to get 0.1 0.975 0.95 0.925 0.9\n"
        "\n"
        "  -e|--max-frame-errors N     Stop simulating after observing N frame errors\n"
        "  -w|--max-decoded-words N    Stop simulating after decoding N words\n"
        "\n\n\n"
        " This program accompanies the paper\n"
        "     S. Miao, A. Schnerring, H. Li and L. Schmalen,\n"
        "     \"Neural belief propagation decoding of quantum LDPC codes using overcomplete check matrices,\"\n"
        "     Proc. IEEE Inform. Theory Workshop (ITW), Saint-Malo, France, Apr. 2023, "
        "https://arxiv.org/abs/2212.10245\n"
    )

def parse_arguments(argument_count, arguments, default_epsilons, maximum_frame_errors=30, maximum_decoded_words=45000000):
    result = ParseResult()
    result.maximum_frame_errors = maximum_frame_errors
    result.maximum_decoded_words = maximum_decoded_words
    result.progname = arguments[0]
    arguments = arguments[1:]
    argument_count -= 1

    if not argument_count:  # no arguments passed – use default epsilons
        result.epsilons = default_epsilons
        return result  # return empty list

    while argument_count:
        argument = arguments[0]
        argument_view = argument

        if argument_view in ("-h", "--help"):
            result.print_help = True
            return result
        if argument_view in ("-r", "--range"):
            if argument_count < 4:  # need three more arguments
                result.print_help = True
                return result
            start = float(arguments[1])
            step = float(arguments[2])
            stop = float(arguments[3])
            argument_count -= 3
            try:
                if stop <= 0.0:
                    raise ValueError("Stop value needs to be positive")
                if step <= 0.0:
                    raise ValueError("step value needs to be positive")
                if start <= stop:
                    raise ValueError("Stop value needs to be smaller than start value")
                result.epsilons = list(np.arange(start, stop, step)[::-1])
            except ValueError as exceptions:
                print(f"Could not parse range: {exceptions}; skipping range.")
        elif argument_view in ("-e", "--max-frame-errors"):
            argument_count -= 1
            result.maximum_frame_errors = int(arguments[1])
        elif argument_view in ("-w", "--max-decoded-words"):
            argument_count -= 1
            result.maximum_decoded_words = int(arguments[1])
        else:
            # we encountered something that doesn't look like a keyword argument -- proceed to parse numbers
            break
        arguments = arguments[1:]
        argument_count -= 1

    if not argument_count and not result.epsilons:  # no arguments passed, and no range found – use default epsilons
        result.epsilons = default_epsilons
        return result

    result.epsilons = []
    for argument in arguments:
        value = float(argument)
        if value <= 0.0:  # couldn't parse number, or actually 0.0 (or below) was passed
            print(f"Ignoring argument '{argument}' as unparseable or non-positive.")
            continue  # skip this
        result.epsilons.append(value)
    return result
